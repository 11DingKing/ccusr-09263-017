"""容量需求预测：采样窗口、模型参数与可解释告警（纯领域层）。

预测只消费已沉淀的历史样本，不直接读取预约库：应用服务负责把历史
预约折算成等宽分桶样本后交给这里的纯函数计算。这样重新计算的结果
完全由“输入版本 + 模型参数”决定，便于复现与对账。

- :class:`ForecastWindow` 定义输入口径：采样起止（UTC）、桶宽与粒度；
- :class:`ModelParameters` 保存模型参数（桶宽、分位数、最小样本数）；
- :class:`ForecastWarning` 携带机器可读的告警级别与可解释说明；
- :func:`build_samples` 把历史观测折算成等宽桶样本；
- :func:`forecast` 仅依据窗口、参数与样本得出分位数预测。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Any

from .models import (
    Booking,
    BookingStatus,
    dt_from_str,
    dt_to_str,
)

#: 容量预测计入的历史预约状态：形成过确定容量承诺，排除候补、取消、过期
#: 与尚未报价锁定的申请（REQUESTED/QUOTED 仅为意向，未承诺容量）。
FORECAST_SAMPLE_STATUSES = frozenset(
    {
        BookingStatus.LOCKED,
        BookingStatus.SHIPPED,
        BookingStatus.CHECKED_IN,
        BookingStatus.SETTLED,
    }
)

SEAT_METRIC = "seats"
BOOKING_METRIC = "bookings"
METRICS = frozenset({SEAT_METRIC, BOOKING_METRIC})

GRANULARITY_DAY = "day"
GRANULARITY_WEEK = "week"
GRANULARITY_MONTH = "month"
GRANULARITIES = frozenset({GRANULARITY_DAY, GRANULARITY_WEEK, GRANULARITY_MONTH})

#: 桶宽下限：过细的桶会制造大量零样本，口径上不允许
_MIN_BUCKET_WIDTH_SECONDS = 3600


class WarningLevel(IntEnum):
    """告警级别：值越大越需要人工介入。"""

    INFO = 10  # 口径提示（如请求范围超出可用历史区间）
    WARNING = 20  # 结果可用但置信度受限（样本偏少/部分区间无数据）
    CRITICAL = 30  # 结果不可采信（完全没有历史数据）


@dataclass(frozen=True)
class ForecastWindow:
    """采样窗口（输入口径）：只采样 ``[start, end)`` 内开课的历史预约。"""

    start: datetime  # UTC，含
    end: datetime  # UTC，不含
    bucket_width_seconds: int
    granularity: str = GRANULARITY_DAY

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("forecast window bounds must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("forecast window end must be after start")
        if self.bucket_width_seconds < _MIN_BUCKET_WIDTH_SECONDS:
            raise ValueError(
                f"bucket_width_seconds must be >= {_MIN_BUCKET_WIDTH_SECONDS}"
            )
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {sorted(GRANULARITIES)}")

    @property
    def bucket_width(self) -> timedelta:
        return timedelta(seconds=self.bucket_width_seconds)

    def bucket_starts(self) -> list[datetime]:
        """窗口内全部桶起点（等宽，最后一个桶可能被窗口右端截断）。"""
        starts: list[datetime] = []
        cursor = self.start
        while cursor < self.end:
            starts.append(cursor)
            cursor += self.bucket_width
        return starts

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": dt_to_str(self.start),
            "end": dt_to_str(self.end),
            "bucket_width_seconds": self.bucket_width_seconds,
            "granularity": self.granularity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForecastWindow":
        return cls(
            start=dt_from_str(data["start"]),
            end=dt_from_str(data["end"]),
            bucket_width_seconds=int(data["bucket_width_seconds"]),
            granularity=data.get("granularity", GRANULARITY_DAY),
        )


@dataclass(frozen=True)
class ModelParameters:
    """模型参数：容量预测取历史桶样本的分位数。"""

    quantile: float = 0.8  # 规划分位数：默认按 P80 预留容量
    min_samples: int = 4  # 低于该样本数给出低置信度告警
    metric: str = SEAT_METRIC  # seats：席位数；bookings：预约场次数

    def __post_init__(self) -> None:
        if not isinstance(self.quantile, (int, float)) or isinstance(self.quantile, bool):
            raise ValueError("quantile must be a number")
        if not 0.0 < float(self.quantile) < 1.0:
            raise ValueError("quantile must be within (0, 1) exclusive")
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {sorted(METRICS)}")

    def to_dict(self) -> dict[str, Any]:
        return {"quantile": self.quantile, "min_samples": self.min_samples, "metric": self.metric}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelParameters":
        return cls(
            quantile=float(data.get("quantile", 0.8)),
            min_samples=int(data.get("min_samples", 4)),
            metric=data.get("metric", SEAT_METRIC),
        )


@dataclass(frozen=True)
class ForecastWarning:
    """可解释告警：级别 + 机器可读代码 + 面向规划会议的说明。"""

    level: WarningLevel
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": int(self.level),
            "level_name": self.level.name,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def canonical_inputs(
    window: ForecastWindow, params: ModelParameters, observations: list[datetime]
) -> str:
    """输入版本指纹：窗口 + 参数 + 全部被采样观测的开课时刻。

    重新计算只有指向同一指纹才视为同一输入版本。
    """
    payload = {
        "window": window.to_dict(),
        "parameters": params.to_dict(),
        "observations": sorted(dt_to_str(o.astimezone(timezone.utc)) for o in observations),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_samples(
    window: ForecastWindow,
    bookings: list[Booking],
    *,
    metric: str = SEAT_METRIC,
    now: datetime | None = None,
) -> tuple[list[int], list[datetime], int]:
    """把历史预约折算为窗口内等宽桶样本。

    只统计：开课时刻落在 ``[start, end)``、状态形成过确定容量承诺的预约。
    开课时刻晚于 ``now`` 的预约属于未来排期，不是历史观测，一律不计入。

    :returns: ``(samples, observation_times, excluded_future)``：
        ``samples`` 与 :meth:`ForecastWindow.bucket_starts` 对齐；
        ``observation_times`` 是被计入预约的开课时刻（用于输入指纹）；
        ``excluded_future`` 是因落在采样窗口之后/未来而被排除的预约数提示用。
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    cutoff = now.astimezone(timezone.utc) if now is not None else None
    starts = window.bucket_starts()
    totals = [0 for _ in starts]
    observation_times: list[datetime] = []
    excluded_future = 0
    for booking in bookings:
        if booking.status not in FORECAST_SAMPLE_STATUSES:
            continue
        slot_start = booking.slot_start.astimezone(timezone.utc)
        if not (window.start <= slot_start < window.end):
            continue
        if cutoff is not None and slot_start > cutoff:
            excluded_future += 1
            continue
        index = int((slot_start - window.start) // window.bucket_width)
        index = min(index, len(starts) - 1)
        totals[index] += 1 if metric == BOOKING_METRIC else booking.seats
        observation_times.append(slot_start)
    return totals, observation_times, excluded_future


def quantile_value(sorted_samples: list[int], quantile: float) -> float:
    """线性插值分位数（与 numpy 默认 'linear' 法一致），空样本抛出。"""
    if not sorted_samples:
        raise ValueError("cannot compute quantile of empty samples")
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    position = (len(sorted_samples) - 1) * quantile
    lower = int(position)
    fraction = position - lower
    upper = min(lower + 1, len(sorted_samples) - 1)
    return sorted_samples[lower] + (sorted_samples[upper] - sorted_samples[lower]) * fraction


def forecast(
    window: ForecastWindow,
    params: ModelParameters,
    samples: list[int],
    *,
    observed_buckets: int | None = None,
    excluded_future: int = 0,
) -> dict[str, Any]:
    """纯计算：由窗口、参数与桶样本得出预测值与告警。

    关键约定：**完全没有历史观测时不产出报告数值**，仅返回
    ``CRITICAL`` 告警；调用方据此跳过报告持久化（“不写空报告”）。
    """
    nonzero = [value for value in samples if value > 0]
    observed_buckets = len(nonzero) if observed_buckets is None else observed_buckets
    warnings: list[ForecastWarning] = []

    if observed_buckets == 0:
        warnings.append(
            ForecastWarning(
                WarningLevel.CRITICAL,
                "no_historical_data",
                "采样窗口内没有任何已形成容量承诺的历史预约，无法进行容量需求预测",
                details={
                    "window_start": dt_to_str(window.start),
                    "window_end": dt_to_str(window.end),
                    "suggestion": "扩大采样窗口或先沉淀历史排期数据后再试算",
                },
            )
        )
        return {
            "status": "insufficient_data",
            "value": None,
            "warnings": [w.to_dict() for w in warnings],
            "max_warning_level": int(WarningLevel.CRITICAL),
        }

    value = quantile_value(sorted(samples), params.quantile)
    if observed_buckets < params.min_samples:
        warnings.append(
            ForecastWarning(
                WarningLevel.WARNING,
                "insufficient_samples",
                f"有效样本仅 {observed_buckets} 个桶，少于模型最小样本数 {params.min_samples}，"
                "预测值仅供参考，不应直接作为容量规划依据",
                details={"observed_buckets": observed_buckets, "min_samples": params.min_samples},
            )
        )
    zero_buckets = len(samples) - observed_buckets
    if zero_buckets > 0:
        warnings.append(
            ForecastWarning(
                WarningLevel.INFO,
                "sparse_window",
                f"采样窗口内 {len(samples)} 个桶中有 {zero_buckets} 个无历史观测，零需求已计入分位数",
                details={"zero_buckets": zero_buckets, "total_buckets": len(samples)},
            )
        )
    if excluded_future > 0:
        warnings.append(
            ForecastWarning(
                WarningLevel.INFO,
                "future_observations_excluded",
                f"采样窗口内 {excluded_future} 条排期尚未开课，不属于历史数据，已从输入中排除",
                details={"excluded_future": excluded_future},
            )
        )

    max_level = max((w.level for w in warnings), default=WarningLevel.INFO)
    return {
        "status": "ok",
        "value": round(value, 6),
        "warnings": [w.to_dict() for w in warnings],
        "max_warning_level": int(max_level),
    }
