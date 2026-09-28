"""容量需求预测的纯领域逻辑。

资源规划会议确定的口径：

- **采样窗口**：半开区间 ``[start, end)``，按固定 ``bucket_seconds`` 切桶，
  每桶聚合“需求席位数 / 课程数”；无课的桶同样是观测值（需求为 0）。
- **计入历史需求的预约**：仅 ``LOCKED / SHIPPED / CHECKED_IN / SETTLED``——
  申请中、报价中可能被取消或过期，候补未确认，取消/过期不代表真实需求。
- **模型参数**：模型类型（移动平均 / 线性趋势）、最小与推荐样本桶数、
  预测 horizon；另带显式 ``MODEL_VERSION``，模型代码变更即升版。
- **输入版本**：对“范围 + 物化窗口 + 参数 + 全部观测点”做规范化哈希，
  重新计算必须指向该指纹；窗口内数据一旦变化指纹即变。
- **警告级别**：``info / warning / error`` 三级，缺历史数据时给出
  可解释的 code 与 details，由应用层决定不写空报告。

本模块不触碰持久化与时钟，全部为可单测的纯函数与值对象。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .errors import ValidationError
from .models import Booking, BookingStatus, dt_to_str

#: 模型代码版本：参数集或算法变更时升版，并纳入输入指纹。
MODEL_VERSION = "capacity-forecast-v1"

MODEL_MOVING_AVERAGE = "moving_average"
MODEL_LINEAR_TREND = "linear_trend"
MODEL_TYPES = frozenset({MODEL_MOVING_AVERAGE, MODEL_LINEAR_TREND})

#: 计入历史需求采样的预约状态。
DEMAND_OBSERVED_STATUSES = frozenset(
    {
        BookingStatus.LOCKED,
        BookingStatus.SHIPPED,
        BookingStatus.CHECKED_IN,
        BookingStatus.SETTLED,
    }
)

#: 允许作为预测范围（scope）过滤的 Booking 字段。
SCOPE_FIELDS = ("resource_id", "window_id", "institution")


class WarningLevel(str, Enum):
    """预测警告级别：逐级严重，``warning`` 及以上阻断报告写入。"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


#: 阻断报告落盘的级别。
BLOCKING_LEVELS = frozenset({WarningLevel.WARNING, WarningLevel.ERROR})

WARNING_NO_HISTORY = "no_history_data"
WARNING_INSUFFICIENT_HISTORY = "insufficient_history"
WARNING_SPARSE_HISTORY = "sparse_history"


@dataclass(frozen=True)
class DemandPoint:
    """单个采样桶的聚合需求。"""

    bucket_start: datetime
    seats_total: int
    bookings_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_start": dt_to_str(self.bucket_start),
            "seats_total": self.seats_total,
            "bookings_count": self.bookings_count,
        }


@dataclass(frozen=True)
class ForecastWarning:
    """可解释警告：级别 + 机器可读 code + 人读说明 + 证据 details。"""

    level: WarningLevel
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class ResolvedWindow:
    """物化为绝对 UTC 时间的采样窗口。"""

    start: datetime
    end: datetime
    bucket_seconds: int

    def bucket_count(self) -> int:
        return int((self.end - self.start).total_seconds() // self.bucket_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": dt_to_str(self.start),
            "end": dt_to_str(self.end),
            "bucket_seconds": self.bucket_seconds,
        }


def validate_window_shape(start: datetime, end: datetime, bucket_seconds: int) -> None:
    """窗口方向、桶大小与整除性校验。"""
    if end <= start:
        raise ValidationError("history window end must be after start")
    if isinstance(bucket_seconds, bool) or not isinstance(bucket_seconds, int) or bucket_seconds <= 0:
        raise ValidationError("bucket_seconds must be a positive integer")
    span_seconds = (end - start).total_seconds()
    if span_seconds % bucket_seconds != 0:
        raise ValidationError(
            "history window length must be divisible by bucket_seconds",
            details={"window_seconds": int(span_seconds), "bucket_seconds": bucket_seconds},
        )


def bucketize(
    bookings: list[Booking],
    window: ResolvedWindow,
    scope: dict[str, str],
) -> list[DemandPoint]:
    """把窗口内、范围匹配、状态已确认的预约按桶聚合（零需求桶保留）。"""
    n_buckets = window.bucket_count()
    seats = [0] * n_buckets
    counts = [0] * n_buckets
    delta_origin = window.start
    bucket_delta = timedelta(seconds=window.bucket_seconds)
    for booking in bookings:
        if booking.status not in DEMAND_OBSERVED_STATUSES:
            continue
        if not window.start <= booking.slot_start < window.end:
            continue
        if any(getattr(booking, field_name) != value for field_name, value in scope.items()):
            continue
        index = int((booking.slot_start - delta_origin) // bucket_delta)
        # 半开区间 + 整除校验保证 index 落在 [0, n_buckets)。
        seats[index] += booking.seats
        counts[index] += 1
    return [
        DemandPoint(
            bucket_start=window.start + timedelta(seconds=window.bucket_seconds * i),
            seats_total=seats[i],
            bookings_count=counts[i],
        )
        for i in range(n_buckets)
    ]


def evaluate_history(
    points: list[DemandPoint],
    *,
    min_data_points: int,
    recommended_data_points: int,
) -> list[ForecastWarning]:
    """按非空桶数量评估历史数据充足度，返回 0/1 条可解释警告。"""
    total = len(points)
    non_empty = sum(1 for p in points if p.bookings_count > 0)
    evidence = {
        "total_buckets": total,
        "non_empty_buckets": non_empty,
        "required_min_data_points": min_data_points,
        "recommended_data_points": recommended_data_points,
    }
    if non_empty == 0:
        return [
            ForecastWarning(
                WarningLevel.ERROR,
                WARNING_NO_HISTORY,
                "no confirmed bookings fall inside the sampling window; "
                "capacity demand cannot be estimated",
                evidence,
            )
        ]
    if non_empty < min_data_points:
        return [
            ForecastWarning(
                WarningLevel.WARNING,
                WARNING_INSUFFICIENT_HISTORY,
                "fewer non-empty sampling buckets than the model minimum; "
                "forecast withheld to avoid an empty/unreliable report",
                evidence,
            )
        ]
    if non_empty < recommended_data_points:
        return [
            ForecastWarning(
                WarningLevel.INFO,
                WARNING_SPARSE_HISTORY,
                "non-empty sampling buckets below the recommended coverage; "
                "forecast emitted but should be read with caution",
                evidence,
            )
        ]
    return []


def _round6(value: float) -> float:
    return round(value, 6)


def forecast_demand(
    points: list[DemandPoint],
    *,
    model_type: str,
    horizon_buckets: int,
    window_end: datetime,
    bucket_seconds: int,
) -> dict[str, Any]:
    """对未来 ``horizon_buckets`` 个桶做确定性需求预测。"""
    n = len(points)
    seats = [float(p.seats_total) for p in points]
    courses = [float(p.bookings_count) for p in points]

    if model_type == MODEL_MOVING_AVERAGE:
        seats_hat = [sum(seats) / n] * horizon_buckets
        courses_hat = [sum(courses) / n] * horizon_buckets
    elif model_type == MODEL_LINEAR_TREND:
        mean_x = (n - 1) / 2.0

        def ols_predict(series: list[float], future_x: list[int]) -> list[float]:
            mean_y = sum(series) / n
            denom = sum((x - mean_x) ** 2 for x in range(n))
            if denom == 0:  # 只有一个桶或所有 x 相同，退化为水平预测
                slope = 0.0
            else:
                slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(range(n), series)) / denom
            intercept = mean_y - slope * mean_x
            return [max(0.0, intercept + slope * x) for x in future_x]

        future_x = list(range(n, n + horizon_buckets))
        seats_hat = ols_predict(seats, future_x)
        courses_hat = ols_predict(courses, future_x)
    else:  # pragma: no cover - 应用层已校验白名单
        raise ValidationError(f"unknown model_type: {model_type}")

    future_points: list[dict[str, Any]] = []
    for offset, (pred_seats, pred_courses) in enumerate(zip(seats_hat, courses_hat)):
        future_points.append(
            {
                "bucket_start": dt_to_str(
                    window_end + timedelta(seconds=bucket_seconds * offset)
                ),
                "predicted_seats": _round6(pred_seats),
                "predicted_bookings": _round6(pred_courses),
            }
        )
    return {
        "model_version": MODEL_VERSION,
        "model_type": model_type,
        "points": future_points,
        "peak_predicted_seats": _round6(max(p["predicted_seats"] for p in future_points)),
        "total_predicted_seats": _round6(sum(p["predicted_seats"] for p in future_points)),
        "total_predicted_bookings": _round6(sum(p["predicted_bookings"] for p in future_points)),
    }


def input_fingerprint(
    *,
    scope: dict[str, str],
    window: ResolvedWindow,
    params: dict[str, Any],
    points: list[DemandPoint],
) -> str:
    """计算预测输入版本：范围 + 物化窗口 + 参数（含模型版本）+ 观测点。"""
    payload = {
        "model_version": MODEL_VERSION,
        "scope": dict(scope),
        "window": window.to_dict(),
        "params": {
            "model_type": params["model_type"],
            "min_data_points": params["min_data_points"],
            "recommended_data_points": params["recommended_data_points"],
            "forecast_horizon_buckets": params["forecast_horizon_buckets"],
        },
        "points": [p.to_dict() for p in points],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
