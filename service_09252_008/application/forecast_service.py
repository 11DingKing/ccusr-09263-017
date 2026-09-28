"""容量需求预测应用服务。

口径管理流程：
1. :meth:`ForecastService.configure` 保存采样窗口与模型参数，形成明确的
   “输入版本”（``fcv_...``），预测输入口径持久化到 SQLite 后重启可见；
2. :meth:`ForecastService.recompute` 必须指向某个输入版本，在该版本固定的
   窗口与参数下重算；结果连同输入指纹（窗口 + 参数 + 实际采样观测）一并
   落库，相同输入版本 + 相同输入指纹必然得到相同预测值；
3. 采样窗口内完全没有历史数据时只返回 ``CRITICAL`` 可解释告警，
   **不写空报告**；样本不足时报告照常落库但带 ``WARNING``。
"""
from __future__ import annotations

from typing import Any

from ..domain.errors import NotFoundError, ValidationError
from ..domain.forecast import (
    GRANULARITIES,
    METRICS,
    ForecastWindow,
    ModelParameters,
    WarningLevel,
    build_samples,
    canonical_inputs,
    forecast,
)
from ..domain.models import Booking, DomainEvent, dt_from_str, dt_to_str
from ..persistence.store import Store
from .booking_service import COLLECTION_BOOKINGS, COLLECTION_EVENTS
from .ports import Clock, IdGenerator

COLLECTION_FORECAST_INPUTS = "forecast_inputs"
COLLECTION_FORECAST_REPORTS = "forecast_reports"


class ForecastService:
    """容量需求预测用例：口径登记、版本化重算、告警返回。"""

    def __init__(self, store: Store, clock: Clock, ids: IdGenerator) -> None:
        self._store = store
        self._clock = clock
        self._ids = ids

    # ------------------------------------------------------------------
    # 输入口径
    # ------------------------------------------------------------------

    def _parse_window(self, data: dict[str, Any]) -> ForecastWindow:
        try:
            start = dt_from_str(data.get("window_start"))
            end = dt_from_str(data.get("window_end"))
        except (ValueError, TypeError) as exc:
            raise ValidationError(f"invalid forecast window: {exc}") from exc
        width = data.get("bucket_width_seconds")
        if isinstance(width, bool) or not isinstance(width, int):
            raise ValidationError("field bucket_width_seconds must be an integer")
        granularity = data.get("granularity", "day")
        if granularity not in GRANULARITIES:
            raise ValidationError(
                f"field granularity must be one of {sorted(GRANULARITIES)}",
                details={"granularity": granularity},
            )
        try:
            return ForecastWindow(
                start=start, end=end, bucket_width_seconds=width, granularity=granularity
            )
        except ValueError as exc:
            raise ValidationError(f"invalid forecast window: {exc}") from exc

    def _parse_parameters(self, data: dict[str, Any]) -> ModelParameters:
        raw = data.get("parameters")
        if raw is None:
            return ModelParameters()
        if not isinstance(raw, dict):
            raise ValidationError("field parameters must be an object")
        quantile = raw.get("quantile", 0.8)
        if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
            raise ValidationError("field parameters.quantile must be a number")
        min_samples = raw.get("min_samples", 4)
        if isinstance(min_samples, bool) or not isinstance(min_samples, int):
            raise ValidationError("field parameters.min_samples must be an integer")
        metric = raw.get("metric", "seats")
        if metric not in METRICS:
            raise ValidationError(f"field parameters.metric must be one of {sorted(METRICS)}")
        try:
            return ModelParameters(quantile=float(quantile), min_samples=min_samples, metric=metric)
        except ValueError as exc:
            raise ValidationError(f"invalid model parameters: {exc}") from exc

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        """登记采样窗口与模型参数，返回可被重算指向的输入版本。"""
        window = self._parse_window(request)
        params = self._parse_parameters(request)
        now = self._clock.now()
        version_id = self._ids.new_id("fcv")
        record = {
            "version_id": version_id,
            "window": window.to_dict(),
            "parameters": params.to_dict(),
            "created_at": dt_to_str(now),
        }
        with self._store.transaction():
            self._store.put(COLLECTION_FORECAST_INPUTS, version_id, record)
            self._emit("forecast_input_configured", {"version_id": version_id})
        return self._input_view(record)

    def list_inputs(self) -> list[dict[str, Any]]:
        records = self._store.query(COLLECTION_FORECAST_INPUTS)
        records.sort(key=lambda r: (r["created_at"], r["version_id"]))
        return [self._input_view(r) for r in records]

    def get_input(self, version_id: str) -> dict[str, Any]:
        record = self._store.get(COLLECTION_FORECAST_INPUTS, version_id)
        if record is None:
            raise NotFoundError(
                f"forecast input version not found: {version_id}",
                details={"version_id": version_id},
            )
        return self._input_view(record)

    def _load_input(self, version_id: str) -> tuple[ForecastWindow, ModelParameters, dict[str, Any]]:
        record = self._store.get(COLLECTION_FORECAST_INPUTS, version_id)
        if record is None:
            raise NotFoundError(
                f"forecast input version not found: {version_id}",
                details={"version_id": version_id},
            )
        return ForecastWindow.from_dict(record["window"]), ModelParameters.from_dict(record["parameters"]), record

    def _input_view(self, record: dict[str, Any]) -> dict[str, Any]:
        report = self._store.get(COLLECTION_FORECAST_REPORTS, record["version_id"])
        return {
            **record,
            "latest_report_at": report["computed_at"] if report else None,
            "latest_warning_level": report["result"]["max_warning_level"] if report else None,
        }

    # ------------------------------------------------------------------
    # 重算
    # ------------------------------------------------------------------

    def recompute(self, version_id: str) -> dict[str, Any]:
        """指向明确输入版本重新计算容量需求预测。"""
        with self._store.transaction():
            window, params, input_record = self._load_input(version_id)
            now = self._clock.now()
            bookings = [Booking.from_dict(b) for b in self._store.query(COLLECTION_BOOKINGS)]
            samples, observation_times, excluded_future = build_samples(
                window, bookings, metric=params.metric, now=now
            )
            observed_buckets = sum(1 for value in samples if value > 0)
            input_hash = canonical_inputs(window, params, observation_times)
            result = forecast(
                window,
                params,
                samples,
                observed_buckets=observed_buckets,
                excluded_future=excluded_future,
            )

            if result["status"] == "insufficient_data":
                # 缺少历史数据：只给出可解释告警，不写空报告
                self._emit(
                    "forecast_insufficient_data",
                    {"version_id": version_id, "input_hash": input_hash},
                )
                return {
                    "version_id": version_id,
                    "input_hash": input_hash,
                    "window": input_record["window"],
                    "parameters": input_record["parameters"],
                    "computed_at": dt_to_str(now),
                    "report_persisted": False,
                    **result,
                }

            bucket_starts = window.bucket_starts()
            report = {
                "version_id": version_id,
                "computed_at": dt_to_str(now),
                "input_hash": input_hash,
                "window": input_record["window"],
                "parameters": input_record["parameters"],
                "sample_count": len(samples),
                "observed_buckets": observed_buckets,
                "excluded_future": excluded_future,
                "samples": [
                    {"bucket_start": dt_to_str(start), "value": value}
                    for start, value in zip(bucket_starts, samples)
                ],
                "result": result,
            }
            self._store.put(COLLECTION_FORECAST_REPORTS, version_id, report)
            self._emit(
                "forecast_computed",
                {
                    "version_id": version_id,
                    "input_hash": input_hash,
                    "value": result["value"],
                    "max_warning_level": result["max_warning_level"],
                },
            )
            return {"report_persisted": True, **report}

    def get_report(self, version_id: str) -> dict[str, Any]:
        self._load_input(version_id)  # 报告必须挂在一个已登记的输入版本上
        report = self._store.get(COLLECTION_FORECAST_REPORTS, version_id)
        if report is None:
            raise NotFoundError(
                f"no forecast report for version: {version_id} (insufficient historical data?)",
                details={"version_id": version_id},
            )
        return report

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = DomainEvent(
            event_id=self._ids.new_id("fevt"),
            type=event_type,
            booking_id=None,
            payload=payload,
            created_at=self._clock.now(),
        )
        self._store.put(COLLECTION_EVENTS, event.event_id, event.to_dict())
