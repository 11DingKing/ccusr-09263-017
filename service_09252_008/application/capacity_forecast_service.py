"""容量需求预测应用服务。

用例：

- ``create_forecast``：登记并持久化**采样窗口与模型参数**（口径输入）；
- ``run_forecast``：按当前配置物化解算——聚合历史需求、评估充足度、
  数据不足时返回带级别的可解释警告且**不写空报告**，充足时落报告；
- ``recompute``：重新计算必须显式指向一个既存的**输入版本**（指纹）；
  指纹与当前数据一致才重算（确定性结果），不一致直接拒绝并给出
  请求版本与当前版本，避免“同版本不同结果”。

每次解算都写一条运行记录（含输入版本与警告），但报告仅在无阻断级
（``warning``/``error``）警告时写入存储。
"""
from __future__ import annotations

from typing import Any

from ..domain.errors import BusinessRuleError, NotFoundError, ValidationError
from ..domain.forecasting import (
    MODEL_TYPES,
    MODEL_VERSION,
    SCOPE_FIELDS,
    BLOCKING_LEVELS,
    ResolvedWindow,
    WarningLevel,
    bucketize,
    evaluate_history,
    forecast_demand,
    input_fingerprint,
    validate_window_shape,
)
from ..domain.models import Booking, dt_from_str
from ..persistence.store import Store
from .booking_service import COLLECTION_BOOKINGS
from .catalog_service import COLLECTION_RESOURCES, COLLECTION_WINDOWS
from .ports import Clock, IdGenerator

COLLECTION_FORECAST_CONFIGS = "capacity_forecast_configs"
COLLECTION_FORECAST_RUNS = "capacity_forecast_runs"
COLLECTION_FORECAST_REPORTS = "capacity_forecast_reports"

DEFAULT_MIN_DATA_POINTS = 3
DEFAULT_RECOMMENDED_DATA_POINTS = 5
DEFAULT_FORECAST_HORIZON_BUCKETS = 3
MAX_FORECAST_HORIZON_BUCKETS = 52


def _require_int(data: dict[str, Any], field: str, *, minimum: int, default: int | None = None) -> int:
    value = data.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"field {field} must be an integer", details={"field": field})
    if value < minimum:
        raise ValidationError(f"field {field} must be >= {minimum}", details={"field": field, "minimum": minimum})
    return value


class CapacityForecastService:
    """容量需求预测用例编排。"""

    def __init__(self, store: Store, clock: Clock, ids: IdGenerator) -> None:
        self._store = store
        self._clock = clock
        self._ids = ids

    # ------------------------------------------------------------------
    # 配置：采样窗口 + 模型参数
    # ------------------------------------------------------------------

    def create_forecast(self, request: dict[str, Any]) -> dict[str, Any]:
        """登记采样窗口与模型参数，返回持久化后的配置。"""
        now = self._clock.now()

        scope_raw = request.get("scope", {})
        if scope_raw is None:
            scope_raw = {}
        if not isinstance(scope_raw, dict):
            raise ValidationError("field scope must be an object of resource_id/window_id/institution filters")
        scope: dict[str, str] = {}
        for field_name, value in scope_raw.items():
            if field_name not in SCOPE_FIELDS:
                raise ValidationError(
                    f"unsupported scope field: {field_name}",
                    details={"allowed": list(SCOPE_FIELDS)},
                )
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"scope filter {field_name} must be a non-empty string")
            scope[field_name] = value.strip()
        # 引用的目录实体必须存在，避免对不存在的资源/窗口做无意义预测。
        if "resource_id" in scope and self._store.get(COLLECTION_RESOURCES, scope["resource_id"]) is None:
            raise ValidationError("scope resource_id does not exist", details={"resource_id": scope["resource_id"]})
        if "window_id" in scope and self._store.get(COLLECTION_WINDOWS, scope["window_id"]) is None:
            raise ValidationError("scope window_id does not exist", details={"window_id": scope["window_id"]})

        try:
            start = dt_from_str(request.get("history_start"))
            end = dt_from_str(request.get("history_end"))
        except (ValueError, TypeError) as exc:
            raise ValidationError(f"invalid history window: {exc}") from exc
        if start >= now:
            raise ValidationError("history_start must be in the past")
        bucket_seconds = _require_int(request, "bucket_seconds", minimum=1)
        validate_window_shape(start, end, bucket_seconds)

        model_type = request.get("model_type", "moving_average")
        if not isinstance(model_type, str) or model_type not in MODEL_TYPES:
            raise ValidationError(
                "field model_type must be one of: " + ", ".join(sorted(MODEL_TYPES)),
                details={"model_type": model_type},
            )
        min_points = _require_int(
            request, "min_data_points", minimum=1, default=DEFAULT_MIN_DATA_POINTS
        )
        recommended_points = _require_int(
            request, "recommended_data_points", minimum=1, default=DEFAULT_RECOMMENDED_DATA_POINTS
        )
        if recommended_points < min_points:
            raise ValidationError(
                "recommended_data_points must be >= min_data_points",
                details={"min_data_points": min_points, "recommended_data_points": recommended_points},
            )
        horizon = _require_int(
            request,
            "forecast_horizon_buckets",
            minimum=1,
            default=DEFAULT_FORECAST_HORIZON_BUCKETS,
        )
        if horizon > MAX_FORECAST_HORIZON_BUCKETS:
            raise ValidationError(
                f"forecast_horizon_buckets must be <= {MAX_FORECAST_HORIZON_BUCKETS}",
                details={"forecast_horizon_buckets": horizon},
            )

        params = {
            "model_type": model_type,
            "min_data_points": min_points,
            "recommended_data_points": recommended_points,
            "forecast_horizon_buckets": horizon,
        }
        forecast_id = self._ids.new_id("cap")
        config = {
            "forecast_id": forecast_id,
            "scope": scope,
            "history_window": {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "bucket_seconds": bucket_seconds,
            },
            "params": params,
            "model_version": MODEL_VERSION,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        with self._store.transaction():
            self._store.put(COLLECTION_FORECAST_CONFIGS, forecast_id, config)
        return config

    def get_forecast(self, forecast_id: str) -> dict[str, Any]:
        return self._load_config(forecast_id)

    # ------------------------------------------------------------------
    # 解算
    # ------------------------------------------------------------------

    def run_forecast(self, forecast_id: str) -> dict[str, Any]:
        """按配置对当前数据解算一次预测；返回运行记录视图（可能无报告）。"""
        config = self._load_config(forecast_id)
        return self._solve(config, recompute_of=None)

    def recompute(self, forecast_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """针对显式输入版本重新计算。

        - 未给 ``input_version`` 或版本在该预测下不存在 → 错误（不允许隐式重算）；
        - 当前数据指纹 == 请求版本：确定性重算，返回既有报告；
        - 不一致：拒绝，提示底层历史输入已变更及当前版本。
        """
        request = request or {}
        requested = request.get("input_version")
        if not isinstance(requested, str) or not requested.strip():
            raise ValidationError(
                "recompute requires an explicit input_version",
                details={"field": "input_version"},
            )
        requested = requested.strip()
        config = self._load_config(forecast_id)
        prior_runs = self._store.query(COLLECTION_FORECAST_RUNS, forecast_id=forecast_id)
        if not any(run["input_version"] == requested for run in prior_runs):
            raise NotFoundError(
                "unknown input_version for this forecast; run a fresh forecast instead",
                details={"forecast_id": forecast_id, "input_version": requested},
            )

        scope, window, params = self._resolved(config)
        points = self._sample(scope, window)
        current_version = input_fingerprint(scope=scope, window=window, params=params, points=points)
        if current_version != requested:
            raise BusinessRuleError(
                "input_version no longer matches the current historical data; "
                "recomputation would produce a different result under the same version",
                details={
                    "forecast_id": forecast_id,
                    "requested_version": requested,
                    "current_version": current_version,
                },
            )
        # 指纹一致 => 模型是确定性的，重算必然得到同一结果（留新运行记录）。
        return self._solve(config, recompute_of=requested)

    def list_runs(self, forecast_id: str) -> dict[str, Any]:
        self._load_config(forecast_id)
        runs = self._store.query(COLLECTION_FORECAST_RUNS, forecast_id=forecast_id)
        runs.sort(key=lambda r: (r["created_at"], r["run_id"]))
        return {"forecast_id": forecast_id, "runs": runs}

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _load_config(self, forecast_id: str) -> dict[str, Any]:
        config = self._store.get(COLLECTION_FORECAST_CONFIGS, forecast_id)
        if config is None:
            raise NotFoundError(
                "capacity forecast not found",
                details={"forecast_id": forecast_id},
            )
        return config

    @staticmethod
    def _resolved(config: dict[str, Any]) -> tuple[dict[str, str], ResolvedWindow, dict[str, Any]]:
        raw_window = config["history_window"]
        window = ResolvedWindow(
            start=dt_from_str(raw_window["start"]),
            end=dt_from_str(raw_window["end"]),
            bucket_seconds=int(raw_window["bucket_seconds"]),
        )
        return dict(config["scope"]), window, dict(config["params"])

    def _sample(self, scope: dict[str, str], window: ResolvedWindow):
        bookings = [Booking.from_dict(b) for b in self._store.query(COLLECTION_BOOKINGS)]
        return bucketize(bookings, window, scope)

    def _solve(self, config: dict[str, Any], *, recompute_of: str | None) -> dict[str, Any]:
        forecast_id = config["forecast_id"]
        now = self._clock.now()
        scope, window, params = self._resolved(config)
        with self._store.transaction():
            points = self._sample(scope, window)
            version = input_fingerprint(scope=scope, window=window, params=params, points=points)
            warnings = evaluate_history(
                points,
                min_data_points=params["min_data_points"],
                recommended_data_points=params["recommended_data_points"],
            )
            blocking = any(w.level in BLOCKING_LEVELS for w in warnings)
            warning_dicts = [w.to_dict() for w in warnings]

            report: dict[str, Any] | None = None
            forecast: dict[str, Any] | None = None
            report_id: str | None = None
            if not blocking:
                forecast = forecast_demand(
                    points,
                    model_type=params["model_type"],
                    horizon_buckets=params["forecast_horizon_buckets"],
                    window_end=window.end,
                    bucket_seconds=window.bucket_seconds,
                )
                report_id = self._ids.new_id("cfr")
                report = {
                    "report_id": report_id,
                    "forecast_id": forecast_id,
                    "input_version": version,
                    "model_version": MODEL_VERSION,
                    "created_at": now.isoformat(),
                    "scope": scope,
                    "history_window": window.to_dict(),
                    "params": params,
                    "history_points": [p.to_dict() for p in points],
                    "forecast": forecast,
                    "warnings": warning_dicts,
                }
                self._store.put(COLLECTION_FORECAST_REPORTS, report_id, report)
            # 数据不足也写运行记录（审计“曾被抑制”），但绝不写空报告。
            run_id = self._ids.new_id("cfrun")
            run = {
                "run_id": run_id,
                "forecast_id": forecast_id,
                "input_version": version,
                "recompute_of": recompute_of,
                "status": "withheld" if blocking else "completed",
                "warnings": warning_dicts,
                "report_id": report_id,
                "created_at": now.isoformat(),
            }
            self._store.put(COLLECTION_FORECAST_RUNS, run_id, run)
        return self._run_view(run, report, recomputed=recompute_of is not None)

    @staticmethod
    def _run_view(run: dict[str, Any], report: dict[str, Any] | None, *, recomputed: bool) -> dict[str, Any]:
        max_level = None
        if run["warnings"]:
            order = {WarningLevel.INFO.value: 1, WarningLevel.WARNING.value: 2, WarningLevel.ERROR.value: 3}
            max_level = max((w["level"] for w in run["warnings"]), key=lambda lvl: order[lvl])
        return {
            "run_id": run["run_id"],
            "forecast_id": run["forecast_id"],
            "input_version": run["input_version"],
            "status": run["status"],
            "recomputed": recomputed,
            "warning_level": max_level,
            "warnings": run["warnings"],
            "report_id": run.get("report_id"),
            "report": report,
        }
