"""容量需求预测：输入口径持久化、显式输入版本重算、缺历史数据的可解释警告。

会议口径：
- 采样窗口与模型参数落库（SQLite 重启可恢复）；
- 重新计算必须指向明确输入版本，版本与当前数据不一致即拒绝；
- 历史数据不足时返回带级别的可解释警告，且不写空报告。
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from service_09252_008.application.booking_service import BookingService
from service_09252_008.application.capacity_forecast_service import (
    COLLECTION_FORECAST_CONFIGS,
    COLLECTION_FORECAST_REPORTS,
    COLLECTION_FORECAST_RUNS,
    CapacityForecastService,
)
from service_09252_008.application.catalog_service import CatalogService
from service_09252_008.application.ports import ManualClock, SequentialIdGenerator, UuidIdGenerator
from service_09252_008.domain.errors import BusinessRuleError, NotFoundError, ValidationError
from service_09252_008.domain.forecasting import (
    DemandPoint,
    WarningLevel,
    evaluate_history,
    forecast_demand,
)
from service_09252_008.persistence.sqlite_store import SQLiteStore
from tests.helpers import NOW, apply_payload, make_services, seed_catalog

AFTER_HISTORY = datetime(2026, 10, 10, 0, 0, 0, tzinfo=timezone.utc)

#: 三天历史窗口（UTC 日桶），覆盖 10-01 ~ 10-03 的课程。
FORECAST_REQ = {
    "history_start": "2026-10-01T00:00:00+00:00",
    "history_end": "2026-10-04T00:00:00+00:00",
    "bucket_seconds": 86400,
}


def _services(**kwargs):
    catalog, bookings, clock, store = make_services(**kwargs)
    forecast = CapacityForecastService(store, clock, SequentialIdGenerator())
    return catalog, bookings, forecast, clock, store


def _multi_day_catalog(catalog):
    """接待窗口覆盖 10-01 ~ 10-03（+08:00），容量充足不触发候补。"""
    return seed_catalog(
        catalog,
        window_capacity=10,
        window_start="2026-10-01T00:00:00+08:00",
        window_end="2026-10-04T00:00:00+08:00",
    )


def _book(bookings, ids, key, day, *, hour=10, seats=10, institution="城北学院"):
    """申请 + 报价 + 锁定，形成一条计入历史需求的预约。"""
    payload = apply_payload(
        ids,
        key,
        seats=seats,
        institution=institution,
        slot_start=f"2026-10-0{day}T{hour:02d}:00:00+08:00",
        slot_end=f"2026-10-0{day}T{hour + 2:02d}:00:00+08:00",
    )
    applied = bookings.apply(payload)
    bookings.quote(applied["booking_id"])
    bookings.lock(applied["booking_id"], {"idempotency_key": f"{key}-lock"})
    return applied


class RunForecastTests(unittest.TestCase):
    def test_run_forecast_persists_report_and_input_version(self) -> None:
        catalog, bookings, forecast, clock, store = _services()
        ids = _multi_day_catalog(catalog)
        for day in (1, 2, 3):
            _book(bookings, ids, f"fc-{day}", day)
        clock.set(AFTER_HISTORY)

        config = forecast.create_forecast(FORECAST_REQ)
        self.assertEqual(store.get(COLLECTION_FORECAST_CONFIGS, config["forecast_id"]), config)
        self.assertEqual(config["params"]["model_type"], "moving_average")
        self.assertEqual(config["history_window"]["bucket_seconds"], 86400)

        view = forecast.run_forecast(config["forecast_id"])
        self.assertEqual(view["status"], "completed")
        self.assertFalse(view["recomputed"])
        # 3 个非空桶 < 推荐 5：info 级提示，但报告照常生成
        self.assertEqual(view["warning_level"], "info")
        self.assertEqual(view["warnings"][0]["code"], "sparse_history")

        report = view["report"]
        self.assertIsNotNone(report)
        self.assertEqual(report["input_version"], view["input_version"])
        self.assertEqual(report["model_version"], "capacity-forecast-v1")
        self.assertEqual([p["seats_total"] for p in report["history_points"]], [10, 10, 10])
        self.assertEqual(len(report["forecast"]["points"]), 3)
        self.assertEqual(report["forecast"]["points"][0]["predicted_seats"], 10.0)
        self.assertEqual(report["forecast"]["total_predicted_seats"], 30.0)
        self.assertEqual(report["forecast"]["peak_predicted_seats"], 10.0)

        # 报告与运行记录均落库
        self.assertEqual(len(store.query(COLLECTION_FORECAST_REPORTS)), 1)
        runs = forecast.list_runs(config["forecast_id"])["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "completed")
        self.assertEqual(runs[0]["report_id"], report["report_id"])

    def test_missing_history_returns_error_warning_and_writes_no_report(self) -> None:
        catalog, bookings, forecast, clock, store = _services()
        _multi_day_catalog(catalog)  # 只登记目录，没有任何预约
        clock.set(AFTER_HISTORY)
        config = forecast.create_forecast(FORECAST_REQ)

        view = forecast.run_forecast(config["forecast_id"])
        self.assertEqual(view["status"], "withheld")
        self.assertEqual(view["warning_level"], "error")
        self.assertIsNone(view["report"])
        self.assertIsNone(view["report_id"])
        warning = view["warnings"][0]
        self.assertEqual(warning["level"], "error")
        self.assertEqual(warning["code"], "no_history_data")
        # 可解释：说明缺什么、窗口内有多少桶
        self.assertIn("no confirmed bookings", warning["message"])
        self.assertEqual(warning["details"]["non_empty_buckets"], 0)
        self.assertEqual(warning["details"]["total_buckets"], 3)

        # 不写空报告，但留下运行记录供审计
        self.assertEqual(store.query(COLLECTION_FORECAST_REPORTS), [])
        runs = store.query(COLLECTION_FORECAST_RUNS)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "withheld")
        self.assertIsNone(runs[0]["report_id"])

    def test_insufficient_history_warning_level_and_no_report(self) -> None:
        catalog, bookings, forecast, clock, store = _services()
        ids = _multi_day_catalog(catalog)
        _book(bookings, ids, "fc-only", 1)  # 只有 1 个非空桶，低于默认最小 3
        clock.set(AFTER_HISTORY)
        config = forecast.create_forecast(FORECAST_REQ)

        view = forecast.run_forecast(config["forecast_id"])
        self.assertEqual(view["status"], "withheld")
        self.assertEqual(view["warning_level"], "warning")
        self.assertIsNone(view["report"])
        warning = view["warnings"][0]
        self.assertEqual(warning["code"], "insufficient_history")
        self.assertEqual(warning["details"]["non_empty_buckets"], 1)
        self.assertEqual(warning["details"]["required_min_data_points"], 3)
        self.assertEqual(store.query(COLLECTION_FORECAST_REPORTS), [])

    def test_scope_filters_history(self) -> None:
        catalog, bookings, forecast, clock, store = _services()
        ids = _multi_day_catalog(catalog)
        for day in (1, 2, 3):
            _book(bookings, ids, f"fc-a-{day}", day, institution="城南大学")
        _book(bookings, ids, "fc-b-1", 1, hour=14, institution="城北学院")
        clock.set(AFTER_HISTORY)

        config_a = forecast.create_forecast({**FORECAST_REQ, "scope": {"institution": "城南大学"}})
        view_a = forecast.run_forecast(config_a["forecast_id"])
        self.assertEqual(view_a["status"], "completed")
        self.assertEqual([p["seats_total"] for p in view_a["report"]["history_points"]], [10, 10, 10])

        config_b = forecast.create_forecast({**FORECAST_REQ, "scope": {"institution": "城北学院"}})
        view_b = forecast.run_forecast(config_b["forecast_id"])
        self.assertEqual(view_b["status"], "withheld")
        self.assertEqual(view_b["warnings"][0]["details"]["non_empty_buckets"], 1)

    def test_linear_trend_model(self) -> None:
        catalog, bookings, forecast, clock, store = _services()
        ids = _multi_day_catalog(catalog)
        _book(bookings, ids, "fc-t-1", 1, seats=4)
        _book(bookings, ids, "fc-t-2", 2, seats=8)
        _book(bookings, ids, "fc-t-3", 3, seats=12)
        clock.set(AFTER_HISTORY)
        config = forecast.create_forecast(
            {
                **FORECAST_REQ,
                "model_type": "linear_trend",
                "min_data_points": 3,
                "recommended_data_points": 3,
            }
        )
        view = forecast.run_forecast(config["forecast_id"])
        self.assertEqual(view["status"], "completed")
        self.assertEqual(view["warnings"], [])
        points = view["report"]["forecast"]["points"]
        self.assertEqual([p["predicted_seats"] for p in points], [16.0, 20.0, 24.0])
        self.assertEqual(view["report"]["forecast"]["peak_predicted_seats"], 24.0)


class RecomputeTests(unittest.TestCase):
    def _completed_forecast(self):
        catalog, bookings, forecast, clock, store = _services()
        ids = _multi_day_catalog(catalog)
        for day in (1, 2, 3):
            _book(bookings, ids, f"fc-{day}", day)
        clock.set(AFTER_HISTORY)
        config = forecast.create_forecast(FORECAST_REQ)
        view = forecast.run_forecast(config["forecast_id"])
        return catalog, bookings, forecast, clock, store, ids, config, view

    def test_recompute_requires_explicit_input_version(self) -> None:
        _, _, forecast, _, _, _, config, _ = self._completed_forecast()
        with self.assertRaises(ValidationError):
            forecast.recompute(config["forecast_id"], {})
        with self.assertRaises(NotFoundError):
            forecast.recompute(config["forecast_id"], {"input_version": "0" * 64})
        with self.assertRaises(NotFoundError):
            forecast.recompute("cap_missing", {"input_version": "0" * 64})

    def test_recompute_with_current_version_reproduces_result(self) -> None:
        _, _, forecast, _, store, _, config, first = self._completed_forecast()
        version = first["input_version"]
        again = forecast.recompute(config["forecast_id"], {"input_version": version})
        self.assertTrue(again["recomputed"])
        self.assertEqual(again["status"], "completed")
        self.assertEqual(again["input_version"], version)
        # 确定性：同一输入版本必然得到同一预测结果
        self.assertEqual(again["report"]["forecast"], first["report"]["forecast"])
        self.assertEqual(len(forecast.list_runs(config["forecast_id"])["runs"]), 2)
        self.assertEqual(len(store.query(COLLECTION_FORECAST_REPORTS)), 2)

    def test_recompute_rejects_stale_version_after_data_change(self) -> None:
        catalog, bookings, forecast, clock, store = _services()
        ids = _multi_day_catalog(catalog)
        for day in (1, 2, 3):
            _book(bookings, ids, f"fc-{day}", day)
        # 时钟停在窗口中间：历史窗口已开始（可登记配置），10-03 仍可补约
        clock.set(datetime(2026, 10, 2, 0, 0, 0, tzinfo=timezone.utc))
        config = forecast.create_forecast(FORECAST_REQ)
        first = forecast.run_forecast(config["forecast_id"])
        stale = first["input_version"]

        # 窗口内新增一条确认预约：底层历史输入已变更
        _book(bookings, ids, "fc-late", 3, hour=14, seats=6)
        with self.assertRaises(BusinessRuleError) as ctx:
            forecast.recompute(config["forecast_id"], {"input_version": stale})
        details = ctx.exception.details
        self.assertEqual(details["requested_version"], stale)
        self.assertNotEqual(details["current_version"], stale)
        # 拒绝重算：不产生新报告与新运行记录
        self.assertEqual(len(store.query(COLLECTION_FORECAST_REPORTS)), 1)
        self.assertEqual(len(store.query(COLLECTION_FORECAST_RUNS)), 1)

        # 新解算产生新版本，之后可针对新版本重算
        fresh = forecast.run_forecast(config["forecast_id"])
        self.assertNotEqual(fresh["input_version"], stale)
        redone = forecast.recompute(config["forecast_id"], {"input_version": fresh["input_version"]})
        self.assertEqual(redone["input_version"], fresh["input_version"])
        self.assertEqual(
            [p["seats_total"] for p in redone["report"]["history_points"]],
            [10, 10, 16],
        )


class ConfigValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        catalog, bookings, forecast, clock, store = _services()
        self.ids = _multi_day_catalog(catalog)
        clock.set(AFTER_HISTORY)
        self.forecast = forecast

    def test_window_shape(self) -> None:
        with self.assertRaises(ValidationError):  # 结束早于开始
            self.forecast.create_forecast(
                {**FORECAST_REQ, "history_end": "2026-09-30T00:00:00+00:00"}
            )
        with self.assertRaises(ValidationError):  # 窗口不能整除桶长
            self.forecast.create_forecast({**FORECAST_REQ, "bucket_seconds": 100000})
        with self.assertRaises(ValidationError):  # 历史窗口必须落在过去
            self.forecast.create_forecast({**FORECAST_REQ, "history_start": "2026-10-11T00:00:00+00:00"})
        with self.assertRaises(ValidationError):  # 朴素时间
            self.forecast.create_forecast({**FORECAST_REQ, "history_start": "2026-10-01T00:00:00"})

    def test_model_params(self) -> None:
        with self.assertRaises(ValidationError):  # 未知模型
            self.forecast.create_forecast({**FORECAST_REQ, "model_type": "prophet"})
        with self.assertRaises(ValidationError):  # 推荐样本数低于最小样本数
            self.forecast.create_forecast(
                {**FORECAST_REQ, "min_data_points": 5, "recommended_data_points": 2}
            )
        with self.assertRaises(ValidationError):  # horizon 非正
            self.forecast.create_forecast({**FORECAST_REQ, "forecast_horizon_buckets": 0})

    def test_scope(self) -> None:
        with self.assertRaises(ValidationError):  # 不支持的过滤字段
            self.forecast.create_forecast({**FORECAST_REQ, "scope": {"package_id": "pkg_0001"}})
        with self.assertRaises(ValidationError):  # 引用的资源不存在
            self.forecast.create_forecast({**FORECAST_REQ, "scope": {"resource_id": "res_missing"}})
        # 合法 scope：引用已登记资源
        config = self.forecast.create_forecast(
            {**FORECAST_REQ, "scope": {"resource_id": self.ids["resource_id"]}}
        )
        self.assertEqual(config["scope"], {"resource_id": self.ids["resource_id"]})


class SqlitePersistenceTests(unittest.TestCase):
    def test_config_and_report_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/forecast.db"
            clock = ManualClock(NOW)

            store = SQLiteStore(db_path)
            catalog = CatalogService(store, clock, UuidIdGenerator())
            bookings = BookingService(store, clock, UuidIdGenerator())
            forecast = CapacityForecastService(store, clock, UuidIdGenerator())
            ids = _multi_day_catalog(catalog)
            for day in (1, 2, 3):
                _book(bookings, ids, f"fc-sql-{day}", day)
            clock.set(AFTER_HISTORY)
            config = forecast.create_forecast(FORECAST_REQ)
            first = forecast.run_forecast(config["forecast_id"])
            version = first["input_version"]
            store.close()

            # 模拟重启：全新服务实例挂载同一数据库
            store2 = SQLiteStore(db_path)
            forecast2 = CapacityForecastService(store2, clock, UuidIdGenerator())
            restored = forecast2.get_forecast(config["forecast_id"])
            self.assertEqual(restored["params"], config["params"])
            self.assertEqual(restored["history_window"], config["history_window"])
            self.assertEqual(len(forecast2.list_runs(config["forecast_id"])["runs"]), 1)
            # 数据未变：同一输入版本可重算，结果一致
            again = forecast2.recompute(config["forecast_id"], {"input_version": version})
            self.assertEqual(again["status"], "completed")
            self.assertEqual(again["report"]["forecast"], first["report"]["forecast"])
            store2.close()


class DomainRuleTests(unittest.TestCase):
    """纯领域逻辑：警告分级与确定性模型。"""

    @staticmethod
    def _points(non_empty: int, total: int = 6) -> list[DemandPoint]:
        base = datetime(2026, 10, 1, tzinfo=timezone.utc)
        return [
            DemandPoint(
                bucket_start=base + timedelta(days=i),
                seats_total=5 if i < non_empty else 0,
                bookings_count=1 if i < non_empty else 0,
            )
            for i in range(total)
        ]

    def test_evaluate_history_levels(self) -> None:
        levels = [
            evaluate_history(self._points(n), min_data_points=3, recommended_data_points=5)
            for n in (0, 1, 4, 6)
        ]
        self.assertEqual(levels[0][0].level, WarningLevel.ERROR)
        self.assertEqual(levels[0][0].code, "no_history_data")
        self.assertEqual(levels[1][0].level, WarningLevel.WARNING)
        self.assertEqual(levels[1][0].code, "insufficient_history")
        self.assertEqual(levels[2][0].level, WarningLevel.INFO)
        self.assertEqual(levels[2][0].code, "sparse_history")
        self.assertEqual(levels[3], [])

    def test_forecast_models_are_deterministic(self) -> None:
        points = self._points(3, total=3)
        for i, seats in enumerate((4, 8, 12)):
            points[i] = DemandPoint(points[i].bucket_start, seats, 1)
        end = datetime(2026, 10, 4, tzinfo=timezone.utc)
        linear = forecast_demand(
            points, model_type="linear_trend", horizon_buckets=2, window_end=end, bucket_seconds=86400
        )
        self.assertEqual([p["predicted_seats"] for p in linear["points"]], [16.0, 20.0])
        again = forecast_demand(
            points, model_type="linear_trend", horizon_buckets=2, window_end=end, bucket_seconds=86400
        )
        self.assertEqual(linear, again)
        average = forecast_demand(
            points, model_type="moving_average", horizon_buckets=2, window_end=end, bucket_seconds=86400
        )
        self.assertEqual([p["predicted_seats"] for p in average["points"]], [8.0, 8.0])


if __name__ == "__main__":
    unittest.main()
