"""容量需求预测：输入口径版本化、可解释告警、缺历史数据不写空报告。"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timezone

from service_09252_008.application.booking_service import COLLECTION_BOOKINGS
from service_09252_008.application.forecast_service import (
    COLLECTION_FORECAST_INPUTS,
    COLLECTION_FORECAST_REPORTS,
    ForecastService,
)
from service_09252_008.application.ports import ManualClock, SequentialIdGenerator
from service_09252_008.domain.errors import NotFoundError
from service_09252_008.domain.forecast import (
    BOOKING_METRIC,
    SEAT_METRIC,
    FORECAST_SAMPLE_STATUSES,
    ForecastWindow,
    ModelParameters,
    WarningLevel,
    build_samples,
    forecast,
    quantile_value,
)
from service_09252_008.domain.models import Booking, BookingStatus
from service_09252_008.interfaces.http_api import create_server
from service_09252_008.persistence.sqlite_store import SQLiteStore
from service_09252_008.persistence.store import InMemoryStore
from tests.helpers import make_services, seed_catalog

NOW = datetime(2026, 9, 25, 0, 0, 0, tzinfo=timezone.utc)
DAY = 86400


def make_forecast_service(store=None, *, now=NOW):
    store = store or InMemoryStore()
    clock = ManualClock(now)
    ids = SequentialIdGenerator()
    forecasts = ForecastService(store, clock, ids)
    return forecasts, clock, ids, store


def historical_booking(
    booking_id: str,
    slot_start: datetime,
    seats: int,
    status: BookingStatus = BookingStatus.SETTLED,
) -> Booking:
    return Booking(
        booking_id=booking_id,
        institution="城南大学",
        package_id="pkg",
        mentor_id="men",
        resource_id="res",
        window_id="win",
        seats=seats,
        slot_start=slot_start,
        slot_end=slot_start.replace(),
        status=status,
        created_at=slot_start,
        updated_at=slot_start,
    )


class ForecastDomainTests(unittest.TestCase):
    def test_window_rejects_naive_bounds_and_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            ForecastWindow(
                start=datetime(2026, 3, 1),
                end=datetime(2026, 9, 1, tzinfo=timezone.utc),
                bucket_width_seconds=DAY,
            )
        with self.assertRaises(ValueError):
            ForecastWindow(
                start=datetime(2026, 9, 1, tzinfo=timezone.utc),
                end=datetime(2026, 3, 1, tzinfo=timezone.utc),
                bucket_width_seconds=DAY,
            )
        with self.assertRaises(ValueError):
            ForecastWindow(
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                end=datetime(2026, 9, 1, tzinfo=timezone.utc),
                bucket_width_seconds=60,
            )
        with self.assertRaises(ValueError):
            ForecastWindow(
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                end=datetime(2026, 9, 1, tzinfo=timezone.utc),
                bucket_width_seconds=DAY,
                granularity="fortnight",
            )

    def test_parameters_validation(self) -> None:
        with self.assertRaises(ValueError):
            ModelParameters(quantile=0.0)
        with self.assertRaises(ValueError):
            ModelParameters(quantile=1.0)
        with self.assertRaises(ValueError):
            ModelParameters(min_samples=0)
        with self.assertRaises(ValueError):
            ModelParameters(metric="revenue")

    def test_quantile_linear_interpolation(self) -> None:
        self.assertEqual(quantile_value([10, 20, 30, 40], 0.8), 34.0)
        self.assertEqual(quantile_value([5], 0.5), 5.0)

    def test_build_samples_buckets_and_filters_status_and_future(self) -> None:
        window = ForecastWindow(
            start=datetime(2026, 3, 1, tzinfo=timezone.utc),
            end=datetime(2026, 3, 7, tzinfo=timezone.utc),
            bucket_width_seconds=DAY,
        )
        at = lambda day, hour=10: datetime(2026, 3, day, hour, tzinfo=timezone.utc)
        bookings = [
            historical_booking("b1", at(1), 10, BookingStatus.SETTLED),
            historical_booking("b2", at(1), 5, BookingStatus.LOCKED),
            historical_booking("b3", at(3), 8, BookingStatus.SHIPPED),
            historical_booking("b4", at(4), 20, BookingStatus.CHECKED_IN),
            # 以下均不计入：取消/候补/过期未形成确定容量承诺
            historical_booking("b5", at(2), 30, BookingStatus.CANCELLED),
            historical_booking("b6", at(2), 30, BookingStatus.WAITLISTED),
            historical_booking("b7", at(2), 30, BookingStatus.EXPIRED),
        ]
        samples, observed, excluded_future = build_samples(
            window, bookings, metric=SEAT_METRIC, now=datetime(2026, 3, 20, tzinfo=timezone.utc)
        )
        self.assertEqual(samples, [15, 0, 8, 20, 0, 0])
        self.assertEqual(len(observed), 4)
        self.assertEqual(excluded_future, 0)

        # 开课时刻晚于 now 的排期属于未来，不进历史样本
        future_bookings = [historical_booking("b9", at(6), 12, BookingStatus.LOCKED)]
        samples_f, _, excluded = build_samples(
            window, future_bookings, metric=SEAT_METRIC, now=datetime(2026, 3, 2, tzinfo=timezone.utc)
        )
        self.assertEqual(samples_f, [0, 0, 0, 0, 0, 0])
        self.assertEqual(excluded, 1)

        # bookings 粒度按场次计数
        samples_b, _, _ = build_samples(window, bookings[:4], metric=BOOKING_METRIC)
        self.assertEqual(samples_b, [2, 0, 1, 1, 0, 0])

    def test_sample_statuses_have_determined_capacity_commitment(self) -> None:
        self.assertEqual(
            FORECAST_SAMPLE_STATUSES,
            {
                BookingStatus.LOCKED,
                BookingStatus.SHIPPED,
                BookingStatus.CHECKED_IN,
                BookingStatus.SETTLED,
            },
        )

    def test_forecast_without_history_is_critical_and_valueless(self) -> None:
        window = ForecastWindow(
            start=datetime(2026, 3, 1, tzinfo=timezone.utc),
            end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            bucket_width_seconds=30 * DAY,
        )
        result = forecast(window, ModelParameters(), [0, 0, 0, 0, 0, 0])
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNone(result["value"])
        self.assertEqual(result["max_warning_level"], int(WarningLevel.CRITICAL))
        warning = result["warnings"][0]
        self.assertEqual(warning["code"], "no_historical_data")
        self.assertTrue(warning["message"])  # 可解释说明非空
        self.assertIn("suggestion", warning["details"])

    def test_forecast_warning_levels(self) -> None:
        window = ForecastWindow(
            start=datetime(2026, 3, 1, tzinfo=timezone.utc),
            end=datetime(2026, 3, 6, tzinfo=timezone.utc),
            bucket_width_seconds=DAY,
        )
        # 仅 2 个有观测的桶，低于默认 min_samples=4 → WARNING；零桶 → INFO
        sparse = forecast(window, ModelParameters(quantile=0.8), [10, 20, 0, 0, 0])
        self.assertEqual(sparse["status"], "ok")
        # 排序后 [0,0,0,10,20]，P80 位于 3.2 → 10 + 0.2*10
        self.assertEqual(sparse["value"], 12.0)
        self.assertEqual(sparse["max_warning_level"], int(WarningLevel.WARNING))
        codes = {w["code"] for w in sparse["warnings"]}
        self.assertIn("insufficient_samples", codes)
        self.assertIn("sparse_window", codes)

        dense = forecast(window, ModelParameters(min_samples=4), [1, 2, 3, 4, 5])
        self.assertEqual(dense["max_warning_level"], int(WarningLevel.INFO))
        self.assertEqual(dense["warnings"], [])


class ForecastServiceTests(unittest.TestCase):
    def _seed_history(self, store, *, seats_by_month=(10, 12, 8, 15, 9, 11)) -> None:
        """直接向预约集合写入 3-8 月每月一场的已结算历史。"""
        for month, seats in enumerate(seats_by_month, start=3):
            booking = historical_booking(
                f"bkg-hist-{month}",
                datetime(2026, month, 5, 2, tzinfo=timezone.utc),
                seats,
            )
            store.put(COLLECTION_BOOKINGS, booking.booking_id, booking.to_dict())

    def _configure(self, forecasts: ForecastService, **overrides) -> dict:
        payload = {
            "window_start": "2026-03-01T00:00:00+00:00",
            "window_end": "2026-09-01T00:00:00+00:00",
            "bucket_width_seconds": 30 * DAY,
            "granularity": "day",
            "parameters": {"quantile": 0.8, "min_samples": 4, "metric": SEAT_METRIC},
        }
        payload.update(overrides)
        return forecasts.configure(payload)

    def test_configure_persists_window_and_parameters_as_version(self) -> None:
        forecasts, clock, ids, store = make_forecast_service()
        view = self._configure(forecasts)
        version_id = view["version_id"]
        self.assertTrue(version_id.startswith("fcv_"))
        self.assertEqual(view["window"]["bucket_width_seconds"], 30 * DAY)
        self.assertEqual(view["parameters"]["quantile"], 0.8)
        # 输入口径已落库
        self.assertIsNotNone(store.get(COLLECTION_FORECAST_INPUTS, version_id))
        self.assertEqual(forecasts.get_input(version_id)["version_id"], version_id)
        self.assertEqual(len(forecasts.list_inputs()), 1)

    def test_configure_validates_window(self) -> None:
        forecasts, *_ = make_forecast_service()
        with self.assertRaises(Exception):
            forecasts.configure(
                {
                    "window_start": "2026-09-01T00:00:00",  # 朴素时间
                    "window_end": "2026-09-02T00:00:00+00:00",
                    "bucket_width_seconds": DAY,
                }
            )

    def test_recompute_requires_existing_input_version(self) -> None:
        forecasts, *_ = make_forecast_service()
        with self.assertRaises(NotFoundError):
            forecasts.recompute("fcv_unknown")

    def test_recompute_is_deterministic_under_pinned_version(self) -> None:
        forecasts, clock, ids, store = make_forecast_service()
        self._seed_history(store)
        version = self._configure(forecasts)["version_id"]

        first = forecasts.recompute(version)
        second = forecasts.recompute(version)
        self.assertTrue(first["report_persisted"])
        self.assertEqual(first["input_hash"], second["input_hash"])
        self.assertEqual(first["result"]["value"], second["result"]["value"])
        self.assertEqual(first["samples"], second["samples"])
        self.assertEqual(first["result"]["status"], "ok")
        # 报告挂在该输入版本下
        report = forecasts.get_report(version)
        self.assertEqual(report["input_hash"], first["input_hash"])
        self.assertEqual(report["observed_buckets"], 6)

    def test_distinct_versions_hold_distinct_parameters(self) -> None:
        forecasts, _, _, store = make_forecast_service()
        self._seed_history(store)
        v_p80 = self._configure(forecasts)["version_id"]
        v_p50 = self._configure(
            forecasts, parameters={"quantile": 0.5, "min_samples": 4, "metric": SEAT_METRIC}
        )["version_id"]
        self.assertNotEqual(v_p80, v_p50)
        self.assertNotEqual(
            forecasts.recompute(v_p80)["result"]["value"],
            forecasts.recompute(v_p50)["result"]["value"],
        )

    def test_missing_history_returns_critical_warning_and_writes_no_report(self) -> None:
        forecasts, _, _, store = make_forecast_service()
        # 全新存储：没有任何历史预约
        version = self._configure(forecasts)["version_id"]
        result = forecasts.recompute(version)

        self.assertFalse(result["report_persisted"])
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNone(result["value"])
        self.assertEqual(result["max_warning_level"], int(WarningLevel.CRITICAL))
        self.assertEqual(result["warnings"][0]["code"], "no_historical_data")
        # 关键断言：不写空报告
        self.assertIsNone(store.get(COLLECTION_FORECAST_REPORTS, version))
        with self.assertRaises(NotFoundError):
            forecasts.get_report(version)

    def test_recompute_after_history_accumulates_then_succeeds(self) -> None:
        forecasts, _, _, store = make_forecast_service()
        version = self._configure(forecasts)["version_id"]
        # 先无历史 → 不写报告
        self.assertFalse(forecasts.recompute(version)["report_persisted"])
        # 历史沉淀后，同一输入版本重算成功
        self._seed_history(store)
        result = forecasts.recompute(version)
        self.assertTrue(result["report_persisted"])
        self.assertEqual(result["result"]["status"], "ok")
        self.assertIsNotNone(forecasts.get_report(version))

    def test_future_schedules_are_excluded_with_info_warning(self) -> None:
        forecasts, clock, ids, store = make_forecast_service()
        self._seed_history(store)
        # 再加一场落在采样窗口内、但尚未开课的未来排期（NOW=2026-09-25）
        future = historical_booking(
            "bkg-future", datetime(2026, 11, 5, tzinfo=timezone.utc), 40, BookingStatus.LOCKED
        )
        store.put(COLLECTION_BOOKINGS, future.booking_id, future.to_dict())
        version = self._configure(
            forecasts, window_end="2026-12-01T00:00:00+00:00"
        )["version_id"]
        result = forecasts.recompute(version)
        self.assertEqual(result["excluded_future"], 1)
        codes = {w["code"] for w in result["result"]["warnings"]}
        self.assertIn("future_observations_excluded", codes)
        # 未来场次的 40 席不得进入任一样本
        self.assertNotIn(40, [s["value"] for s in result["samples"]])


class ForecastSQLitePersistenceTests(unittest.TestCase):
    def test_inputs_and_report_survive_restart_and_recompute_reproduces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/booking.db"
            clock = ManualClock(NOW)
            ids = SequentialIdGenerator()

            store = SQLiteStore(db_path)
            forecasts = ForecastService(store, clock, ids)
            with store.transaction():
                for month, seats in enumerate((10, 12, 8, 15, 9, 11), start=3):
                    booking = historical_booking(
                        f"bkg-hist-{month}",
                        datetime(2026, month, 5, 2, tzinfo=timezone.utc),
                        seats,
                    )
                    store.put(COLLECTION_BOOKINGS, booking.booking_id, booking.to_dict())
            version = forecasts.configure(
                {
                    "window_start": "2026-03-01T00:00:00+00:00",
                    "window_end": "2026-09-01T00:00:00+00:00",
                    "bucket_width_seconds": 30 * DAY,
                    "parameters": {"quantile": 0.8, "min_samples": 4},
                }
            )["version_id"]
            first = forecasts.recompute(version)
            store.close()

            # 重启：新服务挂载同一 SQLite，输入版本仍可指向，重算结果可复现
            store2 = SQLiteStore(db_path)
            forecasts2 = ForecastService(store2, clock, SequentialIdGenerator())
            self.assertEqual(forecasts2.get_input(version)["version_id"], version)
            reopened_report = forecasts2.get_report(version)
            self.assertEqual(reopened_report["input_hash"], first["input_hash"])
            second = forecasts2.recompute(version)
            self.assertEqual(second["input_hash"], first["input_hash"])
            self.assertEqual(second["result"]["value"], first["result"]["value"])
            store2.close()

    def test_empty_history_in_sqlite_writes_no_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(f"{tmp}/booking.db")
            forecasts = ForecastService(store, ManualClock(NOW), SequentialIdGenerator())
            version = forecasts.configure(
                {
                    "window_start": "2026-03-01T00:00:00+00:00",
                    "window_end": "2026-09-01T00:00:00+00:00",
                    "bucket_width_seconds": 30 * DAY,
                }
            )["version_id"]
            result = forecasts.recompute(version)
            self.assertEqual(result["max_warning_level"], int(WarningLevel.CRITICAL))
            self.assertEqual(store.query(COLLECTION_FORECAST_REPORTS), [])
            store.close()


class ForecastHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog, bookings, clock, store = make_services()
        seed_catalog(catalog)
        forecasts = ForecastService(store, clock, SequentialIdGenerator())
        cls.forecasts = forecasts
        cls.server = create_server("127.0.0.1", 0, catalog, bookings, forecasts)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method
        )
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_configure_recompute_insufficient_then_report_404(self) -> None:
        status, created = self._request(
            "POST",
            "/capacity-forecasts",
            {
                "window_start": "2026-03-01T00:00:00+00:00",
                "window_end": "2026-09-01T00:00:00+00:00",
                "bucket_width_seconds": 30 * DAY,
            },
        )
        self.assertEqual(status, 200)
        version = created["version_id"]

        status, recomputed = self._request("POST", f"/capacity-forecasts/{version}/recompute")
        self.assertEqual(status, 200)
        self.assertFalse(recomputed["report_persisted"])
        self.assertEqual(recomputed["max_warning_level"], int(WarningLevel.CRITICAL))
        self.assertEqual(recomputed["warnings"][0]["code"], "no_historical_data")

        status, _ = self._request("GET", f"/capacity-forecasts/{version}/report")
        self.assertEqual(status, 404)

    def test_unknown_version_recompute_is_404(self) -> None:
        status, body = self._request("POST", "/capacity-forecasts/fcv_nope/recompute")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")


if __name__ == "__main__":
    unittest.main()
