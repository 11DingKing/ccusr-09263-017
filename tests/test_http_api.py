"""HTTP 接口边界：路由、幂等头、错误映射与端到端流程。"""
from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

from service_09252_008.application.capacity_forecast_service import CapacityForecastService
from service_09252_008.application.ports import SystemClock, UuidIdGenerator
from service_09252_008.interfaces.http_api import create_server
from tests.helpers import make_services, seed_catalog


class HttpApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog, bookings, clock, store = make_services()
        forecast = CapacityForecastService(store, SystemClock(), UuidIdGenerator())
        cls.ids = seed_catalog(catalog)
        cls.server = create_server("127.0.0.1", 0, catalog, bookings, forecast)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _request(
        self, method: str, path: str, body: dict | None = None, headers: dict | None = None
    ) -> tuple[int, dict]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers=headers or {}
        )
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health(self) -> None:
        status, body = self._request("GET", "/health")
        self.assertEqual((status, body["status"]), (200, "ok"))

    def test_booking_flow_over_http_with_idempotency_header(self) -> None:
        apply_body = {
            "institution": "港城理工学院",
            "package_id": self.ids["package_id"],
            "mentor_id": self.ids["mentor_id"],
            "resource_id": self.ids["resource_id"],
            "window_id": self.ids["window_id"],
            "seats": 6,
            "slot_start": "2026-10-01T02:00:00+00:00",
            "slot_end": "2026-10-01T04:00:00+00:00",
        }
        status, applied = self._request(
            "POST", "/bookings", apply_body, headers={"Idempotency-Key": "http-apply-1"}
        )
        self.assertEqual(status, 201)
        self.assertEqual(applied["status"], "REQUESTED")
        booking_id = applied["booking_id"]

        # 相同幂等键重放：不产生新预约
        status, replay = self._request(
            "POST", "/bookings", apply_body, headers={"Idempotency-Key": "http-apply-1"}
        )
        self.assertEqual(status, 201)
        self.assertEqual(replay["booking_id"], booking_id)
        self.assertTrue(replay["idempotent_replay"])

        # 同一键不同载荷 -> 409
        status, conflict = self._request(
            "POST", "/bookings", {**apply_body, "seats": 8}, headers={"Idempotency-Key": "http-apply-1"}
        )
        self.assertEqual(status, 409)
        self.assertEqual(conflict["error"], "idempotency_conflict")

        status, quoted = self._request("POST", f"/bookings/{booking_id}/quote", {})
        self.assertEqual(status, 200)
        self.assertEqual(quoted["status"], "QUOTED")

        status, locked = self._request(
            "POST", f"/bookings/{booking_id}/lock", {}, headers={"Idempotency-Key": "http-lock-1"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(locked["status"], "LOCKED")

        status, fetched = self._request("GET", f"/bookings/{booking_id}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched["booking_id"], booking_id)
        self.assertEqual(len(fetched["reservations"]), 2)

    def test_error_mapping(self) -> None:
        status, body = self._request("GET", "/bookings/bkg_missing")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

        status, body = self._request("POST", "/bookings", {"institution": "x"})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "validation_error")

        status, body = self._request("POST", "/no-such-route", {})
        self.assertEqual(status, 404)

    def test_admin_recover_endpoint(self) -> None:
        status, body = self._request("POST", "/admin/recover", {})
        self.assertEqual(status, 200)
        self.assertIn("expired_locks", body)
        self.assertIn("expired_quotes", body)


class CapacityForecastHttpTests(unittest.TestCase):
    """容量需求预测的接口边界：登记口径、解算、显式版本重算。"""

    @classmethod
    def setUpClass(cls) -> None:
        catalog, bookings, clock, store = make_services()
        forecast = CapacityForecastService(store, clock, UuidIdGenerator())
        cls.server = create_server("127.0.0.1", 0, catalog, bookings, forecast)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _request(
        self, method: str, path: str, body: dict | None = None, headers: dict | None = None
    ) -> tuple[int, dict]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers=headers or {}
        )
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_forecast_flow_over_http(self) -> None:
        # 登记采样窗口与模型参数（历史窗口落在手动时钟 NOW 之前）
        config_body = {
            "history_start": "2026-09-01T00:00:00+00:00",
            "history_end": "2026-09-04T00:00:00+00:00",
            "bucket_seconds": 86400,
            "model_type": "moving_average",
        }
        status, config = self._request("POST", "/capacity-forecasts", config_body)
        self.assertEqual(status, 201)
        forecast_id = config["forecast_id"]
        self.assertEqual(config["params"]["model_type"], "moving_average")

        status, fetched = self._request("GET", f"/capacity-forecasts/{forecast_id}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched["forecast_id"], forecast_id)

        # 窗口内没有任何历史预约：返回 error 级可解释警告，不写空报告
        status, run = self._request("POST", f"/capacity-forecasts/{forecast_id}/run", {})
        self.assertEqual(status, 200)
        self.assertEqual(run["status"], "withheld")
        self.assertEqual(run["warning_level"], "error")
        self.assertEqual(run["warnings"][0]["code"], "no_history_data")
        self.assertIsNone(run["report"])
        version = run["input_version"]

        status, runs = self._request("GET", f"/capacity-forecasts/{forecast_id}/runs")
        self.assertEqual(status, 200)
        self.assertEqual(len(runs["runs"]), 1)

        # 重算必须显式指向输入版本
        status, missing = self._request("POST", f"/capacity-forecasts/{forecast_id}/recompute", {})
        self.assertEqual(status, 400)
        self.assertEqual(missing["error"], "validation_error")

        status, unknown = self._request(
            "POST", f"/capacity-forecasts/{forecast_id}/recompute", {"input_version": "0" * 64}
        )
        self.assertEqual(status, 404)

        # 数据未变：同一版本可重算，仍然是被抑制的空历史结论
        status, redone = self._request(
            "POST", f"/capacity-forecasts/{forecast_id}/recompute", {"input_version": version}
        )
        self.assertEqual(status, 200)
        self.assertTrue(redone["recomputed"])
        self.assertEqual(redone["input_version"], version)
        self.assertEqual(redone["status"], "withheld")

    def test_forecast_error_mapping(self) -> None:
        status, body = self._request("GET", "/capacity-forecasts/cap_missing")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

        status, body = self._request(
            "POST",
            "/capacity-forecasts",
            {
                "history_start": "2026-09-04T00:00:00+00:00",
                "history_end": "2026-09-01T00:00:00+00:00",
                "bucket_seconds": 86400,
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "validation_error")


if __name__ == "__main__":
    unittest.main()
