import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from riskscope.server import Handler
from riskscope.service import (
    InvalidInput,
    InvalidRequest,
    RequestTooLarge,
    Service,
)


def limit_entry(limit_id, **overrides):
    entry = {
        "id": limit_id,
        "scope": {"type": "book", "id": f"scope-{limit_id}"},
        "metric": "var",
        "unit": "USD",
        "limit": 100.0,
    }
    entry.update(overrides)
    return entry


def request_body(**overrides):
    body = {
        "as_of": "2024-06-30",
        "limits": [
            limit_entry("lim-ok"),
            limit_entry("lim-warning"),
            limit_entry("lim-breach"),
            limit_entry("lim-nodata"),
        ],
        "measurements": [
            {"limit_id": "lim-ok", "value": 50.0},
            {"limit_id": "lim-warning", "value": 85.0},
            {"limit_id": "lim-breach", "value": 120.0},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class LimitCheckTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.limit_check(request_body(**overrides))

    def by_id(self, limits, limit_id):
        matches = [entry for entry in limits if entry["id"] == limit_id]
        self.assertEqual(len(matches), 1)
        return matches[0]

    # ---- success shapes and values ----

    def test_basic_statuses_and_values(self):
        result = self.call()
        self.assertEqual(result["as_of"], "2024-06-30")
        # Limits come back in input order with the definition echoed.
        self.assertEqual(
            [entry["id"] for entry in result["limits"]],
            ["lim-ok", "lim-warning", "lim-breach", "lim-nodata"],
        )

        ok = self.by_id(result["limits"], "lim-ok")
        self.assertEqual(ok["scope"], {"type": "book", "id": "scope-lim-ok"})
        self.assertEqual(ok["metric"], "var")
        self.assertEqual(ok["unit"], "USD")
        self.assertEqual(ok["limit"], 100.0)
        self.assertEqual(ok["warning_ratio"], 0.8)
        self.assertEqual(ok["value"], 50.0)
        self.assertEqual(ok["utilization"], 0.5)
        self.assertEqual(ok["headroom"], 50.0)
        self.assertEqual(ok["status"], "ok")

        warning = self.by_id(result["limits"], "lim-warning")
        self.assertEqual(warning["utilization"], 0.85)
        self.assertEqual(warning["headroom"], 15.0)
        self.assertEqual(warning["status"], "warning")

        breach = self.by_id(result["limits"], "lim-breach")
        self.assertEqual(breach["utilization"], 1.2)
        self.assertEqual(breach["headroom"], -20.0)
        self.assertEqual(breach["status"], "breach")

        nodata = self.by_id(result["limits"], "lim-nodata")
        self.assertIsNone(nodata["value"])
        self.assertIsNone(nodata["utilization"])
        self.assertIsNone(nodata["headroom"])
        self.assertEqual(nodata["status"], "no_data")

    def test_alerts_and_summary(self):
        result = self.call()
        # Alerts keep limit input order and carry only warning and breach.
        self.assertEqual(
            [entry["id"] for entry in result["alerts"]],
            ["lim-warning", "lim-breach"],
        )
        self.assertEqual(
            result["summary"],
            {"ok": 1, "warning": 1, "breach": 1, "no_data": 1},
        )
        self.assertEqual(result["overall_status"], "breach")

    def test_overall_status_priority(self):
        # Warning outranks incomplete.
        result = self.call(
            limits=[limit_entry("a"), limit_entry("b")],
            measurements=[{"limit_id": "a", "value": 90.0}],
        )
        self.assertEqual(result["overall_status"], "warning")
        # No_data alone degrades to incomplete.
        result = self.call(
            limits=[limit_entry("a"), limit_entry("b")],
            measurements=[{"limit_id": "a", "value": 10.0}],
        )
        self.assertEqual(result["overall_status"], "incomplete")
        self.assertEqual(result["alerts"], [])
        # All measured and healthy is ok.
        result = self.call(
            limits=[limit_entry("a")],
            measurements=[{"limit_id": "a", "value": 10.0}],
        )
        self.assertEqual(result["overall_status"], "ok")
        self.assertEqual(
            result["summary"],
            {"ok": 1, "warning": 0, "breach": 0, "no_data": 0},
        )

    def test_breach_and_warning_thresholds_are_inclusive(self):
        result = self.call(
            limits=[limit_entry("a"), limit_entry("b")],
            measurements=[
                # Exactly at the limit breaches.
                {"limit_id": "a", "value": 100.0},
                # Exactly at limit * warning_ratio warns.
                {"limit_id": "b", "value": 80.0},
            ],
        )
        self.assertEqual(self.by_id(result["limits"], "a")["status"], "breach")
        self.assertEqual(self.by_id(result["limits"], "a")["headroom"], 0.0)
        self.assertEqual(self.by_id(result["limits"], "b")["status"], "warning")

    def test_explicit_warning_ratio(self):
        result = self.call(
            limits=[limit_entry("a", warning_ratio=0.5)],
            measurements=[{"limit_id": "a", "value": 60.0}],
        )
        entry = self.by_id(result["limits"], "a")
        self.assertEqual(entry["warning_ratio"], 0.5)
        self.assertEqual(entry["status"], "warning")

    def test_measurements_may_be_omitted_or_empty(self):
        for body in (
            {"as_of": "2024-06-30", "limits": [limit_entry("a")]},
            {
                "as_of": "2024-06-30",
                "limits": [limit_entry("a")],
                "measurements": [],
            },
        ):
            result = self.service.limit_check(json.dumps(body).encode("utf-8"))
            self.assertEqual(result["limits"][0]["status"], "no_data")
            self.assertEqual(result["overall_status"], "incomplete")

    def test_zero_value_is_ok(self):
        result = self.call(
            limits=[limit_entry("a")],
            measurements=[{"limit_id": "a", "value": 0}],
        )
        entry = result["limits"][0]
        self.assertEqual(entry["value"], 0.0)
        self.assertEqual(entry["utilization"], 0.0)
        self.assertEqual(entry["headroom"], 100.0)
        self.assertEqual(entry["status"], "ok")

    def test_extra_fields_ignored(self):
        body = json.loads(request_body())
        body["extra"] = "ignored"
        body["limits"][0]["venue"] = "x"
        body["limits"][0]["scope"]["desk"] = "y"
        body["measurements"][0]["source"] = "z"
        result = self.service.limit_check(json.dumps(body).encode("utf-8"))
        self.assertEqual(len(result["limits"]), 4)
        self.assertNotIn("venue", result["limits"][0])
        self.assertEqual(
            result["limits"][0]["scope"], {"type": "book", "id": "scope-lim-ok"}
        )

    # ---- parse failures ----

    def test_invalid_json_is_invalid_request(self):
        with self.assertRaises(InvalidRequest):
            self.service.limit_check(b"{not json")

    def test_top_level_array_is_invalid_request(self):
        with self.assertRaises(InvalidRequest):
            self.service.limit_check(b"[1, 2]")

    # ---- semantic validation ----

    def test_missing_or_empty_top_level_fields(self):
        cases = [
            {},
            {"as_of": ""},
            {"as_of": None},
            {"as_of": 7},
            {"as_of": "2024-06-30"},
            {"as_of": "2024-06-30", "limits": []},
            {"as_of": "2024-06-30", "limits": None},
            {"as_of": "2024-06-30", "limits": {}},
            {
                "as_of": "2024-06-30",
                "limits": [limit_entry("a")],
                "measurements": None,
            },
            {
                "as_of": "2024-06-30",
                "limits": [limit_entry("a")],
                "measurements": {},
            },
        ]
        for body in cases:
            with self.assertRaises(InvalidInput, msg=repr(body)):
                self.service.limit_check(json.dumps(body).encode("utf-8"))

    def test_malformed_limits(self):
        good = limit_entry("a")
        mutations = [
            [None],
            ["x"],
            [{**good, "id": ""}],
            [{**good, "id": None}],
            [{**good, "scope": None}],
            [{**good, "scope": "book"}],
            [{**good, "scope": {"type": "", "id": "s"}}],
            [{**good, "scope": {"type": "book"}}],
            [{**good, "scope": {"type": "book", "id": ""}}],
            [{**good, "metric": ""}],
            [{**good, "metric": None}],
            [{**good, "unit": ""}],
            [{**good, "unit": None}],
            [{k: v for k, v in good.items() if k != "limit"}],
            [{**good, "limit": 0.0}],
            [{**good, "limit": -1.0}],
            [{**good, "limit": "100"}],
            [{**good, "limit": True}],
            [{**good, "limit": None}],
            [{**good, "warning_ratio": 0.0}],
            [{**good, "warning_ratio": 1.0}],
            [{**good, "warning_ratio": -0.5}],
            [{**good, "warning_ratio": "0.8"}],
            [{**good, "warning_ratio": False}],
        ]
        for limits in mutations:
            body = {"as_of": "2024-06-30", "limits": limits}
            with self.assertRaises(InvalidInput, msg=repr(limits)):
                self.service.limit_check(json.dumps(body).encode("utf-8"))

    def test_malformed_measurements(self):
        mutations = [
            [None],
            ["x"],
            [{"limit_id": "", "value": 1.0}],
            [{"limit_id": None, "value": 1.0}],
            [{"value": 1.0}],
            [{"limit_id": "a"}],
            [{"limit_id": "a", "value": -1.0}],
            [{"limit_id": "a", "value": "1"}],
            [{"limit_id": "a", "value": True}],
            [{"limit_id": "a", "value": None}],
        ]
        for measurements in mutations:
            body = {
                "as_of": "2024-06-30",
                "limits": [limit_entry("a")],
                "measurements": measurements,
            }
            with self.assertRaises(InvalidInput, msg=repr(measurements)):
                self.service.limit_check(json.dumps(body).encode("utf-8"))

    def test_nan_infinity_and_oversized_ints_rejected(self):
        template = (
            '{"as_of": "2024-06-30", "limits": [{"id": "a", '
            '"scope": {"type": "book", "id": "s"}, "metric": "var", '
            '"unit": "USD", "limit": __LIMIT__}], '
            '"measurements": [{"limit_id": "a", "value": __VALUE__}]}'
        )

        def body(limit="100.0", value="50.0"):
            return template.replace("__LIMIT__", limit).replace(
                "__VALUE__", value
            ).encode("utf-8")

        for token in ("NaN", "Infinity", "-Infinity"):
            with self.assertRaises(InvalidInput, msg="limit " + token):
                self.service.limit_check(body(limit=token))
            with self.assertRaises(InvalidInput, msg="value " + token):
                self.service.limit_check(body(value=token))
        huge = json.loads(body())
        huge["limits"][0]["limit"] = 10**400
        with self.assertRaises(InvalidInput):
            self.service.limit_check(json.dumps(huge).encode("utf-8"))
        huge = json.loads(body())
        huge["measurements"][0]["value"] = 10**400
        with self.assertRaises(InvalidInput):
            self.service.limit_check(json.dumps(huge).encode("utf-8"))

    def test_duplicate_limit_id(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                limits=[limit_entry("same"), limit_entry("same")],
                measurements=[],
            )
        self.assertEqual(ctx.exception.code, "duplicate_limit")

    def test_duplicate_measurement(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                limits=[limit_entry("a")],
                measurements=[
                    {"limit_id": "a", "value": 1.0},
                    {"limit_id": "a", "value": 2.0},
                ],
            )
        self.assertEqual(ctx.exception.code, "duplicate_measurement")

    def test_unknown_limit_reference(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                limits=[limit_entry("a")],
                measurements=[{"limit_id": "ghost", "value": 1.0}],
            )
        self.assertEqual(ctx.exception.code, "unknown_limit")

    # ---- size limits ----

    def test_limits_boundary(self):
        limits = [limit_entry(f"lim-{i}") for i in range(10000)]
        result = self.call(limits=limits, measurements=[])
        self.assertEqual(len(result["limits"]), 10000)
        self.assertEqual(result["summary"]["no_data"], 10000)
        with self.assertRaises(RequestTooLarge):
            self.call(
                limits=[limit_entry(f"lim-{i}") for i in range(10001)],
                measurements=[],
            )

    def test_measurements_boundary(self):
        limits = [limit_entry(f"lim-{i}") for i in range(10000)]
        measurements = [
            {"limit_id": f"lim-{i}", "value": 1.0} for i in range(10000)
        ]
        result = self.call(limits=limits, measurements=measurements)
        self.assertEqual(result["summary"]["ok"], 10000)
        # 10001 measurements exceed the cap; the size check fires before
        # the duplicate check the extra entry would otherwise trigger.
        with self.assertRaises(RequestTooLarge):
            self.call(
                limits=limits,
                measurements=measurements + [{"limit_id": "lim-0", "value": 2.0}],
            )

    def test_too_many_measurements_alone(self):
        # 10001 measurements against 2 limits: the size check fires before
        # any duplicate or unknown-reference check can.
        measurements = [
            {"limit_id": "lim-0", "value": 1.0} for _ in range(10001)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(
                limits=[limit_entry("lim-0")],
                measurements=measurements,
            )


class LimitCheckHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def post(self, body, path="/risk-management/limit-check"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_limit_check_route_ok(self):
        status, payload = self.post(request_body())
        self.assertEqual(status, 200)
        self.assertEqual(payload["as_of"], "2024-06-30")
        self.assertEqual(payload["overall_status"], "breach")
        self.assertEqual(len(payload["limits"]), 4)

    def test_invalid_json_returns_400_error_object(self):
        status, payload = self.post(b"nonsense")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")
        self.assertNotIn("limits", payload)

    def test_validation_error_returns_422(self):
        body = json.dumps(
            {
                "as_of": "2024-06-30",
                "limits": [limit_entry("a")],
                "measurements": [{"limit_id": "ghost", "value": 1.0}],
            }
        ).encode("utf-8")
        status, payload = self.post(body)
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "unknown_limit")
        self.assertNotIn("limits", payload)

    def test_too_large_returns_413(self):
        body = json.dumps(
            {
                "as_of": "2024-06-30",
                "limits": [limit_entry(f"lim-{i}") for i in range(10001)],
            }
        ).encode("utf-8")
        status, payload = self.post(body)
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "request_too_large")

    def test_existing_routes_unchanged(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/healthz")
        response = conn.getresponse()
        health = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(health["status"], "ok")

        status, payload = self.post(b"{}", path="/no-such-route")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
