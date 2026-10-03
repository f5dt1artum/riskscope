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


def make_limits():
    return [
        {
            "id": "lim-var",
            "scope": {"type": "book", "id": "book-a"},
            "metric": "var",
            "unit": "USD",
            "limit": 100.0,
        },
        {
            "id": "lim-es",
            "scope": {"type": "book", "id": "book-b"},
            "metric": "expected_shortfall",
            "unit": "USD",
            "limit": 50.0,
            "warning_ratio": 0.5,
        },
        {
            "id": "lim-nodata",
            "scope": {"type": "desk", "id": "desk-1"},
            "metric": "var",
            "unit": "EUR",
            "limit": 10,
        },
    ]


def request_body(**overrides):
    body = {
        "as_of": "2024-06-30",
        "limits": make_limits(),
        "measurements": [
            {"limit_id": "lim-var", "value": 120.0},
            {"limit_id": "lim-es", "value": 30.0},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class LimitCheckTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.limit_check(request_body(**overrides))

    def by_id(self, entries, limit_id):
        matches = [e for e in entries if e["id"] == limit_id]
        self.assertEqual(len(matches), 1)
        return matches[0]

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["as_of"], "2024-06-30")
        # Input order is preserved and the definition is echoed.
        self.assertEqual(
            [entry["id"] for entry in result["limits"]],
            ["lim-var", "lim-es", "lim-nodata"],
        )

        var = self.by_id(result["limits"], "lim-var")
        self.assertEqual(var["scope"], {"type": "book", "id": "book-a"})
        self.assertEqual(var["metric"], "var")
        self.assertEqual(var["unit"], "USD")
        self.assertEqual(var["limit"], 100.0)
        # warning_ratio defaults to 0.8.
        self.assertEqual(var["warning_ratio"], 0.8)
        self.assertEqual(var["value"], 120.0)
        self.assertEqual(var["utilization"], 1.2)
        self.assertEqual(var["headroom"], -20.0)
        self.assertEqual(var["status"], "breach")

        es = self.by_id(result["limits"], "lim-es")
        self.assertEqual(es["warning_ratio"], 0.5)
        self.assertEqual(es["value"], 30.0)
        self.assertEqual(es["utilization"], 0.6)
        self.assertEqual(es["headroom"], 20.0)
        self.assertEqual(es["status"], "warning")

        nodata = self.by_id(result["limits"], "lim-nodata")
        self.assertIsNone(nodata["value"])
        self.assertIsNone(nodata["utilization"])
        self.assertIsNone(nodata["headroom"])
        self.assertEqual(nodata["status"], "no_data")

    def test_status_boundaries(self):
        limits = [
            {
                "id": "ok",
                "scope": {"type": "book", "id": "b"},
                "metric": "var",
                "unit": "USD",
                "limit": 100.0,
            },
            {
                "id": "warn",
                "scope": {"type": "book", "id": "b"},
                "metric": "var",
                "unit": "USD",
                "limit": 100.0,
            },
            {
                "id": "breach",
                "scope": {"type": "book", "id": "b"},
                "metric": "var",
                "unit": "USD",
                "limit": 100.0,
            },
        ]
        measurements = [
            # Just below the 0.8 warning threshold.
            {"limit_id": "ok", "value": 79.999999},
            # Exactly at the warning threshold warns.
            {"limit_id": "warn", "value": 80.0},
            # Exactly at the limit breaches.
            {"limit_id": "breach", "value": 100.0},
        ]
        result = self.call(limits=limits, measurements=measurements)
        self.assertEqual(
            [entry["status"] for entry in result["limits"]],
            ["ok", "warning", "breach"],
        )

    def test_zero_value_is_ok(self):
        result = self.call(
            limits=[make_limits()[0]],
            measurements=[{"limit_id": "lim-var", "value": 0}],
        )
        entry = result["limits"][0]
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["value"], 0)
        self.assertEqual(entry["utilization"], 0.0)
        self.assertEqual(entry["headroom"], 100.0)

    def test_measurements_may_be_omitted_or_empty(self):
        for body in (
            {"as_of": "2024-06-30", "limits": make_limits()},
            {"as_of": "2024-06-30", "limits": make_limits(), "measurements": []},
        ):
            result = self.service.limit_check(json.dumps(body).encode("utf-8"))
            self.assertEqual(
                [entry["status"] for entry in result["limits"]],
                ["no_data", "no_data", "no_data"],
            )
            self.assertEqual(result["alerts"], [])
            self.assertEqual(result["summary"]["no_data"], 3)
            self.assertEqual(result["overall_status"], "incomplete")

    def test_alerts_only_warning_and_breach_in_limit_order(self):
        result = self.call()
        self.assertEqual(
            [(a["id"], a["status"]) for a in result["alerts"]],
            [("lim-var", "breach"), ("lim-es", "warning")],
        )
        # Alert entries carry the full evaluation.
        self.assertEqual(result["alerts"][0]["utilization"], 1.2)

    def test_summary_and_overall_status(self):
        result = self.call()
        self.assertEqual(
            result["summary"], {"ok": 0, "warning": 1, "breach": 1, "no_data": 1}
        )
        self.assertEqual(result["overall_status"], "breach")

        # Warning outranks incomplete.
        result = self.call(
            measurements=[{"limit_id": "lim-es", "value": 30.0}]
        )
        self.assertEqual(result["overall_status"], "warning")

        # Only no_data besides ok gives incomplete.
        result = self.call(
            measurements=[{"limit_id": "lim-var", "value": 1.0}]
        )
        self.assertEqual(result["overall_status"], "incomplete")

        # Everything measured and under the warning threshold is ok.
        result = self.call(
            measurements=[
                {"limit_id": "lim-var", "value": 1.0},
                {"limit_id": "lim-es", "value": 1.0},
                {"limit_id": "lim-nodata", "value": 1.0},
            ]
        )
        self.assertEqual(result["overall_status"], "ok")
        self.assertEqual(
            result["summary"], {"ok": 3, "warning": 0, "breach": 0, "no_data": 0}
        )

    def test_extra_fields_ignored(self):
        body = json.loads(request_body())
        body["extra"] = "ignored"
        body["limits"][0]["venue"] = "x"
        body["limits"][0]["scope"]["desk"] = "y"
        body["measurements"][0]["source"] = "z"
        result = self.service.limit_check(json.dumps(body).encode("utf-8"))
        self.assertEqual(len(result["limits"]), 3)

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
            {"as_of": "2024-06-30", "limits": make_limits(), "measurements": {}},
        ]
        for body in cases:
            with self.assertRaises(InvalidInput, msg=repr(body)):
                self.service.limit_check(json.dumps(body).encode("utf-8"))

    def test_malformed_limits(self):
        good = make_limits()[0]
        mutations = [
            [None],
            ["x"],
            [{k: v for k, v in good.items() if k != "id"}],
            [{**good, "id": ""}],
            [{**good, "id": None}],
            [{**good, "id": 1}],
            [{**good, "scope": None}],
            [{**good, "scope": "book"}],
            [{**good, "scope": {"type": "", "id": "b"}}],
            [{**good, "scope": {"type": "book"}}],
            [{**good, "scope": {"type": "book", "id": ""}}],
            [{**good, "metric": ""}],
            [{**good, "metric": None}],
            [{**good, "unit": ""}],
            [{**good, "unit": None}],
            [{**good, "limit": 0.0}],
            [{**good, "limit": -1.0}],
            [{**good, "limit": "100"}],
            [{**good, "limit": True}],
            [{**good, "limit": None}],
            [{**good, "limit": 10**400}],
            [{**good, "warning_ratio": 0.0}],
            [{**good, "warning_ratio": 1.0}],
            [{**good, "warning_ratio": -0.5}],
            [{**good, "warning_ratio": "0.8"}],
            [{**good, "warning_ratio": True}],
            [{**good, "warning_ratio": None}],
        ]
        for limits in mutations:
            body = {"as_of": "2024-06-30", "limits": limits}
            with self.assertRaises(InvalidInput, msg=repr(limits)):
                self.service.limit_check(json.dumps(body).encode("utf-8"))

    def test_nan_and_infinity_rejected(self):
        template = (
            '{"as_of": "2024-06-30", "limits": [{"id": "l", '
            '"scope": {"type": "book", "id": "b"}, "metric": "var", '
            '"unit": "USD", "limit": __LIMIT__}]}'
        )
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.assertRaises(InvalidInput, msg=token):
                self.service.limit_check(
                    template.replace("__LIMIT__", token).encode("utf-8")
                )

    def test_malformed_measurements(self):
        good = {"limit_id": "lim-var", "value": 1.0}
        mutations = [
            [None],
            ["x"],
            [{k: v for k, v in good.items() if k != "limit_id"}],
            [{**good, "limit_id": ""}],
            [{**good, "limit_id": None}],
            [{**good, "value": None}],
            [{**good, "value": -1.0}],
            [{**good, "value": "1.0"}],
            [{**good, "value": True}],
            [{**good, "value": 10**400}],
        ]
        for measurements in mutations:
            body = {
                "as_of": "2024-06-30",
                "limits": make_limits(),
                "measurements": measurements,
            }
            with self.assertRaises(InvalidInput, msg=repr(measurements)):
                self.service.limit_check(json.dumps(body).encode("utf-8"))

    def test_duplicate_limit_id(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(limits=[make_limits()[0], make_limits()[0]])
        self.assertEqual(ctx.exception.code, "duplicate_limit")

    def test_duplicate_measurement(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                measurements=[
                    {"limit_id": "lim-var", "value": 1.0},
                    {"limit_id": "lim-var", "value": 2.0},
                ]
            )
        self.assertEqual(ctx.exception.code, "duplicate_measurement")

    def test_unknown_limit_reference(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(measurements=[{"limit_id": "nope", "value": 1.0}])
        self.assertEqual(ctx.exception.code, "unknown_limit")

    # ---- size limits ----

    def test_limits_boundary(self):
        def build(n):
            return {
                "as_of": "2024-06-30",
                "limits": [
                    {
                        "id": f"l{i}",
                        "scope": {"type": "book", "id": "b"},
                        "metric": "var",
                        "unit": "USD",
                        "limit": 100.0,
                    }
                    for i in range(n)
                ],
            }

        result = self.service.limit_check(json.dumps(build(10000)).encode("utf-8"))
        self.assertEqual(len(result["limits"]), 10000)
        with self.assertRaises(RequestTooLarge):
            self.service.limit_check(json.dumps(build(10001)).encode("utf-8"))

    def test_measurements_boundary(self):
        def build(n):
            return {
                "as_of": "2024-06-30",
                "limits": [
                    {
                        "id": f"l{i}",
                        "scope": {"type": "book", "id": "b"},
                        "metric": "var",
                        "unit": "USD",
                        "limit": 100.0,
                    }
                    for i in range(n)
                ],
                "measurements": [
                    {"limit_id": f"l{i}", "value": 1.0} for i in range(n)
                ],
            }

        result = self.service.limit_check(json.dumps(build(10000)).encode("utf-8"))
        self.assertEqual(result["summary"]["ok"], 10000)
        with self.assertRaises(RequestTooLarge):
            self.service.limit_check(json.dumps(build(10001)).encode("utf-8"))


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

    def test_invalid_json_returns_400_error_object(self):
        status, payload = self.post(b"nonsense")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")
        self.assertNotIn("limits", payload)

    def test_unknown_limit_returns_422(self):
        status, payload = self.post(
            request_body(measurements=[{"limit_id": "nope", "value": 1.0}])
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "unknown_limit")

    def test_too_large_returns_413(self):
        body = json.dumps(
            {
                "as_of": "2024-06-30",
                "limits": [
                    {
                        "id": f"l{i}",
                        "scope": {"type": "book", "id": "b"},
                        "metric": "var",
                        "unit": "USD",
                        "limit": 100.0,
                    }
                    for i in range(10001)
                ],
            }
        ).encode("utf-8")
        status, payload = self.post(body)
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "request_too_large")


if __name__ == "__main__":
    unittest.main()
