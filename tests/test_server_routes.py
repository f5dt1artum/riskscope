import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from riskscope.server import Handler


def stress_body(**overrides):
    body = {
        "positions": [
            {"id": "p1", "sensitivities": {"eq": 100.0}},
        ],
        "scenarios": [
            {"id": "crash", "factor_shocks": {"eq": -0.1}},
            {"id": "rally", "factor_shocks": {"eq": 0.05}},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class RouteIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()

    def request(self, path, body=None, method="POST"):
        data = body if body is not None else b""
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data if method == "POST" else None,
            method=method,
        )
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health_unchanged(self):
        status, payload = self.request("/healthz", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "riskscope")

    def test_stress_test_success(self):
        status, payload = self.request("/market-risk/stress-test", stress_body())
        self.assertEqual(status, 200)
        self.assertEqual(payload["currency"], "USD")
        self.assertEqual([r["scenario_id"] for r in payload["results"]], ["crash", "rally"])
        self.assertEqual(payload["results"][0]["loss"], 10.0)
        self.assertEqual(payload["results"][1]["loss"], -5.0)
        self.assertEqual(payload["worst_scenario"], {"id": "crash", "loss": 10.0})
        crash = payload["results"][0]
        self.assertEqual(crash["position_loss_contributions"], {"p1": 10.0})
        self.assertEqual(crash["factor_loss_contributions"], {"eq": 10.0})

    def test_stress_test_empty_body_error_wrap(self):
        status, payload = self.request("/market-risk/stress-test", b"{}")
        self.assertEqual(status, 422)
        self.assertEqual(set(payload), {"error"})
        self.assertEqual(payload["error"]["code"], "invalid_input")
        self.assertIsInstance(payload["error"]["message"], str)
        self.assertTrue(payload["error"]["message"])

    def test_stress_test_malformed_json(self):
        status, payload = self.request("/market-risk/stress-test", b"{nope")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")

    def test_stress_test_duplicate_scenario(self):
        scenarios = [
            {"id": "s", "factor_shocks": {"eq": 0.0}},
            {"id": "s", "factor_shocks": {"eq": 1.0}},
        ]
        status, payload = self.request(
            "/market-risk/stress-test", stress_body(scenarios=scenarios)
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "duplicate_scenario")

    def test_stress_test_too_large(self):
        scenarios = [{"id": f"s{i}", "factor_shocks": {"eq": 0.0}} for i in range(1001)]
        status, payload = self.request(
            "/market-risk/stress-test", stress_body(scenarios=scenarios)
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "request_too_large")

    def test_historical_var_route_still_works(self):
        body = json.dumps({
            "confidence": 0.95,
            "positions": [{"id": "p", "sensitivities": {"eq": 100.0}}],
            "observations": [
                {"date": "a", "factor_returns": {"eq": 0.01}},
                {"date": "b", "factor_returns": {"eq": -0.02}},
            ],
        }).encode("utf-8")
        status, payload = self.request("/market-risk/historical-var", body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["currency"], "USD")
        self.assertEqual([item["loss"] for item in payload["losses"]], [-1.0, 2.0])

    def test_unknown_routes_404(self):
        status, payload = self.request("/market-risk/nope", stress_body())
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")
        status, payload = self.request("/healthz/nope", method="GET")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
