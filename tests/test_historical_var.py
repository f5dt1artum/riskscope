import json
import math
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from threading import Thread

from riskscope.server import Handler
from riskscope.service import RiskError, Service


def body(
    confidence=0.95,
    positions=None,
    observations=None,
    currency=None,
):
    payload = {
        "confidence": confidence,
        "positions": positions
        if positions is not None
        else [{"id": "p1", "sensitivities": {"eq": 100.0}}],
        "observations": observations
        if observations is not None
        else [
            {"date": "2024-01-01", "factor_returns": {"eq": 0.01}},
            {"date": "2024-01-02", "factor_returns": {"eq": -0.02}},
        ],
    }
    if currency is not None:
        payload["currency"] = currency
    return payload


class HistoricalVarServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = Service()

    def test_basic_aggregation_and_indices(self) -> None:
        result = self.service.historical_var(
            body(
                confidence=0.75,
                positions=[
                    {"id": "p1", "sensitivities": {"eq": 100.0, "fx": 10.0}},
                    {"id": "p2", "sensitivities": {"eq": 50.0}},
                ],
                observations=[
                    {"date": "d1", "factor_returns": {"eq": 0.01, "fx": 0.0}},
                    {"date": "d2", "factor_returns": {"eq": -0.02, "fx": -1.0}},
                    {"date": "d3", "factor_returns": {"eq": 0.0, "fx": -0.5}},
                    {"date": "d4", "factor_returns": {"eq": -0.10, "fx": 0.0}},
                ],
            )
        )
        # Aggregate sensitivity: eq=150, fx=10 -> losses -1.5, 13, 5, 15.
        self.assertEqual(
            [(item["date"], item["loss"]) for item in result["losses"]],
            [("d1", -1.5), ("d2", 13.0), ("d3", 5.0), ("d4", 15.0)],
        )
        # Ascending order index ceil(0.75*4)-1 = 2 -> 13.
        self.assertEqual(result["var"], 13.0)
        # k = ceil(0.25*4) = 1 -> single worst observation.
        self.assertEqual(result["expected_shortfall"], 15.0)
        self.assertEqual(
            result["factor_expected_shortfall_contributions"], {"eq": 15.0, "fx": 0.0}
        )
        self.assertEqual(result["currency"], "USD")

    def test_contributions_sum_to_expected_shortfall(self) -> None:
        result = self.service.historical_var(
            body(
                confidence=0.5,
                positions=[
                    {"id": "p1", "sensitivities": {"a": 3.0}},
                    {"id": "p2", "sensitivities": {"b": -2.0}},
                ],
                observations=[
                    {"date": "d1", "factor_returns": {"a": 0.1, "b": 0.2}},
                    {"date": "d2", "factor_returns": {"a": -0.3, "b": 0.4}},
                    {"date": "d3", "factor_returns": {"a": 0.2, "b": -0.1}},
                ],
            )
        )
        contributions = result["factor_expected_shortfall_contributions"]
        self.assertEqual(
            math.fsum(contributions.values()), result["expected_shortfall"]
        )
        # Losses are reported in observation input order with their dates.
        self.assertEqual([item["date"] for item in result["losses"]], ["d1", "d2", "d3"])

    def test_tail_ties_resolve_by_input_order(self) -> None:
        result = self.service.historical_var(
            body(
                confidence=0.5,
                positions=[{"id": "p1", "sensitivities": {"x": 1.0, "y": 1.0}}],
                observations=[
                    {"date": "d0", "factor_returns": {"x": -1.0, "y": 0.0}},
                    {"date": "d1", "factor_returns": {"x": 0.0, "y": -1.0}},
                    {"date": "d2", "factor_returns": {"x": -0.5, "y": -0.5}},
                ],
            )
        )
        # k = ceil(0.5 * 3) = 2; every loss is equal, so d0 and d1 win.
        self.assertEqual(
            result["factor_expected_shortfall_contributions"], {"x": 0.5, "y": 0.5}
        )

    def test_extra_unreferenced_factors_are_ignored(self) -> None:
        result = self.service.historical_var(
            body(
                observations=[
                    {"date": "d1", "factor_returns": {"eq": 0.01, "other": 99}},
                    {"date": "d2", "factor_returns": {"eq": -0.02, "other": -99}},
                ]
            )
        )
        self.assertEqual([item["loss"] for item in result["losses"]], [-1.0, 2.0])

    def test_explicit_currency_is_preserved(self) -> None:
        result = self.service.historical_var(body(currency="EUR"))
        self.assertEqual(result["currency"], "EUR")

    def assert_error(self, payload, status, code):
        with self.assertRaises(RiskError) as caught:
            self.service.historical_var(payload)
        self.assertEqual(caught.exception.status, status)
        self.assertEqual(caught.exception.code, code)

    def test_confidence_must_be_strictly_between_zero_and_one(self) -> None:
        for value in (0, 1, -0.1, 1.5):
            self.assert_error(body(confidence=value), 422, "invalid_input")

    def test_collections_must_have_required_sizes(self) -> None:
        self.assert_error(body(positions=[]), 422, "invalid_input")
        self.assert_error(
            body(
                observations=[
                    {"date": "d1", "factor_returns": {"eq": 0.1}}
                ]
            ),
            422,
            "invalid_input",
        )

    def test_non_finite_numbers_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            self.assert_error(body(confidence=value), 422, "invalid_input")
            self.assert_error(
                body(
                    observations=[
                        {"date": "d1", "factor_returns": {"eq": value}},
                        {"date": "d2", "factor_returns": {"eq": 0.1}},
                    ]
                ),
                422,
                "invalid_input",
            )

    def test_field_type_errors_are_rejected(self) -> None:
        self.assert_error(body(confidence="0.9"), 422, "invalid_input")
        self.assert_error(
            body(positions=[{"id": 1, "sensitivities": {"eq": 1.0}}]),
            422,
            "invalid_input",
        )
        self.assert_error(
            body(positions=[{"id": "", "sensitivities": {"eq": 1.0}}]),
            422,
            "invalid_input",
        )
        self.assert_error(
            body(positions=[{"id": "p", "sensitivities": {"eq": True}}]),
            422,
            "invalid_input",
        )
        self.assert_error(
            body(observations=[{"date": "d1", "factor_returns": {}},
                               {"date": "d2", "factor_returns": {"eq": 1.0}}]),
            422,
            "invalid_input",
        )

    def test_duplicate_position_id(self) -> None:
        self.assert_error(
            body(
                positions=[
                    {"id": "p", "sensitivities": {"eq": 1.0}},
                    {"id": "p", "sensitivities": {"eq": 2.0}},
                ]
            ),
            422,
            "duplicate_position",
        )

    def test_duplicate_observation_date(self) -> None:
        self.assert_error(
            body(
                observations=[
                    {"date": "d", "factor_returns": {"eq": 0.1}},
                    {"date": "d", "factor_returns": {"eq": 0.2}},
                ]
            ),
            422,
            "duplicate_observation",
        )

    def test_missing_referenced_factor(self) -> None:
        self.assert_error(
            body(
                positions=[{"id": "p", "sensitivities": {"eq": 1.0, "fx": 1.0}}],
                observations=[
                    {"date": "d1", "factor_returns": {"eq": 0.1}},
                    {"date": "d2", "factor_returns": {"eq": 0.2, "fx": 0.1}},
                ],
            ),
            422,
            "missing_factor",
        )

    def test_too_many_observations(self) -> None:
        self.assert_error(
            body(
                observations=[
                    {"date": f"d{i}", "factor_returns": {"eq": 0.01}}
                    for i in range(10001)
                ]
            ),
            413,
            "request_too_large",
        )


class HistoricalVarHttpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()

    def post(self, raw, path="/market-risk/historical-var"):
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as err:
            payload = json.loads(err.read())
            err.close()
            return err.code, payload

    def test_success_over_http(self) -> None:
        status, payload = self.post(json.dumps(body()))
        self.assertEqual(status, 200)
        self.assertEqual(payload["currency"], "USD")
        self.assertEqual(len(payload["losses"]), 2)

    def test_malformed_json_is_invalid_request(self) -> None:
        status, payload = self.post("{not json")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")

    def test_non_object_top_level_is_invalid_request(self) -> None:
        status, payload = self.post("[1, 2, 3]")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")

    def test_validation_error_shape(self) -> None:
        bad = body(confidence=2)
        status, payload = self.post(json.dumps(bad))
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "invalid_input")
        self.assertNotIn("losses", payload)

    def test_unknown_post_route_is_not_found(self) -> None:
        status, payload = self.post("{}", path="/elsewhere")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")

    def test_health_unchanged(self) -> None:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/healthz"
        ) as response:
            self.assertEqual(response.status, 200)
            payload = json.loads(response.read())
        self.assertEqual(
            payload, {"status": "ok", "service": "riskscope", "version": "0.1.0"}
        )


if __name__ == "__main__":
    unittest.main()
