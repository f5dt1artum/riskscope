import json
import math
import unittest

from riskscope.service import (
    InvalidInput,
    InvalidRequest,
    RequestTooLarge,
    Service,
)


def request_body(**overrides):
    body = {
        "confidence": 0.95,
        "positions": [
            {"id": "p1", "sensitivities": {"eq": 100.0, "ir": -50.0}},
            {"id": "p2", "sensitivities": {"eq": 25.0}},
        ],
        "observations": [
            {"date": "2024-01-01", "factor_returns": {"eq": 0.01, "ir": 0.0, "extra": 9.0}},
            {"date": "2024-01-02", "factor_returns": {"eq": -0.02, "ir": 0.01}},
            {"date": "2024-01-03", "factor_returns": {"eq": 0.03, "ir": -0.02}},
            {"date": "2024-01-04", "factor_returns": {"eq": -0.04, "ir": 0.02}},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class HistoricalVarTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.historical_var(request_body(**overrides))

    def test_basic_shapes_and_values(self):
        result = self.call()
        # Aggregate eq sensitivity = 125, ir = -50.
        # Loss = -(125*eq + -50*ir): [-1.25, 3.0, -4.75, 6.0]
        losses = [item["loss"] for item in result["losses"]]
        self.assertEqual(losses, [-1.25, 3.0, -4.75, 6.0])
        self.assertEqual([item["date"] for item in result["losses"]],
                         ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
        self.assertEqual(result["currency"], "USD")
        # n=4, c=.95 -> sorted index ceil(3.8)-1 = 3 -> worst = 6.0
        self.assertEqual(result["var"], 6.0)
        # k = max(1, ceil(.05*4)=1) -> worst one only
        self.assertEqual(result["expected_shortfall"], 6.0)
        # Worst observation (#4): eq loss = 5.0, ir loss = 1.0
        self.assertEqual(
            result["factor_expected_shortfall_contributions"], {"eq": 5.0, "ir": 1.0}
        )
        contrib_sum = sum(result["factor_expected_shortfall_contributions"].values())
        self.assertTrue(math.isclose(contrib_sum, result["expected_shortfall"], rel_tol=0, abs_tol=1e-12))

    def test_tail_tie_keeps_input_order(self):
        # d1 and d2 both lose 10 overall but through different factors.
        # n=5, c=.8 -> k=ceil(.2*5)=1: the earlier input row owns the tail.
        positions = [{"id": "p", "sensitivities": {"a": 100.0, "b": 100.0}}]
        obs = [
            {"date": "d1", "factor_returns": {"a": -0.10, "b": 0.0}},
            {"date": "d2", "factor_returns": {"a": 0.0, "b": -0.10}},
            {"date": "d3", "factor_returns": {"a": 0.0, "b": 0.0}},
            {"date": "d4", "factor_returns": {"a": 0.05, "b": 0.0}},
            {"date": "d5", "factor_returns": {"a": 0.0, "b": 0.05}},
        ]
        result = self.service.historical_var(
            request_body(confidence=0.8, positions=positions, observations=obs)
        )
        self.assertEqual(result["var"], 10.0)
        self.assertEqual(result["expected_shortfall"], 10.0)
        self.assertEqual(
            result["factor_expected_shortfall_contributions"], {"a": 10.0, "b": 0.0}
        )

    def test_extra_unreferenced_factor_ignored(self):
        result = self.call()
        self.assertNotIn("extra", result["factor_expected_shortfall_contributions"])

    def test_currency_override(self):
        result = self.service.historical_var(request_body(currency="EUR"))
        self.assertEqual(result["currency"], "EUR")

    def test_integer_numbers_accepted(self):
        body = {
            "confidence": 0.9,
            "positions": [{"id": "p", "sensitivities": {"f": 100}}],
            "observations": [
                {"date": "a", "factor_returns": {"f": -1}},
                {"date": "b", "factor_returns": {"f": 1}},
            ],
        }
        result = self.service.historical_var(json.dumps(body).encode())
        self.assertEqual(result["var"], 100.0)

    # ---- parse-level failures -> 400 invalid_request ----

    def test_malformed_json(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.historical_var(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_top_level_array(self):
        with self.assertRaises(InvalidRequest):
            self.service.historical_var(b"[]")

    def test_top_level_scalar(self):
        with self.assertRaises(InvalidRequest):
            self.service.historical_var(b"42")

    # ---- semantic failures -> 422 ----

    def test_confidence_out_of_range(self):
        for bad in (0, 1, -0.5, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput) as ctx:
                    self.call(confidence=bad)
                self.assertEqual(ctx.exception.code, "invalid_input")

    def test_missing_fields_and_wrong_types(self):
        with self.assertRaises(InvalidInput):
            self.service.historical_var(b"{}")
        with self.assertRaises(InvalidInput):
            self.call(positions=[])
        with self.assertRaises(InvalidInput):
            self.call(observations=[
                {"date": "a", "factor_returns": {"f": 0.0}},
            ])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "", "sensitivities": {"f": 1.0}}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p", "sensitivities": {}}])
        with self.assertRaises(InvalidInput):
            self.call(observations=[
                {"date": "a", "factor_returns": {"f": "x"}},
                {"date": "b", "factor_returns": {"f": 1.0}},
            ])
        with self.assertRaises(InvalidInput):
            self.call(confidence=True)

    def test_nan_and_infinity_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                body = request_body().replace(b"0.95", token.encode(), 1)
                with self.assertRaises(InvalidInput) as ctx:
                    self.service.historical_var(body)
                self.assertEqual(ctx.exception.code, "invalid_input")

    def test_duplicate_position(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(positions=[
                {"id": "p", "sensitivities": {"f": 1.0}},
                {"id": "p", "sensitivities": {"g": 2.0}},
            ])
        self.assertEqual(ctx.exception.code, "duplicate_position")

    def test_duplicate_observation(self):
        obs = [
            {"date": "d", "factor_returns": {"eq": 0.0, "ir": 0.0}},
            {"date": "d", "factor_returns": {"eq": 0.0, "ir": 0.0}},
        ]
        with self.assertRaises(InvalidInput) as ctx:
            self.call(observations=obs)
        self.assertEqual(ctx.exception.code, "duplicate_observation")

    def test_missing_factor(self):
        obs = [
            {"date": "a", "factor_returns": {"eq": 0.01}},
            {"date": "b", "factor_returns": {"eq": 0.02, "ir": 0.01}},
        ]
        with self.assertRaises(InvalidInput) as ctx:
            self.call(observations=obs)
        self.assertEqual(ctx.exception.code, "missing_factor")

    def test_too_many_observations(self):
        obs = [{"date": f"d{i}", "factor_returns": {"eq": 0.0, "ir": 0.0}}
               for i in range(10001)]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(observations=obs)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

    def test_exactly_10000_observations_allowed(self):
        obs = [{"date": f"d{i:05d}", "factor_returns": {"eq": 0.01, "ir": 0.0}}
               for i in range(10000)]
        result = self.service.historical_var(
            request_body(observations=obs)
        )
        self.assertEqual(len(result["losses"]), 10000)


if __name__ == "__main__":
    unittest.main()
