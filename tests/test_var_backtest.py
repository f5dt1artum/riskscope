import json
import math
import unittest

from riskscope.service import (
    InvalidInput,
    InvalidRequest,
    RequestTooLarge,
    Service,
)


def observation(date, var, realized_pnl):
    return {"date": date, "var": var, "realized_pnl": realized_pnl}


def request_body(**overrides):
    body = {
        "currency": "USD",
        "confidence": 0.95,
        "significance": 0.05,
        "observations": [
            # loss = -pnl; breach iff loss strictly greater than var.
            observation("2024-01-01", 10.0, -5.0),    # loss 5 <= 10
            observation("2024-01-02", 10.0, -10.0),   # loss 10 == var, no breach
            observation("2024-01-03", 10.0, -12.0),   # loss 12 > 10, breach
            observation("2024-01-04", 10.0, 3.0),     # gain, loss -3
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


def reference_kupiec(n, x, confidence):
    p = 1 - confidence
    q = x / n

    def count_log(count, probability):
        return 0.0 if count == 0 else count * math.log(probability)

    lr = -2 * (
        count_log(n - x, 1 - p)
        + count_log(x, p)
        - count_log(n - x, 1 - q)
        - count_log(x, q)
    )
    if lr < 0:
        lr = 0.0
    return lr, math.erfc(math.sqrt(lr / 2))


class VarBacktestTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.var_backtest(request_body(**overrides))

    def test_basic_shapes_values_and_order(self):
        result = self.call()
        self.assertEqual(
            [item["date"] for item in result["observations"]],
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        )
        first = result["observations"][0]
        self.assertEqual(set(first), {"date", "var", "realized_pnl", "loss", "breach"})
        self.assertEqual(first, {
            "date": "2024-01-01", "var": 10.0,
            "realized_pnl": -5.0, "loss": 5.0, "breach": False,
        })
        # Equality is not a breach; the gain day has a negative loss.
        self.assertEqual(
            [item["breach"] for item in result["observations"]],
            [False, False, True, False],
        )
        self.assertEqual(
            [item["loss"] for item in result["observations"]],
            [5.0, 10.0, 12.0, -3.0],
        )
        self.assertEqual(result["observation_count"], 4)
        self.assertEqual(result["breach_count"], 1)
        self.assertEqual(result["breach_rate"], 0.25)
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["significance"], 0.05)
        self.assertEqual(
            set(result["kupiec"]), {"lr_statistic", "p_value", "accepted"}
        )
        lr, p_value = reference_kupiec(4, 1, 0.95)
        self.assertTrue(math.isclose(result["kupiec"]["lr_statistic"], lr, abs_tol=1e-12))
        self.assertTrue(math.isclose(result["kupiec"]["p_value"], p_value, abs_tol=1e-12))
        # p_value ~0.18 >= 0.05 -> the coverage hypothesis is accepted.
        self.assertEqual(
            result["kupiec"]["accepted"],
            result["kupiec"]["p_value"] >= result["significance"],
        )
        self.assertTrue(result["kupiec"]["accepted"])

    def test_strict_breach_boundary(self):
        obs = [observation("a", 7.0, -7.0), observation("b", 7.0, -7.0000001)]
        result = self.service.var_backtest(request_body(observations=obs))
        self.assertEqual([item["breach"] for item in result["observations"]], [False, True])
        self.assertEqual(result["breach_count"], 1)

    def test_zero_var_breaches_on_any_loss(self):
        obs = [observation("a", 0.0, 0.0), observation("b", 0.0, -1e-300)]
        result = self.service.var_backtest(request_body(observations=obs))
        self.assertEqual([item["breach"] for item in result["observations"]], [False, True])

    def test_integer_numbers_accepted(self):
        body = {
            "confidence": 90 / 100,
            "observations": [
                observation("a", 10, -5),
                observation("b", 10, -20),
            ],
        }
        result = self.service.var_backtest(json.dumps(body).encode())
        self.assertEqual(result["observations"][0]["var"], 10.0)
        self.assertEqual(result["observations"][0]["realized_pnl"], -5.0)
        self.assertEqual(result["significance"], 0.05)
        self.assertEqual(result["currency"], "USD")

    def test_defaults_for_currency_and_significance(self):
        body = {
            "confidence": 0.99,
            "observations": [
                observation("a", 1.0, 0.0),
                observation("b", 1.0, 0.0),
            ],
        }
        result = self.service.var_backtest(json.dumps(body).encode())
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["significance"], 0.05)

    def test_currency_override(self):
        result = self.call(currency="EUR")
        self.assertEqual(result["currency"], "EUR")

    def test_extra_fields_ignored(self):
        result = self.service.var_backtest(
            request_body(note="hi", unknown={"nested": 1},
                         observations=[dict(observation("a", 1.0, 0.0), tag=1),
                                       dict(observation("b", 1.0, 0.0), tag=2)])
        )
        self.assertEqual(result["observation_count"], 2)

    def test_perfect_coverage_kupiec(self):
        # n=100, c=.95, x=5 is exactly the expected breach rate -> lr 0, p 1.
        obs = [observation(f"d{i:03d}", 1.0, -2.0 if i < 5 else 0.0)
               for i in range(100)]
        result = self.service.var_backtest(
            request_body(observations=obs, confidence=0.95)
        )
        self.assertEqual(result["breach_count"], 5)
        self.assertEqual(result["breach_rate"], 0.05)
        self.assertEqual(result["kupiec"]["lr_statistic"], 0.0)
        self.assertEqual(result["kupiec"]["p_value"], 1.0)
        self.assertTrue(result["kupiec"]["accepted"])

    def test_no_breaches_and_all_breaches(self):
        # x == 0 exercises the 0 log 0 convention.
        obs = [observation(f"d{i}", 1.0, 0.0) for i in range(100)]
        result = self.service.var_backtest(request_body(observations=obs))
        self.assertEqual(result["breach_count"], 0)
        self.assertEqual(result["breach_rate"], 0.0)
        lr, p_value = reference_kupiec(100, 0, 0.95)
        self.assertTrue(math.isclose(result["kupiec"]["lr_statistic"], lr, abs_tol=1e-10))
        self.assertTrue(math.isclose(result["kupiec"]["p_value"], p_value, abs_tol=1e-12))
        self.assertFalse(result["kupiec"]["accepted"])

        obs = [observation(f"d{i}", 1.0, -2.0) for i in range(100)]
        result = self.service.var_backtest(request_body(observations=obs))
        self.assertEqual(result["breach_count"], 100)
        self.assertEqual(result["breach_rate"], 1.0)
        lr, p_value = reference_kupiec(100, 100, 0.95)
        self.assertTrue(math.isclose(result["kupiec"]["lr_statistic"], lr, abs_tol=1e-10))
        self.assertTrue(math.isclose(result["kupiec"]["p_value"], p_value, abs_tol=1e-12))

    def test_accepted_uses_p_value_against_significance(self):
        # n=250, c=.95: x=19 -> p_value ~0.079 (accepted at 0.05),
        # x=20 -> p_value ~0.044 (rejected).
        def build(x):
            return [observation(f"d{i:03d}", 1.0, -2.0 if i < x else 0.0)
                    for i in range(250)]

        accepted = self.service.var_backtest(
            request_body(observations=build(19), significance=0.05)
        )
        self.assertGreaterEqual(
            accepted["kupiec"]["p_value"], 0.05)
        self.assertTrue(accepted["kupiec"]["accepted"])

        rejected = self.service.var_backtest(
            request_body(observations=build(20), significance=0.05)
        )
        self.assertLess(rejected["kupiec"]["p_value"], 0.05)
        self.assertFalse(rejected["kupiec"]["accepted"])

    def test_accepted_boundary_is_inclusive(self):
        # Construct a case where p_value is (near) the significance and confirm
        # equality accepts: accepted iff p_value >= significance.
        result = self.call()
        p_value = result["kupiec"]["p_value"]
        again = self.call(significance=p_value)
        self.assertTrue(again["kupiec"]["accepted"])
        self.assertFalse(self.call(significance=p_value + 1e-12)["kupiec"]["accepted"])

    def test_lr_statistic_matches_reference_across_counts(self):
        for n, x, c in [(2, 1, 0.9), (50, 0, 0.99), (50, 10, 0.9),
                        (365, 30, 0.95), (10, 10, 0.5)]:
            with self.subTest(n=n, x=x, c=c):
                obs = [observation(f"d{i:04d}", 1.0, -2.0 if i < x else 0.0)
                       for i in range(n)]
                result = self.service.var_backtest(
                    request_body(observations=obs, confidence=c)
                )
                lr, p_value = reference_kupiec(n, x, c)
                self.assertTrue(
                    math.isclose(result["kupiec"]["lr_statistic"], lr,
                                 rel_tol=1e-12, abs_tol=1e-12)
                )
                self.assertTrue(
                    math.isclose(result["kupiec"]["p_value"], p_value, abs_tol=1e-12)
                )

    def test_result_is_standard_json_serializable(self):
        result = self.call()
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)
        json.loads(encoded)

    # ---- parse-level failures -> 400 invalid_request ----

    def test_malformed_json(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.var_backtest(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_top_level_array(self):
        with self.assertRaises(InvalidRequest):
            self.service.var_backtest(b"[]")

    def test_top_level_scalar(self):
        with self.assertRaises(InvalidRequest):
            self.service.var_backtest(b"42")

    def test_empty_body(self):
        with self.assertRaises(InvalidRequest):
            self.service.var_backtest(b"")

    # ---- semantic failures -> 422 invalid_input ----

    def test_missing_fields(self):
        with self.assertRaises(InvalidInput):
            self.service.var_backtest(b"{}")
        with self.assertRaises(InvalidInput):
            self.service.var_backtest(json.dumps({"confidence": 0.95}))
        with self.assertRaises(InvalidInput):
            self.service.var_backtest(
                json.dumps({"observations": [observation("a", 1.0, 0.0),
                                             observation("b", 1.0, 0.0)]})
            )

    def test_confidence_out_of_range(self):
        for bad in (0, 1, -0.5, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput) as ctx:
                    self.call(confidence=bad)
                self.assertEqual(ctx.exception.code, "invalid_input")

    def test_significance_out_of_range(self):
        for bad in (0, 1, -0.1, 2.0):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(significance=bad)

    def test_wrong_types(self):
        with self.assertRaises(InvalidInput):
            self.call(confidence="0.95")
        with self.assertRaises(InvalidInput):
            self.call(significance="0.05")
        with self.assertRaises(InvalidInput):
            self.call(currency=123)
        with self.assertRaises(InvalidInput):
            self.call(currency="")
        with self.assertRaises(InvalidInput):
            self.call(observations="nope")
        with self.assertRaises(InvalidInput):
            self.call(observations=["nope", observation("b", 1.0, 0.0)])
        with self.assertRaises(InvalidInput):
            self.call(observations=[{"date": "a", "var": 1.0},
                                    observation("b", 1.0, 0.0)])
        with self.assertRaises(InvalidInput):
            self.call(observations=[{"date": "", "var": 1.0, "realized_pnl": 0.0},
                                    observation("b", 1.0, 0.0)])
        with self.assertRaises(InvalidInput):
            self.call(observations=[{"date": 1, "var": 1.0, "realized_pnl": 0.0},
                                    observation("b", 1.0, 0.0)])

    def test_booleans_rejected_everywhere(self):
        with self.assertRaises(InvalidInput):
            self.call(confidence=True)
        with self.assertRaises(InvalidInput):
            self.call(significance=False)
        with self.assertRaises(InvalidInput):
            self.call(observations=[{"date": "a", "var": True, "realized_pnl": 0.0},
                                    observation("b", 1.0, 0.0)])
        with self.assertRaises(InvalidInput):
            self.call(observations=[{"date": "a", "var": 0.0, "realized_pnl": True},
                                    observation("b", 1.0, 0.0)])

    def test_var_must_be_non_negative_and_finite(self):
        for bad in (-0.01, -1.0):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(observations=[observation("a", bad, 0.0),
                                            observation("b", 1.0, 0.0)])
        # zero is allowed
        result = self.call(observations=[observation("a", 0, 0.0),
                                         observation("b", 0.0, 0.0)])
        self.assertEqual(result["breach_count"], 0)

    def test_realized_pnl_must_be_finite_but_may_be_negative(self):
        with self.assertRaises(InvalidInput):
            self.call(observations=[observation("a", 1.0, "x"),
                                    observation("b", 1.0, 0.0)])
        result = self.call(observations=[observation("a", 1.0, -1e300),
                                         observation("b", 1.0, 1e300)])
        self.assertEqual(result["breach_count"], 1)

    def test_nan_and_infinity_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                body = request_body().replace(b"0.95", token.encode(), 1)
                with self.assertRaises(InvalidInput) as ctx:
                    self.service.var_backtest(body)
                self.assertEqual(ctx.exception.code, "invalid_input")
        for field in ("var", "realized_pnl"):
            for token in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(field=field, token=token):
                    obs = [dict(observation("a", 1.0, 0.0)),
                           dict(observation("b", 1.0, 0.0))]
                    obs[0][field] = token  # stays a string in the JSON text
                    text = json.dumps(
                        {"confidence": 0.95, "observations": obs}
                    ).replace(f'"{token}"', token)
                    with self.assertRaises(InvalidInput):
                        self.service.var_backtest(text.encode())

    def test_oversized_integer_rejected(self):
        obs = [observation("a", 10 ** 400, 0.0), observation("b", 1.0, 0.0)]
        with self.assertRaises(InvalidInput):
            self.service.var_backtest(request_body(observations=obs))
        obs = [observation("a", 1.0, 10 ** 400), observation("b", 1.0, 0.0)]
        with self.assertRaises(InvalidInput):
            self.service.var_backtest(request_body(observations=obs))

    def test_observation_count_bounds(self):
        for count in (0, 1):
            with self.subTest(count=count):
                with self.assertRaises(InvalidInput):
                    self.call(observations=[
                        observation(f"d{i}", 1.0, 0.0) for i in range(count)
                    ])

    def test_duplicate_observation(self):
        obs = [observation("2024-01-01", 1.0, 0.0),
               observation("2024-01-01", 1.0, -2.0)]
        with self.assertRaises(InvalidInput) as ctx:
            self.call(observations=obs)
        self.assertEqual(ctx.exception.code, "duplicate_observation")

    # ---- size limits -> 413 ----

    def test_too_many_observations(self):
        obs = [observation(f"d{i:05d}", 1.0, 0.0) for i in range(10001)]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(observations=obs)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

    def test_exactly_10000_observations_allowed(self):
        obs = [observation(f"d{i:05d}", 1.0, -2.0 if i < 500 else 0.0)
               for i in range(10000)]
        result = self.service.var_backtest(
            request_body(observations=obs, confidence=0.95)
        )
        self.assertEqual(len(result["observations"]), 10000)
        self.assertEqual(result["observation_count"], 10000)
        self.assertEqual(result["breach_count"], 500)


if __name__ == "__main__":
    unittest.main()
