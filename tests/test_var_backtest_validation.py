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
        "observations": [
            {"date": "2024-01-01", "var": 10.0, "realized_pnl": 5.0},
            {"date": "2024-01-02", "var": 10.0, "realized_pnl": -15.0},
            {"date": "2024-01-03", "var": 10.0, "realized_pnl": -10.0},
            {"date": "2024-01-04", "var": 1.0, "realized_pnl": -1.0000001},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


def kupiec_lr(n, x, p):
    """Reference Kupiec POF likelihood-ratio statistic."""
    q = x / n

    def term(count, probability):
        return 0.0 if count == 0 else count * math.log(probability)

    lr = -2.0 * (
        term(n - x, 1 - p) + term(x, p)
        - term(n - x, 1 - q) - term(x, q)
    )
    return max(lr, 0.0)


def christoffersen(breaches):
    """Reference independence-test computation from the breach sequence."""
    n = len(breaches)
    n00 = n01 = n10 = n11 = 0
    for previous, current in zip(breaches, breaches[1:]):
        if previous:
            if current:
                n11 += 1
            else:
                n10 += 1
        elif current:
            n01 += 1
        else:
            n00 += 1
    q = (n01 + n11) / (n - 1)
    q0 = n01 / (n00 + n01) if n00 + n01 > 0 else None
    q1 = n11 / (n10 + n11) if n10 + n11 > 0 else None

    def term(count, probability):
        return 0.0 if count == 0 else count * math.log(probability)

    ln_l0 = term(n00 + n10, 1 - q) + term(n01 + n11, q)
    ln_l1 = 0.0
    if q0 is not None:
        ln_l1 += term(n00, 1 - q0) + term(n01, q0)
    if q1 is not None:
        ln_l1 += term(n10, 1 - q1) + term(n11, q1)
    lr = max(2.0 * (ln_l1 - ln_l0), 0.0)
    return (n00, n01, n10, n11), q, q0, q1, lr, math.erfc(math.sqrt(lr / 2.0))


def observations_for(breaches):
    """Build observations whose breach flags follow the given pattern."""
    return [
        {
            "date": f"d{i:02d}",
            "var": 1.0,
            "realized_pnl": -2.0 if breach else 0.0,
        }
        for i, breach in enumerate(breaches)
    ]


class VarBacktestValidationTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.var_backtest_validation(request_body(**overrides))

    # ---- base response preserved ----

    def test_base_content_preserved(self):
        base = self.service.var_backtest(request_body())
        result = self.call()
        for key in (
            "currency", "confidence", "significance", "observations",
            "observation_count", "breach_count", "breach_rate", "kupiec",
        ):
            self.assertEqual(result[key], base[key], key)
        self.assertEqual(
            [row["date"] for row in result["observations"]],
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        )

    def test_new_fields_present(self):
        result = self.call()
        for key in ("n00", "n01", "n10", "n11", "independence",
                    "conditional_coverage"):
            self.assertIn(key, result)
        for key in ("lr_statistic", "p_value", "accepted"):
            self.assertIn(key, result["independence"])
            self.assertIn(key, result["conditional_coverage"])
        for key in ("q", "q0", "q1"):
            self.assertIn(key, result["independence"])

    # ---- transition counts and independence statistics ----

    def test_transition_counts_and_statistics(self):
        # Breach pattern F,T,T,F,T -> n00=0, n01=2, n10=1, n11=1.
        breaches = [False, True, True, False, True]
        result = self.call(observations=observations_for(breaches))
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            (0, 2, 1, 1),
        )
        counts, q, q0, q1, lr, pv = christoffersen(breaches)
        ind = result["independence"]
        self.assertTrue(math.isclose(ind["q"], q, abs_tol=1e-15))
        self.assertTrue(math.isclose(ind["q0"], q0, abs_tol=1e-15))
        self.assertTrue(math.isclose(ind["q1"], q1, abs_tol=1e-15))
        self.assertTrue(math.isclose(ind["lr_statistic"], lr, abs_tol=1e-12))
        self.assertTrue(math.isclose(ind["p_value"], pv, abs_tol=1e-12))

    def test_counts_sum_to_n_minus_one(self):
        breaches = [True, False, False, True, False, True, True, False]
        result = self.call(observations=observations_for(breaches))
        total = result["n00"] + result["n01"] + result["n10"] + result["n11"]
        self.assertEqual(total, len(breaches) - 1)

    def test_no_breaches_q1_null(self):
        # No breach ever occurs: state 1 has no outgoing transitions.
        result = self.call(observations=observations_for([False] * 4))
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            (3, 0, 0, 0),
        )
        ind = result["independence"]
        self.assertEqual(ind["q"], 0.0)
        self.assertEqual(ind["q0"], 0.0)
        self.assertIsNone(ind["q1"])
        self.assertEqual(ind["lr_statistic"], 0.0)
        self.assertEqual(ind["p_value"], 1.0)
        self.assertTrue(ind["accepted"])

    def test_all_breaches_q0_null(self):
        # Every day breaches: state 0 has no outgoing transitions.
        result = self.call(observations=observations_for([True] * 3))
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            (0, 0, 0, 2),
        )
        ind = result["independence"]
        self.assertEqual(ind["q"], 1.0)
        self.assertIsNone(ind["q0"])
        self.assertEqual(ind["q1"], 1.0)
        self.assertEqual(ind["lr_statistic"], 0.0)
        self.assertEqual(ind["p_value"], 1.0)

    def test_clustered_breaches_rejected(self):
        # Strongly clustered breaches: independence must be rejected.
        breaches = [False] * 20 + [True] * 10 + [False] * 20
        result = self.call(observations=observations_for(breaches))
        self.assertEqual((result["n01"], result["n10"]), (1, 1))
        self.assertFalse(result["independence"]["accepted"])
        self.assertLess(result["independence"]["p_value"], 0.05)

    def test_alternating_breaches(self):
        breaches = [True, False] * 5
        result = self.call(observations=observations_for(breaches))
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            (0, 4, 5, 0),
        )
        counts, q, q0, q1, lr, pv = christoffersen(breaches)
        ind = result["independence"]
        self.assertTrue(math.isclose(ind["lr_statistic"], lr, abs_tol=1e-12))
        self.assertTrue(math.isclose(ind["p_value"], pv, abs_tol=1e-12))

    # ---- conditional coverage ----

    def test_conditional_coverage_combines_statistics(self):
        breaches = [False, True, False, False, True, True, False, True]
        result = self.call(observations=observations_for(breaches))
        expected_lr = (
            result["kupiec"]["lr_statistic"]
            + result["independence"]["lr_statistic"]
        )
        cc = result["conditional_coverage"]
        self.assertTrue(math.isclose(cc["lr_statistic"], expected_lr, abs_tol=1e-12))
        self.assertTrue(
            math.isclose(cc["p_value"], math.exp(-expected_lr / 2.0), abs_tol=1e-12)
        )
        self.assertEqual(cc["accepted"], cc["p_value"] >= result["significance"])

    def test_conditional_coverage_reference(self):
        # n=20, one breach: Kupiec lr is exactly 0, so the conditional
        # coverage statistic equals the independence statistic.
        breaches = [False] * 7 + [True] + [False] * 12
        result = self.call(observations=observations_for(breaches))
        self.assertEqual(result["kupiec"]["lr_statistic"], 0.0)
        cc = result["conditional_coverage"]
        self.assertEqual(
            cc["lr_statistic"], result["independence"]["lr_statistic"]
        )
        self.assertEqual(cc["p_value"], math.exp(-cc["lr_statistic"] / 2.0))

    def test_acceptance_boundary(self):
        breaches = [False, True, True, False, True, False, False, True]
        obs = observations_for(breaches)
        result = self.call(observations=obs)
        for name in ("independence", "conditional_coverage"):
            pv = result[name]["p_value"]
            accepted = self.service.var_backtest_validation(
                request_body(observations=obs, significance=pv)
            )
            self.assertTrue(accepted[name]["accepted"])
            rejected = self.service.var_backtest_validation(
                request_body(observations=obs, significance=pv + 1e-9)
            )
            self.assertFalse(rejected[name]["accepted"])

    def test_defaults_echoed(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["significance"], 0.05)

    def test_result_is_standard_json_serializable(self):
        result = self.call()
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)
        roundtrip = json.loads(encoded)
        self.assertEqual(roundtrip["observation_count"], 4)
        self.assertIn("independence", roundtrip)

    # ---- parse-level failures -> 400 invalid_request ----

    def test_malformed_json(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.var_backtest_validation(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_top_level_non_object(self):
        for body in (b"[]", b"42", b"null", b'"x"'):
            with self.subTest(body=body):
                with self.assertRaises(InvalidRequest):
                    self.service.var_backtest_validation(body)

    # ---- semantic failures -> 422 invalid_input ----

    def test_missing_or_invalid_fields(self):
        with self.assertRaises(InvalidInput):
            self.service.var_backtest_validation(b'{"confidence": 0.95}')
        with self.assertRaises(InvalidInput):
            self.call(confidence=1.5)
        with self.assertRaises(InvalidInput):
            self.call(confidence="0.95")
        with self.assertRaises(InvalidInput):
            self.call(significance=0)
        with self.assertRaises(InvalidInput):
            self.call(currency="")
        with self.assertRaises(InvalidInput):
            self.call(observations=[])
        with self.assertRaises(InvalidInput):
            self.call(observations=observations_for([True]))

    def test_observation_field_validation(self):
        for bad_obs in (
            {"date": "", "var": 1.0, "realized_pnl": 0.0},
            {"date": "a", "var": -1.0, "realized_pnl": 0.0},
            {"date": "a", "var": 1.0, "realized_pnl": None},
            {"date": "a", "var": True, "realized_pnl": 0.0},
        ):
            with self.subTest(bad_obs=bad_obs):
                obs = [bad_obs, {"date": "b", "var": 1.0, "realized_pnl": 0.0}]
                with self.assertRaises(InvalidInput):
                    self.call(observations=obs)

    def test_nan_and_infinity_rejected(self):
        body = (
            b'{"confidence":0.95,"observations":['
            b'{"date":"a","var":1.0,"realized_pnl":NaN},'
            b'{"date":"b","var":1.0,"realized_pnl":0.0}]}'
        )
        with self.assertRaises(InvalidInput):
            self.service.var_backtest_validation(body)

    def test_duplicate_observation(self):
        obs = [
            {"date": "d", "var": 1.0, "realized_pnl": 0.0},
            {"date": "d", "var": 1.0, "realized_pnl": 0.0},
        ]
        with self.assertRaises(InvalidInput) as ctx:
            self.call(observations=obs)
        self.assertEqual(ctx.exception.code, "duplicate_observation")

    def test_too_many_observations(self):
        obs = [{"date": f"d{i}", "var": 1.0, "realized_pnl": 0.0}
               for i in range(10001)]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(observations=obs)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

    def test_exactly_10000_observations_allowed(self):
        breaches = [i % 7 == 3 for i in range(10000)]
        result = self.call(observations=observations_for(breaches))
        self.assertEqual(result["observation_count"], 10000)
        total = result["n00"] + result["n01"] + result["n10"] + result["n11"]
        self.assertEqual(total, 9999)


if __name__ == "__main__":
    unittest.main()
