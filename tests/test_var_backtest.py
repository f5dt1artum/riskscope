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


def kupiec(n, x, p, significance=0.05):
    """Reference implementation of the frozen Kupiec POF formula."""
    q = x / n

    def term(count, probability):
        return 0.0 if count == 0 else count * math.log(probability)

    lr = -2.0 * (
        term(n - x, 1 - p) + term(x, p)
        - term(n - x, 1 - q) - term(x, q)
    )
    pv = math.erfc(math.sqrt(max(lr, 0.0) / 2.0))
    return lr, pv, pv >= significance


class VarBacktestTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.var_backtest(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["significance"], 0.05)
        self.assertEqual(result["observation_count"], 4)
        self.assertEqual(result["breach_count"], 2)
        self.assertEqual(result["breach_rate"], 0.5)

        rows = result["observations"]
        self.assertEqual([r["date"] for r in rows],
                         ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
        # loss = -realized_pnl; breach only on strict loss > var.
        self.assertEqual(rows[0], {
            "date": "2024-01-01", "var": 10.0,
            "realized_pnl": 5.0, "loss": -5.0, "breach": False,
        })
        # Equality loss == var is NOT a breach.
        self.assertEqual(rows[2], {
            "date": "2024-01-03", "var": 10.0,
            "realized_pnl": -10.0, "loss": 10.0, "breach": False,
        })
        # Tiny strict excess still breaches.
        self.assertTrue(rows[3]["breach"])
        self.assertEqual(rows[3]["loss"], 1.0000001)

        lr, pv, accepted = kupiec(4, 2, 0.05)
        self.assertTrue(math.isclose(result["kupiec"]["lr_statistic"], lr, abs_tol=1e-12))
        self.assertTrue(math.isclose(result["kupiec"]["p_value"], pv, abs_tol=1e-12))
        self.assertFalse(result["kupiec"]["accepted"])

    def test_detail_rows_carry_exact_input_numbers(self):
        obs = [
            {"date": "d1", "var": 0, "realized_pnl": 0},
            {"date": "d2", "var": 0.0, "realized_pnl": -0.0},
            {"date": "d3", "var": 2, "realized_pnl": -3},
        ]
        result = self.service.var_backtest(request_body(observations=obs))
        rows = result["observations"]
        self.assertFalse(rows[0]["breach"])   # loss 0 == var 0, no breach
        self.assertFalse(rows[1]["breach"])   # -0.0 loss, equality
        self.assertTrue(rows[2]["breach"])    # 3 > 2
        self.assertEqual(rows[0]["var"], 0.0)
        self.assertEqual(rows[2]["realized_pnl"], -3.0)
        self.assertEqual(result["breach_count"], 1)
        self.assertTrue(math.isclose(result["breach_rate"], 1 / 3))

    def test_integer_fields_echoed_as_floats(self):
        obs = [
            {"date": "a", "var": 5, "realized_pnl": 10},
            {"date": "b", "var": 5, "realized_pnl": -6},
        ]
        result = self.service.var_backtest(request_body(observations=obs))
        self.assertEqual(result["observations"][0]["var"], 5.0)
        self.assertEqual(result["observations"][0]["realized_pnl"], 10.0)
        self.assertEqual(result["observations"][1]["loss"], 6.0)
        self.assertEqual(result["breach_count"], 1)

    def test_currency_and_significance_echoed(self):
        result = self.call(currency="EUR", significance=0.5)
        self.assertEqual(result["currency"], "EUR")
        self.assertEqual(result["significance"], 0.5)
        # x=2/4 breaches vs p=.05, p-value ~0.01 < 0.5 -> rejected.
        self.assertFalse(result["kupiec"]["accepted"])

    def test_accepted_when_p_value_at_or_above_significance(self):
        # 2 obs, no breaches, p=.05 -> p-value ~0.65 > .05 -> accepted.
        obs = [
            {"date": "a", "var": 10.0, "realized_pnl": 0.0},
            {"date": "b", "var": 10.0, "realized_pnl": 1.0},
        ]
        result = self.service.var_backtest(request_body(observations=obs))
        self.assertEqual(result["breach_count"], 0)
        lr, pv, accepted = kupiec(2, 0, 0.05)
        self.assertTrue(math.isclose(result["kupiec"]["lr_statistic"], lr, abs_tol=1e-12))
        self.assertTrue(math.isclose(result["kupiec"]["p_value"], pv, abs_tol=1e-12))
        self.assertTrue(result["kupiec"]["accepted"])

    def test_all_breaches_well_defined(self):
        # x == n boundary: 0*ln0 terms must count as zero.
        obs = [
            {"date": "a", "var": 1.0, "realized_pnl": -2.0},
            {"date": "b", "var": 1.0, "realized_pnl": -3.0},
            {"date": "c", "var": 1.0, "realized_pnl": -4.0},
        ]
        result = self.service.var_backtest(request_body(observations=obs))
        self.assertEqual(result["breach_count"], 3)
        self.assertEqual(result["breach_rate"], 1.0)
        lr, pv, accepted = kupiec(3, 3, 0.05)
        self.assertTrue(math.isclose(result["kupiec"]["lr_statistic"], lr, abs_tol=1e-12))
        self.assertTrue(math.isclose(result["kupiec"]["p_value"], pv, abs_tol=1e-12))
        self.assertFalse(result["kupiec"]["accepted"])

    def test_breach_rate_equal_to_predicted_probability(self):
        # n=20, x=1, confidence=.95 -> breach rate equals p=.05; the LR
        # statistic must be exactly 0.0 (0*ln0 terms aside) and accepted.
        obs = [
            {"date": f"d{i:02d}", "var": 1.0,
             "realized_pnl": -2.0 if i == 7 else 0.0}
            for i in range(20)
        ]
        result = self.service.var_backtest(request_body(observations=obs))
        self.assertEqual(result["breach_count"], 1)
        self.assertEqual(result["breach_rate"], 0.05)
        self.assertEqual(result["kupiec"]["lr_statistic"], 0.0)
        self.assertEqual(result["kupiec"]["p_value"], 1.0)
        self.assertTrue(result["kupiec"]["accepted"])

    def test_extra_fields_ignored(self):
        obs = [
            {"date": "a", "var": 1.0, "realized_pnl": 0.0, "book": "eq", "extra": 1},
            {"date": "b", "var": 1.0, "realized_pnl": 0.0, "book": "ir"},
        ]
        result = self.service.var_backtest(request_body(observations=obs, note="x"))
        self.assertEqual(result["observation_count"], 2)
        self.assertNotIn("note", result)
        self.assertNotIn("book", result["observations"][0])

    def test_result_is_standard_json_serializable(self):
        result = self.call()
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)
        roundtrip = json.loads(encoded)
        self.assertEqual(roundtrip["observation_count"], 4)

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

    def test_top_level_null(self):
        with self.assertRaises(InvalidRequest):
            self.service.var_backtest(b"null")

    # ---- semantic failures -> 422 invalid_input ----

    def test_missing_confidence(self):
        body = json.dumps({"observations": [
            {"date": "a", "var": 1.0, "realized_pnl": 0.0},
            {"date": "b", "var": 1.0, "realized_pnl": 0.0},
        ]}).encode()
        with self.assertRaises(InvalidInput) as ctx:
            self.service.var_backtest(body)
        self.assertEqual(ctx.exception.code, "invalid_input")

    def test_missing_observations(self):
        with self.assertRaises(InvalidInput):
            self.service.var_backtest(b'{"confidence": 0.95}')

    def test_confidence_out_of_range(self):
        for bad in (0, 1, -0.5, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(confidence=bad)

    def test_significance_out_of_range(self):
        for bad in (0, 1, -0.1, 2.0):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(significance=bad)

    def test_confidence_wrong_type(self):
        for bad in ("0.95", None, True, False, [], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(confidence=bad)

    def test_significance_wrong_type(self):
        for bad in ("0.05", None, True, [0.05]):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(significance=bad)

    def test_currency_default_and_errors(self):
        self.assertEqual(self.call()["currency"], "USD")
        with self.assertRaises(InvalidInput):
            self.call(currency="")
        for bad in (123, None, True, []):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(currency=bad)

    def test_observation_count_bounds(self):
        with self.assertRaises(InvalidInput):
            self.call(observations=[])
        with self.assertRaises(InvalidInput):
            self.call(observations=[
                {"date": "a", "var": 1.0, "realized_pnl": 0.0},
            ])

    def test_observation_not_object(self):
        for bad in ([1, 2], ["x", "y"], [None, None], [[], []]):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(observations=bad)

    def test_observations_wrong_type(self):
        for bad in ({}, "x", 4, None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(observations=bad)

    def test_date_validation(self):
        good = [
            {"date": "a", "var": 1.0, "realized_pnl": 0.0},
            {"date": "b", "var": 1.0, "realized_pnl": 0.0},
        ]
        self.assertEqual(self.call(observations=good)["observation_count"], 2)
        for bad_date in ("", 1, None, True, []):
            with self.subTest(bad_date=bad_date):
                obs = [
                    {"date": bad_date, "var": 1.0, "realized_pnl": 0.0},
                    {"date": "b", "var": 1.0, "realized_pnl": 0.0},
                ]
                with self.assertRaises(InvalidInput):
                    self.call(observations=obs)

    def test_var_validation(self):
        for bad_var in (-0.01, -1, "1.0", None, True, False):
            with self.subTest(bad_var=bad_var):
                obs = [
                    {"date": "a", "var": bad_var, "realized_pnl": 0.0},
                    {"date": "b", "var": 1.0, "realized_pnl": 0.0},
                ]
                with self.assertRaises(InvalidInput):
                    self.call(observations=obs)

    def test_realized_pnl_validation(self):
        for bad_pnl in ("0.0", None, True, False):
            with self.subTest(bad_pnl=bad_pnl):
                obs = [
                    {"date": "a", "var": 1.0, "realized_pnl": bad_pnl},
                    {"date": "b", "var": 1.0, "realized_pnl": 0.0},
                ]
                with self.assertRaises(InvalidInput):
                    self.call(observations=obs)

    def test_negative_zero_var_allowed(self):
        obs = [
            {"date": "a", "var": -0.0, "realized_pnl": 0.0},
            {"date": "b", "var": 0.0, "realized_pnl": -1e-300},
        ]
        result = self.service.var_backtest(request_body(observations=obs))
        self.assertEqual(result["breach_count"], 1)

    def test_nan_and_infinity_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                body = request_body().replace(b"0.95", token.encode(), 1)
                with self.assertRaises(InvalidInput):
                    self.service.var_backtest(body)

    def test_nan_in_nested_fields_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                body = (
                    b'{"confidence":0.95,"observations":['
                    b'{"date":"a","var":1.0,"realized_pnl":' + token.encode() + b'},'
                    b'{"date":"b","var":1.0,"realized_pnl":0.0}]}'
                )
                with self.assertRaises(InvalidInput):
                    self.service.var_backtest(body)

    def test_oversized_integer_rejected(self):
        huge = b"1" + b"0" * 400  # 10**400, outside the float64 range
        cases = {
            "var": b'{"date":"a","var":' + huge + b',"realized_pnl":0.0}',
            "realized_pnl": b'{"date":"a","var":1.0,"realized_pnl":' + huge + b'}',
        }
        for field, first_obs in cases.items():
            with self.subTest(field=field):
                body = (
                    b'{"confidence":0.95,"observations":[' + first_obs + b','
                    b'{"date":"b","var":1.0,"realized_pnl":0.0}]}'
                )
                with self.assertRaises(InvalidInput):
                    self.service.var_backtest(body)

    def test_huge_int_confidence_overflow(self):
        body = (
            b'{"confidence":1' + b"0" * 400 + b','
            b'"observations":['
            b'{"date":"a","var":1.0,"realized_pnl":0.0},'
            b'{"date":"b","var":1.0,"realized_pnl":0.0}]}'
        )
        with self.assertRaises(InvalidInput):
            self.service.var_backtest(body)

    def test_exp_overflow_confidence_rejected(self):
        # 1e999 silently becomes float infinity in the JSON scanner; the
        # finite-number guard must reject it as invalid_input.
        body = (
            b'{"confidence":1e999,"observations":['
            b'{"date":"a","var":1.0,"realized_pnl":0.0},'
            b'{"date":"b","var":1.0,"realized_pnl":0.0}]}'
        )
        with self.assertRaises(InvalidInput):
            self.service.var_backtest(body)

    def test_confidence_whose_tail_probability_underflows(self):
        # A subnormal confidence makes 1-confidence round to 1.0, so the
        # predicted tail probability is 0; the Kupiec log term is then
        # non-finite and the request must fail rather than raise.
        obs = [
            {"date": "a", "var": 1.0, "realized_pnl": 0.0},
            {"date": "b", "var": 1.0, "realized_pnl": 0.0},
        ]
        with self.assertRaises(InvalidInput) as ctx:
            self.service.var_backtest(
                request_body(confidence=5e-324, observations=obs)
            )
        self.assertEqual(ctx.exception.code, "invalid_input")

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
        obs = [{"date": f"d{i:05d}", "var": 1.0, "realized_pnl": 0.0}
               for i in range(10000)]
        result = self.service.var_backtest(request_body(observations=obs))
        self.assertEqual(result["observation_count"], 10000)
        self.assertEqual(result["breach_count"], 0)
        self.assertEqual(result["breach_rate"], 0.0)

    # ---- boundary acceptance semantics ----

    def test_accepted_boundary_p_value_equals_significance(self):
        # Accepted iff p_value >= significance; equality means accepted.
        obs = [
            {"date": "a", "var": 1.0, "realized_pnl": 0.0},
            {"date": "b", "var": 1.0, "realized_pnl": 0.0},
        ]
        result = self.service.var_backtest(
            request_body(observations=obs, significance=0.5)
        )
        # p-value ~0.65 is strictly above 0.5 -> accepted.
        self.assertTrue(result["kupiec"]["accepted"])
        # Significance just above the p-value -> rejected.
        pv = result["kupiec"]["p_value"]
        rejected = self.service.var_backtest(
            request_body(observations=obs, significance=pv + 1e-9)
        )
        self.assertFalse(rejected["kupiec"]["accepted"])
        # Significance exactly equal to the p-value -> accepted (>= rule).
        equal = self.service.var_backtest(
            request_body(observations=obs, significance=pv)
        )
        self.assertTrue(equal["kupiec"]["accepted"])


if __name__ == "__main__":
    unittest.main()
