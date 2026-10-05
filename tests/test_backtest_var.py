import array
import math
import unittest

from riskscope import backtest_var


def reference(pnl, var, confidence):
    """Independent double-precision reference for the backtest statistics."""
    n = len(pnl)
    flags = [pnl[i] < -var[i] for i in range(n)]
    x = sum(flags)

    def term(count, probability):
        return 0.0 if count == 0 else count * math.log(probability)

    p = 1.0 - confidence
    q = x / n
    lr_uc = max(
        -2.0
        * (
            term(n - x, 1.0 - p) + term(x, p)
            - term(n - x, 1.0 - q) - term(x, q)
        ),
        0.0,
    )
    p_uc = math.erfc(math.sqrt(lr_uc / 2.0))

    result = {
        "flags": flags,
        "x": x,
        "lr_uc": lr_uc,
        "p_uc": p_uc,
        "lr_ind": None,
        "p_ind": None,
        "lr_cc": None,
        "p_cc": None,
    }
    if n < 2:
        return result

    n00 = n01 = n10 = n11 = 0
    for previous, current in zip(flags, flags[1:]):
        if not previous and not current:
            n00 += 1
        elif not previous and current:
            n01 += 1
        elif previous and not current:
            n10 += 1
        else:
            n11 += 1
    qi = (n01 + n11) / (n - 1)
    ln_l0 = term(n00 + n10, 1.0 - qi) + term(n01 + n11, qi)
    ln_l1 = 0.0
    if n00 + n01 > 0:
        q0 = n01 / (n00 + n01)
        ln_l1 += term(n00, 1.0 - q0) + term(n01, q0)
    if n10 + n11 > 0:
        q1 = n11 / (n10 + n11)
        ln_l1 += term(n10, 1.0 - q1) + term(n11, q1)
    lr_ind = max(2.0 * (ln_l1 - ln_l0), 0.0)
    result.update(
        counts=(n00, n01, n10, n11),
        lr_ind=lr_ind,
        p_ind=math.erfc(math.sqrt(lr_ind / 2.0)),
        lr_cc=lr_uc + lr_ind,
        p_cc=math.exp(-(lr_uc + lr_ind) / 2.0),
    )
    return result


def series_from_flags(flags, var=1.0):
    pnl = [(-2.0 if flag else 0.0) for flag in flags]
    return pnl, [var] * len(flags)


class BacktestVarTest(unittest.TestCase):
    def assert_matches_reference(self, pnl, var, confidence=0.99):
        result = backtest_var(pnl, var, confidence_level=confidence)
        expected = reference(list(pnl), list(var), confidence)
        n = len(expected["flags"])
        x = expected["x"]
        self.assertEqual(result["observation_count"], n)
        self.assertEqual(result["breach_count"], x)
        self.assertEqual(result["expected_breach_count"], n * (1.0 - confidence))
        self.assertEqual(result["breach_rate"], x / n)
        self.assertEqual(result["breaches"], expected["flags"])
        self.assertEqual(
            result["breach_indices"],
            [i for i, flag in enumerate(expected["flags"]) if flag],
        )
        self.assertTrue(
            math.isclose(
                result["kupiec"]["lr_statistic"], expected["lr_uc"], abs_tol=1e-12
            )
        )
        self.assertTrue(
            math.isclose(
                result["kupiec"]["p_value"], expected["p_uc"], abs_tol=1e-12
            )
        )
        for key, lr_key, p_key in (
            ("independence", "lr_ind", "p_ind"),
            ("conditional_coverage", "lr_cc", "p_cc"),
        ):
            if expected[lr_key] is None:
                self.assertIsNone(result[key]["lr_statistic"])
                self.assertIsNone(result[key]["p_value"])
            else:
                self.assertTrue(
                    math.isclose(
                        result[key]["lr_statistic"], expected[lr_key], abs_tol=1e-12
                    )
                )
                self.assertTrue(
                    math.isclose(
                        result[key]["p_value"], expected[p_key], abs_tol=1e-12
                    )
                )
        return result

    # ---- success shapes and values ----

    def test_basic_values_default_confidence(self):
        pnl = [5.0, -15.0, -10.0, -1.0000001]
        var = [10.0, 10.0, 10.0, 1.0]
        result = backtest_var(pnl, var)
        self.assertEqual(result["confidence_level"], 0.99)
        self.assertEqual(result["observation_count"], 4)
        self.assertEqual(result["breach_count"], 2)
        self.assertEqual(result["expected_breach_count"], 4 * (1.0 - 0.99))
        self.assertEqual(result["breach_rate"], 0.5)
        # Equality with the VaR forecast (day 3) is not a breach.
        self.assertEqual(result["breaches"], [False, True, False, True])
        self.assertEqual(result["breach_indices"], [1, 3])
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            (0, 2, 1, 0),
        )

    def test_matches_reference_over_many_patterns(self):
        patterns = [
            [False] * 10,
            [True] * 10,
            [True, False] * 10,
            [False, False, True, False, True, True, False, True],
            [True, False, False, False, False],
            [False, True, True, True, True],
            [False] * 250,
        ]
        for flags in patterns:
            for confidence in (0.9, 0.95, 0.99):
                with self.subTest(flags=flags, confidence=confidence):
                    pnl, var = series_from_flags(flags)
                    self.assert_matches_reference(pnl, var, confidence)

    def test_varying_var_forecasts(self):
        pnl = [-0.5, -2.5, -1.5, 0.25, -3.0]
        var = [0.5, 2.0, 1.5, 0.0, 3.5]
        # Only day 2 breaches (-2.5 < -2.0); days 1 and 3 sit exactly on
        # their VaR forecast and equality does not breach.
        result = backtest_var(pnl, var, confidence_level=0.95)
        self.assertEqual(result["breaches"], [False, True, False, False, False])
        self.assertEqual(result["breach_indices"], [1])
        self.assert_matches_reference(pnl, var, 0.95)

    def test_equality_is_not_a_breach(self):
        result = backtest_var([-1.0, -1.0], [1.0, 1.0])
        self.assertEqual(result["breaches"], [False, False])
        self.assertEqual(result["breach_count"], 0)

    def test_negative_zero_var_and_pnl(self):
        result = backtest_var([-0.0, -1e-300], [-0.0, 0.0])
        self.assertEqual(result["breaches"], [False, True])

    def test_integers_accepted(self):
        result = backtest_var([1, -3, 0], [2, 2, 2])
        self.assertEqual(result["breaches"], [False, True, False])
        self.assertEqual(result["breach_count"], 1)

    def test_zero_breaches_finite(self):
        result = self.assert_matches_reference(*series_from_flags([False] * 30))
        self.assertEqual(result["breach_count"], 0)
        for key in ("kupiec", "independence", "conditional_coverage"):
            statistic = result[key]["lr_statistic"]
            p_value = result[key]["p_value"]
            self.assertTrue(math.isfinite(statistic))
            self.assertGreaterEqual(statistic, 0.0)
            self.assertTrue(0.0 <= p_value <= 1.0)
        # No clustering signal: independence statistic is exactly zero.
        self.assertEqual(result["independence"]["lr_statistic"], 0.0)
        self.assertEqual(result["independence"]["p_value"], 1.0)

    def test_all_breaches_finite(self):
        result = self.assert_matches_reference(*series_from_flags([True] * 30))
        self.assertEqual(result["breach_count"], 30)
        self.assertEqual(result["breach_rate"], 1.0)
        for key in ("kupiec", "independence", "conditional_coverage"):
            statistic = result[key]["lr_statistic"]
            p_value = result[key]["p_value"]
            self.assertTrue(math.isfinite(statistic))
            self.assertGreaterEqual(statistic, 0.0)
            self.assertTrue(0.0 <= p_value <= 1.0)
        self.assertEqual(result["independence"]["lr_statistic"], 0.0)

    def test_breach_rate_exactly_at_predicted_probability(self):
        # n=100, x=1, confidence=0.99 -> statistic is exactly 0.0.
        flags = [i == 42 for i in range(100)]
        pnl, var = series_from_flags(flags)
        result = backtest_var(pnl, var)
        self.assertEqual(result["breach_count"], 1)
        self.assertEqual(result["kupiec"]["lr_statistic"], 0.0)
        self.assertEqual(result["kupiec"]["p_value"], 1.0)

    def test_single_observation(self):
        for pnl, var, breached in (([0.0], [1.0], False), ([-2.0], [1.0], True)):
            with self.subTest(breached=breached):
                result = backtest_var(pnl, var)
                self.assertEqual(result["observation_count"], 1)
                self.assertEqual(result["breaches"], [breached])
                self.assertTrue(math.isfinite(result["kupiec"]["lr_statistic"]))
                self.assertTrue(0.0 <= result["kupiec"]["p_value"] <= 1.0)
                self.assertIsNone(result["independence"]["lr_statistic"])
                self.assertIsNone(result["independence"]["p_value"])
                self.assertIsNone(result["conditional_coverage"]["lr_statistic"])
                self.assertIsNone(result["conditional_coverage"]["p_value"])
                self.assertEqual(
                    (result["n00"], result["n01"], result["n10"], result["n11"]),
                    (0, 0, 0, 0),
                )

    def test_two_observations(self):
        result = self.assert_matches_reference([0.0, -2.0], [1.0, 1.0], 0.95)
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            (0, 1, 0, 0),
        )
        self.assertIsNotNone(result["independence"]["lr_statistic"])
        self.assertIsNotNone(result["conditional_coverage"]["p_value"])

    def test_missing_predecessor_state(self):
        # No 1 -> * transitions exist when every period breaches; the
        # zero-probability convention must keep the statistics finite.
        result = self.assert_matches_reference(*series_from_flags([True] * 5))
        self.assertEqual(result["n11"], 4)
        self.assertEqual(result["independence"]["lr_statistic"], 0.0)

    # ---- input containers ----

    def test_accepts_tuples_generators_and_arrays(self):
        pnl = [0.0, -2.0, 0.0, -3.0]
        var = [1.0, 1.0, 1.0, 1.0]
        expected = backtest_var(pnl, var)
        self.assertEqual(backtest_var(tuple(pnl), tuple(var)), expected)
        self.assertEqual(
            backtest_var(iter(pnl), iter(var)),
            expected,
        )
        self.assertEqual(
            backtest_var(array.array("d", pnl), array.array("d", var)),
            expected,
        )
        self.assertEqual(
            backtest_var(array.array("i", [0, -2, 0, -3]), array.array("i", [1] * 4)),
            expected,
        )

    def test_inputs_are_not_mutated(self):
        pnl = [0.0, -2.0, 0.0]
        var = [1.0, 1.0, 1.0]
        pnl_copy, var_copy = list(pnl), list(var)
        backtest_var(pnl, var)
        self.assertEqual(pnl, pnl_copy)
        self.assertEqual(var, var_copy)

    def test_deterministic_for_identical_inputs(self):
        pnl = [0.5, -1.5, -2.5, 0.0, 1.0, -0.5]
        var = [1.0, 1.0, 2.0, 0.5, 1.0, 1.0]
        self.assertEqual(backtest_var(pnl, var), backtest_var(pnl, var))

    # ---- TypeError cases ----

    def test_type_error_on_non_iterable(self):
        for bad in (None, 42, 1.5, True, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    backtest_var(bad, [1.0])
                with self.assertRaises(TypeError):
                    backtest_var([1.0], bad)

    def test_type_error_on_strings(self):
        with self.assertRaises(TypeError):
            backtest_var("0.0,-1.0", [1.0, 1.0])
        with self.assertRaises(TypeError):
            backtest_var(["0.0", "-1.0"], [1.0, 1.0])
        with self.assertRaises(TypeError):
            backtest_var([0.0, -1.0], ["1.0", "1.0"])

    def test_type_error_on_booleans(self):
        with self.assertRaises(TypeError):
            backtest_var([True, False], [1.0, 1.0])
        with self.assertRaises(TypeError):
            backtest_var([0.0, -1.0], [True, False])
        with self.assertRaises(TypeError):
            backtest_var([0.0, -1.0], [1.0, 1.0], confidence_level=True)

    def test_type_error_on_nested_sequences(self):
        with self.assertRaises(TypeError):
            backtest_var([[0.0], [-1.0]], [1.0, 1.0])
        with self.assertRaises(TypeError):
            backtest_var([0.0, -1.0], [[1.0], [1.0]])

    def test_type_error_on_non_numeric_elements(self):
        for bad in (None, {}, object(), [1.0]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    backtest_var([0.0, bad], [1.0, 1.0])

    def test_type_error_on_non_numeric_confidence(self):
        for bad in ("0.99", None, [0.99], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    backtest_var([0.0, -1.0], [1.0, 1.0], confidence_level=bad)

    # ---- ValueError cases ----

    def test_value_error_on_confidence_out_of_range(self):
        for bad in (0.0, 1.0, -0.5, 1.5, 2, math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    backtest_var([0.0, -1.0], [1.0, 1.0], confidence_level=bad)

    def test_value_error_on_empty_sequences(self):
        with self.assertRaises(ValueError):
            backtest_var([], [])
        with self.assertRaises(ValueError):
            backtest_var(iter([]), iter([]))

    def test_value_error_on_length_mismatch(self):
        with self.assertRaises(ValueError):
            backtest_var([0.0, -1.0], [1.0])
        with self.assertRaises(ValueError):
            backtest_var([0.0], [1.0, 1.0])

    def test_value_error_on_nan_and_infinity(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    backtest_var([0.0, bad], [1.0, 1.0])
                with self.assertRaises(ValueError):
                    backtest_var([0.0, -1.0], [1.0, bad])

    def test_value_error_on_negative_var(self):
        for bad in (-0.01, -1.0, -1e300):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    backtest_var([0.0, -1.0], [1.0, bad])

    def test_value_error_on_oversized_integer(self):
        huge = 10**400
        with self.assertRaises(ValueError):
            backtest_var([0.0, huge], [1.0, 1.0])
        with self.assertRaises(ValueError):
            backtest_var([0.0, -1.0], [1.0, huge])


if __name__ == "__main__":
    unittest.main()
