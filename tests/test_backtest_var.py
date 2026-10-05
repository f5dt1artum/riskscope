import array
import math
import unittest

from riskscope import backtest_var


def reference(flags, confidence):
    """Independent reference for the coverage and clustering statistics."""
    n = len(flags)
    x = sum(flags)
    p = 1.0 - confidence

    def term(count, probability):
        return 0.0 if count == 0 else count * math.log(probability)

    q = x / n
    kupiec_lr = max(
        -2.0 * (term(n - x, 1 - p) + term(x, p) - term(n - x, 1 - q) - term(x, q)),
        0.0,
    )
    kupiec_pv = math.erfc(math.sqrt(kupiec_lr / 2.0))
    if n < 2:
        return kupiec_lr, kupiec_pv, None, None, None, None

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
    ind_lr = max(2.0 * (ln_l1 - ln_l0), 0.0)
    ind_pv = math.erfc(math.sqrt(ind_lr / 2.0))
    cc_lr = kupiec_lr + ind_lr
    cc_pv = math.exp(-cc_lr / 2.0)
    return kupiec_lr, kupiec_pv, ind_lr, ind_pv, cc_lr, cc_pv


def series_from_flags(flags, var=1.0):
    pnl = [-2.0 if flag else 0.0 for flag in flags]
    return pnl, [var] * len(flags)


class BacktestVarTest(unittest.TestCase):
    def assert_matches_reference(self, result, flags, confidence):
        kupiec_lr, kupiec_pv, ind_lr, ind_pv, cc_lr, cc_pv = reference(flags, confidence)
        self.assertTrue(
            math.isclose(result["kupiec"]["lr_statistic"], kupiec_lr, abs_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(result["kupiec"]["p_value"], kupiec_pv, abs_tol=1e-12)
        )
        if ind_lr is None:
            self.assertIsNone(result["independence"]["lr_statistic"])
            self.assertIsNone(result["independence"]["p_value"])
            self.assertIsNone(result["conditional_coverage"]["lr_statistic"])
            self.assertIsNone(result["conditional_coverage"]["p_value"])
        else:
            self.assertTrue(
                math.isclose(
                    result["independence"]["lr_statistic"], ind_lr, abs_tol=1e-12
                )
            )
            self.assertTrue(
                math.isclose(
                    result["independence"]["p_value"], ind_pv, abs_tol=1e-12
                )
            )
            self.assertTrue(
                math.isclose(
                    result["conditional_coverage"]["lr_statistic"], cc_lr, abs_tol=1e-12
                )
            )
            self.assertTrue(
                math.isclose(
                    result["conditional_coverage"]["p_value"], cc_pv, abs_tol=1e-12
                )
            )

    # ---- result shape and values ----

    def test_basic_result_fields(self):
        result = backtest_var(
            realized_pnl=[5.0, -15.0, -10.0, -1.0000001],
            var_forecast=[10.0, 10.0, 10.0, 1.0],
            confidence_level=0.95,
        )
        self.assertEqual(result["confidence_level"], 0.95)
        self.assertEqual(result["observation_count"], 4)
        self.assertEqual(result["breach_count"], 2)
        self.assertEqual(result["expected_breach_count"], 4 * (1.0 - 0.95))
        self.assertEqual(result["breach_rate"], 0.5)
        self.assertEqual(result["breaches"], [False, True, False, True])
        self.assertEqual(result["breach_positions"], [1, 3])
        self.assertEqual(
            set(result["kupiec"]), {"lr_statistic", "p_value"}
        )
        self.assert_matches_reference(result, [0, 1, 0, 1], 0.95)

    def test_default_confidence_level_is_0_99(self):
        result = backtest_var([0.0, -2.0], [1.0, 1.0])
        self.assertEqual(result["confidence_level"], 0.99)
        self.assertEqual(result["expected_breach_count"], 2 * (1.0 - 0.99))
        self.assert_matches_reference(result, [0, 1], 0.99)

    def test_equality_is_not_a_breach(self):
        result = backtest_var([-10.0, -0.0, 0.0], [10.0, 0.0, 0.0])
        self.assertEqual(result["breaches"], [False, False, False])
        self.assertEqual(result["breach_count"], 0)
        self.assertEqual(result["breach_positions"], [])

    def test_strict_excess_breaches(self):
        result = backtest_var([-1.0000001, -3], [1.0, 2])
        self.assertEqual(result["breaches"], [True, True])
        self.assertEqual(result["breach_count"], 2)
        self.assertEqual(result["breach_rate"], 1.0)

    def test_input_order_is_preserved(self):
        # No sorting or realignment: positions follow the input sequence.
        result = backtest_var([-2.0, 0.0, -2.0, 0.0], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(result["breaches"], [True, False, True, False])
        self.assertEqual(result["breach_positions"], [0, 2])

    def test_statistics_match_reference_across_patterns(self):
        patterns = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 1, 0],
        ]
        for flags in patterns:
            for confidence in (0.9, 0.95, 0.99):
                with self.subTest(flags=flags, confidence=confidence):
                    pnl, var = series_from_flags(flags)
                    result = backtest_var(pnl, var, confidence_level=confidence)
                    self.assert_matches_reference(result, flags, confidence)

    def test_conditional_coverage_statistic_is_sum(self):
        pnl, var = series_from_flags([0, 1, 0, 1, 0, 0, 1, 0])
        result = backtest_var(pnl, var)
        self.assertAlmostEqual(
            result["conditional_coverage"]["lr_statistic"],
            result["kupiec"]["lr_statistic"] + result["independence"]["lr_statistic"],
            delta=1e-12,
        )

    def test_zero_breaches_stays_finite(self):
        result = backtest_var([0.0] * 5, [1.0] * 5)
        self.assertEqual(result["breach_count"], 0)
        self.assertEqual(result["breach_rate"], 0.0)
        for section in ("kupiec", "independence", "conditional_coverage"):
            statistic = result[section]["lr_statistic"]
            p_value = result[section]["p_value"]
            self.assertTrue(math.isfinite(statistic) and statistic >= 0.0)
            self.assertTrue(0.0 <= p_value <= 1.0)
        self.assertEqual(result["independence"]["lr_statistic"], 0.0)
        self.assertEqual(result["independence"]["p_value"], 1.0)

    def test_all_breaches_stays_finite(self):
        result = backtest_var([-2.0] * 4, [1.0] * 4)
        self.assertEqual(result["breach_count"], 4)
        self.assertEqual(result["breach_rate"], 1.0)
        for section in ("kupiec", "independence", "conditional_coverage"):
            statistic = result[section]["lr_statistic"]
            p_value = result[section]["p_value"]
            self.assertTrue(math.isfinite(statistic) and statistic >= 0.0)
            self.assertTrue(0.0 <= p_value <= 1.0)
        self.assertEqual(result["independence"]["lr_statistic"], 0.0)
        self.assertEqual(result["independence"]["p_value"], 1.0)

    def test_breach_rate_equal_to_tail_probability_gives_zero_statistic(self):
        flags = [1 if i == 7 else 0 for i in range(100)]
        pnl, var = series_from_flags(flags)
        result = backtest_var(pnl, var, confidence_level=0.99)
        self.assertEqual(result["breach_rate"], 0.01)
        self.assertEqual(result["kupiec"]["lr_statistic"], 0.0)
        self.assertEqual(result["kupiec"]["p_value"], 1.0)

    def test_single_observation_returns_none_for_transition_tests(self):
        for pnl, expected_breach in ((0.0, False), (-2.0, True)):
            with self.subTest(pnl=pnl):
                result = backtest_var([pnl], [1.0])
                self.assertEqual(result["observation_count"], 1)
                self.assertEqual(result["breaches"], [expected_breach])
                self.assertTrue(math.isfinite(result["kupiec"]["lr_statistic"]))
                self.assertTrue(0.0 <= result["kupiec"]["p_value"] <= 1.0)
                self.assertIsNone(result["independence"]["lr_statistic"])
                self.assertIsNone(result["independence"]["p_value"])
                self.assertIsNone(result["conditional_coverage"]["lr_statistic"])
                self.assertIsNone(result["conditional_coverage"]["p_value"])

    # ---- accepted input shapes ----

    def test_accepts_tuples_generators_and_array_objects(self):
        expected = backtest_var([-2.0, 0.0, -2.0], [1.0, 1.0, 1.0])
        as_tuples = backtest_var((-2.0, 0.0, -2.0), (1.0, 1.0, 1.0))
        as_generators = backtest_var(
            (v for v in [-2.0, 0.0, -2.0]), (v for v in [1.0, 1.0, 1.0])
        )
        as_arrays = backtest_var(
            array.array("d", [-2.0, 0.0, -2.0]), array.array("d", [1.0, 1.0, 1.0])
        )
        self.assertEqual(as_tuples, expected)
        self.assertEqual(as_generators, expected)
        self.assertEqual(as_arrays, expected)

    def test_accepts_numpy_arrays_when_available(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        result = backtest_var(
            np.array([5.0, -15.0, -10.0]), np.array([10.0, 10.0, 10.0])
        )
        self.assertEqual(result["breaches"], [False, True, False])
        as_lists = backtest_var([5.0, -15.0, -10.0], [10.0, 10.0, 10.0])
        self.assertEqual(result, as_lists)

    def test_caller_objects_are_not_modified(self):
        pnl = [5.0, -15.0, -10.0]
        var = [10.0, 10.0, 10.0]
        backtest_var(pnl, var)
        self.assertEqual(pnl, [5.0, -15.0, -10.0])
        self.assertEqual(var, [10.0, 10.0, 10.0])

    def test_result_is_deterministic(self):
        args = ([5.0, -15.0, -10.0, -1.0000001], [10.0, 10.0, 10.0, 1.0])
        self.assertEqual(backtest_var(*args), backtest_var(*args))

    # ---- ValueError cases ----

    def test_confidence_level_out_of_range(self):
        for bad in (0, 1, -0.5, 1.5, 0.0, 1.0, math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    backtest_var([0.0, 0.0], [1.0, 1.0], confidence_level=bad)

    def test_empty_sequences_rejected(self):
        with self.assertRaises(ValueError):
            backtest_var([], [])
        with self.assertRaises(ValueError):
            backtest_var([], [1.0])
        with self.assertRaises(ValueError):
            backtest_var([0.0], [])

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            backtest_var([0.0, 0.0], [1.0])
        with self.assertRaises(ValueError):
            backtest_var([0.0], [1.0, 1.0])

    def test_nan_and_infinity_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    backtest_var([0.0, bad], [1.0, 1.0])
                with self.assertRaises(ValueError):
                    backtest_var([0.0, 0.0], [1.0, bad])

    def test_negative_var_rejected(self):
        for bad in (-0.01, -1, -1e300):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    backtest_var([0.0, 0.0], [1.0, bad])

    def test_oversized_integer_rejected_as_value_error(self):
        with self.assertRaises(ValueError):
            backtest_var([0.0, 0.0], [1.0, 10**400])

    # ---- TypeError cases ----

    def test_non_iterable_input_rejected(self):
        for bad in (5, 1.5, None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    backtest_var(bad, [1.0])
                with self.assertRaises(TypeError):
                    backtest_var([0.0], bad)

    def test_text_input_rejected(self):
        with self.assertRaises(TypeError):
            backtest_var("0.0, 0.0", [1.0, 1.0])
        with self.assertRaises(TypeError):
            backtest_var([0.0, 0.0], b"11")

    def test_non_numeric_elements_rejected(self):
        for bad_element in ("1.0", None, [1.0], {"v": 1.0}, object()):
            with self.subTest(bad_element=bad_element):
                with self.assertRaises(TypeError):
                    backtest_var([0.0, bad_element], [1.0, 1.0])
                with self.assertRaises(TypeError):
                    backtest_var([0.0, 0.0], [1.0, bad_element])

    def test_nested_sequences_rejected(self):
        with self.assertRaises(TypeError):
            backtest_var([[0.0], [0.0]], [[1.0], [1.0]])

    def test_booleans_rejected(self):
        for bad in (True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    backtest_var([0.0, bad], [1.0, 1.0])
                with self.assertRaises(TypeError):
                    backtest_var([0.0, 0.0], [1.0, bad])
                with self.assertRaises(TypeError):
                    backtest_var([0.0, 0.0], [1.0, 1.0], confidence_level=bad)

    def test_non_numeric_confidence_rejected(self):
        for bad in ("0.99", None, [0.99]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    backtest_var([0.0, 0.0], [1.0, 1.0], confidence_level=bad)


if __name__ == "__main__":
    unittest.main()
