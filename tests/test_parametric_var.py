import json
import math
import unittest

from riskscope.service import (
    InvalidInput,
    InvalidRequest,
    RequestTooLarge,
    Service,
    normal_density,
    normal_quantile,
)


def request_body(**overrides):
    body = {
        "confidence": 0.95,
        "currency": "USD",
        "factors": ["eq", "ir"],
        "positions": [
            {"id": "p1", "sensitivities": {"eq": 100.0, "ir": -50.0}},
            {"id": "p2", "sensitivities": {"eq": 25.0}},
        ],
        "covariance_matrix": [[0.01, 0.002], [0.002, 0.004]],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class ParametricVarTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.parametric_var(request_body(**overrides))

    def assert_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["factors"], ["eq", "ir"])
        self.assertEqual(result["aggregate_sensitivities"], [125.0, -50.0])
        # s = [125, -50], Σ = [[.01, .002], [.002, .004]]
        # variance = 156.25 - 25 + 10 = 141.25
        self.assertAlmostEqual(result["variance"], 141.25)
        volatility = math.sqrt(141.25)
        self.assertAlmostEqual(result["volatility"], volatility)
        z = normal_quantile(0.95)
        self.assertAlmostEqual(result["var"], z * volatility)
        expected_shortfall = normal_density(z) * volatility / 0.05
        self.assertAlmostEqual(result["expected_shortfall"], expected_shortfall)
        self.assertGreater(result["expected_shortfall"], result["var"])
        json.dumps(result, sort_keys=True)

    def test_factor_order_drives_vector_and_matrix(self):
        result = self.call(
            factors=["ir", "eq"],
            covariance_matrix=[[0.004, 0.002], [0.002, 0.01]],
        )
        self.assertEqual(result["factors"], ["ir", "eq"])
        self.assertEqual(result["aggregate_sensitivities"], [-50.0, 125.0])
        self.assertAlmostEqual(result["variance"], 141.25)

    def test_currency_defaults_to_usd_and_overridable(self):
        body = request_body()
        self.assertEqual(self.service.parametric_var(body)["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_missing_sensitivity_entries_are_zero(self):
        result = self.call(
            positions=[{"id": "p", "sensitivities": {"ir": 2.0}}]
        )
        self.assertEqual(result["aggregate_sensitivities"], [0.0, 2.0])

    def test_empty_sensitivities_object_allowed(self):
        result = self.call(
            positions=[{"id": "p", "sensitivities": {}}]
        )
        self.assertEqual(result["aggregate_sensitivities"], [0.0, 0.0])

    def test_integer_numbers_accepted(self):
        result = self.call(
            factors=["f"],
            positions=[{"id": "p", "sensitivities": {"f": 2}}],
            covariance_matrix=[[3]],
        )
        self.assertEqual(result["aggregate_sensitivities"], [2.0])
        self.assertAlmostEqual(result["variance"], 12.0)

    def test_aggregation_accumulates_across_positions(self):
        result = self.call(
            factors=["a", "b"],
            positions=[
                {"id": "p1", "sensitivities": {"a": 1.5, "b": -2.0}},
                {"id": "p2", "sensitivities": {"a": -0.5, "b": 4.0}},
                {"id": "p3", "sensitivities": {}},
            ],
            covariance_matrix=[[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertEqual(result["aggregate_sensitivities"], [1.0, 2.0])
        self.assertAlmostEqual(result["variance"], 5.0)

    def test_extra_fields_ignored(self):
        result = self.call(
            positions=[
                {"id": "p", "sensitivities": {"eq": 1.0}, "book": "eq-1"},
            ],
            method="delta-normal",
        )
        self.assertEqual(result["aggregate_sensitivities"], [1.0, 0.0])

    def test_extra_sensitivity_keys_not_silently_dropped(self):
        # Unknown factors are rejected, not ignored.
        self.assert_code(
            "unknown_factor",
            positions=[{"id": "p", "sensitivities": {"eq": 1.0, "fx": 9.0}}],
        )

    def test_zero_volatility_gives_zero_tail_metrics(self):
        result = self.call(
            confidence=0.99,
            factors=["a"],
            positions=[{"id": "p", "sensitivities": {"a": 3.0}}],
            covariance_matrix=[[0.0]],
        )
        self.assertEqual(result["variance"], 0.0)
        self.assertEqual(result["volatility"], 0.0)
        self.assertEqual(result["var"], 0.0)
        self.assertEqual(result["expected_shortfall"], 0.0)

    def test_all_zero_sensitivities_give_zero_metrics(self):
        result = self.call(
            positions=[{"id": "p", "sensitivities": {"eq": 0.0, "ir": 0.0}}]
        )
        self.assertEqual(result["volatility"], 0.0)
        self.assertEqual(result["var"], 0.0)
        self.assertEqual(result["expected_shortfall"], 0.0)

    def test_zero_diagonal_factor_kept_when_semidefinite(self):
        result = self.call(
            factors=["a", "b"],
            positions=[{"id": "p", "sensitivities": {"b": 3.0}}],
            covariance_matrix=[[0.0, 0.0], [0.0, 1.0]],
        )
        self.assertAlmostEqual(result["variance"], 9.0)

    def test_singular_psd_matrix_accepted(self):
        result = self.call(
            factors=["a", "b"],
            positions=[{"id": "p", "sensitivities": {"a": 1.0, "b": 1.0}}],
            covariance_matrix=[[1.0, 1.0], [1.0, 1.0]],
        )
        self.assertAlmostEqual(result["variance"], 4.0)

    def test_singular_direction_gives_zero_variance(self):
        result = self.call(
            factors=["a", "b"],
            positions=[{"id": "p", "sensitivities": {"a": 1.0, "b": -1.0}}],
            covariance_matrix=[[1.0, 1.0], [1.0, 1.0]],
        )
        self.assertEqual(result["variance"], 0.0)
        self.assertEqual(result["var"], 0.0)

    def test_near_symmetric_matrix_accepted(self):
        result = self.call(
            covariance_matrix=[[1.0, 0.5], [0.5 + 5e-13, 1.0]],
        )
        self.assertTrue(math.isfinite(result["var"]))

    def test_small_negative_rounding_slip_clamped_to_zero(self):
        # A Schur-complement pivot a hair below zero for an otherwise PSD
        # matrix, in a direction the book actually loads.
        nearly = 1.0 - 5e-13
        result = self.call(
            factors=["a", "b"],
            positions=[{"id": "p", "sensitivities": {"a": 1.0, "b": -1.0}}],
            covariance_matrix=[[1.0, 1.0], [1.0, nearly]],
        )
        self.assertEqual(result["variance"], 0.0)
        self.assertEqual(result["var"], 0.0)

    def test_confidence_drives_tail_scaling(self):
        low = self.call(confidence=0.90)
        high = self.call(confidence=0.99)
        self.assertLess(low["var"], high["var"])
        self.assertLess(low["expected_shortfall"], high["expected_shortfall"])

    def test_confidence_near_boundaries(self):
        for confidence in (1e-6, 0.5, 1.0 - 1e-6):
            result = self.call(confidence=confidence)
            self.assertTrue(math.isfinite(result["var"]))
            self.assertTrue(math.isfinite(result["expected_shortfall"]))

    def test_response_has_exactly_the_documented_keys(self):
        result = self.call()
        self.assertEqual(
            sorted(result),
            [
                "aggregate_sensitivities",
                "confidence",
                "currency",
                "expected_shortfall",
                "factors",
                "var",
                "variance",
                "volatility",
            ],
        )

    # ---- 400 invalid_request ----

    def test_unparseable_json(self):
        with self.assertRaises(InvalidRequest):
            self.service.parametric_var(b"{not json")

    def test_top_level_not_object(self):
        for raw in (b"[]", b"42", b'"x"', b"null"):
            with self.subTest(raw=raw):
                with self.assertRaises(InvalidRequest):
                    self.service.parametric_var(raw)

    # ---- 422 invalid_input: scalar fields ----

    def test_confidence_out_of_range(self):
        for bad in (0, 1, -0.5, 1.5):
            with self.subTest(bad=bad):
                self.assert_code("invalid_input", confidence=bad)

    def test_confidence_wrong_type(self):
        for bad in ("0.95", True, False, None, [], {}):
            with self.subTest(bad=bad):
                self.assert_code("invalid_input", confidence=bad)

    def test_confidence_missing(self):
        body = request_body()
        decoded = json.loads(body)
        del decoded["confidence"]
        with self.assertRaises(InvalidInput):
            self.service.parametric_var(json.dumps(decoded).encode())

    def test_confidence_oversized_integer(self):
        self.assert_code("invalid_input", confidence=10**400)

    def test_currency_must_be_nonempty_string(self):
        for bad in ("", 5, True, None, []):
            with self.subTest(bad=bad):
                self.assert_code("invalid_input", currency=bad)

    def test_factors_must_be_nonempty_array(self):
        for bad in (None, [], "eq", 3, {}):
            with self.subTest(bad=bad):
                self.assert_code("invalid_input", factors=bad)

    def test_factor_name_must_be_nonempty_string(self):
        for bad in ("", 1, None, True, [], {}):
            with self.subTest(bad=bad):
                self.assert_code("invalid_input", factors=["eq", bad])

    def test_positions_must_be_nonempty_array(self):
        for bad in (None, [], "x", 3, {}):
            with self.subTest(bad=bad):
                self.assert_code("invalid_input", positions=bad)

    def test_position_must_be_object(self):
        self.assert_code(
            "invalid_input",
            positions=[{"id": "p", "sensitivities": {"eq": 1.0}}, "nope"],
        )

    def test_position_id_must_be_nonempty_string(self):
        for bad in ("", 1, None, True):
            with self.subTest(bad=bad):
                self.assert_code(
                    "invalid_input",
                    positions=[{"id": bad, "sensitivities": {"eq": 1.0}}],
                )

    def test_sensitivities_must_be_object(self):
        for bad in (None, [], 1.0, "x"):
            with self.subTest(bad=bad):
                self.assert_code(
                    "invalid_input",
                    positions=[{"id": "p", "sensitivities": bad}],
                )

    def test_sensitivity_rejects_bool(self):
        self.assert_code(
            "invalid_input",
            positions=[{"id": "p", "sensitivities": {"eq": True}}],
        )

    def test_sensitivity_rejects_oversized_integer(self):
        self.assert_code(
            "invalid_input",
            positions=[{"id": "p", "sensitivities": {"eq": 10**400}}],
        )

    def test_sensitivity_rejects_non_number(self):
        self.assert_code(
            "invalid_input",
            positions=[{"id": "p", "sensitivities": {"eq": "1.0"}}],
        )

    # ---- 422 invalid_input: covariance shape and finiteness ----

    def test_covariance_missing(self):
        decoded = json.loads(request_body())
        del decoded["covariance_matrix"]
        with self.assertRaises(InvalidInput):
            self.service.parametric_var(json.dumps(decoded).encode())

    def test_covariance_wrong_shape(self):
        self.assert_code("invalid_input", covariance_matrix=[])
        self.assert_code("invalid_input", covariance_matrix=[[1.0, 0.0]])
        self.assert_code(
            "invalid_input", covariance_matrix=[[1.0, 0.0], [0.0]]
        )
        self.assert_code("invalid_input", covariance_matrix=[[1.0], [0.0, 1.0]])
        self.assert_code("invalid_input", covariance_matrix="x")

    def test_covariance_element_wrong_type(self):
        self.assert_code(
            "invalid_input", covariance_matrix=[[1.0, "x"], [0.0, 1.0]]
        )

    def test_covariance_element_rejects_bool(self):
        self.assert_code(
            "invalid_input", covariance_matrix=[[True, 0.0], [0.0, 1.0]]
        )

    def test_covariance_element_rejects_oversized_integer(self):
        self.assert_code(
            "invalid_input", covariance_matrix=[[10**400, 0.0], [0.0, 1.0]]
        )

    def test_covariance_rejects_nan_and_infinity(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            raw = (
                b'{"confidence":0.95,"factors":["a"],"positions":['
                b'{"id":"p","sensitivities":{"a":1}}],'
                b'"covariance_matrix":[[' + token.encode() + b"]]}"
            )
            with self.subTest(token=token):
                with self.assertRaises(InvalidInput):
                    self.service.parametric_var(raw)

    # ---- 422 specific codes ----

    def test_duplicate_factor(self):
        self.assert_code("duplicate_factor", factors=["eq", "eq"])

    def test_duplicate_position(self):
        self.assert_code(
            "duplicate_position",
            positions=[
                {"id": "p", "sensitivities": {"eq": 1.0}},
                {"id": "p", "sensitivities": {"ir": 2.0}},
            ],
        )

    def test_unknown_factor(self):
        self.assert_code(
            "unknown_factor",
            positions=[{"id": "p", "sensitivities": {"fx": 1.0}}],
        )

    def test_unknown_factor_reported_before_bad_matrix(self):
        # Even when the matrix is structurally invalid, the unknown factor
        # reference is reported first per the documented order.
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                positions=[{"id": "p", "sensitivities": {"fx": 1.0}}],
                covariance_matrix=[[1.0, 2.0], [2.0, 1.0]],
            )
        self.assertEqual(ctx.exception.code, "unknown_factor")

    def test_duplicate_factor_precedes_duplicate_position(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                factors=["a", "a"],
                positions=[
                    {"id": "p", "sensitivities": {"a": 1.0}},
                    {"id": "p", "sensitivities": {"a": 2.0}},
                ],
            )
        self.assertEqual(ctx.exception.code, "duplicate_factor")

    def test_asymmetric_matrix(self):
        self.assert_code(
            "invalid_covariance",
            covariance_matrix=[[1.0, 0.5], [0.5001, 1.0]],
        )

    def test_indefinite_matrix_negative_diagonal(self):
        self.assert_code(
            "invalid_covariance",
            covariance_matrix=[[-1.0, 0.0], [0.0, 1.0]],
        )

    def test_indefinite_matrix_negative_eigenvalue(self):
        # Eigenvalues 3 and -1.
        self.assert_code(
            "invalid_covariance",
            covariance_matrix=[[1.0, 2.0], [2.0, 1.0]],
        )

    def test_zero_pivot_with_nonzero_cross_row(self):
        # [[0,1],[1,0]] is indefinite; the zero pivot pairs with a load.
        self.assert_code(
            "invalid_covariance",
            covariance_matrix=[[0.0, 1.0], [1.0, 0.0]],
        )

    def test_negative_variance_beyond_tolerance(self):
        # Semidefinite-looking but the loaded diagonal is materially negative.
        self.assert_code(
            "invalid_covariance",
            factors=["a", "b"],
            positions=[{"id": "p", "sensitivities": {"b": 1.0}}],
            covariance_matrix=[[1.0, 0.0], [0.0, -1e-6]],
        )

    # ---- 413 request_too_large ----

    def test_too_many_factors(self):
        factors = [f"f{i}" for i in range(101)]
        covariance = [
            [1.0 if i == j else 0.0 for j in range(101)] for i in range(101)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(factors=factors, covariance_matrix=covariance)

    def test_exactly_100_factors_allowed(self):
        factors = [f"f{i}" for i in range(100)]
        covariance = [
            [1.0 if i == j else 0.0 for j in range(100)] for i in range(100)
        ]
        result = self.call(
            factors=factors,
            positions=[{"id": "p", "sensitivities": {}}],
            covariance_matrix=covariance,
        )
        self.assertEqual(len(result["factors"]), 100)

    def test_too_many_positions(self):
        positions = [
            {"id": f"p{i}", "sensitivities": {}} for i in range(10001)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(positions=positions)

    def test_exactly_10000_positions_allowed(self):
        positions = [
            {"id": f"p{i:05d}", "sensitivities": {}} for i in range(10000)
        ]
        result = self.call(positions=positions)
        self.assertEqual(result["aggregate_sensitivities"], [0.0, 0.0])

    def test_factor_position_product_boundary_allowed(self):
        factors = [f"f{i}" for i in range(100)]
        covariance = [
            [1.0 if i == j else 0.0 for j in range(100)] for i in range(100)
        ]
        positions = [
            {"id": f"p{i:05d}", "sensitivities": {}} for i in range(10000)
        ]  # 100 * 10000 = 1,000,000, the allowed boundary.
        result = self.call(
            factors=factors, positions=positions, covariance_matrix=covariance
        )
        self.assertEqual(len(result["aggregate_sensitivities"]), 100)

    # ---- normal helpers ----

    def test_normal_quantile_known_values(self):
        self.assertAlmostEqual(normal_quantile(0.5), 0.0)
        self.assertAlmostEqual(normal_quantile(0.95), 1.6448536269514722)
        self.assertAlmostEqual(normal_quantile(0.99), 2.3263478740408408)
        self.assertAlmostEqual(normal_quantile(0.975), 1.959963984540054)
        self.assertAlmostEqual(normal_quantile(0.01), -2.3263478740408408)
        self.assertAlmostEqual(normal_quantile(0.999), 3.0902323061678135, places=12)

    def test_normal_density_symmetric(self):
        self.assertAlmostEqual(normal_density(1.0), normal_density(-1.0))
        self.assertAlmostEqual(
            normal_density(0.0), 1.0 / math.sqrt(2.0 * math.pi)
        )


if __name__ == "__main__":
    unittest.main()
