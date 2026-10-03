import json
import math
import unittest
from statistics import NormalDist

from riskscope.service import (
    InvalidInput,
    InvalidRequest,
    RequestTooLarge,
    Service,
)


def request_body(**overrides):
    body = {
        "confidence": 0.99,
        "factors": ["eq", "ir"],
        "positions": [
            {"id": "p1", "sensitivities": {"eq": 100.0, "ir": 50.0}},
            {"id": "p2", "sensitivities": {"eq": -20.0}},
        ],
        "covariance_matrix": [
            [0.04, 0.0],
            [0.0, 0.01],
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class ParametricVarTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.parametric_var(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["confidence"], 0.99)
        self.assertEqual(result["factors"], ["eq", "ir"])
        self.assertEqual(result["aggregate_sensitivities"], [80.0, 50.0])
        # variance = 80^2 * 0.04 + 50^2 * 0.01
        self.assertEqual(result["variance"], 281.0)
        volatility = math.sqrt(281.0)
        self.assertEqual(result["volatility"], volatility)
        z = NormalDist().inv_cdf(0.99)
        self.assertEqual(result["var"], z * volatility)
        density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        self.assertEqual(
            result["expected_shortfall"], density * volatility / (1.0 - 0.99)
        )

    def test_factor_order_drives_output_positions(self):
        result = self.call(
            factors=["ir", "eq"],
            covariance_matrix=[
                [0.01, 0.0],
                [0.0, 0.04],
            ],
        )
        self.assertEqual(result["factors"], ["ir", "eq"])
        self.assertEqual(result["aggregate_sensitivities"], [50.0, 80.0])
        self.assertEqual(result["variance"], 281.0)

    def test_currency_default_and_echo(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_missing_sensitivity_is_zero(self):
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {"ir": 10.0}}]
        )
        self.assertEqual(result["aggregate_sensitivities"], [0.0, 10.0])
        self.assertEqual(result["variance"], 1.0)

    def test_empty_sensitivities_allowed(self):
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {}}]
        )
        self.assertEqual(result["aggregate_sensitivities"], [0.0, 0.0])
        self.assertEqual(result["variance"], 0.0)

    def test_zero_volatility_zeroes_tail_metrics(self):
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {"eq": 5.0}}],
            covariance_matrix=[
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        )
        self.assertEqual(result["variance"], 0.0)
        self.assertEqual(result["volatility"], 0.0)
        self.assertEqual(result["var"], 0.0)
        self.assertEqual(result["expected_shortfall"], 0.0)

    def test_correlated_factors(self):
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {"eq": 1.0, "ir": 1.0}}],
            covariance_matrix=[
                [1.0, 0.5],
                [0.5, 1.0],
            ],
        )
        # variance = 1 + 1 + 2 * 0.5
        self.assertEqual(result["variance"], 3.0)

    def test_nearly_symmetric_matrix_accepted(self):
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {"eq": 1.0, "ir": 1.0}}],
            covariance_matrix=[
                [1.0, 0.5],
                [0.5 * (1.0 + 1e-13), 1.0],
            ],
        )
        self.assertTrue(math.isfinite(result["var"]))

    def test_extra_fields_ignored(self):
        result = self.call(
            note="ignored",
            positions=[
                {
                    "id": "p1",
                    "sensitivities": {"eq": 100.0, "ir": 50.0},
                    "extra": True,
                },
                {"id": "p2", "sensitivities": {"eq": -20.0}},
            ],
        )
        self.assertEqual(result["aggregate_sensitivities"], [80.0, 50.0])

    def test_result_is_json_serializable(self):
        json.dumps(self.call(), sort_keys=True)

    # ---- 400 invalid_request ----

    def test_unparseable_json(self):
        with self.assertRaises(InvalidRequest):
            self.service.parametric_var(b"{not json")

    def test_top_level_not_object(self):
        with self.assertRaises(InvalidRequest):
            self.service.parametric_var(b"[1, 2]")

    # ---- 422 invalid_input ----

    def test_bad_currency(self):
        for bad in ("", 1, None, True):
            with self.assertRaises(InvalidInput):
                self.call(currency=bad)

    def test_missing_confidence(self):
        with self.assertRaises(InvalidInput):
            self.call(confidence=None)

    def test_confidence_out_of_range(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaises(InvalidInput):
                self.call(confidence=bad)

    def test_confidence_rejects_bool_and_string(self):
        for bad in (True, "0.99"):
            with self.assertRaises(InvalidInput):
                self.call(confidence=bad)

    def test_missing_factors(self):
        with self.assertRaises(InvalidInput):
            self.call(factors=None)

    def test_empty_factors(self):
        with self.assertRaises(InvalidInput):
            self.call(factors=[])

    def test_factor_not_nonempty_string(self):
        for bad in ("", 1, None, True):
            with self.assertRaises(InvalidInput):
                self.call(factors=["eq", bad])

    def test_missing_positions(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=None)

    def test_empty_positions(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=[])

    def test_position_not_object(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=["oops"])

    def test_position_bad_id(self):
        for bad in ("", 1, None, True):
            with self.assertRaises(InvalidInput):
                self.call(positions=[{"id": bad, "sensitivities": {"eq": 1}}])

    def test_sensitivities_not_object(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p1", "sensitivities": [1, 2]}])

    def test_sensitivity_rejects_bool(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p1", "sensitivities": {"eq": True}}])

    def test_sensitivity_rejects_nan_and_infinity(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            body = (
                b'{"confidence": 0.99, "factors": ["eq"], "positions": ['
                b'{"id": "p1", "sensitivities": {"eq": ' + token.encode()
                + b'}}], "covariance_matrix": [[1.0]]}'
            )
            with self.assertRaises(InvalidInput):
                self.service.parametric_var(body)

    def test_sensitivity_rejects_oversized_int(self):
        with self.assertRaises(InvalidInput):
            self.call(
                positions=[{"id": "p1", "sensitivities": {"eq": 10**400}}]
            )

    def test_sensitivity_rejects_string(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p1", "sensitivities": {"eq": "1"}}])

    def test_missing_covariance_matrix(self):
        with self.assertRaises(InvalidInput):
            self.call(covariance_matrix=None)

    def test_covariance_matrix_wrong_shape(self):
        with self.assertRaises(InvalidInput):
            self.call(covariance_matrix=[[1.0]])
        with self.assertRaises(InvalidInput):
            self.call(covariance_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with self.assertRaises(InvalidInput):
            self.call(covariance_matrix=[[1.0, 0.0], [0.0, "1.0"]])

    def test_covariance_matrix_rejects_bool_and_oversized_int(self):
        with self.assertRaises(InvalidInput):
            self.call(covariance_matrix=[[1.0, True], [False, 1.0]])
        with self.assertRaises(InvalidInput):
            self.call(covariance_matrix=[[10**400, 0.0], [0.0, 1.0]])

    # ---- 422 invalid_covariance ----

    def assert_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)

    def test_asymmetric_matrix(self):
        self.assert_code(
            "invalid_covariance",
            covariance_matrix=[[1.0, 0.5], [0.4, 1.0]],
        )

    def test_not_positive_semidefinite(self):
        self.assert_code(
            "invalid_covariance",
            covariance_matrix=[[1.0, 2.0], [2.0, 1.0]],
        )

    def test_negative_diagonal(self):
        self.assert_code(
            "invalid_covariance",
            covariance_matrix=[[-1.0, 0.0], [0.0, 1.0]],
        )

    def test_zero_pivot_with_nonzero_column(self):
        self.assert_code(
            "invalid_covariance",
            covariance_matrix=[[0.0, 1.0], [1.0, 1.0]],
        )

    # ---- 422 specific codes ----

    def test_duplicate_factor(self):
        self.assert_code("duplicate_factor", factors=["eq", "eq"])

    def test_duplicate_position(self):
        self.assert_code(
            "duplicate_position",
            positions=[
                {"id": "p1", "sensitivities": {"eq": 1.0}},
                {"id": "p1", "sensitivities": {"ir": 2.0}},
            ],
        )

    def test_unknown_factor(self):
        self.assert_code(
            "unknown_factor",
            positions=[{"id": "p1", "sensitivities": {"fx": 1.0}}],
        )

    # ---- 413 request_too_large ----

    def test_too_many_factors(self):
        with self.assertRaises(RequestTooLarge):
            self.call(factors=[f"f{i}" for i in range(101)])

    def test_too_many_positions(self):
        positions = [
            {"id": f"p{i}", "sensitivities": {"eq": 1.0}}
            for i in range(10001)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(positions=positions)

    def test_factor_position_product_boundary_allowed(self):
        # 100 factors x 10000 positions is the allowed boundary.
        factors = [f"f{i}" for i in range(100)]
        matrix = [[0.0] * 100 for _ in range(100)]
        for i in range(100):
            matrix[i][i] = 1.0
        positions = [
            {"id": f"p{i}", "sensitivities": {"f0": 1.0}}
            for i in range(10000)
        ]
        result = self.call(
            factors=factors, positions=positions, covariance_matrix=matrix
        )
        self.assertEqual(result["aggregate_sensitivities"][0], 10000.0)

    # ---- non-finite computation fails the whole request ----

    def test_non_finite_computation_fails(self):
        with self.assertRaises(InvalidInput):
            self.call(
                factors=["eq"],
                positions=[
                    {"id": "p1", "sensitivities": {"eq": 1e308}},
                    {"id": "p2", "sensitivities": {"eq": 1e308}},
                ],
                covariance_matrix=[[1.0]],
            )


if __name__ == "__main__":
    unittest.main()
