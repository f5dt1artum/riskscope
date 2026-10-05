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
        "factors": ["eq", "ir"],
        "observations": [
            {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
            {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
            {"date": "2024-01-03", "factor_returns": {"eq": 3, "ir": 6}},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


def expected_ewma(rows, decay):
    k = len(rows[0])
    matrix = [[rows[0][i] * rows[0][j] for j in range(k)] for i in range(k)]
    for row in rows[1:]:
        for i in range(k):
            for j in range(k):
                matrix[i][j] = decay * matrix[i][j] + (1.0 - decay) * row[i] * row[j]
    return matrix


class EwmaCovarianceEstimateTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.ewma_covariance_estimate(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_default_decay(self):
        result = self.call()
        self.assertEqual(result["factors"], ["eq", "ir"])
        self.assertEqual(result["decay"], 0.94)
        self.assertEqual(result["observation_count"], 3)
        expected = expected_ewma([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], 0.94)
        self.assertEqual(result["covariance_matrix"], expected)
        self.assertEqual(
            result["volatilities"],
            [math.sqrt(expected[0][0]), math.sqrt(expected[1][1])],
        )
        self.assertEqual(
            result["correlation_matrix"],
            [[1.0, 1.0], [1.0, 1.0]],
        )

    def test_custom_decay_recursion(self):
        result = self.call(
            decay=0.5,
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
                {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
            ],
        )
        self.assertEqual(result["decay"], 0.5)
        # Σ₁ = [[1, 2], [2, 4]]; Σ₂ = 0.5·Σ₁ + 0.5·[[4, 8], [8, 16]].
        self.assertEqual(
            result["covariance_matrix"],
            [[2.5, 5.0], [5.0, 10.0]],
        )
        self.assertEqual(
            result["volatilities"],
            [math.sqrt(2.5), math.sqrt(10.0)],
        )

    def test_recent_observations_weigh_more(self):
        # Same returns in opposite order: the last return dominates.
        forward = self.call(
            decay=0.7,
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 0}},
                {"date": "2024-01-02", "factor_returns": {"eq": 10, "ir": 0}},
            ],
        )
        backward = self.call(
            decay=0.7,
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 10, "ir": 0}},
                {"date": "2024-01-02", "factor_returns": {"eq": 1, "ir": 0}},
            ],
        )
        self.assertGreater(
            backward["covariance_matrix"][0][0], forward["covariance_matrix"][0][0]
        )

    def test_input_order_is_recursion_order_not_date_order(self):
        result = self.call(
            decay=0.5,
            observations=[
                {"date": "2024-01-03", "factor_returns": {"eq": 1, "ir": 0}},
                {"date": "2024-01-01", "factor_returns": {"eq": 10, "ir": 0}},
            ],
        )
        # The "2024-01-01" row arrives last, so it weighs more despite
        # carrying the earliest date.
        self.assertEqual(
            result["covariance_matrix"],
            [[0.5 * 1 + 0.5 * 100, 0.0], [0.0, 0.0]],
        )

    def test_factor_order_drives_output_positions(self):
        result = self.call(factors=["ir", "eq"], decay=0.5)
        expected = expected_ewma([[2.0, 1.0], [4.0, 2.0], [6.0, 3.0]], 0.5)
        self.assertEqual(result["factors"], ["ir", "eq"])
        self.assertEqual(result["covariance_matrix"], expected)

    def test_zero_volatility_factor(self):
        result = self.call(
            factors=["eq", "flat"],
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 1, "flat": 0}},
                {"date": "2024-01-02", "factor_returns": {"eq": 2, "flat": 0}},
                {"date": "2024-01-03", "factor_returns": {"eq": 3, "flat": 0}},
            ],
        )
        self.assertEqual(result["volatilities"][1], 0.0)
        self.assertEqual(result["covariance_matrix"][1], [0.0, 0.0])
        self.assertEqual(
            result["correlation_matrix"],
            [[1.0, 0.0], [0.0, 1.0]],
        )

    def test_all_zero_returns(self):
        result = self.call(
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 0, "ir": 0}},
                {"date": "2024-01-02", "factor_returns": {"eq": 0.0, "ir": 0.0}},
            ]
        )
        self.assertEqual(result["covariance_matrix"], [[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(result["volatilities"], [0.0, 0.0])
        self.assertEqual(
            result["correlation_matrix"],
            [[1.0, 0.0], [0.0, 1.0]],
        )

    def test_matrices_are_exactly_symmetric(self):
        result = self.call(
            factors=["a", "b", "c"],
            observations=[
                {
                    "date": "2024-01-01",
                    "factor_returns": {"a": 0.1, "b": -0.2, "c": 0.3},
                },
                {
                    "date": "2024-01-02",
                    "factor_returns": {"a": -0.4, "b": 0.5, "c": -0.1},
                },
                {
                    "date": "2024-01-03",
                    "factor_returns": {"a": 0.2, "b": 0.1, "c": -0.3},
                },
                {
                    "date": "2024-01-04",
                    "factor_returns": {"a": 0.0, "b": -0.1, "c": 0.2},
                },
            ],
        )
        for matrix in (result["covariance_matrix"], result["correlation_matrix"]):
            for i in range(3):
                for j in range(3):
                    self.assertEqual(matrix[i][j], matrix[j][i])

    def test_extra_factors_and_fields_ignored(self):
        result = self.call(
            currency="EUR",
            observations=[
                {
                    "date": "2024-01-01",
                    "factor_returns": {"eq": 1, "ir": 2, "fx": 99},
                    "note": "ignored",
                },
                {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
            ],
        )
        self.assertEqual(result["factors"], ["eq", "ir"])
        self.assertEqual(result["observation_count"], 2)

    def test_result_feeds_parametric_var(self):
        result = self.call()
        var_result = self.service.parametric_var(
            json.dumps(
                {
                    "confidence": 0.99,
                    "factors": result["factors"],
                    "positions": [{"id": "p1", "sensitivities": {"eq": 100, "ir": 50}}],
                    "covariance_matrix": result["covariance_matrix"],
                }
            ).encode("utf-8")
        )
        self.assertGreaterEqual(var_result["variance"], 0.0)
        self.assertTrue(math.isfinite(var_result["var"]))

    def test_result_is_json_serializable(self):
        json.dumps(self.call(), sort_keys=True)

    # ---- 400 invalid_request ----

    def test_unparseable_json(self):
        with self.assertRaises(InvalidRequest):
            self.service.ewma_covariance_estimate(b"{not json")

    def test_top_level_not_object(self):
        with self.assertRaises(InvalidRequest):
            self.service.ewma_covariance_estimate(b"[1, 2]")

    # ---- 422 invalid_input ----

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

    def test_missing_observations(self):
        with self.assertRaises(InvalidInput):
            self.call(observations=None)

    def test_too_few_observations(self):
        with self.assertRaises(InvalidInput):
            self.call(
                observations=[
                    {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}}
                ]
            )

    def test_observation_not_object(self):
        with self.assertRaises(InvalidInput):
            self.call(observations=[{"date": "2024-01-01"}, "oops"])

    def test_observation_bad_date(self):
        with self.assertRaises(InvalidInput):
            self.call(
                observations=[
                    {"date": "", "factor_returns": {"eq": 1, "ir": 2}},
                    {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
                ]
            )

    def test_factor_returns_not_object(self):
        with self.assertRaises(InvalidInput):
            self.call(
                observations=[
                    {"date": "2024-01-01", "factor_returns": [1, 2]},
                    {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
                ]
            )

    def test_return_rejects_bool(self):
        with self.assertRaises(InvalidInput):
            self.call(
                observations=[
                    {"date": "2024-01-01", "factor_returns": {"eq": True, "ir": 2}},
                    {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
                ]
            )

    def test_return_rejects_nan_and_infinity(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            body = (
                b'{"factors": ["eq"], "observations": ['
                b'{"date": "2024-01-01", "factor_returns": {"eq": 1}},'
                b'{"date": "2024-01-02", "factor_returns": {"eq": ' + token.encode() + b"}}]}"
            )
            with self.assertRaises(InvalidInput):
                self.service.ewma_covariance_estimate(body)

    def test_return_rejects_oversized_int(self):
        with self.assertRaises(InvalidInput):
            self.call(
                observations=[
                    {"date": "2024-01-01", "factor_returns": {"eq": 10**400, "ir": 2}},
                    {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
                ]
            )

    def test_return_rejects_string(self):
        with self.assertRaises(InvalidInput):
            self.call(
                observations=[
                    {"date": "2024-01-01", "factor_returns": {"eq": "1", "ir": 2}},
                    {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
                ]
            )

    def test_decay_out_of_range(self):
        for bad in (0, 0.0, 1, 1.0, -0.5, 1.5):
            with self.assertRaises(InvalidInput):
                self.call(decay=bad)

    def test_decay_rejects_non_number(self):
        for bad in (True, "0.94", None, [0.94]):
            with self.assertRaises(InvalidInput):
                self.call(decay=bad)

    def test_decay_boundary_inside_range_ok(self):
        result = self.call(decay=0.000001)
        self.assertEqual(result["decay"], 0.000001)
        result = self.call(decay=0.999999)
        self.assertEqual(result["decay"], 0.999999)

    # ---- 422 specific codes ----

    def assert_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)

    def test_duplicate_factor(self):
        self.assert_code("duplicate_factor", factors=["eq", "eq"])

    def test_duplicate_observation(self):
        self.assert_code(
            "duplicate_observation",
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
                {"date": "2024-01-01", "factor_returns": {"eq": 2, "ir": 4}},
            ],
        )

    def test_missing_factor(self):
        self.assert_code(
            "missing_factor",
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 1}},
                {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
            ],
        )

    # ---- 413 request_too_large ----

    def test_too_many_factors(self):
        with self.assertRaises(RequestTooLarge):
            self.call(factors=[f"f{i}" for i in range(101)])

    def test_too_many_observations(self):
        observations = [
            {"date": f"2024-01-{i:05d}", "factor_returns": {"eq": 1, "ir": 2}}
            for i in range(10001)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(observations=observations)

    def test_factor_observation_product_limit(self):
        # 100 factors x 10000 observations is the allowed boundary.
        factors = [f"f{i}" for i in range(100)]
        returns = {factor: 1 for factor in factors}
        observations = [
            {"date": f"2024-01-{i:05d}", "factor_returns": returns}
            for i in range(10000)
        ]
        result = self.call(factors=factors, observations=observations)
        self.assertEqual(result["observation_count"], 10000)

    def test_factor_observation_product_over_limit(self):
        factors = [f"f{i}" for i in range(101 - 1)]
        returns = {factor: 1 for factor in factors}
        observations = [
            {"date": f"2024-01-{i:05d}", "factor_returns": returns}
            for i in range(10001)
        ]
        # 100 x 10001 exceeds both individual caps' product bound.
        with self.assertRaises(RequestTooLarge):
            self.call(factors=factors, observations=observations)

    # ---- non-finite computation fails the whole request ----

    def test_non_finite_computation_fails(self):
        with self.assertRaises(InvalidInput):
            self.call(
                observations=[
                    {"date": "2024-01-01", "factor_returns": {"eq": 1e308, "ir": 2}},
                    {"date": "2024-01-02", "factor_returns": {"eq": 1e308, "ir": 4}},
                ]
            )


if __name__ == "__main__":
    unittest.main()
