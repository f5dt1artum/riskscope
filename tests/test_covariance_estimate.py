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


class CovarianceEstimateTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.covariance_estimate(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["factors"], ["eq", "ir"])
        self.assertEqual(result["observation_count"], 3)
        self.assertEqual(result["means"], [2.0, 4.0])
        self.assertEqual(result["volatilities"], [1.0, 2.0])
        self.assertEqual(
            result["covariance_matrix"],
            [[1.0, 2.0], [2.0, 4.0]],
        )
        self.assertEqual(
            result["correlation_matrix"],
            [[1.0, 1.0], [1.0, 1.0]],
        )

    def test_factor_order_drives_output_positions(self):
        result = self.call(factors=["ir", "eq"])
        self.assertEqual(result["factors"], ["ir", "eq"])
        self.assertEqual(result["means"], [4.0, 2.0])
        self.assertEqual(result["volatilities"], [2.0, 1.0])
        self.assertEqual(
            result["covariance_matrix"],
            [[4.0, 2.0], [2.0, 1.0]],
        )

    def test_uncorrelated_factors(self):
        result = self.call(
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 1.0, "ir": 1.0}},
                {"date": "2024-01-02", "factor_returns": {"eq": -1.0, "ir": 1.0}},
                {"date": "2024-01-03", "factor_returns": {"eq": 0.0, "ir": -2.0}},
            ]
        )
        self.assertEqual(result["means"], [0.0, 0.0])
        self.assertEqual(result["covariance_matrix"][0][1], 0.0)
        self.assertEqual(result["covariance_matrix"][1][0], 0.0)
        self.assertEqual(result["correlation_matrix"][0][1], 0.0)
        self.assertEqual(result["correlation_matrix"][1][0], 0.0)

    def test_zero_volatility_factor(self):
        result = self.call(
            factors=["eq", "flat"],
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 1, "flat": 5}},
                {"date": "2024-01-02", "factor_returns": {"eq": 2, "flat": 5}},
                {"date": "2024-01-03", "factor_returns": {"eq": 3, "flat": 5}},
            ],
        )
        self.assertEqual(result["volatilities"], [1.0, 0.0])
        self.assertEqual(result["covariance_matrix"][1], [0.0, 0.0])
        self.assertEqual(
            result["correlation_matrix"],
            [[1.0, 0.0], [0.0, 1.0]],
        )

    def test_matrices_are_symmetric(self):
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
        for value in result["correlation_matrix"][0] + result["correlation_matrix"][1]:
            self.assertTrue(math.isfinite(value))

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
                {"date": "2024-01-03", "factor_returns": {"eq": 3, "ir": 6}},
            ],
        )
        self.assertEqual(result["factors"], ["eq", "ir"])
        self.assertEqual(result["means"], [2.0, 4.0])

    def test_dates_are_not_sorted(self):
        result = self.call(
            observations=[
                {"date": "2024-01-03", "factor_returns": {"eq": 3, "ir": 6}},
                {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
                {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
            ]
        )
        self.assertEqual(result["means"], [2.0, 4.0])

    def test_result_is_json_serializable(self):
        json.dumps(self.call(), sort_keys=True)

    # ---- 400 invalid_request ----

    def test_unparseable_json(self):
        with self.assertRaises(InvalidRequest):
            self.service.covariance_estimate(b"{not json")

    def test_top_level_not_object(self):
        with self.assertRaises(InvalidRequest):
            self.service.covariance_estimate(b"[1, 2]")

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
                self.service.covariance_estimate(body)

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
