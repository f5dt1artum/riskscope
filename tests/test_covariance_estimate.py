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
            {"date": "2024-01-01", "factor_returns": {"eq": 0.01, "ir": 0.0, "extra": 9.0}},
            {"date": "2024-01-02", "factor_returns": {"eq": -0.02, "ir": 0.01}},
            {"date": "2024-01-03", "factor_returns": {"eq": 0.03, "ir": -0.02}},
            {"date": "2024-01-04", "factor_returns": {"eq": -0.04, "ir": 0.02}},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class CovarianceEstimateTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.covariance_estimate(request_body(**overrides))

    def test_basic_shapes_and_order(self):
        result = self.call()
        self.assertEqual(result["factors"], ["eq", "ir"])
        self.assertEqual(result["observation_count"], 4)
        self.assertEqual(len(result["means"]), 2)
        self.assertEqual(len(result["volatilities"]), 2)
        self.assertEqual(
            [len(row) for row in result["covariance_matrix"]], [2, 2]
        )
        self.assertEqual(
            [len(row) for row in result["correlation_matrix"]], [2, 2]
        )
        # Means: eq = -.02/4 = -.005, ir = .01/4 = .0025
        self.assertTrue(
            math.isclose(result["means"][0], -0.005, abs_tol=1e-15)
        )
        self.assertTrue(
            math.isclose(result["means"][1], 0.0025, abs_tol=1e-15)
        )

    def test_sample_covariance_divides_by_n_minus_one(self):
        # Hand-computed two-column series.
        body = {
            "factors": ["a", "b"],
            "observations": [
                {"date": "d1", "factor_returns": {"a": 1, "b": 2}},
                {"date": "d2", "factor_returns": {"a": 3, "b": 4}},
                {"date": "d3", "factor_returns": {"a": 5, "b": 6}},
            ],
        }
        result = self.service.covariance_estimate(json.dumps(body).encode())
        self.assertEqual(result["means"], [3.0, 4.0])
        self.assertEqual(
            result["covariance_matrix"], [[4.0, 4.0], [4.0, 4.0]]
        )
        self.assertEqual(result["volatilities"], [2.0, 2.0])
        self.assertEqual(
            result["correlation_matrix"], [[1.0, 1.0], [1.0, 1.0]]
        )

    def test_matrices_are_symmetric_and_correlation_has_unit_diagonal(self):
        result = self.call()
        covariance = result["covariance_matrix"]
        correlation = result["correlation_matrix"]
        for i in range(2):
            self.assertEqual(correlation[i][i], 1.0)
            for j in range(2):
                self.assertEqual(covariance[i][j], covariance[j][i])
                self.assertEqual(correlation[i][j], correlation[j][i])
                self.assertTrue(math.isfinite(covariance[i][j]))
                self.assertTrue(math.isfinite(correlation[i][j]))

    def test_factor_input_order_drives_vector_and_matrix_positions(self):
        body = {
            "factors": ["z", "a", "m"],
            "observations": [
                {"date": "d1", "factor_returns": {"z": 1.0, "a": 0.0, "m": -1.0}},
                {"date": "d2", "factor_returns": {"z": 3.0, "a": 2.0, "m": 1.0}},
            ],
        }
        result = self.service.covariance_estimate(json.dumps(body).encode())
        self.assertEqual(result["factors"], ["z", "a", "m"])
        self.assertEqual(result["means"], [2.0, 1.0, 0.0])
        # Columns are [z,a,m]; rows/columns must follow that order, not sorted.
        self.assertEqual(
            result["covariance_matrix"],
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        )

    def test_observation_order_is_kept_not_date_sorted(self):
        # Same returns in a different date/row permutation must give the same
        # statistics, and dates never sort the computation.
        body_a = {
            "factors": ["f"],
            "observations": [
                {"date": "2024-03-01", "factor_returns": {"f": 3.0}},
                {"date": "2024-01-01", "factor_returns": {"f": 1.0}},
                {"date": "2024-02-01", "factor_returns": {"f": 2.0}},
            ],
        }
        body_b = {
            "factors": ["f"],
            "observations": [
                {"date": "2024-01-01", "factor_returns": {"f": 1.0}},
                {"date": "2024-02-01", "factor_returns": {"f": 2.0}},
                {"date": "2024-03-01", "factor_returns": {"f": 3.0}},
            ],
        }
        ra = self.service.covariance_estimate(json.dumps(body_a).encode())
        rb = self.service.covariance_estimate(json.dumps(body_b).encode())
        self.assertEqual(ra["means"], rb["means"])
        self.assertEqual(ra["covariance_matrix"], rb["covariance_matrix"])

    def test_undeclared_factors_and_extra_fields_ignored(self):
        result = self.call()
        self.assertEqual(result["factors"], ["eq", "ir"])
        self.assertEqual(len(result["covariance_matrix"]), 2)

    def test_integer_returns_accepted(self):
        body = {
            "factors": ["f"],
            "observations": [
                {"date": "a", "factor_returns": {"f": -1}},
                {"date": "b", "factor_returns": {"f": 1}},
            ],
        }
        result = self.service.covariance_estimate(json.dumps(body).encode())
        self.assertEqual(result["means"], [0.0])
        self.assertEqual(result["covariance_matrix"], [[2.0]])
        self.assertEqual(result["volatilities"], [math.sqrt(2.0)])
        self.assertEqual(result["correlation_matrix"], [[1.0]])

    def test_zero_volatility_factor_correlations(self):
        body = {
            "factors": ["const", "v"],
            "observations": [
                {"date": "d1", "factor_returns": {"const": 0.0, "v": 1.0}},
                {"date": "d2", "factor_returns": {"const": 0.0, "v": -3.0}},
            ],
        }
        result = self.service.covariance_estimate(json.dumps(body).encode())
        self.assertEqual(result["volatilities"][0], 0.0)
        self.assertEqual(
            result["correlation_matrix"], [[1.0, 0.0], [0.0, 1.0]]
        )

    def test_everything_serializes_as_finite_json(self):
        result = self.call()
        text = json.dumps(result, sort_keys=True)
        for token in ("NaN", "Infinity"):
            self.assertNotIn(token, text)

    # ---- parse-level failures -> 400 invalid_request ----

    def test_malformed_json(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.covariance_estimate(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_top_level_non_object(self):
        for raw in (b"[]", b"42", b'"x"', b"null"):
            with self.subTest(raw=raw):
                with self.assertRaises(InvalidRequest):
                    self.service.covariance_estimate(raw)

    # ---- semantic failures -> 422 ----

    def test_missing_fields_and_wrong_types(self):
        with self.assertRaises(InvalidInput):
            self.service.covariance_estimate(b"{}")
        with self.assertRaises(InvalidInput):
            self.call(factors=[])
        with self.assertRaises(InvalidInput):
            self.call(observations=[])
        with self.assertRaises(InvalidInput):
            self.call(factors="eq")
        with self.assertRaises(InvalidInput):
            self.call(observations={})
        with self.assertRaises(InvalidInput):
            self.call(factors=[""])
        with self.assertRaises(InvalidInput):
            self.call(factors=[7])
        with self.assertRaises(InvalidInput):
            self.call(observations=[{"date": "d", "factor_returns": []}])
        with self.assertRaises(InvalidInput):
            self.call(
                observations=[
                    {"date": "", "factor_returns": {"eq": 0.0, "ir": 0.0}},
                    {"date": "b", "factor_returns": {"eq": 0.0, "ir": 0.0}},
                ]
            )

    def test_single_observation_rejected(self):
        with self.assertRaises(InvalidInput):
            self.call(
                observations=[
                    {"date": "a", "factor_returns": {"eq": 0.0, "ir": 0.0}}
                ]
            )

    def test_illegal_returns(self):
        good_two = [
            {"date": "a", "factor_returns": {"eq": 0.0, "ir": 0.0}},
            {"date": "b", "factor_returns": {"eq": 0.0, "ir": 0.0}},
        ]

        def body_with(value):
            obs = json.loads(request_body(observations=good_two))
            obs["observations"][0]["factor_returns"]["eq"] = value
            return json.dumps(obs).encode()

        with self.assertRaises(InvalidInput):
            self.service.covariance_estimate(body_with(True))
        with self.assertRaises(InvalidInput):
            self.service.covariance_estimate(body_with("0.1"))
        with self.assertRaises(InvalidInput):
            self.service.covariance_estimate(body_with(None))
        with self.assertRaises(InvalidInput):
            self.service.covariance_estimate(
                body_with(10**400)
            )

    def test_nan_and_infinity_tokens_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                raw = (
                    b'{"factors":["eq","ir"],"observations":['
                    b'{"date":"a","factor_returns":{"eq":'
                    + token.encode()
                    + b',"ir":0.0}},'
                    b'{"date":"b","factor_returns":{"eq":0.0,"ir":0.0}}]}'
                )
                with self.assertRaises(InvalidInput) as ctx:
                    self.service.covariance_estimate(raw)
                self.assertEqual(ctx.exception.code, "invalid_input")

    def test_duplicate_factor(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                factors=["eq", "eq"],
                observations=[
                    {"date": "a", "factor_returns": {"eq": 0.0}},
                    {"date": "b", "factor_returns": {"eq": 1.0}},
                ],
            )
        self.assertEqual(ctx.exception.code, "duplicate_factor")

    def test_duplicate_observation(self):
        obs = [
            {"date": "d", "factor_returns": {"eq": 0.0, "ir": 0.0}},
            {"date": "d", "factor_returns": {"eq": 1.0, "ir": 1.0}},
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

    # ---- size limits -> 413 ----

    def test_too_many_factors(self):
        factors = [f"f{i}" for i in range(101)]
        obs = [
            {
                "date": "d1",
                "factor_returns": {f: 0.0 for f in factors},
            },
            {
                "date": "d2",
                "factor_returns": {f: 0.0 for f in factors},
            },
        ]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.service.covariance_estimate(
                json.dumps({"factors": factors, "observations": obs}).encode()
            )
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

    def test_too_many_observations(self):
        body = {
            "factors": ["eq"],
            "observations": [
                {"date": f"d{i}", "factor_returns": {"eq": 0.0}}
                for i in range(10001)
            ],
        }
        with self.assertRaises(RequestTooLarge) as ctx:
            self.service.covariance_estimate(json.dumps(body).encode())
        self.assertEqual(ctx.exception.code, "request_too_large")

    def test_boundary_limits_allowed(self):
        factors = [f"f{i:03d}" for i in range(100)]
        observations = [
            {
                "date": f"d{j:05d}",
                "factor_returns": {f"f{i:03d}": float((i - j) % 7) for i in range(100)},
            }
            for j in range(10000)
        ]
        result = self.service.covariance_estimate(
            json.dumps({"factors": factors, "observations": observations}).encode()
        )
        self.assertEqual(result["observation_count"], 10000)
        self.assertEqual(len(result["covariance_matrix"]), 100)


if __name__ == "__main__":
    unittest.main()
