import http.client
import json
import math
import threading
import unittest
from http.server import ThreadingHTTPServer

from riskscope.server import Handler
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


class EwmaCovarianceEstimateTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.ewma_covariance_estimate(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_defaults(self):
        result = self.call()
        self.assertEqual(result["factors"], ["eq", "ir"])
        self.assertEqual(result["decay"], 0.94)
        self.assertEqual(result["observation_count"], 3)
        self.assertEqual(
            set(result),
            {
                "factors",
                "decay",
                "observation_count",
                "covariance_matrix",
                "volatilities",
                "correlation_matrix",
            },
        )
        for row in result["covariance_matrix"]:
            self.assertEqual(len(row), 2)
        self.assertEqual(len(result["volatilities"]), 2)
        self.assertEqual(len(result["correlation_matrix"]), 2)

    def test_recursion_seeds_on_first_outer_product(self):
        # decay 0.5, two observations: Σ = 0.5·r1r1ᵀ + 0.5·r2r2ᵀ.
        result = self.call(
            decay=0.5,
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
                {"date": "2024-01-02", "factor_returns": {"eq": 3, "ir": 4}},
            ],
        )
        self.assertEqual(result["decay"], 0.5)
        self.assertEqual(result["observation_count"], 2)
        # r1r1ᵀ = [[1, 2], [2, 4]], r2r2ᵀ = [[9, 12], [12, 16]]
        self.assertEqual(
            result["covariance_matrix"],
            [[5.0, 7.0], [7.0, 10.0]],
        )
        self.assertEqual(result["volatilities"], [math.sqrt(5.0), math.sqrt(10.0)])

    def test_three_step_recursion_values(self):
        # decay 0.9: Σ3 = 0.9·(0.9·r1r1ᵀ + 0.1·r2r2ᵀ) + 0.1·r3r3ᵀ.
        result = self.call(
            decay=0.9,
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 1.0, "ir": 0.0}},
                {"date": "2024-01-02", "factor_returns": {"eq": 0.0, "ir": 1.0}},
                {"date": "2024-01-03", "factor_returns": {"eq": 1.0, "ir": 1.0}},
            ],
        )
        matrix = result["covariance_matrix"]
        self.assertAlmostEqual(matrix[0][0], 0.9 * 0.9 + 0.1)
        self.assertAlmostEqual(matrix[1][1], 0.9 * 0.1 + 0.1)
        self.assertAlmostEqual(matrix[0][1], 0.1)
        self.assertAlmostEqual(matrix[1][0], 0.1)

    def test_recent_observations_dominate(self):
        # With a long series the seed weight has decayed away, so the same
        # shock landing on the last observation outweighs it on the first.
        calm = [
            {"date": f"2024-{i:03d}", "factor_returns": {"eq": 0.01, "ir": 0.01}}
            for i in range(1, 101)
        ]
        wild = {"date": "2024-wild", "factor_returns": {"eq": 0.2, "ir": 0.2}}
        calm_then_wild = self.call(
            decay=0.94, observations=calm + [wild]
        )
        wild_then_calm = self.call(
            decay=0.94, observations=[wild] + calm
        )
        self.assertGreater(
            calm_then_wild["covariance_matrix"][0][0],
            wild_then_calm["covariance_matrix"][0][0],
        )

    def test_factor_order_drives_output_positions(self):
        result = self.call(factors=["ir", "eq"])
        self.assertEqual(result["factors"], ["ir", "eq"])
        matrix = result["covariance_matrix"]
        self.assertEqual(matrix[0][0], matrix[1][1] * 4.0)
        self.assertEqual(result["volatilities"][0], result["volatilities"][1] * 2.0)

    def test_all_zero_returns(self):
        result = self.call(
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 0.0, "ir": 0.0}},
                {"date": "2024-01-02", "factor_returns": {"eq": 0.0, "ir": 0.0}},
            ],
        )
        self.assertEqual(result["covariance_matrix"], [[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(result["volatilities"], [0.0, 0.0])
        self.assertEqual(
            result["correlation_matrix"],
            [[1.0, 0.0], [0.0, 1.0]],
        )

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

    def test_matrices_are_exactly_symmetric(self):
        result = self.call(
            factors=["a", "b", "c"],
            decay=0.97,
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
        # Unit diagonal on the correlation matrix.
        for i in range(3):
            self.assertEqual(result["correlation_matrix"][i][i], 1.0)
        for value in result["correlation_matrix"][0] + result["correlation_matrix"][1]:
            self.assertTrue(math.isfinite(value))

    def test_extra_factors_and_fields_ignored(self):
        result = self.call(
            currency="EUR",
            decay=0.9,
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
        self.assertEqual(result["decay"], 0.9)

    def test_dates_are_not_sorted(self):
        in_order = self.call(
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
                {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
                {"date": "2024-01-03", "factor_returns": {"eq": 3, "ir": 6}},
            ]
        )
        shuffled = self.call(
            observations=[
                {"date": "2024-01-03", "factor_returns": {"eq": 3, "ir": 6}},
                {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
                {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
            ]
        )
        self.assertNotEqual(
            in_order["covariance_matrix"], shuffled["covariance_matrix"]
        )

    def test_result_is_json_serializable(self):
        json.dumps(self.call(), sort_keys=True)

    def test_covariance_feeds_parametric_var_directly(self):
        estimate = self.call(decay=0.94)
        var_request = {
            "confidence": 0.99,
            "factors": estimate["factors"],
            "positions": [
                {"id": "p1", "sensitivities": {"eq": 100.0, "ir": -50.0}}
            ],
            "covariance_matrix": estimate["covariance_matrix"],
        }
        result = self.service.parametric_var(json.dumps(var_request).encode("utf-8"))
        self.assertTrue(math.isfinite(result["var"]))
        self.assertGreaterEqual(result["var"], 0.0)
        self.assertTrue(math.isfinite(result["expected_shortfall"]))

    # ---- decay validation ----

    def test_decay_boundaries_inside_open_interval(self):
        result = self.call(decay=0.01, observations=[
            {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
            {"date": "2024-01-02", "factor_returns": {"eq": 3, "ir": 4}},
        ])
        self.assertEqual(result["decay"], 0.01)

    def test_decay_outside_interval(self):
        observations = [
            {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
            {"date": "2024-01-02", "factor_returns": {"eq": 3, "ir": 4}},
        ]
        for bad in (0, 1, 0.0, 1.0, -0.1, 1.01):
            with self.assertRaises(InvalidInput):
                self.call(decay=bad, observations=observations)

    def test_decay_wrong_type(self):
        observations = [
            {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
            {"date": "2024-01-02", "factor_returns": {"eq": 3, "ir": 4}},
        ]
        for bad in ("0.94", True, None, [0.94], {}):
            with self.assertRaises(InvalidInput):
                self.call(decay=bad, observations=observations)

    def test_decay_rejects_nan_and_infinity(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            body = (
                b'{"factors": ["eq"], "decay": ' + token.encode()
                + b', "observations": ['
                b'{"date": "2024-01-01", "factor_returns": {"eq": 1}},'
                b'{"date": "2024-01-02", "factor_returns": {"eq": 2}}]}'
            )
            with self.assertRaises(InvalidInput):
                self.service.ewma_covariance_estimate(body)

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

    def test_two_observations_is_the_boundary(self):
        result = self.call(
            observations=[
                {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
                {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
            ]
        )
        self.assertEqual(result["observation_count"], 2)

    def test_observation_not_object(self):
        with self.assertRaises(InvalidInput):
            self.call(
                observations=[
                    {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
                    "oops",
                ]
            )

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

    def test_factor_observation_product_boundary(self):
        # 100 factors x 10000 observations is the allowed boundary.
        factors = [f"f{i}" for i in range(100)]
        returns = {factor: 1 for factor in factors}
        observations = [
            {"date": f"2024-01-{i:05d}", "factor_returns": returns}
            for i in range(10000)
        ]
        result = self.call(factors=factors, observations=observations)
        self.assertEqual(result["observation_count"], 10000)
        self.assertEqual(len(result["covariance_matrix"]), 100)

    # ---- non-finite computation fails the whole request ----

    def test_non_finite_computation_fails(self):
        with self.assertRaises(InvalidInput):
            self.call(
                observations=[
                    {"date": "2024-01-01", "factor_returns": {"eq": 1e308, "ir": 2}},
                    {"date": "2024-01-02", "factor_returns": {"eq": 1e308, "ir": 4}},
                ]
            )


class EwmaCovarianceEstimateHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def post(self, body, path="/market-risk/ewma-covariance-estimate"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_route_ok_with_default_decay(self):
        status, payload = self.post(request_body())
        self.assertEqual(status, 200)
        self.assertEqual(payload["factors"], ["eq", "ir"])
        self.assertEqual(payload["decay"], 0.94)
        self.assertEqual(payload["observation_count"], 3)
        self.assertIn("covariance_matrix", payload)
        self.assertIn("volatilities", payload)
        self.assertIn("correlation_matrix", payload)

    def test_invalid_json_returns_400_error_object(self):
        status, payload = self.post(b"nonsense")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")
        self.assertIn("message", payload["error"])

    def test_duplicate_factor_returns_422(self):
        status, payload = self.post(request_body(factors=["eq", "eq"]))
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "duplicate_factor")

    def test_duplicate_observation_returns_422(self):
        status, payload = self.post(
            request_body(
                observations=[
                    {"date": "2024-01-01", "factor_returns": {"eq": 1, "ir": 2}},
                    {"date": "2024-01-01", "factor_returns": {"eq": 2, "ir": 4}},
                ]
            )
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "duplicate_observation")

    def test_missing_factor_returns_422(self):
        status, payload = self.post(
            request_body(
                observations=[
                    {"date": "2024-01-01", "factor_returns": {"eq": 1}},
                    {"date": "2024-01-02", "factor_returns": {"eq": 2, "ir": 4}},
                ]
            )
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "missing_factor")

    def test_bad_decay_returns_422(self):
        status, payload = self.post(request_body(decay=1))
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "invalid_input")

    def test_too_large_returns_413(self):
        status, payload = self.post(
            request_body(factors=[f"f{i}" for i in range(101)])
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "request_too_large")

    def test_unknown_route_still_404(self):
        status, payload = self.post(request_body(), path="/market-risk/nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
