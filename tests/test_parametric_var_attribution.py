import http.client
import json
import math
import threading
import unittest
from http.server import ThreadingHTTPServer
from statistics import NormalDist

from riskscope.server import Handler
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


class ParametricVarAttributionTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.parametric_var_attribution(request_body(**overrides))

    def assert_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)

    # ---- portfolio metrics are preserved verbatim ----

    def test_portfolio_metrics_match_parametric_var(self):
        baseline = self.service.parametric_var(request_body())
        result = self.call()
        for key in (
            "currency",
            "confidence",
            "factors",
            "aggregate_sensitivities",
            "variance",
            "volatility",
            "var",
            "expected_shortfall",
        ):
            self.assertEqual(result[key], baseline[key], key)

    def test_basic_factor_values(self):
        result = self.call()
        # aggregate s = [80, 50]; diagonal covariance gives c = [3.2, 0.5].
        self.assertEqual(result["aggregate_sensitivities"], [80.0, 50.0])
        factors = result["factor_attributions"]
        self.assertEqual([f["factor"] for f in factors], ["eq", "ir"])
        self.assertEqual(factors[0]["aggregate_sensitivity"], 80.0)
        self.assertEqual(factors[1]["aggregate_sensitivity"], 50.0)
        self.assertEqual(factors[0]["covariance_loading"], 3.2)
        self.assertEqual(factors[1]["covariance_loading"], 0.5)
        # variance_contribution = s_i * c_i
        self.assertEqual(factors[0]["variance_contribution"], 256.0)
        self.assertEqual(factors[1]["variance_contribution"], 25.0)

    def test_component_metrics_values(self):
        result = self.call()
        volatility = result["volatility"]
        z = NormalDist().inv_cdf(0.99)
        density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        first = result["factor_attributions"][0]
        self.assertEqual(first["component_var"], z * 256.0 / volatility)
        self.assertEqual(
            first["component_expected_shortfall"],
            density * 256.0 / ((1.0 - 0.99) * volatility),
        )

    def test_position_variance_contributions(self):
        result = self.call()
        positions = result["position_attributions"]
        # c = [3.2, 0.5]; p1 = [100, 50] -> 345, p2 = [-20, 0] -> -64.
        self.assertEqual([p["id"] for p in positions], ["p1", "p2"])
        self.assertEqual(positions[0]["variance_contribution"], 345.0)
        self.assertEqual(positions[1]["variance_contribution"], -64.0)
        volatility = result["volatility"]
        z = NormalDist().inv_cdf(0.99)
        self.assertEqual(positions[0]["component_var"], z * 345.0 / volatility)
        self.assertEqual(positions[1]["component_var"], z * -64.0 / volatility)

    def test_covariance_loading_uses_cross_terms(self):
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {"eq": 1.0, "ir": 1.0}}],
            covariance_matrix=[[1.0, 0.5], [0.5, 1.0]],
        )
        factors = result["factor_attributions"]
        self.assertEqual(factors[0]["covariance_loading"], 1.5)
        self.assertEqual(factors[1]["covariance_loading"], 1.5)
        self.assertEqual(factors[0]["variance_contribution"], 1.5)
        self.assertEqual(factors[1]["variance_contribution"], 1.5)
        self.assertEqual(result["position_attributions"][0]["variance_contribution"], 3.0)

    def test_factor_and_position_orders_preserved(self):
        result = self.call(
            factors=["ir", "eq"],
            covariance_matrix=[[0.01, 0.0], [0.0, 0.04]],
            positions=[
                {"id": "z", "sensitivities": {"ir": 1.0}},
                {"id": "a", "sensitivities": {"eq": 2.0}},
                {"id": "m", "sensitivities": {}},
            ],
        )
        self.assertEqual([f["factor"] for f in result["factor_attributions"]], ["ir", "eq"])
        self.assertEqual(
            [p["id"] for p in result["position_attributions"]], ["z", "a", "m"]
        )

    def test_negative_contributions_kept(self):
        # M = [[1, 0.5], [0.5, 1]], s = [-1, 3] gives c = [0.5, 2.5] and the
        # eq contribution is -0.5; it must be reported, not clipped.
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {"eq": -1.0, "ir": 3.0}}],
            covariance_matrix=[[1.0, 0.5], [0.5, 1.0]],
        )
        factor = result["factor_attributions"][0]
        self.assertEqual(factor["covariance_loading"], 0.5)
        self.assertEqual(factor["variance_contribution"], -0.5)
        self.assertLess(factor["component_var"], 0.0)
        self.assertLess(factor["component_expected_shortfall"], 0.0)
        self.assertEqual(result["variance"], 7.0)

    def test_zero_rows_and_zero_factors_kept(self):
        result = self.call(
            positions=[
                {"id": "p1", "sensitivities": {"eq": 1.0}},
                {"id": "empty", "sensitivities": {}},
            ],
        )
        factors = result["factor_attributions"]
        # The ir factor has zero aggregate sensitivity.
        self.assertEqual(len(factors), 2)
        self.assertEqual(factors[1]["aggregate_sensitivity"], 0.0)
        self.assertEqual(factors[1]["covariance_loading"], 0.0)
        self.assertEqual(factors[1]["variance_contribution"], 0.0)
        self.assertEqual(factors[1]["component_var"], 0.0)
        self.assertEqual(factors[1]["component_expected_shortfall"], 0.0)
        positions = result["position_attributions"]
        self.assertEqual(len(positions), 2)
        empty = positions[1]
        self.assertEqual(empty["id"], "empty")
        self.assertEqual(empty["variance_contribution"], 0.0)
        self.assertEqual(empty["component_var"], 0.0)
        self.assertEqual(empty["component_expected_shortfall"], 0.0)

    def test_partition_identities_within_tolerance(self):
        # Both layers, all three contribution kinds, sum to the portfolio
        # value within 1e-12 * max(1, |portfolio value|).
        result = self.call()
        cases = [
            ("variance_contribution", result["variance"]),
            ("component_var", result["var"]),
            ("component_expected_shortfall", result["expected_shortfall"]),
        ]
        for layer in ("factor_attributions", "position_attributions"):
            for key, target in cases:
                total = sum(entry[key] for entry in result[layer])
                self.assertLessEqual(
                    abs(total - target),
                    1e-12 * max(1.0, abs(target)),
                    (layer, key, total, target),
                )

    def test_zero_volatility_loads_matrix_but_zeroes_contributions(self):
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {"eq": 5.0}}],
            covariance_matrix=[[0.0, 0.0], [0.0, 0.0]],
        )
        self.assertEqual(result["volatility"], 0.0)
        self.assertEqual(result["var"], 0.0)
        self.assertEqual(result["expected_shortfall"], 0.0)
        factor = result["factor_attributions"][0]
        # covariance_loading is still the matrix product (0 here).
        self.assertEqual(factor["covariance_loading"], 0.0)
        for entry in result["factor_attributions"]:
            self.assertEqual(entry["variance_contribution"], 0.0)
            self.assertEqual(entry["component_var"], 0.0)
            self.assertEqual(entry["component_expected_shortfall"], 0.0)
        for entry in result["position_attributions"]:
            self.assertEqual(entry["variance_contribution"], 0.0)
            self.assertEqual(entry["component_var"], 0.0)
            self.assertEqual(entry["component_expected_shortfall"], 0.0)

    def test_integers_accepted(self):
        result = self.service.parametric_var_attribution(
            json.dumps(
                {
                    "confidence": 0.99,
                    "factors": ["eq"],
                    "positions": [{"id": "p1", "sensitivities": {"eq": 2}}],
                    "covariance_matrix": [[1]],
                }
            ).encode("utf-8")
        )
        self.assertEqual(result["factor_attributions"][0]["variance_contribution"], 4.0)
        self.assertIsInstance(
            result["factor_attributions"][0]["covariance_loading"], float
        )

    def test_currency_default_and_echo(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

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
            self.service.parametric_var_attribution(b"{not json")

    def test_top_level_not_object(self):
        with self.assertRaises(InvalidRequest):
            self.service.parametric_var_attribution(b"[1, 2]")

    # ---- 422 invalid_input ----

    def test_bad_currency(self):
        for bad in ("", 1, None, True):
            with self.assertRaises(InvalidInput):
                self.call(currency=bad)

    def test_confidence_out_of_range(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaises(InvalidInput):
                self.call(confidence=bad)

    def test_sensitivity_rejects_bool_nan_and_oversized_int(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p1", "sensitivities": {"eq": True}}])
        with self.assertRaises(InvalidInput):
            self.call(
                positions=[{"id": "p1", "sensitivities": {"eq": 10**400}}]
            )
        for token in ("NaN", "Infinity", "-Infinity"):
            body = (
                b'{"confidence": 0.99, "factors": ["eq"], "positions": ['
                b'{"id": "p1", "sensitivities": {"eq": ' + token.encode()
                + b'}}], "covariance_matrix": [[1.0]]}'
            )
            with self.assertRaises(InvalidInput):
                self.service.parametric_var_attribution(body)

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

    def test_invalid_covariance_codes(self):
        self.assert_code(
            "invalid_covariance",
            covariance_matrix=[[1.0, 0.5], [0.4, 1.0]],
        )
        self.assert_code(
            "invalid_covariance",
            covariance_matrix=[[-1.0, 0.0], [0.0, 1.0]],
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

    def test_size_boundary_allowed(self):
        factors = [f"f{i}" for i in range(100)]
        matrix = [[0.0] * 100 for _ in range(100)]
        matrix[0][0] = 1.0
        positions = [
            {"id": f"p{i}", "sensitivities": {"f0": 1.0}}
            for i in range(10000)
        ]
        result = self.call(
            factors=factors, positions=positions, covariance_matrix=matrix
        )
        self.assertEqual(len(result["position_attributions"]), 10000)

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


class ParametricVarAttributionHttpTest(unittest.TestCase):
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

    def post(self, body, path="/market-risk/parametric-var-attribution"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_attribution_route_ok(self):
        status, payload = self.post(request_body())
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["factor_attributions"]), 2)
        self.assertEqual(len(payload["position_attributions"]), 2)
        self.assertEqual(payload["variance"], 281.0)

    def test_invalid_json_returns_400_without_partial_result(self):
        status, payload = self.post(b"nonsense")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")
        self.assertNotIn("factor_attributions", payload)

    def test_top_level_array_returns_400(self):
        status, payload = self.post(b"[1, 2]")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")

    def test_invalid_covariance_returns_422(self):
        status, payload = self.post(
            request_body(covariance_matrix=[[1.0, 0.5], [0.4, 1.0]])
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "invalid_covariance")

    def test_duplicate_position_returns_422(self):
        status, payload = self.post(
            request_body(
                positions=[
                    {"id": "p1", "sensitivities": {"eq": 1.0}},
                    {"id": "p1", "sensitivities": {"eq": 2.0}},
                ]
            )
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "duplicate_position")

    def test_too_large_returns_413(self):
        status, payload = self.post(
            request_body(factors=[f"f{i}" for i in range(101)])
        )
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "request_too_large")

    def test_original_parametric_var_route_unchanged(self):
        status, payload = self.post(
            request_body(), path="/market-risk/parametric-var"
        )
        self.assertEqual(status, 200)
        self.assertNotIn("factor_attributions", payload)
        self.assertNotIn("position_attributions", payload)
        self.assertEqual(payload["variance"], 281.0)


if __name__ == "__main__":
    unittest.main()
