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


def assert_sum_close(test_case, values, target):
    tolerance = 1e-12 * max(1.0, abs(target))
    test_case.assertLessEqual(abs(sum(values) - target), tolerance)


class ParametricVarAttributionTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.parametric_var_attribution(request_body(**overrides))

    # ---- portfolio metrics match the plain parametric route ----

    def test_portfolio_metrics_match_parametric_var(self):
        body = request_body()
        attributed = self.service.parametric_var_attribution(body)
        plain = self.service.parametric_var(body)
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
            self.assertEqual(attributed[key], plain[key])

    # ---- factor attributions ----

    def test_factor_attributions_values(self):
        result = self.call()
        # s = (80, 50); c = Σs = (3.2, 0.5)
        attributions = result["factor_attributions"]
        self.assertEqual([a["factor"] for a in attributions], ["eq", "ir"])
        self.assertEqual(
            [a["aggregate_sensitivity"] for a in attributions], [80.0, 50.0]
        )
        self.assertEqual(
            [a["covariance_loading"] for a in attributions], [3.2, 0.5]
        )
        self.assertEqual(
            [a["variance_contribution"] for a in attributions], [256.0, 25.0]
        )
        volatility = math.sqrt(281.0)
        z = NormalDist().inv_cdf(0.99)
        density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        for attribution, contribution in zip(attributions, (256.0, 25.0)):
            self.assertEqual(
                attribution["component_var"], z * contribution / volatility
            )
            self.assertEqual(
                attribution["component_expected_shortfall"],
                density * contribution / ((1.0 - 0.99) * volatility),
            )

    def test_factor_attributions_follow_factor_order(self):
        result = self.call(
            factors=["ir", "eq"],
            covariance_matrix=[
                [0.01, 0.0],
                [0.0, 0.04],
            ],
        )
        attributions = result["factor_attributions"]
        self.assertEqual([a["factor"] for a in attributions], ["ir", "eq"])
        self.assertEqual(
            [a["aggregate_sensitivity"] for a in attributions], [50.0, 80.0]
        )
        self.assertEqual(
            [a["variance_contribution"] for a in attributions], [25.0, 256.0]
        )

    def test_factor_attributions_correlated(self):
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {"eq": 1.0, "ir": 1.0}}],
            covariance_matrix=[
                [1.0, 0.5],
                [0.5, 1.0],
            ],
        )
        attributions = result["factor_attributions"]
        # c = (1.5, 1.5); contributions 1.5 each, summing to variance 3.
        self.assertEqual(
            [a["covariance_loading"] for a in attributions], [1.5, 1.5]
        )
        self.assertEqual(
            [a["variance_contribution"] for a in attributions], [1.5, 1.5]
        )

    def test_factor_contributions_can_be_negative(self):
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {"eq": 1.0, "ir": -0.5}}],
            covariance_matrix=[
                [1.0, 0.9],
                [0.9, 1.0],
            ],
        )
        attributions = result["factor_attributions"]
        # c = (0.55, 0.4); contributions 0.55 and -0.2.
        self.assertEqual(
            [a["variance_contribution"] for a in attributions], [0.55, -0.2]
        )
        self.assertLess(attributions[1]["component_var"], 0.0)

    # ---- position attributions ----

    def test_position_attributions_values(self):
        result = self.call()
        # c = Σs = (3.2, 0.5); p1 = (100, 50), p2 = (-20, 0)
        attributions = result["position_attributions"]
        self.assertEqual([a["id"] for a in attributions], ["p1", "p2"])
        self.assertEqual(
            [a["variance_contribution"] for a in attributions], [345.0, -64.0]
        )
        volatility = math.sqrt(281.0)
        z = NormalDist().inv_cdf(0.99)
        density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        for attribution, contribution in zip(attributions, (345.0, -64.0)):
            self.assertEqual(
                attribution["component_var"], z * contribution / volatility
            )
            self.assertEqual(
                attribution["component_expected_shortfall"],
                density * contribution / ((1.0 - 0.99) * volatility),
            )

    def test_position_attributions_follow_position_order(self):
        result = self.call(
            positions=[
                {"id": "b", "sensitivities": {"eq": 1.0}},
                {"id": "a", "sensitivities": {"eq": 2.0}},
            ]
        )
        self.assertEqual(
            [a["id"] for a in result["position_attributions"]], ["b", "a"]
        )

    def test_zero_contributions_not_omitted(self):
        result = self.call(
            positions=[
                {"id": "p1", "sensitivities": {"eq": 1.0}},
                {"id": "p2", "sensitivities": {"ir": 5.0}},
            ],
            covariance_matrix=[
                [1.0, 0.0],
                [0.0, 0.0],
            ],
        )
        attributions = result["position_attributions"]
        self.assertEqual(len(attributions), 2)
        self.assertEqual(attributions[1]["variance_contribution"], 0.0)
        self.assertEqual(attributions[1]["component_var"], 0.0)
        self.assertEqual(attributions[1]["component_expected_shortfall"], 0.0)
        factor_attributions = result["factor_attributions"]
        self.assertEqual(len(factor_attributions), 2)
        self.assertEqual(factor_attributions[1]["variance_contribution"], 0.0)

    # ---- contribution sums equal the portfolio values ----

    def test_contribution_sums_match_portfolio(self):
        result = self.call(
            covariance_matrix=[
                [0.04, 0.01],
                [0.01, 0.01],
            ],
        )
        for attributions in (
            result["factor_attributions"],
            result["position_attributions"],
        ):
            assert_sum_close(
                self,
                [a["variance_contribution"] for a in attributions],
                result["variance"],
            )
            assert_sum_close(
                self,
                [a["component_var"] for a in attributions],
                result["var"],
            )
            assert_sum_close(
                self,
                [a["component_expected_shortfall"] for a in attributions],
                result["expected_shortfall"],
            )

    # ---- zero volatility ----

    def test_zero_volatility_zeroes_contributions_keeps_loadings(self):
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {"eq": 5.0}}],
            covariance_matrix=[
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        )
        self.assertEqual(result["volatility"], 0.0)
        self.assertEqual(result["var"], 0.0)
        self.assertEqual(result["expected_shortfall"], 0.0)
        for attribution in result["factor_attributions"]:
            self.assertEqual(attribution["covariance_loading"], 0.0)
            self.assertEqual(attribution["variance_contribution"], 0.0)
            self.assertEqual(attribution["component_var"], 0.0)
            self.assertEqual(attribution["component_expected_shortfall"], 0.0)
        for attribution in result["position_attributions"]:
            self.assertEqual(attribution["variance_contribution"], 0.0)
            self.assertEqual(attribution["component_var"], 0.0)
            self.assertEqual(attribution["component_expected_shortfall"], 0.0)

    def test_result_is_json_serializable(self):
        json.dumps(self.call(), sort_keys=True)

    # ---- 400 invalid_request ----

    def test_unparseable_json(self):
        with self.assertRaises(InvalidRequest):
            self.service.parametric_var_attribution(b"{not json")

    def test_top_level_not_object(self):
        with self.assertRaises(InvalidRequest):
            self.service.parametric_var_attribution(b"[1, 2]")

    # ---- 422 invalid_input / specific codes ----

    def assert_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)

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

    def test_invalid_covariance(self):
        self.assert_code(
            "invalid_covariance",
            covariance_matrix=[[1.0, 0.5], [0.4, 1.0]],
        )

    def test_invalid_input_generic(self):
        self.assert_code("invalid_input", confidence=1.5)

    def test_sensitivity_rejects_bool_nan_infinity_and_oversized_int(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p1", "sensitivities": {"eq": True}}])
        for token in ("NaN", "Infinity", "-Infinity"):
            body = (
                b'{"confidence": 0.99, "factors": ["eq"], "positions": ['
                b'{"id": "p1", "sensitivities": {"eq": ' + token.encode()
                + b'}}], "covariance_matrix": [[1.0]]}'
            )
            with self.assertRaises(InvalidInput):
                self.service.parametric_var_attribution(body)
        with self.assertRaises(InvalidInput):
            self.call(
                positions=[{"id": "p1", "sensitivities": {"eq": 10**400}}]
            )

    def test_integers_accepted(self):
        result = self.call(
            positions=[{"id": "p1", "sensitivities": {"eq": 2, "ir": 1}}],
            covariance_matrix=[[1, 0], [0, 1]],
        )
        self.assertEqual(result["variance"], 5.0)

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
