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
        "currency": "USD",
        "positions": [
            {"id": "p1", "sensitivities": {"eq": 100.0, "ir": -50.0}},
            {"id": "p2", "sensitivities": {"eq": 25.0, "fx": 10.0}},
        ],
        "scenarios": [
            {"id": "s1", "factor_shocks": {"eq": 0.01, "ir": 0.02, "extra": 9.0}},
            {"id": "s2", "factor_shocks": {"eq": -0.02}},
            {"id": "s3", "factor_shocks": {"ir": -0.03, "fx": 0.5}},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class StressTestTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.stress_test(request_body(**overrides))

    def test_basic_shapes_values_and_order(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        self.assertEqual([item["id"] for item in result["results"]], ["s1", "s2", "s3"])

        s1, s2, s3 = result["results"]

        # s1: eq aggregate 125 * .01 -> -1.25; ir -50 * .02 -> +1.0
        self.assertTrue(math.isclose(s1["loss"], -0.25, abs_tol=1e-12))
        # p1: -(100*.01 + -50*.02) = 0.0; p2: -(25*.01) = -0.25
        self.assertEqual(
            s1["position_loss_contributions"], {"p1": 0.0, "p2": -0.25}
        )
        self.assertEqual(
            s1["factor_loss_contributions"], {"eq": -1.25, "ir": 1.0, "fx": 0.0}
        )
        self.assertNotIn("extra", s1["factor_loss_contributions"])

        # s2: missing ir/fx are zero shocks. eq shock -0.02 -> +2.5
        self.assertTrue(math.isclose(s2["loss"], 2.5, abs_tol=1e-12))
        self.assertEqual(
            s2["position_loss_contributions"], {"p1": 2.0, "p2": 0.5}
        )
        self.assertEqual(
            s2["factor_loss_contributions"], {"eq": 2.5, "ir": 0.0, "fx": 0.0}
        )

        # s3: ir shock -0.03 -> -1.5 (a loss, given negative sensitivity);
        # fx shock 0.5 -> -5.0
        self.assertTrue(math.isclose(s3["loss"], -6.5, abs_tol=1e-12))
        self.assertEqual(
            s3["position_loss_contributions"], {"p1": -1.5, "p2": -5.0}
        )
        self.assertEqual(
            s3["factor_loss_contributions"], {"eq": 0.0, "ir": -1.5, "fx": -5.0}
        )

    def test_contributions_sum_to_loss_for_every_scenario(self):
        result = self.call()
        for item in result["results"]:
            position_sum = sum(item["position_loss_contributions"].values())
            factor_sum = sum(item["factor_loss_contributions"].values())
            self.assertTrue(math.isclose(position_sum, item["loss"], rel_tol=0, abs_tol=1e-12))
            self.assertTrue(math.isclose(factor_sum, item["loss"], rel_tol=0, abs_tol=1e-12))

    def test_contribution_identities_survive_json_wire_order(self):
        # The server serialises with sort_keys; a client sums in that order.
        positions = [
            {"id": "p0", "sensitivities": {"fx": 81.3709, "ir": -68.1684, "cmd": -79}},
            {"id": "p1", "sensitivities": {"fx": -52.1508, "cmd": -842, "ir": -444}},
            {"id": "p2", "sensitivities": {"ir": -527}},
            {"id": "p3", "sensitivities": {"ir": -33.8237, "eq": -87.5469, "cmd": -203}},
            {"id": "p4", "sensitivities": {"cmd": -29.6482, "ir": 78.3283, "eq": 102}},
            {"id": "p5", "sensitivities": {"ir": -2.8716, "cmd": 33.7747}},
        ]
        scenarios = [{"id": "s3", "factor_shocks": {"eq": -2, "cmd": 0.448, "extra": 1}}]
        result = self.service.stress_test(
            request_body(positions=positions, scenarios=scenarios)
        )
        for item in result["results"]:
            for group in ("position_loss_contributions", "factor_loss_contributions"):
                wire = json.loads(json.dumps(item[group], sort_keys=True))
                self.assertEqual(sum(wire.values()), item["loss"])

    def test_many_factors_identity_within_float_tolerance(self):
        positions = [
            {"id": f"p{i}",
             "sensitivities": {"eq": 1.0 + i, "ir": -0.5 * i, "fx": 3.25, "cmd": -7.0}}
            for i in range(10)
        ]
        scenarios = [
            {"id": "s", "factor_shocks": {"eq": 0.01, "ir": 0.02, "fx": -0.03, "cmd": 0.04}},
        ]
        result = self.service.stress_test(
            request_body(positions=positions, scenarios=scenarios)
        )
        item = result["results"][0]
        self.assertTrue(
            math.isclose(
                sum(item["position_loss_contributions"].values()),
                item["loss"], rel_tol=0, abs_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                sum(item["factor_loss_contributions"].values()),
                item["loss"], rel_tol=0, abs_tol=1e-12,
            )
        )

    def test_zero_entries_are_retained(self):
        result = self.call()
        s1 = result["results"][0]
        self.assertIn("fx", s1["factor_loss_contributions"])
        self.assertEqual(s1["factor_loss_contributions"]["fx"], 0.0)
        self.assertEqual(set(s1["position_loss_contributions"]), {"p1", "p2"})

    def test_worst_scenario_is_max_loss(self):
        result = self.call()
        # losses: -0.25, 2.5, -6.5 -> worst is s2
        self.assertEqual(result["worst_scenario"], {"id": "s2", "loss": 2.5})

    def test_worst_scenario_tie_keeps_earliest_input(self):
        scenarios = [
            {"id": "a", "factor_shocks": {"f": 1.0}},
            {"id": "b", "factor_shocks": {"f": 1.0}},
        ]
        positions = [{"id": "p", "sensitivities": {"f": -2.0}}]
        result = self.service.stress_test(
            request_body(positions=positions, scenarios=scenarios)
        )
        self.assertEqual(result["worst_scenario"]["id"], "a")
        self.assertEqual(result["worst_scenario"]["loss"], 2.0)

    def test_negative_losses_are_not_floored_at_zero(self):
        positions = [{"id": "p", "sensitivities": {"f": 10.0}}]
        scenarios = [{"id": "only", "factor_shocks": {"f": 0.1}}]
        result = self.service.stress_test(
            request_body(positions=positions, scenarios=scenarios)
        )
        self.assertEqual(result["results"][0]["loss"], -1.0)
        self.assertEqual(result["worst_scenario"], {"id": "only", "loss": -1.0})

    def test_currency_defaults_to_usd(self):
        body = {
            "positions": [{"id": "p", "sensitivities": {"f": 1.0}}],
            "scenarios": [{"id": "s", "factor_shocks": {"f": 1.0}}],
        }
        result = self.service.stress_test(json.dumps(body).encode())
        self.assertEqual(result["currency"], "USD")

    def test_currency_override(self):
        result = self.call(currency="EUR")
        self.assertEqual(result["currency"], "EUR")

    def test_integer_numbers_accepted(self):
        positions = [{"id": "p", "sensitivities": {"f": 100}}]
        scenarios = [{"id": "s", "factor_shocks": {"f": -1}}]
        result = self.service.stress_test(
            request_body(positions=positions, scenarios=scenarios)
        )
        self.assertEqual(result["results"][0]["loss"], 100.0)
        self.assertEqual(result["results"][0]["position_loss_contributions"], {"p": 100.0})

    def test_booleans_rejected(self):
        with self.assertRaises(InvalidInput):
            self.call(currency=True)
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p", "sensitivities": {"f": True}}])
        with self.assertRaises(InvalidInput):
            self.call(scenarios=[{"id": "s", "factor_shocks": {"f": False}}])

    def test_extra_fields_ignored(self):
        result = self.service.stress_test(
            request_body(confidence=0.99, note="hi", unknown={"nested": 1})
        )
        self.assertEqual(result["currency"], "USD")

    # ---- parse-level failures -> 400 invalid_request ----

    def test_malformed_json(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.stress_test(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_top_level_array(self):
        with self.assertRaises(InvalidRequest):
            self.service.stress_test(b"[]")

    def test_top_level_scalar(self):
        with self.assertRaises(InvalidRequest):
            self.service.stress_test(b"42")

    def test_empty_body(self):
        with self.assertRaises(InvalidRequest):
            self.service.stress_test(b"")

    # ---- semantic failures -> 422 ----

    def test_missing_fields_and_wrong_types(self):
        with self.assertRaises(InvalidInput):
            self.service.stress_test(b"{}")
        with self.assertRaises(InvalidInput):
            self.call(positions=[])
        with self.assertRaises(InvalidInput):
            self.call(scenarios=[])
        with self.assertRaises(InvalidInput):
            self.call(positions="nope")
        with self.assertRaises(InvalidInput):
            self.call(scenarios="nope")
        with self.assertRaises(InvalidInput):
            self.call(positions=[{}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "", "sensitivities": {"f": 1.0}}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p", "sensitivities": {}}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p", "sensitivities": {"f": "x"}}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": 4, "sensitivities": {"f": 1.0}}])
        with self.assertRaises(InvalidInput):
            self.call(scenarios=[{"id": "s"}])
        with self.assertRaises(InvalidInput):
            self.call(scenarios=[{"id": "s", "factor_shocks": {}}])
        with self.assertRaises(InvalidInput):
            self.call(scenarios=[{"id": "", "factor_shocks": {"f": 1.0}}])
        with self.assertRaises(InvalidInput):
            self.call(currency="")

    def test_nan_and_infinity_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                body = request_body().replace(b"100.0", token.encode(), 1)
                with self.assertRaises(InvalidInput) as ctx:
                    self.service.stress_test(body)
                self.assertEqual(ctx.exception.code, "invalid_input")

    def test_oversized_integer_rejected(self):
        positions = [{"id": "p", "sensitivities": {"f": 10 ** 400}}]
        with self.assertRaises(InvalidInput):
            self.service.stress_test(request_body(positions=positions))

    def test_computation_overflow_rejected(self):
        positions = [{"id": "p", "sensitivities": {"f": 1.7e308}}]
        scenarios = [{"id": "s", "factor_shocks": {"f": 10.0}}]
        with self.assertRaises(InvalidInput) as ctx:
            self.service.stress_test(request_body(positions=positions, scenarios=scenarios))
        self.assertEqual(ctx.exception.code, "invalid_input")

    def test_duplicate_position(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(positions=[
                {"id": "p", "sensitivities": {"f": 1.0}},
                {"id": "p", "sensitivities": {"g": 2.0}},
            ])
        self.assertEqual(ctx.exception.code, "duplicate_position")

    def test_duplicate_scenario(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(scenarios=[
                {"id": "s", "factor_shocks": {"f": 1.0}},
                {"id": "s", "factor_shocks": {"g": 2.0}},
            ])
        self.assertEqual(ctx.exception.code, "duplicate_scenario")

    # ---- size limits -> 413 ----

    def test_too_many_scenarios(self):
        scenarios = [
            {"id": f"s{i}", "factor_shocks": {"eq": 0.0}} for i in range(1001)
        ]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(scenarios=scenarios)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

    def test_exactly_1000_scenarios_allowed(self):
        scenarios = [
            {"id": f"s{i:04d}", "factor_shocks": {"eq": 0.01}} for i in range(1000)
        ]
        result = self.call(scenarios=scenarios)
        self.assertEqual(len(result["results"]), 1000)
        self.assertEqual(result["worst_scenario"]["id"], "s0000")

    def test_scenario_position_product_too_large(self):
        positions = [
            {"id": f"p{i}", "sensitivities": {"eq": 0.0}} for i in range(101)
        ]
        scenarios = [
            {"id": f"s{i}", "factor_shocks": {"eq": 0.0}} for i in range(1000)
        ]
        # 101 * 1000 = 101000 > 100000, while scenario count alone is allowed.
        with self.assertRaises(RequestTooLarge):
            self.call(positions=positions, scenarios=scenarios)

    def test_product_boundary_allowed(self):
        # 100 positions * 1000 scenarios == 100000 -> accepted.
        positions = [
            {"id": f"p{i:03d}", "sensitivities": {"eq": 1.0}} for i in range(100)
        ]
        scenarios = [
            {"id": f"s{i:04d}", "factor_shocks": {"eq": 0.01}} for i in range(1000)
        ]
        result = self.call(positions=positions, scenarios=scenarios)
        self.assertEqual(len(result["results"]), 1000)
        self.assertTrue(math.isclose(result["results"][0]["loss"], -1.0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
