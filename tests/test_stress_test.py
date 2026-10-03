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
        "positions": [
            {"id": "p1", "sensitivities": {"a": 100.0, "b": -50.0}},
            {"id": "p2", "sensitivities": {"a": 25.0}},
        ],
        "scenarios": [
            {"id": "s1", "factor_shocks": {"a": 0.01, "b": 0.02, "extra": 9.0}},
            {"id": "s2", "factor_shocks": {"a": -0.02}},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class StressTestTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.stress_test(request_body(**overrides))

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        self.assertEqual([item["scenario_id"] for item in result["results"]], ["s1", "s2"])
        s1, s2 = result["results"]

        # s1: p1 = -100*.01 + -(-50)*.02 = -1 + 1 = 0
        #     p2 = -25*.01 = -0.25
        self.assertTrue(math.isclose(s1["position_loss_contributions"]["p1"], 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(s1["position_loss_contributions"]["p2"], -0.25, abs_tol=1e-12))
        self.assertTrue(math.isclose(s1["factor_loss_contributions"]["a"], -1.25, abs_tol=1e-12))
        self.assertTrue(math.isclose(s1["factor_loss_contributions"]["b"], 1.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(s1["loss"], -0.25, abs_tol=1e-12))

        # s2: missing b -> zero shock; p1 = -100*-.02 = 2, p2 = -25*-.02 = .5
        self.assertTrue(math.isclose(s2["position_loss_contributions"]["p1"], 2.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(s2["position_loss_contributions"]["p2"], 0.5, abs_tol=1e-12))
        self.assertTrue(math.isclose(s2["factor_loss_contributions"]["a"], 2.5, abs_tol=1e-12))
        self.assertTrue(math.isclose(s2["factor_loss_contributions"]["b"], 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(s2["loss"], 2.5, abs_tol=1e-12))

        self.assertEqual(result["worst_scenario"], {"id": "s2", "loss": s2["loss"]})

    def test_extra_scenario_factor_ignored(self):
        result = self.call()
        for item in result["results"]:
            self.assertNotIn("extra", item["factor_loss_contributions"])

    def test_zero_entries_retained(self):
        # A factor referenced by positions but absent from a scenario keeps a
        # zero contribution, as does a position whose losses net to zero.
        result = self.call()
        s1, s2 = result["results"]
        self.assertIn("p1", s1["position_loss_contributions"])
        self.assertEqual(set(s2["factor_loss_contributions"]), {"a", "b"})
        self.assertEqual(s2["factor_loss_contributions"]["b"], 0.0)
        # Every position id appears for every scenario.
        for item in result["results"]:
            self.assertEqual(
                set(item["position_loss_contributions"]), {"p1", "p2"}
            )

    def test_contribution_groups_total_loss(self):
        result = self.call()
        for item in result["results"]:
            # The position fold is the loss accumulator itself: bit-exact.
            position_total = 0.0
            for value in item["position_loss_contributions"].values():
                position_total += value
            self.assertEqual(position_total, item["loss"])
            # The factor fold traverses the same cells in another order, so it
            # agrees within floating-point rounding, as for the VaR identity.
            factor_total = math.fsum(item["factor_loss_contributions"].values())
            self.assertTrue(
                math.isclose(factor_total, item["loss"], rel_tol=0, abs_tol=1e-12)
            )

    def test_worst_scenario_tie_keeps_input_order(self):
        positions = [{"id": "p", "sensitivities": {"f": 100.0}}]
        scenarios = [
            {"id": "first", "factor_shocks": {"f": -0.1}},
            {"id": "second", "factor_shocks": {"f": -0.1}},
            {"id": "flat", "factor_shocks": {"f": 0.0}},
        ]
        result = self.service.stress_test(
            request_body(positions=positions, scenarios=scenarios)
        )
        self.assertEqual(result["worst_scenario"]["id"], "first")
        self.assertTrue(math.isclose(result["worst_scenario"]["loss"], 10.0, abs_tol=1e-12))

    def test_all_negative_losses_not_clamped(self):
        positions = [{"id": "p", "sensitivities": {"f": 1.0}}]
        scenarios = [
            {"id": "s1", "factor_shocks": {"f": 1.0}},
            {"id": "s2", "factor_shocks": {"f": 2.0}},
        ]
        result = self.service.stress_test(
            request_body(positions=positions, scenarios=scenarios)
        )
        self.assertEqual(result["worst_scenario"]["id"], "s1")
        self.assertEqual(result["worst_scenario"]["loss"], -1.0)
        self.assertEqual([item["loss"] for item in result["results"]], [-1.0, -2.0])

    def test_currency_override_and_default(self):
        self.assertEqual(self.call()["currency"], "USD")
        body = json.loads(request_body())
        body["currency"] = "EUR"
        result = self.service.stress_test(json.dumps(body).encode())
        self.assertEqual(result["currency"], "EUR")

    def test_integer_numbers_accepted(self):
        body = {
            "positions": [{"id": "p", "sensitivities": {"f": 100}}],
            "scenarios": [{"id": "s", "factor_shocks": {"f": -1}}],
        }
        result = self.service.stress_test(json.dumps(body).encode())
        self.assertEqual(result["results"][0]["loss"], 100.0)
        self.assertEqual(result["worst_scenario"]["loss"], 100.0)

    def test_extra_top_level_fields_ignored(self):
        body = json.loads(request_body())
        body["unexpected"] = {"nested": True}
        result = self.service.stress_test(json.dumps(body).encode())
        self.assertEqual([r["scenario_id"] for r in result["results"]], ["s1", "s2"])

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

    # ---- semantic failures -> 422 ----

    def test_missing_fields_and_wrong_types(self):
        with self.assertRaises(InvalidInput):
            self.service.stress_test(b"{}")
        with self.assertRaises(InvalidInput):
            self.call(positions=[])
        with self.assertRaises(InvalidInput):
            self.call(scenarios=[])
        with self.assertRaises(InvalidInput):
            self.call(positions="x")
        with self.assertRaises(InvalidInput):
            self.call(scenarios={"id": "s"})
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p", "sensitivities": {}}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "", "sensitivities": {"f": 1.0}}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"sensitivities": {"f": 1.0}}])
        with self.assertRaises(InvalidInput):
            self.call(scenarios=[{"id": "s", "factor_shocks": {}}])
        with self.assertRaises(InvalidInput):
            self.call(scenarios=[{"id": "", "factor_shocks": {"f": 1.0}}])
        with self.assertRaises(InvalidInput):
            self.call(scenarios=[{"factor_shocks": {"f": 1.0}}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p", "sensitivities": {"f": "x"}}])
        with self.assertRaises(InvalidInput):
            self.call(scenarios=[{"id": "s", "factor_shocks": {"f": "x"}}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p", "sensitivities": {"f": None}}])

    def test_booleans_rejected(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p", "sensitivities": {"f": True}}])
        with self.assertRaises(InvalidInput):
            self.call(scenarios=[{"id": "s", "factor_shocks": {"f": False}}])

    def test_empty_currency_rejected(self):
        with self.assertRaises(InvalidInput):
            self.call(currency="")
        with self.assertRaises(InvalidInput):
            self.call(currency=123)

    def test_nan_and_infinity_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token, where="sensitivity"):
                body = request_body(
                    positions=[{"id": "p", "sensitivities": {"f": 1.0}}],
                    scenarios=[{"id": "s", "factor_shocks": {"f": 0.0}}],
                )
                body = body.replace(b"1.0", token.encode(), 1)
                with self.assertRaises(InvalidInput) as ctx:
                    self.service.stress_test(body)
                self.assertEqual(ctx.exception.code, "invalid_input")
            with self.subTest(token=token, where="shock"):
                body = request_body(
                    positions=[{"id": "p", "sensitivities": {"f": 0.0}}],
                    scenarios=[{"id": "s", "factor_shocks": {"f": 1.0}}],
                )
                body = body.replace(b"1.0", token.encode(), 1)
                with self.assertRaises(InvalidInput) as ctx:
                    self.service.stress_test(body)
                self.assertEqual(ctx.exception.code, "invalid_input")

    def test_huge_json_integer_rejected(self):
        body = {
            "positions": [{"id": "p", "sensitivities": {"f": 10**400}}],
            "scenarios": [{"id": "s", "factor_shocks": {"f": 1.0}}],
        }
        with self.assertRaises(InvalidInput) as ctx:
            self.service.stress_test(json.dumps(body).encode())
        self.assertEqual(ctx.exception.code, "invalid_input")

    def test_computation_overflow_rejected(self):
        body = {
            "positions": [{"id": "p", "sensitivities": {"f": 1e308}}],
            "scenarios": [{"id": "s", "factor_shocks": {"f": 1e308}}],
        }
        with self.assertRaises(InvalidInput) as ctx:
            self.service.stress_test(json.dumps(body).encode())
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
                {"id": "s", "factor_shocks": {"f": 2.0}},
            ])
        self.assertEqual(ctx.exception.code, "duplicate_scenario")

    # ---- size limits -> 413 ----

    def test_too_many_scenarios(self):
        scenarios = [{"id": f"s{i}", "factor_shocks": {"f": 0.0}} for i in range(1001)]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(scenarios=scenarios)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

    def test_exactly_1000_scenarios_allowed(self):
        scenarios = [{"id": f"s{i:04d}", "factor_shocks": {"f": 0.0}}
                     for i in range(1000)]
        result = self.call(scenarios=scenarios)
        self.assertEqual(len(result["results"]), 1000)

    def test_too_many_scenario_position_pairs(self):
        # 101 scenarios * 1000 positions = 101000 pairs.
        positions = [{"id": f"p{i:04d}", "sensitivities": {"f": 0.0}}
                     for i in range(1000)]
        scenarios = [{"id": f"s{i:03d}", "factor_shocks": {"f": 0.0}}
                     for i in range(101)]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(positions=positions, scenarios=scenarios)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

    def test_exactly_100000_pairs_allowed(self):
        positions = [{"id": f"p{i:04d}", "sensitivities": {"f": 0.0}}
                     for i in range(1000)]
        scenarios = [{"id": f"s{i:03d}", "factor_shocks": {"f": 0.0}}
                     for i in range(100)]
        result = self.call(positions=positions, scenarios=scenarios)
        self.assertEqual(len(result["results"]), 100)

    def test_no_partial_results_on_failure(self):
        # The overflow happens during computation; nothing should be returned.
        body = {
            "positions": [{"id": "p", "sensitivities": {"f": 1e308}}],
            "scenarios": [
                {"id": "ok", "factor_shocks": {"f": 0.0}},
                {"id": "boom", "factor_shocks": {"f": 1e308}},
            ],
        }
        with self.assertRaises(InvalidInput):
            self.service.stress_test(json.dumps(body).encode())


if __name__ == "__main__":
    unittest.main()
