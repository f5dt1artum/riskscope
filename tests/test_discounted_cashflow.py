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
        "curve_points": [
            {"day": 365, "zero_rate": 0.05},
            {"day": 730, "zero_rate": 0.04},
        ],
        "positions": [
            {
                "id": "p-1",
                "cashflows": [
                    {"day": 365, "amount": 1000.0},
                    {"day": 730, "amount": -500.0},
                ],
            },
            {
                "id": "p-2",
                "cashflows": [
                    {"day": 365, "amount": 250.0},
                ],
            },
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class DiscountedCashflowTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.discounted_cashflow(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(
            result["curve_points"],
            [
                {"day": 365, "zero_rate": 0.05},
                {"day": 730, "zero_rate": 0.04},
            ],
        )
        self.assertEqual([p["id"] for p in result["positions"]], ["p-1", "p-2"])

        # Cashflows land exactly on the nodes, so each takes the node rate.
        pv1 = 1000.0 * math.exp(-0.05 * 1.0)
        pv2 = -500.0 * math.exp(-0.04 * 2.0)
        pv3 = 250.0 * math.exp(-0.05 * 1.0)
        p1 = result["positions"][0]
        self.assertEqual(p1["present_value"], pv1 + pv2)
        self.assertEqual(
            p1["key_rate_dv01"],
            [
                {"day": 365, "dv01": pv1 * 1.0 * 0.0001},
                {"day": 730, "dv01": pv2 * 2.0 * 0.0001},
            ],
        )
        self.assertEqual(p1["dv01"], pv1 * 0.0001 + pv2 * 2.0 * 0.0001)

        p2 = result["positions"][1]
        self.assertEqual(p2["present_value"], pv3)
        self.assertEqual(
            p2["key_rate_dv01"],
            [
                {"day": 365, "dv01": pv3 * 0.0001},
                {"day": 730, "dv01": 0.0},
            ],
        )
        self.assertEqual(p2["dv01"], pv3 * 0.0001)

        totals = result["portfolio_totals"]
        self.assertEqual(totals["present_value"], pv1 + pv2 + pv3)
        self.assertEqual(
            totals["key_rate_dv01"],
            [
                {"day": 365, "dv01": pv1 * 0.0001 + pv3 * 0.0001},
                {"day": 730, "dv01": pv2 * 2.0 * 0.0001},
            ],
        )
        self.assertEqual(
            totals["dv01"], (pv1 * 0.0001 + pv3 * 0.0001) + pv2 * 2.0 * 0.0001
        )

    def test_unsorted_curve_is_sorted_and_interpolated(self):
        result = self.call(
            curve_points=[
                {"day": 720, "zero_rate": 0.04},
                {"day": 360, "zero_rate": 0.02},
            ],
            positions=[
                {"id": "p-1", "cashflows": [{"day": 540, "amount": 1000.0}]}
            ],
        )
        self.assertEqual(
            result["curve_points"],
            [
                {"day": 360, "zero_rate": 0.02},
                {"day": 720, "zero_rate": 0.04},
            ],
        )
        # Midpoint between the nodes: equal weights, average rate.
        t = 540 / 365
        pv = 1000.0 * math.exp(-0.03 * t)
        position = result["positions"][0]
        self.assertEqual(position["present_value"], pv)
        self.assertEqual(
            position["key_rate_dv01"],
            [
                {"day": 360, "dv01": pv * t * 0.0001 * 0.5},
                {"day": 720, "dv01": pv * t * 0.0001 * 0.5},
            ],
        )
        self.assertEqual(position["dv01"], pv * t * 0.0001)

    def test_interpolation_weights_follow_the_day(self):
        result = self.call(
            curve_points=[
                {"day": 100, "zero_rate": 0.01},
                {"day": 200, "zero_rate": 0.03},
            ],
            positions=[
                {"id": "p-1", "cashflows": [{"day": 125, "amount": 100.0}]}
            ],
        )
        t = 125 / 365
        zero_rate = 0.75 * 0.01 + 0.25 * 0.03
        pv = 100.0 * math.exp(-zero_rate * t)
        position = result["positions"][0]
        self.assertEqual(position["present_value"], pv)
        self.assertEqual(
            position["key_rate_dv01"],
            [
                {"day": 100, "dv01": pv * t * 0.0001 * 0.75},
                {"day": 200, "dv01": pv * t * 0.0001 * 0.25},
            ],
        )

    def test_negative_zero_rate_and_negative_amounts(self):
        result = self.call(
            curve_points=[
                {"day": 365, "zero_rate": -0.01},
                {"day": 730, "zero_rate": 0.02},
            ],
            positions=[
                {"id": "p-1", "cashflows": [{"day": 365, "amount": -100.0}]}
            ],
        )
        pv = -100.0 * math.exp(0.01 * 1.0)
        position = result["positions"][0]
        self.assertEqual(position["present_value"], pv)
        self.assertEqual(position["dv01"], pv * 0.0001)

    def test_integer_numbers_are_accepted(self):
        result = self.call(
            curve_points=[
                {"day": 365, "zero_rate": 0},
                {"day": 730, "zero_rate": 0},
            ],
            positions=[{"id": "p-1", "cashflows": [{"day": 365, "amount": 100}]}],
        )
        self.assertEqual(result["positions"][0]["present_value"], 100.0)
        self.assertEqual(result["positions"][0]["dv01"], 100.0 * 0.0001)

    def test_currency_defaults_and_echoes(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_extra_fields_are_ignored(self):
        result = self.call(
            as_of="2024-01-01",
            curve_points=[
                {"day": 365, "zero_rate": 0.05, "source": "desk"},
                {"day": 730, "zero_rate": 0.04},
            ],
            positions=[
                {
                    "id": "p-1",
                    "book": "book-a",
                    "cashflows": [{"day": 365, "amount": 1000.0, "label": "x"}],
                }
            ],
        )
        self.assertEqual(
            result["positions"][0]["present_value"], 1000.0 * math.exp(-0.05)
        )

    def test_layer_totals_match_details(self):
        result = self.call(
            positions=[
                {
                    "id": "p-1",
                    "cashflows": [
                        {"day": 400, "amount": 100.0},
                        {"day": 700, "amount": -40.0},
                    ],
                },
                {"id": "p-2", "cashflows": [{"day": 365, "amount": 10.0}]},
                {"id": "p-3", "cashflows": [{"day": 730, "amount": 5.0}]},
            ]
        )
        positions = result["positions"]
        totals = result["portfolio_totals"]
        self.assertEqual(
            totals["present_value"], sum(p["present_value"] for p in positions)
        )
        for index in range(2):
            self.assertEqual(
                totals["key_rate_dv01"][index]["dv01"],
                sum(p["key_rate_dv01"][index]["dv01"] for p in positions),
            )
        for position in positions:
            self.assertEqual(
                position["dv01"],
                sum(entry["dv01"] for entry in position["key_rate_dv01"]),
            )
        self.assertEqual(
            totals["dv01"], sum(entry["dv01"] for entry in totals["key_rate_dv01"])
        )

    def test_boundary_sizes_are_processed(self):
        curve = [{"day": i + 1, "zero_rate": 0.01} for i in range(100)]
        result = self.call(
            curve_points=curve,
            positions=[{"id": "p-1", "cashflows": [{"day": 1, "amount": 1.0}]}],
        )
        self.assertEqual(len(result["curve_points"]), 100)
        self.assertEqual(len(result["portfolio_totals"]["key_rate_dv01"]), 100)

    # ---- error semantics ----

    def test_invalid_json_and_non_object_raise_invalid_request(self):
        with self.assertRaises(InvalidRequest):
            self.service.discounted_cashflow(b"{not json")
        with self.assertRaises(InvalidRequest):
            self.service.discounted_cashflow(b"[1, 2]")
        with self.assertRaises(InvalidRequest):
            self.service.discounted_cashflow(b"null")

    def test_currency_must_be_nonempty_string(self):
        with self.assertRaises(InvalidInput):
            self.call(currency="")
        with self.assertRaises(InvalidInput):
            self.call(currency=12)

    def test_curve_requires_two_points(self):
        with self.assertRaises(InvalidInput):
            self.call(curve_points=[{"day": 365, "zero_rate": 0.05}])
        with self.assertRaises(InvalidInput):
            self.call(curve_points=[])
        with self.assertRaises(InvalidInput):
            self.call(curve_points="flat")

    def test_curve_point_validation(self):
        with self.assertRaises(InvalidInput):
            self.call(curve_points=[{"day": 0, "zero_rate": 0.01},
                                    {"day": 365, "zero_rate": 0.01}])
        with self.assertRaises(InvalidInput):
            self.call(curve_points=[{"day": -3, "zero_rate": 0.01},
                                    {"day": 365, "zero_rate": 0.01}])
        with self.assertRaises(InvalidInput):
            self.call(curve_points=[{"day": 1.5, "zero_rate": 0.01},
                                    {"day": 365, "zero_rate": 0.01}])
        with self.assertRaises(InvalidInput):
            self.call(curve_points=[{"day": 100, "zero_rate": "0.01"},
                                    {"day": 365, "zero_rate": 0.01}])
        with self.assertRaises(InvalidInput):
            self.call(curve_points=[{"day": 100},
                                    {"day": 365, "zero_rate": 0.01}])
        with self.assertRaises(InvalidInput):
            self.call(curve_points=[{"day": 100, "zero_rate": True},
                                    {"day": 365, "zero_rate": 0.01}])
        with self.assertRaises(InvalidInput):
            self.call(curve_points=[{"day": 100, "zero_rate": 10**400},
                                    {"day": 365, "zero_rate": 0.01}])

    def test_duplicate_curve_point(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(curve_points=[{"day": 365, "zero_rate": 0.05},
                                    {"day": 365, "zero_rate": 0.04}])
        self.assertEqual(ctx.exception.code, "duplicate_curve_point")

    def test_positions_validation(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=[])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "", "cashflows": [{"day": 365, "amount": 1.0}]}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p-1", "cashflows": []}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p-1"}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p-1", "cashflows": [{"day": 365, "amount": 1.0}]},
                                 {"id": "p-1", "cashflows": [{"day": 365, "amount": 2.0}]}])

    def test_duplicate_position_code(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(positions=[{"id": "p-1", "cashflows": [{"day": 365, "amount": 1.0}]},
                                 {"id": "p-1", "cashflows": [{"day": 365, "amount": 2.0}]}])
        self.assertEqual(ctx.exception.code, "duplicate_position")

    def test_cashflow_validation(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p-1", "cashflows": [{"day": 0, "amount": 1.0}]}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p-1", "cashflows": [{"day": 365, "amount": "1"}]}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p-1", "cashflows": [{"day": 365, "amount": False}]}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p-1", "cashflows": [{"day": 365, "amount": 10**400}]}])
        with self.assertRaises(InvalidInput):
            self.call(positions=[{"id": "p-1", "cashflows": [{"day": 365}]}])

    def test_cashflow_outside_curve_range(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(positions=[{"id": "p-1", "cashflows": [{"day": 364, "amount": 1.0}]}])
        self.assertEqual(ctx.exception.code, "curve_out_of_range")
        with self.assertRaises(InvalidInput) as ctx:
            self.call(positions=[{"id": "p-1", "cashflows": [{"day": 731, "amount": 1.0}]}])
        self.assertEqual(ctx.exception.code, "curve_out_of_range")

    def test_non_finite_computation_fails(self):
        with self.assertRaises(InvalidInput):
            self.call(
                curve_points=[{"day": 365, "zero_rate": -1e308},
                              {"day": 730, "zero_rate": 0.0}],
                positions=[{"id": "p-1", "cashflows": [{"day": 365, "amount": 1.0}]}],
            )

    def test_request_too_large(self):
        with self.assertRaises(RequestTooLarge):
            self.call(curve_points=[
                {"day": i + 1, "zero_rate": 0.01} for i in range(101)
            ])
        with self.assertRaises(RequestTooLarge):
            self.call(positions=[
                {"id": f"p-{i}", "cashflows": [{"day": 365, "amount": 1.0}]}
                for i in range(10001)
            ])
        with self.assertRaises(RequestTooLarge):
            self.call(positions=[
                {"id": "p-1",
                 "cashflows": [{"day": 365, "amount": 1.0}] * 100001}
            ])

    def test_nan_and_infinity_tokens_rejected(self):
        body = (
            b'{"curve_points": [{"day": 365, "zero_rate": NaN},'
            b' {"day": 730, "zero_rate": 0.04}],'
            b' "positions": [{"id": "p-1", "cashflows": [{"day": 365, "amount": 1.0}]}]}'
        )
        with self.assertRaises(InvalidInput):
            self.service.discounted_cashflow(body)
        body = (
            b'{"curve_points": [{"day": 365, "zero_rate": 0.05},'
            b' {"day": 730, "zero_rate": 0.04}],'
            b' "positions": [{"id": "p-1", "cashflows": [{"day": 365, "amount": Infinity}]}]}'
        )
        with self.assertRaises(InvalidInput):
            self.service.discounted_cashflow(body)


if __name__ == "__main__":
    unittest.main()
