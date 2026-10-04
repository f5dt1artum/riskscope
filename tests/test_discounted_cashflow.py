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
            {"day": 365, "zero_rate": 0.02},
            {"day": 730, "zero_rate": 0.03},
        ],
        "positions": [
            {"id": "p1", "cashflows": [{"day": 365, "amount": 100.0}]},
            {
                "id": "p2",
                "cashflows": [
                    {"day": 730, "amount": -50.0},
                    {"day": 365, "amount": 25.0},
                ],
            },
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


def pv_of(amount, day, rate):
    return amount * math.exp(-rate * (day / 365.0))


def sens_of(amount, day, rate, weight=1.0):
    t = day / 365.0
    return amount * math.exp(-rate * t) * t * 0.0001 * weight


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
                {"day": 365, "zero_rate": 0.02},
                {"day": 730, "zero_rate": 0.03},
            ],
        )

        positions = result["positions"]
        self.assertEqual([p["id"] for p in positions], ["p1", "p2"])

        p1 = positions[0]
        p1_pv = pv_of(100.0, 365, 0.02)
        p1_sens = sens_of(100.0, 365, 0.02)
        self.assertEqual(p1["present_value"], p1_pv)
        self.assertEqual(
            p1["key_rate_dv01"],
            [
                {"day": 365, "key_rate_dv01": p1_sens},
                {"day": 730, "key_rate_dv01": 0.0},
            ],
        )
        self.assertEqual(p1["dv01"], p1_sens + 0.0)

        p2 = positions[1]
        p2_pv = pv_of(-50.0, 730, 0.03) + pv_of(25.0, 365, 0.02)
        p2_node_365 = sens_of(25.0, 365, 0.02)
        p2_node_730 = sens_of(-50.0, 730, 0.03)
        self.assertEqual(p2["present_value"], p2_pv)
        self.assertEqual(
            p2["key_rate_dv01"],
            [
                {"day": 365, "key_rate_dv01": p2_node_365},
                {"day": 730, "key_rate_dv01": p2_node_730},
            ],
        )
        self.assertEqual(p2["dv01"], p2_node_365 + p2_node_730)

    def test_portfolio_totals(self):
        result = self.call()
        positions = result["positions"]
        totals = result["portfolio_totals"]

        self.assertEqual(
            totals["present_value"],
            positions[0]["present_value"] + positions[1]["present_value"],
        )
        for index, day in enumerate([365, 730]):
            expected = (
                positions[0]["key_rate_dv01"][index]["key_rate_dv01"]
                + positions[1]["key_rate_dv01"][index]["key_rate_dv01"]
            )
            self.assertEqual(totals["key_rate_dv01"][index]["day"], day)
            self.assertEqual(totals["key_rate_dv01"][index]["key_rate_dv01"], expected)
        self.assertEqual(
            totals["dv01"],
            sum(entry["key_rate_dv01"] for entry in totals["key_rate_dv01"]),
        )

    def test_layer_dv01_equals_node_sensitivity_sum(self):
        result = self.call(
            positions=[
                {
                    "id": "p1",
                    "cashflows": [
                        {"day": 365, "amount": 100.0},
                        {"day": 548, "amount": 200.0},
                        {"day": 730, "amount": -40.0},
                    ],
                }
            ]
        )
        for level in [result["positions"][0], result["portfolio_totals"]]:
            self.assertEqual(
                level["dv01"],
                sum(entry["key_rate_dv01"] for entry in level["key_rate_dv01"]),
            )

    def test_interpolation_between_nodes(self):
        result = self.call(
            curve_points=[
                {"day": 365, "zero_rate": 0.02},
                {"day": 730, "zero_rate": 0.04},
            ],
            positions=[
                {"id": "p1", "cashflows": [{"day": 548, "amount": 1000.0}]}
            ],
        )
        span = 730 - 365
        weight_left = (730 - 548) / span
        weight_right = (548 - 365) / span
        rate = 0.02 * weight_left + 0.04 * weight_right
        t = 548 / 365.0
        pv = 1000.0 * math.exp(-rate * t)
        base = pv * t * 0.0001

        position = result["positions"][0]
        self.assertEqual(position["present_value"], pv)
        self.assertEqual(
            position["key_rate_dv01"],
            [
                {"day": 365, "key_rate_dv01": base * weight_left},
                {"day": 730, "key_rate_dv01": base * weight_right},
            ],
        )
        self.assertEqual(position["dv01"], base * weight_left + base * weight_right)

    def test_unsorted_curve_is_sorted_in_response(self):
        result = self.call(
            curve_points=[
                {"day": 730, "zero_rate": 0.03},
                {"day": 365, "zero_rate": 0.02},
            ]
        )
        self.assertEqual(
            [point["day"] for point in result["curve_points"]], [365, 730]
        )
        self.assertEqual(
            [entry["day"] for entry in result["positions"][0]["key_rate_dv01"]],
            [365, 730],
        )

    def test_negative_rates_and_amounts_are_kept(self):
        result = self.call(
            curve_points=[
                {"day": 365, "zero_rate": -0.01},
                {"day": 730, "zero_rate": 0.05},
            ],
            positions=[
                {"id": "p1", "cashflows": [{"day": 365, "amount": -250.0}]}
            ],
        )
        position = result["positions"][0]
        # A negative zero rate discounts upwards; nothing is floored.
        self.assertEqual(position["present_value"], pv_of(-250.0, 365, -0.01))
        self.assertEqual(position["dv01"], sens_of(-250.0, 365, -0.01) + 0.0)
        self.assertLess(position["present_value"], 0.0)
        self.assertLess(position["dv01"], 0.0)

    def test_currency_echo_and_default(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_extra_fields_are_ignored(self):
        result = self.call(
            unexpected="top-level",
            curve_points=[
                {"day": 365, "zero_rate": 0.02, "source": "desk"},
                {"day": 730, "zero_rate": 0.03, "source": "desk"},
            ],
            positions=[
                {
                    "id": "p1",
                    "cashflows": [{"day": 365, "amount": 100.0, "note": "x"}],
                    "book": "ir",
                }
            ],
        )
        self.assertEqual(
            result["curve_points"],
            [
                {"day": 365, "zero_rate": 0.02},
                {"day": 730, "zero_rate": 0.03},
            ],
        )
        self.assertEqual(
            result["positions"][0]["present_value"], pv_of(100.0, 365, 0.02)
        )

    def test_integer_numbers_are_accepted(self):
        result = self.call(
            curve_points=[
                {"day": 365, "zero_rate": 0},
                {"day": 730, "zero_rate": 1},
            ],
            positions=[{"id": "p1", "cashflows": [{"day": 365, "amount": 100}]}],
        )
        position = result["positions"][0]
        self.assertEqual(position["present_value"], 100.0)
        self.assertEqual(position["dv01"], 100.0 * 1.0 * 0.0001 + 0.0)

    def test_many_node_curve(self):
        curve = [{"day": day, "zero_rate": 0.01} for day in (30, 90, 365, 730, 3650)]
        result = self.call(
            curve_points=curve,
            positions=[{"id": "p1", "cashflows": [{"day": 90, "amount": 10.0}]}],
        )
        position = result["positions"][0]
        self.assertEqual(
            [entry["day"] for entry in position["key_rate_dv01"]],
            [30, 90, 365, 730, 3650],
        )
        self.assertEqual(
            [entry["key_rate_dv01"] for entry in position["key_rate_dv01"]],
            [0.0, sens_of(10.0, 90, 0.01), 0.0, 0.0, 0.0],
        )

    # ---- 400 invalid_request ----

    def test_unparseable_json(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.discounted_cashflow(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_top_level_must_be_object(self):
        for raw in (b"[1]", b"1", b'"x"', b"null"):
            with self.assertRaises(InvalidRequest):
                self.service.discounted_cashflow(raw)

    # ---- 422 invalid_input ----

    def test_nan_and_infinity_literals(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            raw = (
                '{"curve_points": [{"day": 365, "zero_rate": TOKEN},'
                ' {"day": 730, "zero_rate": 0.03}],'
                ' "positions": [{"id": "p1",'
                ' "cashflows": [{"day": 365, "amount": 1.0}]}]}'
            ).replace("TOKEN", token).encode("utf-8")
            with self.assertRaises(InvalidInput, msg=token):
                self.service.discounted_cashflow(raw)

    def test_currency_validation(self):
        for currency in ("", 1, None, True):
            with self.assertRaises(InvalidInput, msg=repr(currency)):
                self.call(currency=currency)

    def test_curve_structure_validation(self):
        for curve_points in (
            None,
            "x",
            [],
            [{"day": 365, "zero_rate": 0.02}],
            [1, 2],
            [{"day": 365}],
            [{"zero_rate": 0.02}],
            [{"day": 365, "zero_rate": 0.02}, {"day": 730}],
        ):
            with self.assertRaises(InvalidInput, msg=repr(curve_points)):
                self.call(curve_points=curve_points)

    def test_curve_day_validation(self):
        for day in (0, -1, 1.5, 365.0, True, "365", None, 10**400):
            curve_points = [
                {"day": day, "zero_rate": 0.02},
                {"day": 730, "zero_rate": 0.03},
            ]
            with self.assertRaises(InvalidInput, msg=repr(day)):
                self.call(curve_points=curve_points)

    def test_zero_rate_validation(self):
        for zero_rate in (True, "0.02", None, [0.02], 10**400):
            curve_points = [
                {"day": 365, "zero_rate": zero_rate},
                {"day": 730, "zero_rate": 0.03},
            ]
            with self.assertRaises(InvalidInput, msg=repr(zero_rate)):
                self.call(curve_points=curve_points)

    def test_positions_structure_validation(self):
        for positions in (
            None,
            "x",
            [],
            [1],
            [{"id": "p1"}],
            [{"cashflows": [{"day": 365, "amount": 1.0}]}],
            [{"id": "", "cashflows": [{"day": 365, "amount": 1.0}]}],
            [{"id": 1, "cashflows": [{"day": 365, "amount": 1.0}]}],
            [{"id": "p1", "cashflows": []}],
            [{"id": "p1", "cashflows": "x"}],
            [{"id": "p1", "cashflows": [1]}],
            [{"id": "p1", "cashflows": [{"day": 365}]}],
            [{"id": "p1", "cashflows": [{"amount": 1.0}]}],
        ):
            with self.assertRaises(InvalidInput, msg=repr(positions)):
                self.call(positions=positions)

    def test_cashflow_number_validation(self):
        for cashflow in (
            {"day": 0, "amount": 1.0},
            {"day": -3, "amount": 1.0},
            {"day": 365.0, "amount": 1.0},
            {"day": True, "amount": 1.0},
            {"day": 10**400, "amount": 1.0},
            {"day": 365, "amount": True},
            {"day": 365, "amount": "1"},
            {"day": 365, "amount": None},
            {"day": 365, "amount": 10**400},
        ):
            positions = [{"id": "p1", "cashflows": [cashflow]}]
            with self.assertRaises(InvalidInput, msg=repr(cashflow)):
                self.call(positions=positions)

    def test_non_finite_computation_fails(self):
        # A huge negative zero rate makes exp(-rate * t) overflow.
        with self.assertRaises(InvalidInput):
            self.call(
                curve_points=[
                    {"day": 365, "zero_rate": -1e6},
                    {"day": 730, "zero_rate": 0.03},
                ]
            )

    # ---- 422 duplicate_* / curve_out_of_range ----

    def check_invalid_input_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)
        self.assertEqual(ctx.exception.status, 422)

    def test_duplicate_curve_point(self):
        self.check_invalid_input_code(
            "duplicate_curve_point",
            curve_points=[
                {"day": 365, "zero_rate": 0.02},
                {"day": 365, "zero_rate": 0.05},
            ],
        )

    def test_duplicate_position(self):
        self.check_invalid_input_code(
            "duplicate_position",
            positions=[
                {"id": "p1", "cashflows": [{"day": 365, "amount": 1.0}]},
                {"id": "p1", "cashflows": [{"day": 730, "amount": 2.0}]},
            ],
        )

    def test_curve_out_of_range(self):
        for day in (1, 364, 731, 10000):
            self.check_invalid_input_code(
                "curve_out_of_range",
                positions=[{"id": "p1", "cashflows": [{"day": day, "amount": 1.0}]}],
            )

    def test_curve_endpoint_days_are_in_range(self):
        result = self.call(
            positions=[
                {
                    "id": "p1",
                    "cashflows": [
                        {"day": 365, "amount": 1.0},
                        {"day": 730, "amount": 1.0},
                    ],
                }
            ]
        )
        self.assertEqual(
            result["positions"][0]["present_value"],
            pv_of(1.0, 365, 0.02) + pv_of(1.0, 730, 0.03),
        )

    # ---- 413 request_too_large ----

    def check_too_large(self, **overrides):
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

    def test_curve_point_limit(self):
        curve = [{"day": i + 1, "zero_rate": 0.01} for i in range(101)]
        self.check_too_large(curve_points=curve)

    def test_curve_point_boundary_is_allowed(self):
        curve = [{"day": i + 1, "zero_rate": 0.01} for i in range(100)]
        result = self.call(
            curve_points=curve,
            positions=[{"id": "p1", "cashflows": [{"day": 50, "amount": 1.0}]}],
        )
        self.assertEqual(len(result["curve_points"]), 100)

    def test_position_limit(self):
        positions = [
            {"id": f"p{i}", "cashflows": [{"day": 365, "amount": 1.0}]}
            for i in range(10001)
        ]
        self.check_too_large(positions=positions)

    def test_cashflow_limit(self):
        cashflows = [{"day": 365, "amount": 1.0} for _ in range(100001)]
        self.check_too_large(positions=[{"id": "p1", "cashflows": cashflows}])

    def test_cashflow_boundary_is_allowed(self):
        curve = [
            {"day": 365, "zero_rate": 0.0},
            {"day": 730, "zero_rate": 0.0},
        ]
        cashflows = [{"day": 365, "amount": 1.0} for _ in range(100000)]
        result = self.call(
            curve_points=curve,
            positions=[{"id": "p1", "cashflows": cashflows}],
        )
        # A zero rate discounts to exactly 1, so the sequential sum is exact.
        self.assertEqual(result["positions"][0]["present_value"], 100000.0)


if __name__ == "__main__":
    unittest.main()
