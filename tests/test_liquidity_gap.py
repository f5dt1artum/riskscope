import json
import unittest

from riskscope.service import (
    InvalidInput,
    InvalidRequest,
    Service,
)


def request_body(**overrides):
    body = {
        "buckets": [7, 30, 90],
        "cashflows": [
            {"id": "cf-1", "day": 5, "amount": 100.0},
            {"id": "cf-2", "day": 10, "amount": -250.0},
            {"id": "cf-3", "day": 90, "amount": 400.0},
        ],
        "liquid_assets": [
            {"id": "la-1", "market_value": 200.0, "haircut": 0.1, "available_day": 0},
            {"id": "la-2", "market_value": 100.0, "haircut": 0.5, "available_day": 31},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class LiquidityGapTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.liquidity_gap(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        self.assertEqual([bucket["day"] for bucket in result["buckets"]], [7, 30, 90])

        # cf-1 (day 5) lands in bucket 7; la-1 discounted 200*0.9 = 180
        # available from bucket 7 (available_day 0).
        first = result["buckets"][0]
        self.assertEqual(first["net_cashflow"], 100.0)
        self.assertEqual(first["cumulative_net_cashflow"], 100.0)
        self.assertEqual(first["available_liquidity"], 180.0)
        self.assertEqual(first["surplus"], 280.0)
        self.assertEqual(first["required_funding"], 0.0)

        # cf-2 (day 10) lands in bucket 30; la-2 (available_day 31) not yet.
        second = result["buckets"][1]
        self.assertEqual(second["net_cashflow"], -250.0)
        self.assertEqual(second["cumulative_net_cashflow"], -150.0)
        self.assertEqual(second["available_liquidity"], 180.0)
        self.assertEqual(second["surplus"], 30.0)
        self.assertEqual(second["required_funding"], 0.0)

        # cf-3 (day 90) lands in bucket 90; la-2 discounted 100*0.5 = 50.
        third = result["buckets"][2]
        self.assertEqual(third["net_cashflow"], 400.0)
        self.assertEqual(third["cumulative_net_cashflow"], 250.0)
        self.assertEqual(third["available_liquidity"], 230.0)
        self.assertEqual(third["surplus"], 480.0)
        self.assertEqual(third["required_funding"], 0.0)

        self.assertIsNone(result["earliest_shortfall"])

    def test_shortfall_and_required_funding(self):
        result = self.call(
            cashflows=[
                {"id": "cf-1", "day": 1, "amount": -500.0},
                {"id": "cf-2", "day": 90, "amount": 100.0},
            ],
            liquid_assets=[],
        )
        first, second, third = result["buckets"]
        self.assertEqual(first["surplus"], -500.0)
        self.assertEqual(first["required_funding"], 500.0)
        self.assertEqual(second["surplus"], -500.0)
        self.assertEqual(second["required_funding"], 500.0)
        self.assertEqual(third["surplus"], -400.0)
        self.assertEqual(third["required_funding"], 400.0)
        # Earliest negative-surplus bucket wins.
        self.assertEqual(
            result["earliest_shortfall"], {"day": 7, "required_funding": 500.0}
        )

    def test_currency_echo_and_default(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_cashflow_on_bucket_boundary(self):
        # A cashflow exactly on a bucket day lands in that bucket.
        result = self.call(
            cashflows=[{"id": "cf-1", "day": 30, "amount": 5.0}],
            liquid_assets=[],
        )
        self.assertEqual(result["buckets"][0]["net_cashflow"], 0.0)
        self.assertEqual(result["buckets"][1]["net_cashflow"], 5.0)
        self.assertEqual(result["buckets"][2]["net_cashflow"], 0.0)

    def test_asset_available_on_bucket_boundary(self):
        result = self.call(
            cashflows=[{"id": "cf-1", "day": 1, "amount": 1.0}],
            liquid_assets=[
                {"id": "la-1", "market_value": 10.0, "haircut": 0.0, "available_day": 30}
            ],
        )
        self.assertEqual(result["buckets"][0]["available_liquidity"], 0.0)
        self.assertEqual(result["buckets"][1]["available_liquidity"], 10.0)
        self.assertEqual(result["buckets"][2]["available_liquidity"], 10.0)

    def test_asset_beyond_final_bucket_never_contributes(self):
        result = self.call(
            cashflows=[{"id": "cf-1", "day": 1, "amount": 1.0}],
            liquid_assets=[
                {"id": "la-1", "market_value": 10.0, "haircut": 0.0, "available_day": 91}
            ],
        )
        for bucket in result["buckets"]:
            self.assertEqual(bucket["available_liquidity"], 0.0)

    def test_liquid_assets_default_empty(self):
        body = json.loads(request_body())
        del body["liquid_assets"]
        result = self.service.liquidity_gap(json.dumps(body))
        for bucket in result["buckets"]:
            self.assertEqual(bucket["available_liquidity"], 0.0)

    def test_zero_values_preserved(self):
        result = self.call(
            cashflows=[{"id": "cf-1", "day": 1, "amount": 0.0}],
            liquid_assets=[
                {"id": "la-1", "market_value": 0.0, "haircut": 1.0, "available_day": 0}
            ],
        )
        for bucket in result["buckets"]:
            self.assertEqual(bucket["net_cashflow"], 0.0)
            self.assertEqual(bucket["surplus"], 0.0)
            self.assertEqual(bucket["required_funding"], 0.0)
        self.assertIsNone(result["earliest_shortfall"])

    def test_integer_amounts_accepted(self):
        result = self.call(
            cashflows=[{"id": "cf-1", "day": 1, "amount": -3}],
            liquid_assets=[
                {"id": "la-1", "market_value": 10, "haircut": 0, "available_day": 0}
            ],
        )
        self.assertEqual(result["buckets"][0]["surplus"], 7.0)

    # ---- request-level errors ----

    def test_unparseable_json_is_invalid_request(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.liquidity_gap(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_non_object_payload_is_invalid_request(self):
        for raw in ("[1, 2]", '"text"', "42", "null"):
            with self.subTest(raw=raw):
                with self.assertRaises(InvalidRequest):
                    self.service.liquidity_gap(raw)

    # ---- field validation ----

    def test_currency_must_be_nonempty_string(self):
        for bad in ("", 1, None, True, ["USD"]):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(currency=bad)

    def test_buckets_must_be_nonempty_list(self):
        for bad in (None, [], "7", 7):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(buckets=bad)

    def test_buckets_must_be_positive_integers(self):
        for bad in (0, -1, 1.5, 7.0, "7", True, None):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(buckets=[bad])

    def test_buckets_must_be_strictly_increasing(self):
        for bad in ([30, 7, 90], [7, 7, 90], [90, 30, 7], [7, 30, 30]):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(buckets=bad)

    def test_cashflows_must_be_nonempty_list(self):
        for bad in (None, [], "x", 1):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(cashflows=bad)

    def test_cashflow_fields_validated(self):
        bad_cashflows = [
            {"day": 1, "amount": 1.0},  # missing id
            {"id": "", "day": 1, "amount": 1.0},
            {"id": 1, "day": 1, "amount": 1.0},
            {"id": "cf", "amount": 1.0},  # missing day
            {"id": "cf", "day": 0, "amount": 1.0},
            {"id": "cf", "day": -1, "amount": 1.0},
            {"id": "cf", "day": 1.5, "amount": 1.0},
            {"id": "cf", "day": True, "amount": 1.0},
            {"id": "cf", "day": 1},  # missing amount
            {"id": "cf", "day": 1, "amount": "1"},
            {"id": "cf", "day": 1, "amount": True},
            {"id": "cf", "day": 1, "amount": None},
            "not-an-object",
        ]
        for bad in bad_cashflows:
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(cashflows=[bad])

    def test_cashflow_day_beyond_final_bucket_rejected(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(cashflows=[{"id": "cf-1", "day": 91, "amount": 1.0}])
        self.assertEqual(ctx.exception.code, "invalid_input")
        # Exactly the final bucket day is fine.
        self.call(cashflows=[{"id": "cf-1", "day": 90, "amount": 1.0}])

    def test_liquid_assets_must_be_list(self):
        for bad in ("x", 1, {}):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(liquid_assets=bad)

    def test_asset_fields_validated(self):
        bad_assets = [
            {"market_value": 1.0, "haircut": 0.0, "available_day": 0},
            {"id": "", "market_value": 1.0, "haircut": 0.0, "available_day": 0},
            {"id": "la", "haircut": 0.0, "available_day": 0},
            {"id": "la", "market_value": -1.0, "haircut": 0.0, "available_day": 0},
            {"id": "la", "market_value": "1", "haircut": 0.0, "available_day": 0},
            {"id": "la", "market_value": True, "haircut": 0.0, "available_day": 0},
            {"id": "la", "market_value": 1.0, "available_day": 0},
            {"id": "la", "market_value": 1.0, "haircut": -0.1, "available_day": 0},
            {"id": "la", "market_value": 1.0, "haircut": 1.1, "available_day": 0},
            {"id": "la", "market_value": 1.0, "haircut": True, "available_day": 0},
            {"id": "la", "market_value": 1.0, "haircut": 0.0},
            {"id": "la", "market_value": 1.0, "haircut": 0.0, "available_day": -1},
            {"id": "la", "market_value": 1.0, "haircut": 0.0, "available_day": 1.5},
            {"id": "la", "market_value": 1.0, "haircut": 0.0, "available_day": False},
            "not-an-object",
        ]
        for bad in bad_assets:
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(liquid_assets=[bad])

    def test_haircut_boundaries_allowed(self):
        self.call(
            liquid_assets=[
                {"id": "la-1", "market_value": 1.0, "haircut": 0.0, "available_day": 0},
                {"id": "la-2", "market_value": 1.0, "haircut": 1.0, "available_day": 0},
            ]
        )

    def test_duplicate_cashflow_ids(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                cashflows=[
                    {"id": "cf-1", "day": 1, "amount": 1.0},
                    {"id": "cf-1", "day": 2, "amount": 2.0},
                ]
            )
        self.assertEqual(ctx.exception.code, "duplicate_cashflow")
        self.assertEqual(ctx.exception.status, 422)

    def test_duplicate_asset_ids(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                liquid_assets=[
                    {"id": "la-1", "market_value": 1.0, "haircut": 0.0, "available_day": 0},
                    {"id": "la-1", "market_value": 2.0, "haircut": 0.0, "available_day": 0},
                ]
            )
        self.assertEqual(ctx.exception.code, "duplicate_asset")
        self.assertEqual(ctx.exception.status, 422)

    def test_nan_and_infinity_rejected(self):
        with self.assertRaises(InvalidInput):
            self.service.liquidity_gap(
                b'{"buckets": [7], "cashflows": [{"id": "c", "day": 1, "amount": NaN}]}'
            )
        with self.assertRaises(InvalidInput):
            self.service.liquidity_gap(
                b'{"buckets": [7], "cashflows": [{"id": "c", "day": 1, "amount": Infinity}]}'
            )

    def test_oversized_integer_amount_rejected(self):
        with self.assertRaises(InvalidInput):
            self.call(cashflows=[{"id": "cf-1", "day": 1, "amount": 10**400}])

    def test_non_finite_computation_fails(self):
        with self.assertRaises(InvalidInput):
            self.call(
                cashflows=[
                    {"id": "cf-1", "day": 1, "amount": 1.7e308},
                    {"id": "cf-2", "day": 1, "amount": 1.7e308},
                ]
            )


if __name__ == "__main__":
    unittest.main()
