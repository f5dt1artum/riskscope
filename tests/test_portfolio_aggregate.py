import json
import unittest

from riskscope.service import (
    InvalidInput,
    InvalidRequest,
    RequestTooLarge,
    Service,
)


def request_body(**overrides):
    body = {
        "reporting_currency": "USD",
        "fx_rates": {"EUR": 1.2, "JPY": 0.01},
        "positions": [
            {
                "id": "p1",
                "book": "equity",
                "currency": "EUR",
                "market_value": 100.0,
                "sensitivities": {"eq": 10.0, "ir": -5.0},
            },
            {
                "id": "p2",
                "book": "rates",
                "currency": "USD",
                "market_value": -50.0,
                "sensitivities": {"ir": 2.0},
            },
            {
                "id": "p3",
                "book": "equity",
                "currency": "JPY",
                "market_value": 1000,
                "sensitivities": {"eq": 300},
            },
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class PortfolioAggregateTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.portfolio_aggregate(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["reporting_currency"], "USD")
        self.assertEqual(result["position_count"], 3)

        currencies = result["currencies"]
        self.assertEqual([c["currency"] for c in currencies], ["EUR", "USD", "JPY"])
        eur, usd, jpy = currencies
        self.assertEqual(eur["fx_rate"], 1.2)
        self.assertEqual(eur["position_count"], 1)
        self.assertEqual(eur["converted_market_value"], 120.0)
        self.assertEqual(eur["converted_sensitivities"], {"eq": 12.0, "ir": -6.0})
        self.assertEqual(usd["fx_rate"], 1.0)
        self.assertEqual(usd["position_count"], 1)
        self.assertEqual(usd["converted_market_value"], -50.0)
        self.assertEqual(usd["converted_sensitivities"], {"ir": 2.0})
        self.assertEqual(jpy["fx_rate"], 0.01)
        self.assertEqual(jpy["position_count"], 1)
        self.assertEqual(jpy["converted_market_value"], 10.0)
        self.assertEqual(jpy["converted_sensitivities"], {"eq": 3.0})

        books = result["books"]
        self.assertEqual([b["book"] for b in books], ["equity", "rates"])
        equity, rates = books
        self.assertEqual(equity["position_count"], 2)
        self.assertEqual(equity["converted_market_value"], 130.0)
        self.assertEqual(equity["converted_sensitivities"], {"eq": 15.0, "ir": -6.0})
        self.assertEqual(rates["position_count"], 1)
        self.assertEqual(rates["converted_market_value"], -50.0)
        self.assertEqual(rates["converted_sensitivities"], {"ir": 2.0})

        totals = result["portfolio_totals"]
        self.assertEqual(totals["converted_market_value"], 80.0)
        self.assertEqual(totals["converted_sensitivities"], {"eq": 15.0, "ir": -4.0})

    def test_first_appearance_order_preserved(self):
        result = self.call(
            positions=[
                {
                    "id": "p1",
                    "book": "b2",
                    "currency": "JPY",
                    "market_value": 1.0,
                    "sensitivities": {"zeta": 1.0, "alpha": 2.0},
                },
                {
                    "id": "p2",
                    "book": "b1",
                    "currency": "EUR",
                    "market_value": 1.0,
                    "sensitivities": {"mid": 1.0},
                },
            ]
        )
        self.assertEqual(
            [c["currency"] for c in result["currencies"]], ["JPY", "EUR"]
        )
        self.assertEqual([b["book"] for b in result["books"]], ["b2", "b1"])
        self.assertEqual(
            list(result["portfolio_totals"]["converted_sensitivities"]),
            ["zeta", "alpha", "mid"],
        )
        self.assertEqual(
            list(result["currencies"][0]["converted_sensitivities"]),
            ["zeta", "alpha"],
        )

    def test_totals_equal_sum_of_details(self):
        result = self.call()
        for key, details in (
            ("currency", result["currencies"]),
            ("book", result["books"]),
        ):
            self.assertAlmostEqual(
                sum(d["converted_market_value"] for d in details),
                result["portfolio_totals"]["converted_market_value"],
            )
            for factor, total in result["portfolio_totals"][
                "converted_sensitivities"
            ].items():
                self.assertAlmostEqual(
                    sum(
                        d["converted_sensitivities"].get(factor, 0.0)
                        for d in details
                    ),
                    total,
                    msg=f"{key} detail mismatch on factor {factor}",
                )

    def test_zero_sum_factor_retained(self):
        result = self.call(
            positions=[
                {
                    "id": "p1",
                    "book": "b1",
                    "currency": "USD",
                    "market_value": 1.0,
                    "sensitivities": {"eq": 5.0},
                },
                {
                    "id": "p2",
                    "book": "b1",
                    "currency": "USD",
                    "market_value": 1.0,
                    "sensitivities": {"eq": -5.0},
                },
            ]
        )
        self.assertEqual(
            result["portfolio_totals"]["converted_sensitivities"], {"eq": 0.0}
        )
        self.assertEqual(
            result["books"][0]["converted_sensitivities"], {"eq": 0.0}
        )

    def test_reporting_currency_only_needs_no_fx_rates(self):
        result = self.call(
            fx_rates={},
            positions=[
                {
                    "id": "p1",
                    "book": "b1",
                    "currency": "USD",
                    "market_value": 7.5,
                    "sensitivities": {"eq": 1.0},
                }
            ],
        )
        self.assertEqual(result["currencies"][0]["fx_rate"], 1.0)
        self.assertEqual(
            result["portfolio_totals"]["converted_market_value"], 7.5
        )

    def test_reporting_currency_explicit_rate_one_accepted(self):
        result = self.call(fx_rates={"USD": 1, "EUR": 1.2, "JPY": 0.01})
        self.assertEqual(result["position_count"], 3)

    def test_unreferenced_fx_rates_ignored(self):
        result = self.call(
            fx_rates={"EUR": 1.2, "JPY": 0.01, "GBP": "not-a-number", "": None}
        )
        self.assertEqual(result["position_count"], 3)

    def test_extra_fields_ignored(self):
        result = self.call(
            note="ignored",
            positions=[
                {
                    "id": "p1",
                    "book": "b1",
                    "currency": "USD",
                    "market_value": 1.0,
                    "sensitivities": {"eq": 1.0},
                    "extra": True,
                }
            ],
        )
        self.assertEqual(result["position_count"], 1)

    def test_result_is_json_serializable(self):
        json.dumps(self.call(), sort_keys=True)

    # ---- 400 invalid_request ----

    def test_unparseable_json(self):
        with self.assertRaises(InvalidRequest):
            self.service.portfolio_aggregate(b"{not json")

    def test_top_level_not_object(self):
        with self.assertRaises(InvalidRequest):
            self.service.portfolio_aggregate(b"[1, 2]")

    # ---- 422 invalid_input ----

    def test_bad_reporting_currency(self):
        for bad in ("", 1, None, True):
            with self.assertRaises(InvalidInput):
                self.call(reporting_currency=bad)

    def test_missing_fx_rates(self):
        with self.assertRaises(InvalidInput):
            self.call(fx_rates=None)

    def test_fx_rates_not_object(self):
        with self.assertRaises(InvalidInput):
            self.call(fx_rates=[["EUR", 1.2]])

    def test_reporting_currency_rate_not_one(self):
        for bad in (1.5, 0.5, 0, -1):
            with self.assertRaises(InvalidInput):
                self.call(fx_rates={"USD": bad, "EUR": 1.2, "JPY": 0.01})

    def test_reporting_currency_rate_rejects_bool_and_string(self):
        for bad in (True, "1"):
            with self.assertRaises(InvalidInput):
                self.call(fx_rates={"USD": bad, "EUR": 1.2, "JPY": 0.01})

    def test_referenced_rate_not_positive(self):
        for bad in (0.0, -1.2):
            with self.assertRaises(InvalidInput):
                self.call(fx_rates={"EUR": bad, "JPY": 0.01})

    def test_referenced_rate_rejects_bool_string_and_oversized(self):
        for bad in (True, "1.2", 10**400):
            with self.assertRaises(InvalidInput):
                self.call(fx_rates={"EUR": bad, "JPY": 0.01})

    def test_missing_positions(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=None)

    def test_empty_positions(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=[])

    def test_position_not_object(self):
        with self.assertRaises(InvalidInput):
            self.call(positions=["oops"])

    def test_position_bad_id(self):
        for bad in ("", 1, None, True):
            with self.assertRaises(InvalidInput):
                self.call(
                    positions=[
                        {
                            "id": bad,
                            "book": "b1",
                            "currency": "USD",
                            "market_value": 1.0,
                            "sensitivities": {"eq": 1.0},
                        }
                    ]
                )

    def test_position_bad_book(self):
        for bad in ("", 1, None, True):
            with self.assertRaises(InvalidInput):
                self.call(
                    positions=[
                        {
                            "id": "p1",
                            "book": bad,
                            "currency": "USD",
                            "market_value": 1.0,
                            "sensitivities": {"eq": 1.0},
                        }
                    ]
                )

    def test_position_bad_currency(self):
        for bad in ("", 1, None, True):
            with self.assertRaises(InvalidInput):
                self.call(
                    positions=[
                        {
                            "id": "p1",
                            "book": "b1",
                            "currency": bad,
                            "market_value": 1.0,
                            "sensitivities": {"eq": 1.0},
                        }
                    ]
                )

    def test_market_value_rejects_bool_string_and_oversized(self):
        for bad in (True, "1.0", 10**400, None):
            with self.assertRaises(InvalidInput):
                self.call(
                    positions=[
                        {
                            "id": "p1",
                            "book": "b1",
                            "currency": "USD",
                            "market_value": bad,
                            "sensitivities": {"eq": 1.0},
                        }
                    ]
                )

    def test_sensitivities_missing_or_empty(self):
        for bad in (None, {}, [1, 2]):
            with self.assertRaises(InvalidInput):
                self.call(
                    positions=[
                        {
                            "id": "p1",
                            "book": "b1",
                            "currency": "USD",
                            "market_value": 1.0,
                            "sensitivities": bad,
                        }
                    ]
                )

    def test_sensitivity_factor_name_nonempty(self):
        with self.assertRaises(InvalidInput):
            self.call(
                positions=[
                    {
                        "id": "p1",
                        "book": "b1",
                        "currency": "USD",
                        "market_value": 1.0,
                        "sensitivities": {"": 1.0},
                    }
                ]
            )

    def test_sensitivity_rejects_bool_string_and_oversized(self):
        for bad in (True, "1", 10**400):
            with self.assertRaises(InvalidInput):
                self.call(
                    positions=[
                        {
                            "id": "p1",
                            "book": "b1",
                            "currency": "USD",
                            "market_value": 1.0,
                            "sensitivities": {"eq": bad},
                        }
                    ]
                )

    def test_numbers_reject_nan_and_infinity(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            body = (
                b'{"reporting_currency": "USD", "fx_rates": {}, "positions": ['
                b'{"id": "p1", "book": "b1", "currency": "USD", '
                b'"market_value": ' + token.encode()
                + b', "sensitivities": {"eq": 1.0}}]}'
            )
            with self.assertRaises(InvalidInput):
                self.service.portfolio_aggregate(body)

    # ---- 422 specific codes ----

    def assert_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)

    def test_duplicate_position(self):
        self.assert_code(
            "duplicate_position",
            positions=[
                {
                    "id": "p1",
                    "book": "b1",
                    "currency": "USD",
                    "market_value": 1.0,
                    "sensitivities": {"eq": 1.0},
                },
                {
                    "id": "p1",
                    "book": "b2",
                    "currency": "EUR",
                    "market_value": 2.0,
                    "sensitivities": {"ir": 1.0},
                },
            ],
        )

    def test_missing_fx_rate(self):
        self.assert_code("missing_fx_rate", fx_rates={"JPY": 0.01})

    # ---- 413 request_too_large ----

    def test_too_many_positions(self):
        positions = [
            {
                "id": f"p{i}",
                "book": "b1",
                "currency": "USD",
                "market_value": 1.0,
                "sensitivities": {"eq": 1.0},
            }
            for i in range(10001)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(positions=positions)

    def test_too_many_fx_rates(self):
        fx_rates = {f"C{i}": 1.0 for i in range(1001)}
        with self.assertRaises(RequestTooLarge):
            self.call(fx_rates=fx_rates)

    def test_too_many_sensitivity_entries(self):
        positions = [
            {
                "id": f"p{i}",
                "book": "b1",
                "currency": "USD",
                "market_value": 1.0,
                "sensitivities": {f"f{j}": 1.0 for j in range(101)},
            }
            for i in range(10000)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(positions=positions)

    def test_boundaries_allowed(self):
        fx_rates = {f"C{i}": 1.0 for i in range(1000)}
        positions = [
            {
                "id": f"p{i}",
                "book": "b1",
                "currency": "USD",
                "market_value": 1.0,
                "sensitivities": {f"f{j}": 1.0 for j in range(100)},
            }
            for i in range(10000)
        ]
        result = self.call(fx_rates=fx_rates, positions=positions)
        self.assertEqual(result["position_count"], 10000)
        self.assertEqual(
            result["portfolio_totals"]["converted_market_value"], 10000.0
        )

    # ---- non-finite computation fails the whole request ----

    def test_non_finite_computation_fails(self):
        with self.assertRaises(InvalidInput):
            self.call(
                positions=[
                    {
                        "id": "p1",
                        "book": "b1",
                        "currency": "USD",
                        "market_value": 1e308,
                        "sensitivities": {"eq": 1.0},
                    },
                    {
                        "id": "p2",
                        "book": "b1",
                        "currency": "USD",
                        "market_value": 1e308,
                        "sensitivities": {"eq": 1.0},
                    },
                ]
            )

    def test_non_finite_conversion_fails(self):
        with self.assertRaises(InvalidInput):
            self.call(
                fx_rates={"EUR": 1e308},
                positions=[
                    {
                        "id": "p1",
                        "book": "b1",
                        "currency": "EUR",
                        "market_value": 1e308,
                        "sensitivities": {"eq": 1.0},
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
