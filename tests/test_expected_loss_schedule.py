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
        "periods": [1, 2, 3],
        "discount_factors": [1.0, 0.5, 0.25],
        "counterparties": [
            {"id": "cp-a", "cumulative_pd": [0.125, 0.25, 0.5]},
            {"id": "cp-b", "cumulative_pd": [0.0, 0.0, 0.5]},
        ],
        "facilities": [
            {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.5,
             "ead": [100.0, 200.0, 400.0]},
            {"id": "f-2", "counterparty_id": "cp-a", "lgd": 1.0,
             "ead": [8.0, 16.0, 32.0]},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class ExpectedLossScheduleTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.expected_loss_schedule(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["periods"], [1, 2, 3])

        facilities = result["facilities"]
        self.assertEqual([f["id"] for f in facilities], ["f-1", "f-2"])

        # Marginal pd of cp-a: [0.125, 0.125, 0.25].
        f1 = facilities[0]
        self.assertEqual(f1["counterparty_id"], "cp-a")
        self.assertEqual(
            [c["period"] for c in f1["contributions"]], [1, 2, 3]
        )
        self.assertEqual(
            [c["discounted_expected_loss"] for c in f1["contributions"]],
            [6.25, 6.25, 12.5],
        )
        self.assertEqual(f1["total_discounted_expected_loss"], 25.0)

        f2 = facilities[1]
        self.assertEqual(
            [c["discounted_expected_loss"] for c in f2["contributions"]],
            [1.0, 1.0, 2.0],
        )
        self.assertEqual(f2["total_discounted_expected_loss"], 4.0)

    def test_counterparty_and_portfolio_aggregation(self):
        result = self.call()
        counterparties = result["counterparties"]
        self.assertEqual([cp["id"] for cp in counterparties], ["cp-a", "cp-b"])

        cp_a = counterparties[0]
        self.assertEqual(
            [c["discounted_expected_loss"] for c in cp_a["contributions"]],
            [7.25, 7.25, 14.5],
        )
        self.assertEqual(cp_a["total_discounted_expected_loss"], 29.0)

        # cp-b has no facilities: all-zero results are kept, not omitted.
        cp_b = counterparties[1]
        self.assertEqual(
            [c["discounted_expected_loss"] for c in cp_b["contributions"]],
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(cp_b["total_discounted_expected_loss"], 0.0)

        portfolio = result["portfolio_total"]
        self.assertEqual(
            [c["period"] for c in portfolio["contributions"]], [1, 2, 3]
        )
        self.assertEqual(
            [c["discounted_expected_loss"] for c in portfolio["contributions"]],
            [7.25, 7.25, 14.5],
        )
        self.assertEqual(portfolio["total_discounted_expected_loss"], 29.0)

    def test_layer_totals_match_details(self):
        result = self.call()
        facilities = result["facilities"]
        counterparties = result["counterparties"]
        portfolio = result["portfolio_total"]

        for facility in facilities:
            self.assertEqual(
                facility["total_discounted_expected_loss"],
                sum(c["discounted_expected_loss"] for c in facility["contributions"]),
            )
        for cp_row, cp_id in zip(counterparties, ["cp-a", "cp-b"]):
            owned = [f for f in facilities if f["counterparty_id"] == cp_id]
            for index in range(3):
                self.assertEqual(
                    cp_row["contributions"][index]["discounted_expected_loss"],
                    sum(
                        f["contributions"][index]["discounted_expected_loss"]
                        for f in owned
                    ),
                )
            self.assertEqual(
                cp_row["total_discounted_expected_loss"],
                sum(f["total_discounted_expected_loss"] for f in owned),
            )
            self.assertEqual(
                cp_row["total_discounted_expected_loss"],
                sum(c["discounted_expected_loss"] for c in cp_row["contributions"]),
            )
        for index in range(3):
            self.assertEqual(
                portfolio["contributions"][index]["discounted_expected_loss"],
                sum(
                    cp["contributions"][index]["discounted_expected_loss"]
                    for cp in counterparties
                ),
            )
        self.assertEqual(
            portfolio["total_discounted_expected_loss"],
            sum(cp["total_discounted_expected_loss"] for cp in counterparties),
        )
        self.assertEqual(
            portfolio["total_discounted_expected_loss"],
            sum(c["discounted_expected_loss"] for c in portfolio["contributions"]),
        )

    def test_first_period_marginal_pd_is_the_cumulative_value(self):
        result = self.call(
            periods=[5],
            discount_factors=[1.0],
            counterparties=[{"id": "c", "cumulative_pd": [0.25]}],
            facilities=[
                {"id": "f", "counterparty_id": "c", "lgd": 0.5, "ead": [40.0]}
            ],
        )
        facility = result["facilities"][0]
        # 40.0 * 0.5 * 0.25 * 1.0
        self.assertEqual(
            facility["contributions"][0]["discounted_expected_loss"], 5.0
        )
        self.assertEqual(facility["total_discounted_expected_loss"], 5.0)

    def test_zero_marginal_pd_periods_keep_zero_contributions(self):
        result = self.call(
            periods=[1, 2],
            discount_factors=[1.0, 1.0],
            counterparties=[{"id": "c", "cumulative_pd": [0.5, 0.5]}],
            facilities=[
                {"id": "f", "counterparty_id": "c", "lgd": 1.0, "ead": [10.0, 10.0]}
            ],
        )
        facility = result["facilities"][0]
        # Marginal pd is [0.5, 0.0]: the flat cumulative curve defaults
        # only in the first period.
        self.assertEqual(
            [c["discounted_expected_loss"] for c in facility["contributions"]],
            [5.0, 0.0],
        )

    def test_currency_echo_and_default(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_extra_fields_are_ignored(self):
        result = self.call(
            unexpected="top-level",
            counterparties=[
                {"id": "c", "cumulative_pd": [0.5, 0.5, 0.5], "rating": "AAA"}
            ],
            facilities=[
                {"id": "f", "counterparty_id": "c", "lgd": 1.0,
                 "ead": [2.0, 2.0, 2.0], "note": 1}
            ],
        )
        facility = result["facilities"][0]
        self.assertEqual(
            [c["discounted_expected_loss"] for c in facility["contributions"]],
            [1.0, 0.0, 0.0],
        )

    def test_integer_numbers_are_accepted(self):
        result = self.call(
            periods=[1, 2],
            discount_factors=[1, 1],
            counterparties=[{"id": "c", "cumulative_pd": [1, 1]}],
            facilities=[
                {"id": "f", "counterparty_id": "c", "lgd": 1, "ead": [10, 20]}
            ],
        )
        facility = result["facilities"][0]
        self.assertEqual(
            [c["discounted_expected_loss"] for c in facility["contributions"]],
            [10.0, 0.0],
        )
        self.assertEqual(facility["total_discounted_expected_loss"], 10.0)

    # ---- 400 invalid_request ----

    def test_unparseable_json_is_invalid_request(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.expected_loss_schedule(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_non_object_payload_is_invalid_request(self):
        for raw in (b"[1, 2]", b"42", b'"text"', b"null"):
            with self.assertRaises(InvalidRequest):
                self.service.expected_loss_schedule(raw)

    # ---- 422 invalid_input: structure, types, ranges ----

    def test_missing_and_empty_arrays(self):
        for key in ("periods", "discount_factors", "counterparties", "facilities"):
            with self.assertRaises(InvalidInput):
                self.call(**{key: None})
            with self.assertRaises(InvalidInput):
                self.call(**{key: []})
            with self.assertRaises(InvalidInput):
                self.call(**{key: "not-a-list"})

    def test_currency_must_be_nonempty_string(self):
        for bad in ("", 3, None, True):
            with self.assertRaises(InvalidInput):
                self.call(currency=bad)

    def test_periods_validation(self):
        bad_periods = [
            [0],            # not positive
            [-1],           # negative
            [1.5],          # not an integer
            [True],         # bool is not an integer term
            ["1"],          # string
            [2, 1],         # not increasing
            [1, 1],         # duplicate
            [1, 3, 2],      # not increasing
            [None],         # missing value
        ]
        for periods in bad_periods:
            with self.assertRaises(InvalidInput, msg=repr(periods)):
                self.call(periods=periods)

    def test_discount_factors_validation(self):
        bad_factors = [
            [1.0, 0.5],             # length mismatch
            [1.0, 0.5, 0.25, 0.1],  # length mismatch
            [1.0, 0.0, 0.25],       # zero not allowed
            [1.0, -0.5, 0.25],      # negative
            [1.0, 1.5, 0.25],       # above 1
            [1.0, True, 0.25],      # bool
            [1.0, "0.5", 0.25],     # string
            [1.0, 10**400, 0.25],   # oversized integer
            [1.0, None, 0.25],      # missing value
        ]
        for factors in bad_factors:
            with self.assertRaises(InvalidInput, msg=repr(factors)):
                self.call(discount_factors=factors)

    def test_discount_factor_boundaries_are_allowed(self):
        result = self.call(discount_factors=[1.0, 1.0, 1.0])
        # f-1: [6.25, 12.5, 50.0]; f-2: [1.0, 2.0, 8.0].
        self.assertEqual(
            result["portfolio_total"]["total_discounted_expected_loss"], 79.75
        )

    def test_counterparty_field_validation(self):
        bad_counterparties = [
            [{"cumulative_pd": [0.1, 0.2, 0.3]}],  # missing id
            [{"id": "", "cumulative_pd": [0.1, 0.2, 0.3]}],
            [{"id": "c"}],  # missing cumulative_pd
            [{"id": "c", "cumulative_pd": "not-a-list"}],
            [{"id": "c", "cumulative_pd": [0.1, 0.2]}],        # length mismatch
            [{"id": "c", "cumulative_pd": [0.1, 0.2, 0.3, 0.4]}],
            [{"id": "c", "cumulative_pd": [-0.1, 0.2, 0.3]}],  # below 0
            [{"id": "c", "cumulative_pd": [0.1, 0.2, 1.1]}],   # above 1
            [{"id": "c", "cumulative_pd": [0.3, 0.2, 0.3]}],   # decreasing
            [{"id": "c", "cumulative_pd": [True, 0.2, 0.3]}],  # bool
            [{"id": "c", "cumulative_pd": ["0.1", 0.2, 0.3]}],  # string
            [{"id": "c", "cumulative_pd": [10**400, 1.0, 1.0]}],  # oversized
            ["not-an-object"],
        ]
        for counterparties in bad_counterparties:
            with self.assertRaises(InvalidInput, msg=repr(counterparties)):
                self.call(counterparties=counterparties)

    def test_cumulative_pd_boundaries_are_allowed(self):
        result = self.call(
            counterparties=[{"id": "c", "cumulative_pd": [0.0, 1.0, 1.0]}],
            facilities=[
                {"id": "f", "counterparty_id": "c", "lgd": 1.0,
                 "ead": [10.0, 10.0, 10.0]}
            ],
        )
        facility = result["facilities"][0]
        self.assertEqual(
            [c["discounted_expected_loss"] for c in facility["contributions"]],
            [0.0, 5.0, 0.0],
        )

    def test_facility_field_validation(self):
        bad_facilities = [
            [{"counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]}],
            [{"id": "", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]}],
            [{"id": "f", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]}],  # missing cp id
            [{"id": "f", "counterparty_id": "", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "ead": [1.0, 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": -0.1, "ead": [1.0, 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 1.1, "ead": [1.0, 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": True, "ead": [1.0, 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5}],  # missing ead
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": "x"}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, -1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, False, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, "1", 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, 10**400, 1.0]}],
            ["not-an-object"],
        ]
        for facilities in bad_facilities:
            with self.assertRaises(InvalidInput, msg=repr(facilities)):
                self.call(facilities=facilities)

    def test_nan_and_infinity_tokens_are_invalid_input(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            raw = request_body().replace(b"100.0", token.encode())
            with self.assertRaises(InvalidInput):
                self.service.expected_loss_schedule(raw)

    def test_non_finite_computation_is_invalid_input(self):
        with self.assertRaises(InvalidInput):
            self.call(
                periods=[1],
                discount_factors=[1.0],
                counterparties=[{"id": "c", "cumulative_pd": [1.0]}],
                facilities=[
                    {"id": "f-1", "counterparty_id": "c", "lgd": 1.0, "ead": [1e308]},
                    {"id": "f-2", "counterparty_id": "c", "lgd": 1.0, "ead": [1e308]},
                ],
            )

    # ---- 422 duplicate_* / unknown_counterparty ----

    def assert_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)
        self.assertEqual(ctx.exception.status, 422)

    def test_duplicate_ids(self):
        self.assert_code(
            "duplicate_counterparty",
            counterparties=[
                {"id": "c", "cumulative_pd": [0.1, 0.2, 0.3]},
                {"id": "c", "cumulative_pd": [0.1, 0.2, 0.3]},
            ],
        )
        self.assert_code(
            "duplicate_facility",
            facilities=[
                {"id": "f", "counterparty_id": "cp-a", "lgd": 0.5,
                 "ead": [1.0, 1.0, 1.0]},
                {"id": "f", "counterparty_id": "cp-b", "lgd": 0.5,
                 "ead": [1.0, 1.0, 1.0]},
            ],
        )

    def test_unknown_counterparty(self):
        self.assert_code(
            "unknown_counterparty",
            facilities=[
                {"id": "f", "counterparty_id": "ghost", "lgd": 0.5,
                 "ead": [1.0, 1.0, 1.0]}
            ],
        )

    # ---- 413 request_too_large ----

    def test_size_limits(self):
        counterparties = [
            {"id": f"cp-{i}", "cumulative_pd": [0.1, 0.2, 0.3]}
            for i in range(1001)
        ]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(counterparties=counterparties)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

        facilities = [
            {"id": f"f-{i}", "counterparty_id": "cp-a", "lgd": 0.5,
             "ead": [1.0, 1.0, 1.0]}
            for i in range(10001)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(facilities=facilities)

        # 1001 periods x 1000 facilities exceeds the pair budget.
        with self.assertRaises(RequestTooLarge):
            self.call(
                periods=list(range(1, 1002)),
                discount_factors=[1.0] * 1001,
                counterparties=[{"id": "cp-a", "cumulative_pd": [0.5] * 1001}],
                facilities=[
                    {"id": f"f-{i}", "counterparty_id": "cp-a", "lgd": 0.5,
                     "ead": [1.0] * 1001}
                    for i in range(1000)
                ],
            )

    def test_size_boundaries_are_allowed(self):
        counterparties = [
            {"id": f"cp-{i}", "cumulative_pd": [0.1, 0.2, 0.3]}
            for i in range(1000)
        ]
        facilities = [
            {"id": "f-1", "counterparty_id": "cp-0", "lgd": 0.5,
             "ead": [1.0, 1.0, 1.0]}
        ]
        result = self.call(counterparties=counterparties, facilities=facilities)
        self.assertEqual(len(result["counterparties"]), 1000)

        facilities = [
            {"id": f"f-{i}", "counterparty_id": "cp-a", "lgd": 0.5,
             "ead": [1.0, 1.0, 1.0]}
            for i in range(10000)
        ]
        result = self.call(facilities=facilities)
        self.assertEqual(len(result["facilities"]), 10000)

        # Exactly 1000 periods x 1000 facilities = 1000000 pairs.
        result = self.call(
            periods=list(range(1, 1001)),
            discount_factors=[1.0] * 1000,
            counterparties=[{"id": "cp-a", "cumulative_pd": [0.5] * 1000}],
            facilities=[
                {"id": f"f-{i}", "counterparty_id": "cp-a", "lgd": 0.5,
                 "ead": [1.0] * 1000}
                for i in range(1000)
            ],
        )
        self.assertEqual(len(result["facilities"]), 1000)
        self.assertEqual(len(result["facilities"][0]["contributions"]), 1000)


if __name__ == "__main__":
    unittest.main()
