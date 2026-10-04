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
        "discount_factors": [0.99, 0.98, 0.97],
        "counterparties": [
            {"id": "cp-a", "cumulative_pd": [0.01, 0.03, 0.06]},
            {"id": "cp-b", "cumulative_pd": [0.0, 0.0, 0.5]},
            {"id": "cp-c", "cumulative_pd": [0.2, 0.2, 0.2]},
        ],
        "facilities": [
            {
                "id": "f-1",
                "counterparty_id": "cp-a",
                "lgd": 0.5,
                "ead": [100.0, 200.0, 300.0],
            },
            {
                "id": "f-2",
                "counterparty_id": "cp-a",
                "lgd": 1.0,
                "ead": [10.0, 20.0, 30.0],
            },
            {"id": "f-3", "counterparty_id": "cp-b", "lgd": 0.25, "ead": [0.0, 0.0, 8.0]},
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
        self.assertEqual([f["id"] for f in facilities], ["f-1", "f-2", "f-3"])

        # cp-a marginal pds: 0.01, 0.03 - 0.01, 0.06 - 0.03
        mpd_a = [0.01, 0.03 - 0.01, 0.06 - 0.03]
        f1 = facilities[0]
        self.assertEqual(f1["counterparty_id"], "cp-a")
        expected_f1 = [
            100.0 * 0.5 * mpd_a[0] * 0.99,
            200.0 * 0.5 * mpd_a[1] * 0.98,
            300.0 * 0.5 * mpd_a[2] * 0.97,
        ]
        self.assertEqual(f1["discounted_expected_losses"], expected_f1)
        self.assertEqual(f1["total_discounted_expected_loss"], sum(expected_f1))

        f2 = facilities[1]
        expected_f2 = [
            10.0 * 1.0 * mpd_a[0] * 0.99,
            20.0 * 1.0 * mpd_a[1] * 0.98,
            30.0 * 1.0 * mpd_a[2] * 0.97,
        ]
        self.assertEqual(f2["discounted_expected_losses"], expected_f2)
        self.assertEqual(f2["total_discounted_expected_loss"], sum(expected_f2))

        # cp-b marginal pds: 0.0, 0.0, 0.5; only the last period loses.
        f3 = facilities[2]
        expected_f3 = [0.0, 0.0, 8.0 * 0.25 * 0.5 * 0.97]
        self.assertEqual(f3["discounted_expected_losses"], expected_f3)
        self.assertEqual(f3["total_discounted_expected_loss"], sum(expected_f3))

    def test_counterparty_aggregation_and_portfolio_total(self):
        result = self.call()
        facilities = result["facilities"]
        counterparties = result["counterparties"]
        self.assertEqual([cp["id"] for cp in counterparties], ["cp-a", "cp-b", "cp-c"])

        for cp_row, cp_id in zip(counterparties, ["cp-a", "cp-b", "cp-c"]):
            owned = [f for f in facilities if f["counterparty_id"] == cp_id]
            expected_per_period = [
                sum(f["discounted_expected_losses"][i] for f in owned) for i in range(3)
            ]
            self.assertEqual(cp_row["discounted_expected_losses"], expected_per_period)
            self.assertEqual(
                cp_row["total_discounted_expected_loss"],
                sum(f["total_discounted_expected_loss"] for f in owned),
            )

        # cp-c declares no facilities: all-zero results are kept.
        cp_c = counterparties[2]
        self.assertEqual(cp_c["discounted_expected_losses"], [0.0, 0.0, 0.0])
        self.assertEqual(cp_c["total_discounted_expected_loss"], 0.0)

        portfolio = result["portfolio_total"]
        self.assertEqual(
            portfolio["discounted_expected_losses"],
            [
                sum(cp["discounted_expected_losses"][i] for cp in counterparties)
                for i in range(3)
            ],
        )
        self.assertEqual(
            portfolio["total_discounted_expected_loss"],
            sum(cp["total_discounted_expected_loss"] for cp in counterparties),
        )

    def test_first_period_marginal_pd_is_first_cumulative_value(self):
        result = self.call(
            periods=[5],
            discount_factors=[1.0],
            counterparties=[{"id": "cp", "cumulative_pd": [0.4]}],
            facilities=[
                {"id": "f", "counterparty_id": "cp", "lgd": 0.5, "ead": [100.0]}
            ],
        )
        facility = result["facilities"][0]
        self.assertEqual(facility["discounted_expected_losses"], [20.0])
        self.assertEqual(facility["total_discounted_expected_loss"], 20.0)

    def test_zero_values_are_not_omitted(self):
        result = self.call(
            periods=[1, 2],
            discount_factors=[1.0, 1.0],
            counterparties=[{"id": "cp", "cumulative_pd": [0.0, 0.0]}],
            facilities=[
                {"id": "f", "counterparty_id": "cp", "lgd": 0.5, "ead": [100.0, 50.0]}
            ],
        )
        facility = result["facilities"][0]
        self.assertEqual(facility["discounted_expected_losses"], [0.0, 0.0])
        self.assertEqual(facility["total_discounted_expected_loss"], 0.0)
        self.assertEqual(
            result["portfolio_total"],
            {"discounted_expected_losses": [0.0, 0.0], "total_discounted_expected_loss": 0.0},
        )

    def test_currency_echo_and_default(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_extra_fields_are_ignored(self):
        result = self.call(
            unexpected="top-level",
            counterparties=[
                {"id": "c", "cumulative_pd": [0.1, 0.1, 0.1], "rating": "AAA"}
            ],
            facilities=[
                {
                    "id": "f",
                    "counterparty_id": "c",
                    "lgd": 0.5,
                    "ead": [10.0, 10.0, 10.0],
                    "note": 1,
                }
            ],
        )
        facility = result["facilities"][0]
        self.assertEqual(facility["discounted_expected_losses"][0], 10.0 * 0.5 * 0.1 * 0.99)

    def test_integer_numbers_are_accepted(self):
        result = self.call(
            discount_factors=[1, 1, 1],
            counterparties=[{"id": "c", "cumulative_pd": [1, 1, 1]}],
            facilities=[{"id": "f", "counterparty_id": "c", "lgd": 1, "ead": [4, 5, 6]}],
        )
        facility = result["facilities"][0]
        self.assertEqual(facility["discounted_expected_losses"], [4.0, 0.0, 0.0])
        self.assertEqual(facility["total_discounted_expected_loss"], 4.0)

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
            [0],  # not positive
            [-1],
            [1.5],  # floats are not integers
            [True],
            ["1"],
            [None],
            [2, 1],  # not increasing
            [1, 1],  # duplicates
            [3, 3, 4],
            [1, 3, 2],
        ]
        for periods in bad_periods:
            with self.assertRaises(InvalidInput, msg=repr(periods)):
                self.call(periods=periods)

    def test_discount_factors_validation(self):
        bad_discount_factors = [
            [0.99, 0.98],  # length mismatch
            [0.99, 0.98, 0.97, 1.0],
            [0.0, 0.98, 0.97],  # must be positive
            [-0.1, 0.98, 0.97],
            [1.1, 0.98, 0.97],  # must not exceed 1
            [True, 0.98, 0.97],
            ["0.99", 0.98, 0.97],
            [None, 0.98, 0.97],
            [10**400, 0.98, 0.97],
        ]
        for discount_factors in bad_discount_factors:
            with self.assertRaises(InvalidInput, msg=repr(discount_factors)):
                self.call(discount_factors=discount_factors)

    def test_discount_factor_boundaries_are_allowed(self):
        result = self.call(discount_factors=[1.0, 1.0, 1.0])
        self.assertEqual(len(result["facilities"]), 3)

    def test_counterparty_field_validation(self):
        bad_counterparties = [
            [{"cumulative_pd": [0.1, 0.2, 0.3]}],  # missing id
            [{"id": "", "cumulative_pd": [0.1, 0.2, 0.3]}],
            [{"id": "c"}],  # missing cumulative_pd
            [{"id": "c", "cumulative_pd": [0.1, 0.2]}],  # length mismatch
            [{"id": "c", "cumulative_pd": [0.1, 0.2, 0.3, 0.4]}],
            [{"id": "c", "cumulative_pd": "not-a-list"}],
            [{"id": "c", "cumulative_pd": [-0.1, 0.2, 0.3]}],
            [{"id": "c", "cumulative_pd": [0.1, 0.2, 1.1]}],
            [{"id": "c", "cumulative_pd": [0.3, 0.2, 0.3]}],  # decreasing
            [{"id": "c", "cumulative_pd": [True, 0.2, 0.3]}],
            [{"id": "c", "cumulative_pd": ["0.1", 0.2, 0.3]}],
            [{"id": "c", "cumulative_pd": [10**400, 0.2, 0.3]}],
            ["not-an-object"],
        ]
        for counterparties in bad_counterparties:
            with self.assertRaises(InvalidInput, msg=repr(counterparties)):
                self.call(counterparties=counterparties)

    def test_cumulative_pd_boundaries_are_allowed(self):
        result = self.call(
            counterparties=[{"id": "c", "cumulative_pd": [0.0, 0.5, 1.0]}],
            facilities=[{"id": "f", "counterparty_id": "c", "lgd": 1.0, "ead": [1.0, 1.0, 1.0]}],
        )
        self.assertEqual(len(result["facilities"]), 1)

    def test_facility_field_validation(self):
        bad_facilities = [
            [{"counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]}],  # no id
            [{"id": "", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]}],
            [{"id": "f", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]}],  # no counterparty_id
            [{"id": "f", "counterparty_id": "", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "ead": [1.0, 1.0, 1.0]}],  # no lgd
            [{"id": "f", "counterparty_id": "cp-a", "lgd": -0.1, "ead": [1.0, 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 1.1, "ead": [1.0, 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": True, "ead": [1.0, 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5}],  # missing ead
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": "not-a-list"}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [-1.0, 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [False, 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": ["1", 1.0, 1.0]}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [10**400, 1.0, 1.0]}],
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
                discount_factors=[1.0, 1.0, 1.0],
                counterparties=[{"id": "cp-a", "cumulative_pd": [1.0, 1.0, 1.0]}],
                facilities=[
                    {
                        "id": "f-1",
                        "counterparty_id": "cp-a",
                        "lgd": 1.0,
                        "ead": [1e308, 0.0, 0.0],
                    },
                    {
                        "id": "f-2",
                        "counterparty_id": "cp-a",
                        "lgd": 1.0,
                        "ead": [1e308, 0.0, 0.0],
                    },
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
                {"id": "f", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]},
                {"id": "f", "counterparty_id": "cp-b", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]},
            ],
        )

    def test_unknown_counterparty(self):
        self.assert_code(
            "unknown_counterparty",
            facilities=[
                {"id": "f", "counterparty_id": "ghost", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]}
            ],
        )

    # ---- 413 request_too_large ----

    def test_size_limits(self):
        counterparties = [
            {"id": f"cp-{i}", "cumulative_pd": [0.1, 0.2, 0.3]} for i in range(1001)
        ]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(counterparties=counterparties)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

        facilities = [
            {"id": f"f-{i}", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]}
            for i in range(10001)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(facilities=facilities)

        # periods times facilities must not exceed 1000000
        periods = list(range(1, 1001))
        with self.assertRaises(RequestTooLarge):
            self.call(
                periods=periods,
                discount_factors=[1.0] * 1000,
                counterparties=[{"id": "cp-a", "cumulative_pd": [0.1] * 1000}],
                facilities=[
                    {
                        "id": f"f-{i}",
                        "counterparty_id": "cp-a",
                        "lgd": 0.5,
                        "ead": [1.0] * 1000,
                    }
                    for i in range(1001)
                ],
            )

    def test_size_boundaries_are_allowed(self):
        counterparties = [
            {"id": f"cp-{i}", "cumulative_pd": [0.1, 0.2, 0.3]} for i in range(1000)
        ]
        facilities = [
            {"id": f"f-{i}", "counterparty_id": "cp-0", "lgd": 0.5, "ead": [1.0, 1.0, 1.0]}
            for i in range(1000)
        ]
        result = self.call(counterparties=counterparties, facilities=facilities)
        self.assertEqual(len(result["counterparties"]), 1000)
        self.assertEqual(len(result["facilities"]), 1000)

        # Exactly 1000000 period-facility pairs is allowed.
        periods = list(range(1, 1001))
        result = self.call(
            periods=periods,
            discount_factors=[1.0] * 1000,
            counterparties=[{"id": "cp-a", "cumulative_pd": [0.1] * 1000}],
            facilities=[
                {
                    "id": f"f-{i}",
                    "counterparty_id": "cp-a",
                    "lgd": 0.5,
                    "ead": [1.0] * 1000,
                }
                for i in range(1000)
            ],
        )
        self.assertEqual(len(result["facilities"]), 1000)
        self.assertEqual(len(result["facilities"][0]["discounted_expected_losses"]), 1000)


if __name__ == "__main__":
    unittest.main()
