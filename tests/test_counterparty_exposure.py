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
        "counterparties": [
            {"id": "cp-a", "pd": 0.02, "lgd": 0.5},
            {"id": "cp-b", "pd": 0.1, "lgd": 0.25},
        ],
        "netting_sets": [
            {"id": "ns-1", "counterparty_id": "cp-a", "collateral": 10.0},
            {"id": "ns-2", "counterparty_id": "cp-a", "collateral": 1000.0},
            {"id": "ns-3", "counterparty_id": "cp-b", "collateral": 0.0},
        ],
        "trades": [
            {"id": "t-1", "netting_set_id": "ns-1", "mtm": 100.0, "add_on": 5.0},
            {"id": "t-2", "netting_set_id": "ns-1", "mtm": -30.0, "add_on": 2.5},
            {"id": "t-3", "netting_set_id": "ns-3", "mtm": -50.0, "add_on": 0.0},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class CounterpartyExposureTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.counterparty_exposure(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")

        netting_sets = result["netting_sets"]
        self.assertEqual([ns["id"] for ns in netting_sets], ["ns-1", "ns-2", "ns-3"])

        ns1 = netting_sets[0]
        self.assertEqual(ns1["counterparty_id"], "cp-a")
        self.assertEqual(ns1["trade_count"], 2)
        self.assertEqual(ns1["gross_exposure"], 100.0)  # max(mtm, 0) sums
        self.assertEqual(ns1["net_mtm"], 70.0)
        self.assertEqual(ns1["potential_future_exposure"], 7.5)
        self.assertEqual(ns1["collateral"], 10.0)
        # max(70 + 7.5 - 10, 0) = 67.5; EL = 67.5 * 0.02 * 0.5
        self.assertEqual(ns1["exposure_at_default"], 67.5)
        self.assertEqual(ns1["expected_loss"], 0.675)

        # Declared netting set with no trades keeps zero values; heavy
        # collateral still cannot push the exposure below zero.
        ns2 = netting_sets[1]
        self.assertEqual(ns2["trade_count"], 0)
        self.assertEqual(ns2["gross_exposure"], 0.0)
        self.assertEqual(ns2["net_mtm"], 0.0)
        self.assertEqual(ns2["potential_future_exposure"], 0.0)
        self.assertEqual(ns2["exposure_at_default"], 0.0)
        self.assertEqual(ns2["expected_loss"], 0.0)

        # Negative net mtm with no add-on floors at zero as well.
        ns3 = netting_sets[2]
        self.assertEqual(ns3["trade_count"], 1)
        self.assertEqual(ns3["gross_exposure"], 0.0)
        self.assertEqual(ns3["net_mtm"], -50.0)
        self.assertEqual(ns3["exposure_at_default"], 0.0)
        self.assertEqual(ns3["expected_loss"], 0.0)

    def test_counterparty_aggregation_and_portfolio_totals(self):
        result = self.call()
        counterparties = result["counterparties"]
        self.assertEqual([cp["id"] for cp in counterparties], ["cp-a", "cp-b"])

        amount_keys = (
            "gross_exposure",
            "net_mtm",
            "potential_future_exposure",
            "collateral",
            "exposure_at_default",
            "expected_loss",
        )
        netting_sets = result["netting_sets"]
        for cp_row, cp_id in zip(counterparties, ["cp-a", "cp-b"]):
            owned = [ns for ns in netting_sets if ns["counterparty_id"] == cp_id]
            for key in amount_keys:
                self.assertEqual(
                    cp_row[key], sum(ns[key] for ns in owned), f"{cp_id}.{key}"
                )

        totals = result["portfolio_totals"]
        for key in amount_keys:
            # Totals equal the sum of the counterparty details exactly.
            self.assertEqual(
                totals[key], sum(cp[key] for cp in counterparties), f"totals.{key}"
            )
            self.assertEqual(
                totals[key], sum(ns[key] for ns in netting_sets), f"totals.{key}"
            )

    def test_currency_echo_and_default(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_extra_fields_are_ignored(self):
        result = self.call(
            unexpected="top-level",
            counterparties=[{"id": "c", "pd": 0.5, "lgd": 0.5, "rating": "AAA"}],
            netting_sets=[
                {"id": "n", "counterparty_id": "c", "collateral": 0, "note": 1}
            ],
            trades=[{"id": "t", "netting_set_id": "n", "mtm": 4, "add_on": 1, "x": 2}],
        )
        ns = result["netting_sets"][0]
        self.assertEqual(ns["exposure_at_default"], 5.0)
        self.assertEqual(ns["expected_loss"], 1.25)

    def test_integer_numbers_are_accepted(self):
        result = self.call(
            counterparties=[{"id": "c", "pd": 1, "lgd": 1}],
            netting_sets=[{"id": "n", "counterparty_id": "c", "collateral": 2}],
            trades=[{"id": "t", "netting_set_id": "n", "mtm": 10, "add_on": 3}],
        )
        ns = result["netting_sets"][0]
        self.assertEqual(ns["exposure_at_default"], 11.0)
        self.assertEqual(ns["expected_loss"], 11.0)

    # ---- 400 invalid_request ----

    def test_unparseable_json_is_invalid_request(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.counterparty_exposure(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_non_object_payload_is_invalid_request(self):
        for raw in (b"[1, 2]", b"42", b'"text"', b"null"):
            with self.assertRaises(InvalidRequest):
                self.service.counterparty_exposure(raw)

    # ---- 422 invalid_input: structure, types, ranges ----

    def test_missing_and_empty_arrays(self):
        for key in ("counterparties", "netting_sets", "trades"):
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

    def test_counterparty_field_validation(self):
        bad_counterparties = [
            [{"pd": 0.1, "lgd": 0.5}],  # missing id
            [{"id": "", "pd": 0.1, "lgd": 0.5}],
            [{"id": "c", "lgd": 0.5}],  # missing pd
            [{"id": "c", "pd": 0.1}],  # missing lgd
            [{"id": "c", "pd": -0.1, "lgd": 0.5}],
            [{"id": "c", "pd": 1.1, "lgd": 0.5}],
            [{"id": "c", "pd": 0.1, "lgd": -0.1}],
            [{"id": "c", "pd": 0.1, "lgd": 1.1}],
            [{"id": "c", "pd": True, "lgd": 0.5}],
            [{"id": "c", "pd": 0.1, "lgd": "0.5"}],
            [{"id": "c", "pd": 10**400, "lgd": 0.5}],
            ["not-an-object"],
        ]
        for counterparties in bad_counterparties:
            with self.assertRaises(InvalidInput, msg=repr(counterparties)):
                self.call(counterparties=counterparties)

    def test_pd_lgd_boundaries_are_allowed(self):
        result = self.call(
            counterparties=[{"id": "c", "pd": 0.0, "lgd": 1.0}],
            netting_sets=[{"id": "n", "counterparty_id": "c", "collateral": 0.0}],
            trades=[{"id": "t", "netting_set_id": "n", "mtm": 7.0, "add_on": 0.0}],
        )
        ns = result["netting_sets"][0]
        self.assertEqual(ns["exposure_at_default"], 7.0)
        self.assertEqual(ns["expected_loss"], 0.0)  # pd == 0
        result = self.call(
            counterparties=[{"id": "c", "pd": 1.0, "lgd": 1.0}],
            netting_sets=[{"id": "n", "counterparty_id": "c", "collateral": 0.0}],
            trades=[{"id": "t", "netting_set_id": "n", "mtm": 7.0, "add_on": 0.0}],
        )
        self.assertEqual(result["netting_sets"][0]["expected_loss"], 7.0)

    def test_netting_set_field_validation(self):
        bad_netting_sets = [
            [{"counterparty_id": "cp-a", "collateral": 0.0}],  # missing id
            [{"id": "", "counterparty_id": "cp-a", "collateral": 0.0}],
            [{"id": "n", "collateral": 0.0}],  # missing counterparty_id
            [{"id": "n", "counterparty_id": "cp-a"}],  # missing collateral
            [{"id": "n", "counterparty_id": "cp-a", "collateral": -1.0}],
            [{"id": "n", "counterparty_id": "cp-a", "collateral": False}],
            [{"id": "n", "counterparty_id": "cp-a", "collateral": "0"}],
            [{"id": "n", "counterparty_id": "cp-a", "collateral": 10**400}],
            ["not-an-object"],
        ]
        for netting_sets in bad_netting_sets:
            with self.assertRaises(InvalidInput, msg=repr(netting_sets)):
                self.call(netting_sets=netting_sets)

    def test_trade_field_validation(self):
        bad_trades = [
            [{"netting_set_id": "ns-1", "mtm": 1.0, "add_on": 0.0}],  # missing id
            [{"id": "", "netting_set_id": "ns-1", "mtm": 1.0, "add_on": 0.0}],
            [{"id": "t", "mtm": 1.0, "add_on": 0.0}],  # missing netting_set_id
            [{"id": "t", "netting_set_id": "ns-1", "add_on": 0.0}],  # missing mtm
            [{"id": "t", "netting_set_id": "ns-1", "mtm": 1.0}],  # missing add_on
            [{"id": "t", "netting_set_id": "ns-1", "mtm": 1.0, "add_on": -0.5}],
            [{"id": "t", "netting_set_id": "ns-1", "mtm": True, "add_on": 0.0}],
            [{"id": "t", "netting_set_id": "ns-1", "mtm": 1.0, "add_on": False}],
            [{"id": "t", "netting_set_id": "ns-1", "mtm": "1", "add_on": 0.0}],
            [{"id": "t", "netting_set_id": "ns-1", "mtm": 10**400, "add_on": 0.0}],
            [{"id": "t", "netting_set_id": "ns-1", "mtm": 1.0, "add_on": 10**400}],
            ["not-an-object"],
        ]
        for trades in bad_trades:
            with self.assertRaises(InvalidInput, msg=repr(trades)):
                self.call(trades=trades)

    def test_nan_and_infinity_tokens_are_invalid_input(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            raw = request_body().replace(b"100.0", token.encode())
            with self.assertRaises(InvalidInput):
                self.service.counterparty_exposure(raw)

    def test_unknown_references_are_invalid_input(self):
        with self.assertRaises(InvalidInput):
            self.call(
                netting_sets=[
                    {"id": "ns-1", "counterparty_id": "ghost", "collateral": 0.0}
                ]
            )
        with self.assertRaises(InvalidInput):
            self.call(
                trades=[{"id": "t", "netting_set_id": "ghost", "mtm": 1.0, "add_on": 0.0}]
            )

    def test_non_finite_computation_is_invalid_input(self):
        with self.assertRaises(InvalidInput):
            self.call(
                trades=[
                    {"id": "t-1", "netting_set_id": "ns-1", "mtm": 1e308, "add_on": 0.0},
                    {"id": "t-2", "netting_set_id": "ns-1", "mtm": 1e308, "add_on": 0.0},
                ]
            )

    # ---- 422 duplicate_* ----

    def assert_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)
        self.assertEqual(ctx.exception.status, 422)

    def test_duplicate_ids(self):
        self.assert_code(
            "duplicate_counterparty",
            counterparties=[
                {"id": "c", "pd": 0.1, "lgd": 0.5},
                {"id": "c", "pd": 0.2, "lgd": 0.5},
            ],
        )
        self.assert_code(
            "duplicate_netting_set",
            netting_sets=[
                {"id": "n", "counterparty_id": "cp-a", "collateral": 0.0},
                {"id": "n", "counterparty_id": "cp-b", "collateral": 0.0},
            ],
        )
        self.assert_code(
            "duplicate_trade",
            trades=[
                {"id": "t", "netting_set_id": "ns-1", "mtm": 1.0, "add_on": 0.0},
                {"id": "t", "netting_set_id": "ns-3", "mtm": 2.0, "add_on": 0.0},
            ],
        )

    # ---- 413 request_too_large ----

    def test_size_limits(self):
        trades = [
            {"id": f"t-{i}", "netting_set_id": "ns-1", "mtm": 1.0, "add_on": 0.0}
            for i in range(10001)
        ]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(trades=trades)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

        netting_sets = [
            {"id": f"ns-{i}", "counterparty_id": "cp-a", "collateral": 0.0}
            for i in range(1001)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(netting_sets=netting_sets)

    def test_size_boundaries_are_allowed(self):
        trades = [
            {"id": f"t-{i}", "netting_set_id": "ns-1", "mtm": 1.0, "add_on": 0.0}
            for i in range(10000)
        ]
        result = self.call(trades=trades)
        self.assertEqual(result["netting_sets"][0]["trade_count"], 10000)

        netting_sets = [
            {"id": f"ns-{i}", "counterparty_id": "cp-a", "collateral": 0.0}
            for i in range(1000)
        ]
        result = self.call(netting_sets=netting_sets)
        self.assertEqual(len(result["netting_sets"]), 1000)


if __name__ == "__main__":
    unittest.main()
