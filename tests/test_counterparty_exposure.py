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
        "counterparties": [
            {"id": "cp1", "pd": 0.02, "lgd": 0.6},
            {"id": "cp2", "pd": 0.1, "lgd": 0.5},
        ],
        "netting_sets": [
            {"id": "ns1", "counterparty_id": "cp1", "collateral": 5.0},
            {"id": "ns2", "counterparty_id": "cp1", "collateral": 0.0},
            {"id": "ns3", "counterparty_id": "cp2", "collateral": 100.0},
        ],
        "trades": [
            {"id": "t1", "netting_set_id": "ns1", "mtm": 10.0, "add_on": 2.0},
            {"id": "t2", "netting_set_id": "ns1", "mtm": -4.0, "add_on": 1.0},
            {"id": "t3", "netting_set_id": "ns2", "mtm": 20.0, "add_on": 0.0},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class CounterpartyExposureTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.counterparty_exposure(request_body(**overrides))

    def by_id(self, result, key):
        return {row["id"]: row for row in result[key]}

    def test_basic_metrics_and_order(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        sets = result["netting_set_results"]
        self.assertEqual([s["id"] for s in sets], ["ns1", "ns2", "ns3"])

        ns1, ns2, ns3 = sets
        # ns1: gross = 10 + max(-4,0) = 10; net = 6; pfe = 3
        self.assertEqual(ns1["gross_exposure"], 10.0)
        self.assertEqual(ns1["net_mtm"], 6.0)
        self.assertEqual(ns1["potential_future_exposure"], 3.0)
        # ead = max(6 + 3 - 5, 0) = 4; el = 4 * .02 * .6 = .048
        self.assertEqual(ns1["exposure_at_default"], 4.0)
        self.assertAlmostEqual(ns1["expected_loss"], 0.048)
        self.assertEqual(ns1["trade_count"], 2)

        # ns2: ead = 20 + 0 - 0 = 20; el = 20*.02*.6 = .24
        self.assertEqual(ns2["gross_exposure"], 20.0)
        self.assertEqual(ns2["exposure_at_default"], 20.0)
        self.assertAlmostEqual(ns2["expected_loss"], 0.24)
        self.assertEqual(ns2["trade_count"], 1)

        # ns3 declared but has no trades -> zero values retained.
        self.assertEqual(ns3["gross_exposure"], 0.0)
        self.assertEqual(ns3["net_mtm"], 0.0)
        self.assertEqual(ns3["potential_future_exposure"], 0.0)
        self.assertEqual(ns3["exposure_at_default"], 0.0)
        self.assertEqual(ns3["expected_loss"], 0.0)
        self.assertEqual(ns3["trade_count"], 0)

    def test_overcollateralisation_floors_at_zero(self):
        result = self.call()
        ns3 = self.by_id(result, "netting_set_results")["ns3"]
        # 0 + 0 - 100 would be -100 but must floor to zero, hence el zero.
        self.assertEqual(ns3["exposure_at_default"], 0.0)
        self.assertEqual(ns3["expected_loss"], 0.0)

    def test_counterparty_totals_input_order(self):
        result = self.call()
        cps = result["counterparty_totals"]
        self.assertEqual([c["id"] for c in cps], ["cp1", "cp2"])
        cp1, cp2 = cps
        # cp1 owns ns1 + ns2.
        self.assertEqual(cp1["gross_exposure"], 30.0)
        self.assertEqual(cp1["net_mtm"], 26.0)
        self.assertEqual(cp1["potential_future_exposure"], 3.0)
        self.assertEqual(cp1["exposure_at_default"], 24.0)
        self.assertAlmostEqual(cp1["expected_loss"], 0.288)
        # cp2 owns only the over-collateralised ns3.
        self.assertEqual(cp2["exposure_at_default"], 0.0)
        self.assertEqual(cp2["expected_loss"], 0.0)

    def test_portfolio_totals_equal_detail_sum(self):
        result = self.call()
        for metric in (
            "gross_exposure",
            "net_mtm",
            "potential_future_exposure",
            "exposure_at_default",
            "expected_loss",
        ):
            detail_sum = sum(s[metric] for s in result["netting_set_results"])
            # The portfolio total is the canonical total of the details.
            self.assertEqual(detail_sum, result["portfolio_totals"][metric], msg=metric)
            # Counterparties partition the same money; agree to machine
            # precision (exact wherever float64 can express the partition).
            cp_sum = sum(c[metric] for c in result["counterparty_totals"])
            self.assertTrue(
                math.isclose(cp_sum, detail_sum, rel_tol=1e-9, abs_tol=1e-9),
                msg=metric,
            )
        self.assertEqual(result["portfolio_totals"]["exposure_at_default"], 24.0)
        self.assertAlmostEqual(result["portfolio_totals"]["expected_loss"], 0.288)

    def test_each_counterparty_matches_its_netting_sets(self):
        result = self.call()
        # Ownership per request_body(): ns1/ns2 belong to cp1, ns3 to cp2.
        owner_of = {"ns1": "cp1", "ns2": "cp1", "ns3": "cp2"}
        sets_by_owner = {"cp1": [], "cp2": []}
        for s in result["netting_set_results"]:
            sets_by_owner[owner_of[s["id"]]].append(s)
        for cp in result["counterparty_totals"]:
            owned = sets_by_owner[cp["id"]]
            for metric in (
                "gross_exposure",
                "net_mtm",
                "potential_future_exposure",
                "exposure_at_default",
                "expected_loss",
            ):
                self.assertTrue(
                    math.isclose(
                        cp[metric],
                        sum(s[metric] for s in owned),
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ),
                    msg=(cp["id"], metric),
                )

    def test_currency_defaults_to_usd(self):
        body = json.loads(request_body())
        del body["currency"]
        result = self.service.counterparty_exposure(json.dumps(body).encode())
        self.assertEqual(result["currency"], "USD")

    def test_currency_override(self):
        result = self.call(currency="EUR")
        self.assertEqual(result["currency"], "EUR")

    def test_extra_fields_ignored(self):
        body = json.loads(request_body())
        body["extra_top"] = 1
        body["counterparties"][0]["rating"] = "A"
        body["netting_sets"][0]["custodian"] = "x"
        body["trades"][0]["desk"] = "rates"
        result = self.service.counterparty_exposure(json.dumps(body).encode())
        self.assertEqual(result["netting_set_results"][0]["id"], "ns1")

    def test_boundary_pd_lgd_allowed(self):
        for pd, lgd in ((0.0, 0.0), (1.0, 1.0)):
            with self.subTest(pd=pd, lgd=lgd):
                result = self.call(
                    counterparties=[{"id": "cp1", "pd": pd, "lgd": lgd}],
                    netting_sets=[
                        {"id": "ns1", "counterparty_id": "cp1", "collateral": 0.0}
                    ],
                    trades=[
                        {"id": "t1", "netting_set_id": "ns1", "mtm": 10.0, "add_on": 0.0}
                    ],
                )
                self.assertEqual(
                    result["netting_set_results"][0]["expected_loss"], 10.0 * pd * lgd
                )

    # ---- parse failures -> 400 ----

    def test_malformed_json(self):
        with self.assertRaises(InvalidRequest):
            self.service.counterparty_exposure(b"{nope")
        with self.assertRaises(InvalidRequest):
            self.service.counterparty_exposure(b"[]")
        with self.assertRaises(InvalidRequest):
            self.service.counterparty_exposure(b"42")

    # ---- semantic failures -> 422 ----

    def test_missing_fields(self):
        with self.assertRaises(InvalidInput):
            self.service.counterparty_exposure(b"{}")

    def test_empty_arrays(self):
        with self.assertRaises(InvalidInput):
            self.call(counterparties=[])
        with self.assertRaises(InvalidInput):
            self.call(netting_sets=[])
        with self.assertRaises(InvalidInput):
            self.call(trades=[])

    def test_currency_validation(self):
        with self.assertRaises(InvalidInput):
            self.call(currency="")
        with self.assertRaises(InvalidInput):
            self.call(currency=123)

    def test_pd_lgd_range(self):
        for bad in (-0.1, 1.01, "0.5", True, None):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInput):
                    self.call(counterparties=[{"id": "cp1", "pd": bad, "lgd": 0.5}])
                with self.assertRaises(InvalidInput):
                    self.call(counterparties=[{"id": "cp1", "pd": 0.5, "lgd": bad}])

    def test_collateral_non_negative(self):
        with self.assertRaises(InvalidInput):
            self.call(
                netting_sets=[
                    {"id": "ns1", "counterparty_id": "cp1", "collateral": -0.01}
                ]
            )

    def test_add_on_non_negative(self):
        with self.assertRaises(InvalidInput):
            self.call(
                trades=[
                    {"id": "t1", "netting_set_id": "ns1", "mtm": 1.0, "add_on": -1.0}
                ]
            )

    def test_mtm_must_be_finite(self):
        for bad in ("x", True):
            with self.assertRaises(InvalidInput):
                self.call(
                    trades=[
                        {"id": "t1", "netting_set_id": "ns1", "mtm": bad, "add_on": 0.0}
                    ]
                )

    def test_boolean_numbers_rejected(self):
        with self.assertRaises(InvalidInput):
            self.call(counterparties=[{"id": "cp1", "pd": 0, "lgd": False}])
        with self.assertRaises(InvalidInput):
            self.call(
                trades=[
                    {"id": "t1", "netting_set_id": "ns1", "mtm": True, "add_on": 0.0}
                ]
            )

    def test_nan_infinity_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            body = request_body().replace(b"0.02", token.encode(), 1)
            with self.assertRaises(InvalidInput):
                self.service.counterparty_exposure(body)

    def test_oversized_integer_rejected(self):
        big = "9" * 400
        body = json.loads(request_body())
        body["trades"][0]["mtm"] = int(big)
        with self.assertRaises(InvalidInput):
            self.service.counterparty_exposure(json.dumps(body).encode())

    def test_empty_ids_rejected(self):
        with self.assertRaises(InvalidInput):
            self.call(counterparties=[{"id": "", "pd": 0.1, "lgd": 0.5}])
        with self.assertRaises(InvalidInput):
            self.call(
                netting_sets=[{"id": "", "counterparty_id": "cp1", "collateral": 0.0}]
            )
        with self.assertRaises(InvalidInput):
            self.call(
                trades=[{"id": "", "netting_set_id": "ns1", "mtm": 1.0, "add_on": 0.0}]
            )

    def test_unknown_references(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                netting_sets=[
                    {"id": "ns1", "counterparty_id": "ghost", "collateral": 0.0}
                ]
            )
        self.assertEqual(ctx.exception.code, "invalid_input")
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                trades=[
                    {"id": "t1", "netting_set_id": "ghost", "mtm": 1.0, "add_on": 0.0}
                ]
            )
        self.assertEqual(ctx.exception.code, "invalid_input")

    def test_duplicate_ids(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                counterparties=[
                    {"id": "cp1", "pd": 0.1, "lgd": 0.5},
                    {"id": "cp1", "pd": 0.1, "lgd": 0.5},
                ]
            )
        self.assertEqual(ctx.exception.code, "duplicate_counterparty")

        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                netting_sets=[
                    {"id": "ns1", "counterparty_id": "cp1", "collateral": 0.0},
                    {"id": "ns1", "counterparty_id": "cp1", "collateral": 0.0},
                ]
            )
        self.assertEqual(ctx.exception.code, "duplicate_netting_set")

        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                trades=[
                    {"id": "t1", "netting_set_id": "ns1", "mtm": 1.0, "add_on": 0.0},
                    {"id": "t1", "netting_set_id": "ns1", "mtm": 1.0, "add_on": 0.0},
                ]
            )
        self.assertEqual(ctx.exception.code, "duplicate_trade")

    def test_too_many_trades(self):
        trades = [
            {"id": f"t{i}", "netting_set_id": "ns1", "mtm": 1.0, "add_on": 0.0}
            for i in range(10001)
        ]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(trades=trades)
        self.assertEqual(ctx.exception.status, 413)

    def test_exactly_10000_trades_allowed(self):
        trades = [
            {"id": f"t{i:05d}", "netting_set_id": "ns1", "mtm": 1.0, "add_on": 0.0}
            for i in range(10000)
        ]
        result = self.call(trades=trades)
        self.assertEqual(result["netting_set_results"][0]["trade_count"], 10000)

    def test_too_many_netting_sets(self):
        ns = [
            {"id": f"ns{i}", "counterparty_id": "cp1", "collateral": 0.0}
            for i in range(1001)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(netting_sets=ns)

    def test_exactly_1000_netting_sets_allowed(self):
        ns = [
            {"id": f"ns{i:04d}", "counterparty_id": "cp1", "collateral": 0.0}
            for i in range(1000)
        ]
        trades = [
            {"id": "t0", "netting_set_id": "ns0000", "mtm": 1.0, "add_on": 0.0}
        ]
        result = self.call(netting_sets=ns, trades=trades)
        self.assertEqual(len(result["netting_set_results"]), 1000)

    def test_no_partial_results_on_failure(self):
        # Service raises; nothing returned.
        with self.assertRaises(InvalidInput):
            self.call(trades=[])


if __name__ == "__main__":
    unittest.main()
