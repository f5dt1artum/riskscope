import json
import math
import unittest
from statistics import NormalDist

from riskscope.service import (
    InvalidInput,
    InvalidRequest,
    RequestTooLarge,
    Service,
)

NORMAL = NormalDist()


def irb_expected(pd, lgd, ead, maturity):
    """Independent restatement of the Basel F-IRB formulas."""
    a = (1.0 - math.exp(-50.0 * pd)) / (1.0 - math.exp(-50.0))
    r = 0.12 * a + 0.24 * (1.0 - a)
    b = (0.11852 - 0.05478 * math.log(pd)) ** 2
    ma = (1.0 + (maturity - 2.5) * b) / (1.0 - 1.5 * b)
    k = lgd * (
        NORMAL.cdf((NORMAL.inv_cdf(pd) + math.sqrt(r) * NORMAL.inv_cdf(0.999)) / math.sqrt(1.0 - r))
        - pd
    ) * ma
    capital = ead * k
    return r, ma, k, capital, 12.5 * capital


def request_body(**overrides):
    body = {
        "counterparties": [
            {"id": "cp-a", "pd": 0.01},
            {"id": "cp-b", "pd": 0.2},
        ],
        "facilities": [
            {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.45,
             "ead": 1000.0, "maturity": 2.5},
            {"id": "f-2", "counterparty_id": "cp-a", "lgd": 1.0,
             "ead": 500.0, "maturity": 5.0},
            {"id": "f-3", "counterparty_id": "cp-b", "lgd": 0.0,
             "ead": 250.0, "maturity": 1.0},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class IrbCapitalTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.irb_capital(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")

        facilities = result["facilities"]
        self.assertEqual([f["id"] for f in facilities], ["f-1", "f-2", "f-3"])
        self.assertEqual(
            [f["counterparty_id"] for f in facilities],
            ["cp-a", "cp-a", "cp-b"],
        )

        cases = [
            (0.01, 0.45, 1000.0, 2.5),
            (0.01, 1.0, 500.0, 5.0),
            (0.2, 0.0, 250.0, 1.0),
        ]
        for facility, (pd, lgd, ead, maturity) in zip(facilities, cases):
            r, ma, k, capital, rwa = irb_expected(pd, lgd, ead, maturity)
            self.assertEqual(facility["pd"], pd)
            self.assertEqual(facility["ead"], ead)
            self.assertEqual(facility["r"], r)
            self.assertEqual(facility["ma"], ma)
            self.assertEqual(facility["k"], k)
            self.assertEqual(facility["capital_requirement"], capital)
            self.assertEqual(facility["risk_weighted_assets"], rwa)

        # A zero lgd zeroes the capital factor and everything downstream.
        self.assertEqual(facilities[2]["k"], 0.0)
        self.assertEqual(facilities[2]["capital_requirement"], 0.0)
        self.assertEqual(facilities[2]["risk_weighted_assets"], 0.0)

    def test_known_value(self):
        # pd = 0.01, lgd = 0.45, ead = 1000, maturity = 2.5.
        result = self.call(
            counterparties=[{"id": "cp-a", "pd": 0.01}],
            facilities=[
                {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.45,
                 "ead": 1000.0, "maturity": 2.5},
            ],
        )
        facility = result["facilities"][0]
        self.assertAlmostEqual(facility["r"], 0.192783679165516, places=12)
        self.assertAlmostEqual(facility["ma"], 1.2598095009238282, places=12)
        self.assertAlmostEqual(facility["k"], 0.07385344111364117, places=12)
        self.assertAlmostEqual(
            facility["capital_requirement"], 73.85344111364117, places=9
        )
        self.assertAlmostEqual(
            facility["risk_weighted_assets"], 923.1680139205147, places=9
        )

    def test_counterparty_and_portfolio_aggregation(self):
        result = self.call()
        facilities = result["facilities"]
        counterparties = result["counterparties"]
        self.assertEqual([cp["id"] for cp in counterparties], ["cp-a", "cp-b"])

        keys = ("ead", "capital_requirement", "risk_weighted_assets")
        for cp_row, cp_id in zip(counterparties, ["cp-a", "cp-b"]):
            owned = [f for f in facilities if f["counterparty_id"] == cp_id]
            for key in keys:
                self.assertEqual(cp_row[key], sum(f[key] for f in owned))

        portfolio = result["portfolio_totals"]
        for key in keys:
            self.assertEqual(
                portfolio[key], sum(cp[key] for cp in counterparties)
            )
            self.assertEqual(
                portfolio[key], sum(f[key] for f in facilities)
            )

    def test_counterparty_without_facilities_keeps_zero_totals(self):
        result = self.call(
            facilities=[
                {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.45,
                 "ead": 1000.0, "maturity": 2.5},
            ],
        )
        counterparties = result["counterparties"]
        self.assertEqual([cp["id"] for cp in counterparties], ["cp-a", "cp-b"])
        cp_b = counterparties[1]
        self.assertEqual(cp_b["ead"], 0.0)
        self.assertEqual(cp_b["capital_requirement"], 0.0)
        self.assertEqual(cp_b["risk_weighted_assets"], 0.0)
        # Portfolio totals still equal the sum over all counterparties.
        for key in ("ead", "capital_requirement", "risk_weighted_assets"):
            self.assertEqual(
                result["portfolio_totals"][key],
                sum(cp[key] for cp in counterparties),
            )

    def test_currency_default_and_override(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_extra_fields_ignored(self):
        result = self.call(
            counterparties=[{"id": "cp-a", "pd": 0.01, "rating": "AAA"}],
            facilities=[
                {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.45,
                 "ead": 100.0, "maturity": 3.0, "sector": "retail"},
            ],
            unexpected=True,
        )
        self.assertEqual([f["id"] for f in result["facilities"]], ["f-1"])

    def test_integer_numbers_accepted(self):
        result = self.call(
            facilities=[
                {"id": "f-1", "counterparty_id": "cp-a", "lgd": 1,
                 "ead": 100, "maturity": 2},
            ],
        )
        facility = result["facilities"][0]
        self.assertEqual(facility["ead"], 100.0)
        _, _, _, capital, rwa = irb_expected(0.01, 1.0, 100.0, 2.0)
        self.assertEqual(facility["capital_requirement"], capital)
        self.assertEqual(facility["risk_weighted_assets"], rwa)

    # ---- request-level errors ----

    def test_invalid_json_is_invalid_request(self):
        with self.assertRaises(InvalidRequest):
            self.service.irb_capital(b"{not json")

    def test_non_object_payload_is_invalid_request(self):
        for payload in (b"[1]", b"1", b'"x"', b"null"):
            with self.assertRaises(InvalidRequest):
                self.service.irb_capital(payload)

    def test_non_finite_constant_is_invalid_input(self):
        with self.assertRaises(InvalidInput):
            self.service.irb_capital(b'{"counterparties": NaN, "facilities": []}')

    # ---- field and range validation ----

    def test_currency_must_be_nonempty_string(self):
        for currency in ("", 1, None, []):
            with self.assertRaises(InvalidInput):
                self.call(currency=currency)

    def test_counterparties_must_be_nonempty_array(self):
        for counterparties in (None, [], {}, "cp-a"):
            with self.assertRaises(InvalidInput):
                self.call(counterparties=counterparties)

    def test_facilities_must_be_nonempty_array(self):
        for facilities in (None, [], {}, "f-1"):
            with self.assertRaises(InvalidInput):
                self.call(facilities=facilities)

    def test_counterparty_id_must_be_nonempty_string(self):
        for bad in ("", 1, None):
            with self.assertRaises(InvalidInput):
                self.call(counterparties=[{"id": bad, "pd": 0.01}])

    def test_pd_must_be_strictly_between_zero_and_one(self):
        for bad in (0.0, 1.0, -0.1, 1.1, "0.5", None, True):
            with self.assertRaises(InvalidInput):
                self.call(counterparties=[{"id": "cp-a", "pd": bad}])

    def test_facility_fields_validated(self):
        base = {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.5,
                "ead": 100.0, "maturity": 2.5}

        def facility(**overrides):
            row = dict(base)
            row.update(overrides)
            return [row]

        for bad in ("", 1, None):
            with self.assertRaises(InvalidInput):
                self.call(facilities=facility(id=bad))
            with self.assertRaises(InvalidInput):
                self.call(facilities=facility(counterparty_id=bad))
        for bad in (-0.1, 1.1, "0.5", None, False):
            with self.assertRaises(InvalidInput):
                self.call(facilities=facility(lgd=bad))
        for bad in (-1.0, "100", None, True):
            with self.assertRaises(InvalidInput):
                self.call(facilities=facility(ead=bad))
        for bad in (0.9, 5.1, "2.5", None, True):
            with self.assertRaises(InvalidInput):
                self.call(facilities=facility(maturity=bad))

    def test_oversized_integer_rejected(self):
        huge = 10 ** 400
        with self.assertRaises(InvalidInput):
            self.call(facilities=[
                {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.5,
                 "ead": huge, "maturity": 2.5},
            ])

    # ---- duplicate and reference errors ----

    def assert_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as caught:
            self.call(**overrides)
        self.assertEqual(caught.exception.code, code)

    def test_duplicate_counterparty(self):
        self.assert_code(
            "duplicate_counterparty",
            counterparties=[{"id": "cp-a", "pd": 0.01}, {"id": "cp-a", "pd": 0.02}],
        )

    def test_duplicate_facility(self):
        self.assert_code(
            "duplicate_facility",
            facilities=[
                {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.5,
                 "ead": 1.0, "maturity": 2.5},
                {"id": "f-1", "counterparty_id": "cp-b", "lgd": 0.5,
                 "ead": 2.0, "maturity": 3.0},
            ],
        )

    def test_unknown_counterparty(self):
        self.assert_code(
            "unknown_counterparty",
            facilities=[
                {"id": "f-1", "counterparty_id": "cp-zz", "lgd": 0.5,
                 "ead": 1.0, "maturity": 2.5},
            ],
        )

    def test_error_ordering_duplicate_before_unknown(self):
        # Duplicate counterparties are reported before facility references
        # are resolved.
        self.assert_code(
            "duplicate_counterparty",
            counterparties=[{"id": "cp-a", "pd": 0.01}, {"id": "cp-a", "pd": 0.02}],
            facilities=[
                {"id": "f-1", "counterparty_id": "cp-zz", "lgd": 0.5,
                 "ead": 1.0, "maturity": 2.5},
            ],
        )

    # ---- size limits ----

    def test_counterparty_limit_boundary(self):
        counterparties = [{"id": f"cp-{i}", "pd": 0.01} for i in range(1000)]
        facilities = [
            {"id": "f-1", "counterparty_id": "cp-0", "lgd": 0.5,
             "ead": 1.0, "maturity": 2.5},
        ]
        result = self.call(counterparties=counterparties, facilities=facilities)
        self.assertEqual(len(result["counterparties"]), 1000)
        counterparties.append({"id": "cp-1000", "pd": 0.01})
        with self.assertRaises(RequestTooLarge):
            self.call(counterparties=counterparties, facilities=facilities)

    def test_facility_limit_boundary(self):
        facilities = [
            {"id": f"f-{i}", "counterparty_id": "cp-a", "lgd": 0.5,
             "ead": 1.0, "maturity": 2.5}
            for i in range(10000)
        ]
        result = self.call(facilities=facilities)
        self.assertEqual(len(result["facilities"]), 10000)
        facilities.append(
            {"id": "f-10000", "counterparty_id": "cp-a", "lgd": 0.5,
             "ead": 1.0, "maturity": 2.5}
        )
        with self.assertRaises(RequestTooLarge):
            self.call(facilities=facilities)


if __name__ == "__main__":
    unittest.main()
