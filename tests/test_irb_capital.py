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


def request_body(**overrides):
    body = {
        "counterparties": [
            {"id": "cp-a", "pd": 0.02},
            {"id": "cp-b", "pd": 0.001},
        ],
        "facilities": [
            {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.45,
             "ead": 1_000_000.0, "maturity": 2.5},
            {"id": "f-2", "counterparty_id": "cp-a", "lgd": 0.5,
             "ead": 500_000.0, "maturity": 1},
            {"id": "f-3", "counterparty_id": "cp-b", "lgd": 0.25,
             "ead": 2_000_000.0, "maturity": 5},
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
            set(facilities[0]),
            {
                "id",
                "counterparty_id",
                "pd",
                "R",
                "MA",
                "K",
                "ead",
                "capital_requirement",
                "risk_weighted_assets",
            },
        )

        f1 = facilities[0]
        self.assertEqual(f1["counterparty_id"], "cp-a")
        self.assertEqual(f1["pd"], 0.02)
        self.assertEqual(f1["ead"], 1_000_000.0)
        self.assertAlmostEqual(f1["R"], 0.16414553294057307)
        self.assertAlmostEqual(f1["MA"], 1.1992627142216061)
        self.assertAlmostEqual(f1["K"], 0.09188338300660007)
        self.assertAlmostEqual(f1["capital_requirement"], 91883.38300660007)
        self.assertAlmostEqual(f1["risk_weighted_assets"], 1148542.2875825008)

        # maturity 1 and 5 are the allowed endpoints; at M = 1 the
        # maturity adjustment collapses to exactly 1.
        self.assertEqual(facilities[1]["MA"], 1.0)
        self.assertAlmostEqual(facilities[1]["K"], 0.0851295104687512)
        self.assertAlmostEqual(
            facilities[1]["capital_requirement"], 42564.755234375596
        )

        f3 = facilities[2]
        self.assertEqual(f3["counterparty_id"], "cp-b")
        self.assertEqual(f3["pd"], 0.001)
        self.assertAlmostEqual(f3["R"], 0.23414753094008567)
        self.assertAlmostEqual(f3["MA"], 2.568856488264487)
        self.assertAlmostEqual(f3["K"], 0.021315826771455118)
        self.assertAlmostEqual(f3["capital_requirement"], 42631.65354291024)
        self.assertAlmostEqual(f3["risk_weighted_assets"], 532895.669286378)

    def test_capital_and_rwa_relationship(self):
        result = self.call()
        for facility in result["facilities"]:
            self.assertEqual(
                facility["risk_weighted_assets"],
                12.5 * facility["capital_requirement"],
            )
            self.assertEqual(
                facility["capital_requirement"], facility["ead"] * facility["K"]
            )

    def test_matches_independent_formula(self):
        normal = NormalDist()
        counterparties = [
            {"id": "cp-a", "pd": 0.02},
            {"id": "cp-b", "pd": 0.001},
        ]
        sent_facilities = [
            {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.45,
             "ead": 1_000_000.0, "maturity": 2.5},
            {"id": "f-2", "counterparty_id": "cp-a", "lgd": 0.5,
             "ead": 500_000.0, "maturity": 1},
            {"id": "f-3", "counterparty_id": "cp-b", "lgd": 0.25,
             "ead": 2_000_000.0, "maturity": 5},
        ]
        pd_by_id = {cp["id"]: cp["pd"] for cp in counterparties}
        result = self.call(
            counterparties=counterparties, facilities=sent_facilities
        )

        for sent, got in zip(sent_facilities, result["facilities"]):
            pd = pd_by_id[sent["counterparty_id"]]
            a = (1.0 - math.exp(-50.0 * pd)) / (1.0 - math.exp(-50.0))
            correlation = 0.12 * a + 0.24 * (1.0 - a)
            b = (0.11852 - 0.05478 * math.log(pd)) ** 2
            ma = (1.0 + (sent["maturity"] - 2.5) * b) / (1.0 - 1.5 * b)
            k = sent["lgd"] * (
                normal.cdf(
                    (
                        normal.inv_cdf(pd)
                        + math.sqrt(correlation) * normal.inv_cdf(0.999)
                    )
                    / math.sqrt(1.0 - correlation)
                )
                - pd
            ) * ma
            self.assertEqual(got["pd"], pd)
            self.assertAlmostEqual(got["R"], correlation, places=14)
            self.assertAlmostEqual(got["MA"], ma, places=14)
            self.assertAlmostEqual(got["K"], k, places=14)
            self.assertAlmostEqual(got["capital_requirement"], sent["ead"] * k, places=8)

    def test_counterparty_and_portfolio_aggregation(self):
        result = self.call()
        counterparties = result["counterparties"]
        self.assertEqual([cp["id"] for cp in counterparties], ["cp-a", "cp-b"])
        self.assertEqual(
            set(counterparties[0]),
            {"id", "ead", "capital_requirement", "risk_weighted_assets"},
        )

        cp_a, cp_b = counterparties
        self.assertEqual(cp_a["ead"], 1_500_000.0)
        self.assertAlmostEqual(cp_a["capital_requirement"], 134448.13824097568)
        self.assertAlmostEqual(cp_a["risk_weighted_assets"], 1680601.7280121958)

        self.assertEqual(cp_b["ead"], 2_000_000.0)
        self.assertAlmostEqual(cp_b["capital_requirement"], 42631.65354291024)
        self.assertAlmostEqual(cp_b["risk_weighted_assets"], 532895.669286378)

        totals = result["portfolio_totals"]
        self.assertEqual(totals["ead"], 3_500_000.0)
        self.assertAlmostEqual(totals["capital_requirement"], 177079.7917838859)
        self.assertAlmostEqual(totals["risk_weighted_assets"], 2213497.3972985735)

    def test_layer_totals_match_details(self):
        result = self.call()
        facilities = result["facilities"]
        counterparties = result["counterparties"]
        totals = result["portfolio_totals"]

        for cp_row in counterparties:
            owned = [f for f in facilities if f["counterparty_id"] == cp_row["id"]]
            self.assertEqual(cp_row["ead"], sum(f["ead"] for f in owned))
            self.assertEqual(
                cp_row["capital_requirement"],
                sum(f["capital_requirement"] for f in owned),
            )
            self.assertEqual(
                cp_row["risk_weighted_assets"],
                sum(f["risk_weighted_assets"] for f in owned),
            )
        self.assertEqual(
            totals["ead"], sum(cp["ead"] for cp in counterparties)
        )
        self.assertEqual(
            totals["capital_requirement"],
            sum(cp["capital_requirement"] for cp in counterparties),
        )
        self.assertEqual(
            totals["risk_weighted_assets"],
            sum(cp["risk_weighted_assets"] for cp in counterparties),
        )

    def test_counterparty_without_facilities_keeps_zeros(self):
        result = self.call(
            counterparties=[
                {"id": "cp-a", "pd": 0.02},
                {"id": "cp-b", "pd": 0.05},
            ],
            facilities=[
                {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.45,
                 "ead": 1000.0, "maturity": 2.5}
            ],
        )
        cp_b = result["counterparties"][1]
        self.assertEqual(cp_b["id"], "cp-b")
        self.assertEqual(cp_b["ead"], 0.0)
        self.assertEqual(cp_b["capital_requirement"], 0.0)
        self.assertEqual(cp_b["risk_weighted_assets"], 0.0)

        cp_a = result["counterparties"][0]
        totals = result["portfolio_totals"]
        self.assertEqual(totals["ead"], cp_a["ead"])
        self.assertEqual(totals["capital_requirement"], cp_a["capital_requirement"])
        self.assertEqual(totals["risk_weighted_assets"], cp_a["risk_weighted_assets"])

    def test_currency_echo_and_default(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_extra_fields_are_ignored(self):
        result = self.call(
            unexpected="top-level",
            counterparties=[
                {"id": "c", "pd": 0.03, "rating": "AAA"}
            ],
            facilities=[
                {"id": "f", "counterparty_id": "c", "lgd": 0.45,
                 "ead": 1000.0, "maturity": 2.5, "note": "x"}
            ],
        )
        self.assertEqual(len(result["facilities"]), 1)
        self.assertEqual(result["facilities"][0]["counterparty_id"], "c")

    def test_integer_numbers_are_accepted(self):
        result = self.call(
            counterparties=[{"id": "c", "pd": 0.05}],
            facilities=[
                {"id": "f", "counterparty_id": "c", "lgd": 1,
                 "ead": 1000, "maturity": 2}
            ],
        )
        facility = result["facilities"][0]
        self.assertEqual(facility["ead"], 1000.0)
        self.assertEqual(
            facility["capital_requirement"], facility["ead"] * facility["K"]
        )

    def test_boundary_values_are_allowed(self):
        result = self.call(
            counterparties=[{"id": "c", "pd": 0.5}],
            facilities=[
                {"id": "f-zero", "counterparty_id": "c", "lgd": 0.0,
                 "ead": 1000.0, "maturity": 1},
                {"id": "f-full", "counterparty_id": "c", "lgd": 1.0,
                 "ead": 0.0, "maturity": 5},
            ],
        )
        zero, full = result["facilities"]
        self.assertEqual(zero["K"], 0.0)
        self.assertEqual(zero["capital_requirement"], 0.0)
        self.assertEqual(zero["risk_weighted_assets"], 0.0)
        # Zero exposure means zero capital regardless of K.
        self.assertEqual(full["capital_requirement"], 0.0)
        self.assertEqual(full["risk_weighted_assets"], 0.0)

    def test_extreme_pd_stays_finite(self):
        # The spec applies no PD floor: at very small PD the literal
        # maturity adjustment is well defined and the request succeeds as
        # long as every result is finite.
        result = self.call(
            counterparties=[{"id": "c", "pd": 1e-9}],
            facilities=[
                {"id": "f", "counterparty_id": "c", "lgd": 0.45,
                 "ead": 1000.0, "maturity": 2.5}
            ],
        )
        facility = result["facilities"][0]
        for key in ("R", "MA", "K", "capital_requirement", "risk_weighted_assets"):
            self.assertTrue(math.isfinite(facility[key]), key)
        self.assertEqual(
            facility["capital_requirement"], facility["ead"] * facility["K"]
        )
        self.assertEqual(
            facility["risk_weighted_assets"],
            12.5 * facility["capital_requirement"],
        )

    # ---- 400 invalid_request ----

    def test_unparseable_json_is_invalid_request(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.irb_capital(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_non_object_payload_is_invalid_request(self):
        for raw in (b"[1, 2]", b"42", b'"text"', b"null"):
            with self.assertRaises(InvalidRequest):
                self.service.irb_capital(raw)

    # ---- 422 invalid_input: structure, types, ranges ----

    def test_missing_and_empty_arrays(self):
        for key in ("counterparties", "facilities"):
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
            [{"pd": 0.02}],                       # missing id
            [{"id": "", "pd": 0.02}],             # empty id
            [{"id": "c"}],                        # missing pd
            [{"id": "c", "pd": 0.0}],             # pd must be > 0
            [{"id": "c", "pd": 1.0}],             # pd must be < 1
            [{"id": "c", "pd": -0.1}],            # below 0
            [{"id": "c", "pd": 1.1}],             # above 1
            [{"id": "c", "pd": True}],            # bool
            [{"id": "c", "pd": "0.02"}],          # string
            [{"id": "c", "pd": None}],            # null
            [{"id": "c", "pd": 10**400}],         # oversized integer
            ["not-an-object"],
        ]
        for counterparties in bad_counterparties:
            with self.assertRaises(InvalidInput, msg=repr(counterparties)):
                self.call(counterparties=counterparties)

    def test_facility_field_validation(self):
        bad_facilities = [
            [{"counterparty_id": "cp-a", "lgd": 0.45, "ead": 1.0,
              "maturity": 2.5}],                                  # missing id
            [{"id": "", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": 1.0, "maturity": 2.5}],                      # empty id
            [{"id": "f", "lgd": 0.45, "ead": 1.0,
              "maturity": 2.5}],                                  # missing cp id
            [{"id": "f", "counterparty_id": "", "lgd": 0.45,
              "ead": 1.0, "maturity": 2.5}],                      # empty cp id
            [{"id": "f", "counterparty_id": "cp-a",
              "ead": 1.0, "maturity": 2.5}],                      # missing lgd
            [{"id": "f", "counterparty_id": "cp-a", "lgd": -0.1,
              "ead": 1.0, "maturity": 2.5}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 1.1,
              "ead": 1.0, "maturity": 2.5}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": True,
              "ead": 1.0, "maturity": 2.5}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": "0.45",
              "ead": 1.0, "maturity": 2.5}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 10**400,
              "ead": 1.0, "maturity": 2.5}],
            [{"id": "f", "counterparty_id": "cp-a",
              "lgd": 0.45, "maturity": 2.5}],                    # missing ead
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": -1.0, "maturity": 2.5}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": False, "maturity": 2.5}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": "1", "maturity": 2.5}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": 10**400, "maturity": 2.5}],
            [{"id": "f", "counterparty_id": "cp-a",
              "lgd": 0.45, "ead": 1.0}],                         # missing maturity
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": 1.0, "maturity": 0.0}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": 1.0, "maturity": 6.0}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": 1.0, "maturity": 0.999}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": 1.0, "maturity": 5.0001}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": 1.0, "maturity": True}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": 1.0, "maturity": "2.5"}],
            [{"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
              "ead": 1.0, "maturity": 10**400}],
            ["not-an-object"],
        ]
        for facilities in bad_facilities:
            with self.assertRaises(InvalidInput, msg=repr(facilities)):
                self.call(facilities=facilities)

    def test_nan_and_infinity_tokens_are_invalid_input(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            raw = request_body().replace(b"0.02", token.encode(), 1)
            with self.assertRaises(InvalidInput):
                self.service.irb_capital(raw)

    def test_non_finite_computation_is_invalid_input(self):
        with self.assertRaises(InvalidInput):
            self.call(
                counterparties=[{"id": "c", "pd": 0.02}],
                facilities=[
                    {"id": "f-1", "counterparty_id": "c", "lgd": 1.0,
                     "ead": 1e308, "maturity": 2.5},
                    {"id": "f-2", "counterparty_id": "c", "lgd": 1.0,
                     "ead": 1e308, "maturity": 2.5},
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
                {"id": "c", "pd": 0.02},
                {"id": "c", "pd": 0.03},
            ],
        )
        self.assert_code(
            "duplicate_facility",
            facilities=[
                {"id": "f", "counterparty_id": "cp-a", "lgd": 0.45,
                 "ead": 1.0, "maturity": 2.5},
                {"id": "f", "counterparty_id": "cp-b", "lgd": 0.45,
                 "ead": 1.0, "maturity": 2.5},
            ],
        )

    def test_unknown_counterparty(self):
        self.assert_code(
            "unknown_counterparty",
            facilities=[
                {"id": "f", "counterparty_id": "ghost", "lgd": 0.45,
                 "ead": 1.0, "maturity": 2.5}
            ],
        )

    # ---- 413 request_too_large ----

    def test_size_limits(self):
        counterparties = [
            {"id": f"cp-{i}", "pd": 0.02} for i in range(1001)
        ]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(counterparties=counterparties)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

        facilities = [
            {"id": f"f-{i}", "counterparty_id": "cp-a", "lgd": 0.45,
             "ead": 1.0, "maturity": 2.5}
            for i in range(10001)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(facilities=facilities)

    def test_size_boundaries_are_allowed(self):
        counterparties = [
            {"id": f"cp-{i}", "pd": 0.02} for i in range(1000)
        ]
        facilities = [
            {"id": "f-1", "counterparty_id": "cp-0", "lgd": 0.45,
             "ead": 1.0, "maturity": 2.5}
        ]
        result = self.call(counterparties=counterparties, facilities=facilities)
        self.assertEqual(len(result["counterparties"]), 1000)

        facilities = [
            {"id": f"f-{i}", "counterparty_id": "cp-a", "lgd": 0.45,
             "ead": 1.0, "maturity": 2.5}
            for i in range(10000)
        ]
        result = self.call(facilities=facilities)
        self.assertEqual(len(result["facilities"]), 10000)


if __name__ == "__main__":
    unittest.main()
