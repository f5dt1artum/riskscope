import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from riskscope.server import Handler
from riskscope.service import (
    InvalidInput,
    InvalidRequest,
    Service,
)


def request_body(**overrides):
    body = {
        "currency": "USD",
        "buckets": [1, 7, 30],
        "cashflows": [
            {"id": "in-on-1", "day": 1, "amount": 100.0},
            {"id": "out-on-2", "day": 2, "amount": -40.0},
            {"id": "in-on-7", "day": 7, "amount": 50.0},
            {"id": "out-on-30", "day": 30, "amount": -200.0},
        ],
        "liquid_assets": [
            {"id": "cash", "market_value": 100.0, "haircut": 0.0, "available_day": 0},
            {"id": "bonds", "market_value": 200.0, "haircut": 0.1, "available_day": 7},
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

    def test_basic_bucketing_and_totals(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        buckets = result["buckets"]
        self.assertEqual([b["day"] for b in buckets], [1, 7, 30])

        # day 1 lands on bucket 1; day 2 on the first bucket >= 2 (7);
        # the remaining flows land exactly on their own buckets.
        self.assertEqual(buckets[0]["net_cashflow"], 100.0)
        self.assertEqual(buckets[1]["net_cashflow"], 10.0)   # -40 + 50
        self.assertEqual(buckets[2]["net_cashflow"], -200.0)

        self.assertEqual(buckets[0]["cumulative_net_cashflow"], 100.0)
        self.assertEqual(buckets[1]["cumulative_net_cashflow"], 110.0)
        self.assertEqual(buckets[2]["cumulative_net_cashflow"], -90.0)

        # Cash is available from the first bucket; bonds (200 * 0.9 = 180)
        # join from bucket 7 and stay available afterwards.
        self.assertEqual(buckets[0]["available_liquidity"], 100.0)
        self.assertEqual(buckets[1]["available_liquidity"], 280.0)
        self.assertEqual(buckets[2]["available_liquidity"], 280.0)

        self.assertEqual(buckets[0]["surplus"], 200.0)
        self.assertEqual(buckets[1]["surplus"], 390.0)
        self.assertEqual(buckets[2]["surplus"], 190.0)
        self.assertEqual([b["required_funding"] for b in buckets], [0.0, 0.0, 0.0])
        self.assertIsNone(result["earliest_shortfall"])

    def test_shortfall_tracks_first_negative_surplus(self):
        result = self.call(
            cashflows=[
                {"id": "c1", "day": 1, "amount": -50.0},
                {"id": "c2", "day": 7, "amount": -100.0},
                {"id": "c3", "day": 30, "amount": -20.0},
            ],
            liquid_assets=[
                {"id": "a1", "market_value": 40.0, "haircut": 0.0, "available_day": 0},
                {"id": "a2", "market_value": 1000.0, "haircut": 0.0, "available_day": 30},
            ],
        )
        buckets = result["buckets"]
        # Bucket 1: -50 + 40 = -10 -> funding 10; earlier buckets win on ties.
        self.assertEqual(buckets[0]["surplus"], -10.0)
        self.assertEqual(buckets[0]["required_funding"], 10.0)
        # Bucket 7: -150 + 40 = -110; a later, deeper gap must not replace it.
        self.assertEqual(buckets[1]["surplus"], -110.0)
        self.assertEqual(buckets[1]["required_funding"], 110.0)
        # Bucket 30: -170 + 1040 = 870, recovered.
        self.assertEqual(buckets[2]["surplus"], 870.0)
        self.assertEqual(buckets[2]["required_funding"], 0.0)
        self.assertEqual(
            result["earliest_shortfall"],
            {"day": 1, "required_funding": 10.0},
        )

    def test_zero_surplus_is_not_a_shortfall(self):
        result = self.call(
            cashflows=[{"id": "c1", "day": 1, "amount": -10.0}],
            liquid_assets=[
                {"id": "a1", "market_value": 10.0, "haircut": 0.0, "available_day": 0}
            ],
        )
        self.assertEqual(result["buckets"][0]["surplus"], 0.0)
        self.assertEqual(result["buckets"][0]["required_funding"], 0.0)
        self.assertIsNone(result["earliest_shortfall"])

    def test_empty_buckets_keep_zero_values(self):
        result = self.call(
            cashflows=[{"id": "c1", "day": 30, "amount": 5.0}],
            liquid_assets=[],
        )
        buckets = result["buckets"]
        self.assertEqual(
            (
                buckets[0]["net_cashflow"],
                buckets[0]["cumulative_net_cashflow"],
                buckets[0]["available_liquidity"],
                buckets[0]["surplus"],
                buckets[0]["required_funding"],
            ),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(buckets[1]["cumulative_net_cashflow"], 0.0)
        self.assertEqual(buckets[2]["net_cashflow"], 5.0)

    def test_asset_availability_edges(self):
        # available_day equal to a bucket day lands on that bucket; a day past
        # the last bucket means the asset never contributes.
        result = self.call(
            cashflows=[{"id": "c1", "day": 1, "amount": 0.0}],
            liquid_assets=[
                {"id": "on-7", "market_value": 10.0, "haircut": 0.0, "available_day": 7},
                {"id": "after-8", "market_value": 10.0, "haircut": 0.0, "available_day": 8},
                {"id": "too-late", "market_value": 999.0, "haircut": 0.0, "available_day": 31},
            ],
        )
        buckets = result["buckets"]
        self.assertEqual(buckets[0]["available_liquidity"], 0.0)
        self.assertEqual(buckets[1]["available_liquidity"], 10.0)
        self.assertEqual(buckets[2]["available_liquidity"], 20.0)

    def test_haircut_boundaries(self):
        result = self.call(
            cashflows=[{"id": "c1", "day": 1, "amount": 0.0}],
            liquid_assets=[
                {"id": "full", "market_value": 100.0, "haircut": 0.0, "available_day": 0},
                {"id": "wiped", "market_value": 100.0, "haircut": 1.0, "available_day": 1},
            ],
        )
        buckets = result["buckets"]
        self.assertEqual(buckets[0]["available_liquidity"], 100.0)
        self.assertEqual(buckets[1]["available_liquidity"], 100.0)

    def test_defaults_and_currency_echo(self):
        body = json.loads(request_body())
        del body["currency"]
        del body["liquid_assets"]
        result = self.service.liquidity_gap(json.dumps(body).encode("utf-8"))
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(
            [b["available_liquidity"] for b in result["buckets"]], [0.0, 0.0, 0.0]
        )
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_integer_amounts_are_accepted(self):
        result = self.call(
            cashflows=[
                {"id": "c1", "day": 1, "amount": 10},
                {"id": "c2", "day": 7, "amount": -3},
            ],
            liquid_assets=[
                {"id": "a1", "market_value": 5, "haircut": 0, "available_day": 0}
            ],
        )
        buckets = result["buckets"]
        self.assertEqual(buckets[0]["net_cashflow"], 10.0)
        self.assertEqual(buckets[0]["surplus"], 15.0)
        self.assertEqual(buckets[1]["cumulative_net_cashflow"], 7.0)

    def test_extra_fields_are_ignored(self):
        result = self.call(
            unexpected="top-level",
            cashflows=[
                {"id": "c1", "day": 1, "amount": 1.0, "note": "x"},
            ],
            liquid_assets=[
                {
                    "id": "a1",
                    "market_value": 2.0,
                    "haircut": 0.0,
                    "available_day": 0,
                    "tier": "cash",
                }
            ],
        )
        self.assertEqual(result["buckets"][0]["surplus"], 3.0)

    # ---- 400 invalid_request ----

    def test_unparseable_json_is_invalid_request(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.liquidity_gap(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_non_object_payload_is_invalid_request(self):
        for raw in (b"[1, 2]", b"42", b'"text"', b"null"):
            with self.assertRaises(InvalidRequest):
                self.service.liquidity_gap(raw)

    # ---- 422 invalid_input ----

    def test_currency_must_be_nonempty_string(self):
        for bad in ("", 3, None, True):
            with self.assertRaises(InvalidInput, msg=repr(bad)):
                self.call(currency=bad)

    def test_buckets_validation(self):
        bad_buckets = [
            None,
            [],
            "not-a-list",
            [0],
            [-1],
            [1.5],
            [True],
            ["1"],
            [1, 1],       # duplicate
            [7, 1],       # not increasing
            [1, 30, 7],   # not increasing
            [1, None],
            [1, "x"],
        ]
        for buckets in bad_buckets:
            with self.assertRaises(InvalidInput, msg=repr(buckets)):
                self.call(buckets=buckets)

    def test_cashflow_day_must_fit_a_bucket(self):
        with self.assertRaises(InvalidInput):
            self.call(cashflows=[{"id": "c1", "day": 31, "amount": 1.0}])

    def test_cashflow_field_validation(self):
        base = [{"id": "c1", "day": 1, "amount": 1.0}]
        bad_cashflows = [
            None,
            [],
            "not-a-list",
            ["not-an-object"],
            [{"day": 1, "amount": 1.0}],          # missing id
            [{"id": "", "day": 1, "amount": 1.0}],
            [{"id": 7, "day": 1, "amount": 1.0}],
            [{"id": "c1", "amount": 1.0}],        # missing day
            [{"id": "c1", "day": 0, "amount": 1.0}],
            [{"id": "c1", "day": -1, "amount": 1.0}],
            [{"id": "c1", "day": 1.5, "amount": 1.0}],
            [{"id": "c1", "day": True, "amount": 1.0}],
            [{"id": "c1", "day": "1", "amount": 1.0}],
            [{"id": "c1", "day": 1}],             # missing amount
            [{"id": "c1", "day": 1, "amount": True}],
            [{"id": "c1", "day": 1, "amount": "1"}],
            [{"id": "c1", "day": 1, "amount": 10**400}],
        ]
        for cashflows in bad_cashflows:
            with self.assertRaises(InvalidInput, msg=repr(cashflows)):
                self.call(cashflows=cashflows)
        # Baseline itself must be valid.
        self.call(cashflows=base)

    def test_liquid_assets_validation(self):
        bad_assets = [
            [{"market_value": 1.0, "haircut": 0.0, "available_day": 0}],  # missing id
            [{"id": "", "market_value": 1.0, "haircut": 0.0, "available_day": 0}],
            [{"id": "a1", "haircut": 0.0, "available_day": 0}],           # missing mv
            [{"id": "a1", "market_value": -0.01, "haircut": 0.0, "available_day": 0}],
            [{"id": "a1", "market_value": True, "haircut": 0.0, "available_day": 0}],
            [{"id": "a1", "market_value": "1", "haircut": 0.0, "available_day": 0}],
            [{"id": "a1", "market_value": 10**400, "haircut": 0.0, "available_day": 0}],
            [{"id": "a1", "market_value": 1.0, "available_day": 0}],      # missing haircut
            [{"id": "a1", "market_value": 1.0, "haircut": -0.01, "available_day": 0}],
            [{"id": "a1", "market_value": 1.0, "haircut": 1.01, "available_day": 0}],
            [{"id": "a1", "market_value": 1.0, "haircut": True, "available_day": 0}],
            [{"id": "a1", "market_value": 1.0, "haircut": "0", "available_day": 0}],
            [{"id": "a1", "market_value": 1.0, "haircut": 0.0}],          # missing day
            [{"id": "a1", "market_value": 1.0, "haircut": 0.0, "available_day": -1}],
            [{"id": "a1", "market_value": 1.0, "haircut": 0.0, "available_day": 1.5}],
            [{"id": "a1", "market_value": 1.0, "haircut": 0.0, "available_day": True}],
            [{"id": "a1", "market_value": 1.0, "haircut": 0.0, "available_day": "0"}],
        ]
        for assets in bad_assets:
            with self.assertRaises(InvalidInput, msg=repr(assets)):
                self.call(liquid_assets=assets)

    def test_nan_and_infinity_tokens_are_invalid_input(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            raw = request_body().replace(b"100.0", token.encode(), 1)
            with self.assertRaises(InvalidInput):
                self.service.liquidity_gap(raw)

    def test_non_finite_computation_is_invalid_input(self):
        with self.assertRaises(InvalidInput):
            self.call(
                cashflows=[
                    {"id": "c1", "day": 1, "amount": 1e308},
                    {"id": "c2", "day": 1, "amount": 1e308},
                ],
                liquid_assets=[],
            )
        with self.assertRaises(InvalidInput):
            self.call(
                cashflows=[
                    {"id": "c1", "day": 1, "amount": -1e308},
                    {"id": "c2", "day": 7, "amount": -1e308},
                ],
                liquid_assets=[],
            )

    # ---- 422 duplicate_* ----

    def assert_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)
        self.assertEqual(ctx.exception.status, 422)

    def test_duplicate_cashflow_id(self):
        self.assert_code(
            "duplicate_cashflow",
            cashflows=[
                {"id": "dup", "day": 1, "amount": 1.0},
                {"id": "dup", "day": 7, "amount": 2.0},
            ],
        )

    def test_duplicate_asset_id(self):
        self.assert_code(
            "duplicate_asset",
            liquid_assets=[
                {"id": "dup", "market_value": 1.0, "haircut": 0.0, "available_day": 0},
                {"id": "dup", "market_value": 2.0, "haircut": 0.0, "available_day": 1},
            ],
        )


class LiquidityGapHttpTest(unittest.TestCase):
    """Route wiring: new endpoint works and frozen behavior stays intact."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def post(self, path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=body if isinstance(body, bytes) else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = json.loads(exc.read())
            exc.close()
            return exc.code, body

    def test_liquidity_gap_route(self):
        status, body = self.post(
            "/liquidity-risk/liquidity-gap",
            {
                "buckets": [1, 7],
                "cashflows": [{"id": "c1", "day": 1, "amount": -50.0}],
                "liquid_assets": [
                    {"id": "a1", "market_value": 40.0, "haircut": 0.0,
                     "available_day": 0}
                ],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["currency"], "USD")
        self.assertEqual(body["earliest_shortfall"],
                         {"day": 1, "required_funding": 10.0})

    def test_liquidity_gap_route_errors(self):
        status, body = self.post(
            "/liquidity-risk/liquidity-gap", b"{broken"
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_request")

        status, body = self.post(
            "/liquidity-risk/liquidity-gap",
            {"buckets": [1], "cashflows": [
                {"id": "x", "day": 1, "amount": 1.0},
                {"id": "x", "day": 1, "amount": 2.0},
            ]},
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "duplicate_cashflow")
        self.assertEqual(set(body), {"error"})

    def test_frozen_routes_remain_compatible(self):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/healthz"
        ) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(json.loads(resp.read())["status"], "ok")

        status, body = self.post("/no/such/route", {})
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "not_found")

        status, body = self.post(
            "/credit-risk/counterparty-exposure", b"{broken"
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
