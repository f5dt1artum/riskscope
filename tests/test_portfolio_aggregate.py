import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

from riskscope.server import Handler
from riskscope.service import (
    InvalidInput,
    InvalidRequest,
    RequestTooLarge,
    Service,
)


def request_body(**overrides):
    body = {
        "reporting_currency": "USD",
        "fx_rates": {"EUR": 1.1, "JPY": 0.01},
        "positions": [
            {
                "id": "p1",
                "book": "book-a",
                "currency": "USD",
                "market_value": 100.0,
                "sensitivities": {"eq": 10.0, "ir": -2.0},
            },
            {
                "id": "p2",
                "book": "book-b",
                "currency": "EUR",
                "market_value": 50.0,
                "sensitivities": {"eq": 5.0},
            },
            {
                "id": "p3",
                "book": "book-a",
                "currency": "JPY",
                "market_value": -200.0,
                "sensitivities": {"ir": 3.0, "fx": 7.0},
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

    def by(self, details, key, name):
        matches = [d for d in details if d[key] == name]
        self.assertEqual(len(matches), 1)
        return matches[0]

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["reporting_currency"], "USD")
        self.assertEqual(result["position_count"], 3)
        self.assertEqual([c["currency"] for c in result["currencies"]], [
            "USD",
            "EUR",
            "JPY",
        ])
        self.assertEqual([b["book"] for b in result["books"]], ["book-a", "book-b"])

        usd = self.by(result["currencies"], "currency", "USD")
        eur = self.by(result["currencies"], "currency", "EUR")
        jpy = self.by(result["currencies"], "currency", "JPY")
        self.assertEqual(usd["fx_rate"], 1.0)
        self.assertEqual(eur["fx_rate"], 1.1)
        self.assertEqual(jpy["fx_rate"], 0.01)
        self.assertEqual(usd["position_count"], 1)
        self.assertEqual(eur["position_count"], 1)
        self.assertEqual(jpy["position_count"], 1)
        # Converted values are the plain float products, negatives preserved.
        self.assertEqual(usd["converted_market_value"], 100.0 * 1.0)
        self.assertEqual(eur["converted_market_value"], 50.0 * 1.1)
        self.assertEqual(jpy["converted_market_value"], -200.0 * 0.01)

        book_a = self.by(result["books"], "book", "book-a")
        book_b = self.by(result["books"], "book", "book-b")
        self.assertEqual(book_a["position_count"], 2)
        self.assertEqual(book_b["position_count"], 1)
        self.assertAlmostEqual(book_a["converted_market_value"], 98.0, places=10)
        self.assertAlmostEqual(book_b["converted_market_value"], 55.0, places=10)
        self.assertAlmostEqual(
            result["portfolio_totals"]["converted_market_value"], 153.0, places=10
        )

    def test_factors_follow_first_appearance_and_zero_sums_kept(self):
        result = self.call()
        # First position establishes eq, ir; the JPY position adds fx.
        expected_factors = ["eq", "ir", "fx"]
        totals = result["portfolio_totals"]["converted_sensitivities"]
        self.assertEqual(list(totals), expected_factors)
        self.assertEqual(totals["eq"], 10.0 + 5.0 * 1.1)
        self.assertEqual(totals["ir"], -2.0 + 3.0 * 0.01)
        self.assertEqual(totals["fx"], 7.0 * 0.01)
        # Every detail carries every factor, even absent ones.
        eur = self.by(result["currencies"], "currency", "EUR")
        self.assertEqual(list(eur["converted_sensitivities"]), expected_factors)
        self.assertEqual(eur["converted_sensitivities"]["ir"], 0.0)
        self.assertEqual(eur["converted_sensitivities"]["fx"], 0.0)
        # A genuinely zero aggregate keeps the factor.
        result_zero = self.service.portfolio_aggregate(
            json.dumps(
                {
                    "reporting_currency": "USD",
                    "fx_rates": {},
                    "positions": [
                        {
                            "id": "p1",
                            "book": "b",
                            "currency": "USD",
                            "market_value": 0.0,
                            "sensitivities": {"eq": 0.0},
                        }
                    ],
                }
            ).encode("utf-8")
        )
        self.assertEqual(
            result_zero["portfolio_totals"]["converted_sensitivities"],
            {"eq": 0.0},
        )

    def test_negative_values_are_preserved(self):
        result = self.call()
        self.assertEqual(
            self.by(result["currencies"], "currency", "JPY")[
                "converted_market_value"
            ],
            -200.0 * 0.01,
        )
        self.assertLess(
            result["portfolio_totals"]["converted_market_value"], 154.0
        )
        # The JPY position's ir is positive; the sign-preservation of a
        # converted negative sensitivity is checked via its market value.
        self.assertLess(
            self.by(result["currencies"], "currency", "JPY")[
                "converted_market_value"
            ],
            0.0,
        )

    def test_reporting_currency_rate_rules(self):
        # Implicit rate when fx_rates omits the reporting currency.
        result = self.service.portfolio_aggregate(
            json.dumps(
                {
                    "reporting_currency": "USD",
                    "fx_rates": {"EUR": 1.1},
                    "positions": [
                        {
                            "id": "p1",
                            "book": "b",
                            "currency": "USD",
                            "market_value": 10.0,
                            "sensitivities": {"eq": 1.0},
                        },
                        {
                            "id": "p2",
                            "book": "b",
                            "currency": "EUR",
                            "market_value": 10.0,
                            "sensitivities": {"eq": 1.0},
                        },
                    ],
                }
            ).encode("utf-8")
        )
        self.assertEqual(
            self.by(result["currencies"], "currency", "USD")["fx_rate"], 1.0
        )
        # An explicit numeric 1 (int or float) is accepted.
        for explicit in (1, 1.0):
            result = self.call(fx_rates={"EUR": 1.1, "JPY": 0.01, "USD": explicit})
            self.assertEqual(
                self.by(result["currencies"], "currency", "USD")["fx_rate"], 1.0
            )

    def test_unreferenced_fx_rates_are_ignored(self):
        result = self.call(
            fx_rates={"EUR": 1.1, "JPY": 0.01, "XXX": -5.0, "GBP": "nope"}
        )
        self.assertEqual(result["position_count"], 3)
        self.assertNotIn("XXX", [c["currency"] for c in result["currencies"]])

    def test_detail_sums_equal_portfolio_totals_exactly(self):
        # Powers that invite binary rounding: the reconciliation still keeps
        # the layer identities exact.
        positions = []
        for i, (currency, rate) in enumerate(
            [("EUR", 0.3), ("JPY", 0.7), ("USD", 1.0), ("EUR", 1.1)]
        ):
            positions.append(
                {
                    "id": f"p{i}",
                    "book": f"book-{i % 3}",
                    "currency": currency,
                    "market_value": 0.1,
                    "sensitivities": {"eq": 0.1, "ir": 0.2},
                }
            )
        result = self.service.portfolio_aggregate(
            request_body(fx_rates={"EUR": 1.1, "JPY": 0.01}, positions=positions)
        )
        totals = result["portfolio_totals"]
        self.assertEqual(
            sum(c["converted_market_value"] for c in result["currencies"]),
            totals["converted_market_value"],
        )
        self.assertEqual(
            sum(b["converted_market_value"] for b in result["books"]),
            totals["converted_market_value"],
        )
        for factor in ("eq", "ir"):
            self.assertEqual(
                sum(
                    c["converted_sensitivities"][factor]
                    for c in result["currencies"]
                ),
                totals["converted_sensitivities"][factor],
            )
            self.assertEqual(
                sum(
                    b["converted_sensitivities"][factor] for b in result["books"]
                ),
                totals["converted_sensitivities"][factor],
            )
        # Position counts partition the total too.
        self.assertEqual(sum(c["position_count"] for c in result["currencies"]), 4)
        self.assertEqual(sum(b["position_count"] for b in result["books"]), 4)

    def test_extra_fields_ignored(self):
        body = json.loads(request_body())
        body["extra"] = "ignored"
        body["positions"][0]["venue"] = "x"
        result = self.service.portfolio_aggregate(
            json.dumps(body).encode("utf-8")
        )
        self.assertEqual(result["position_count"], 3)

    # ---- parse failures ----

    def test_invalid_json_is_invalid_request(self):
        with self.assertRaises(InvalidRequest):
            self.service.portfolio_aggregate(b"{not json")

    def test_top_level_array_is_invalid_request(self):
        with self.assertRaises(InvalidRequest):
            self.service.portfolio_aggregate(b"[1, 2]")

    # ---- semantic validation ----

    def test_missing_or_empty_fields(self):
        cases = [
            {},
            {"fx_rates": {}, "positions": []},
            {"reporting_currency": "", "fx_rates": {}, "positions": []},
            {"reporting_currency": None, "fx_rates": {}, "positions": []},
            {"reporting_currency": "USD", "positions": []},
            {"reporting_currency": "USD", "fx_rates": {}},
            {"reporting_currency": "USD", "fx_rates": []},
        ]
        for body in cases:
            with self.assertRaises(InvalidInput, msg=repr(body)):
                self.service.portfolio_aggregate(
                    json.dumps(body).encode("utf-8")
                )

    def test_malformed_positions(self):
        good_position = {
            "id": "p1",
            "book": "b",
            "currency": "USD",
            "market_value": 1.0,
            "sensitivities": {"eq": 1.0},
        }
        mutations = [
            [],
            [None],
            ["x"],
            [{**good_position, "id": ""}],
            [{**good_position, "id": None}],
            [{**good_position, "book": ""}],
            [{**good_position, "currency": ""}],
            [{**good_position, "market_value": None}],
            [{**good_position, "sensitivities": {}}],
            [{**good_position, "sensitivities": None}],
            [{**good_position, "sensitivities": {"": 1.0}}],
            [{**good_position, "sensitivities": {"eq": None}}],
        ]
        for positions in mutations:
            body = {
                "reporting_currency": "USD",
                "fx_rates": {},
                "positions": positions,
            }
            with self.assertRaises(InvalidInput, msg=repr(positions)):
                self.service.portfolio_aggregate(
                    json.dumps(body).encode("utf-8")
                )

    def test_booleans_nan_infinity_and_oversized_ints_rejected(self):
        template = (
            '{"reporting_currency": "USD", "fx_rates": {"EUR": 1.1}, '
            '"positions": [{"id": "p1", "book": "b", "currency": "EUR", '
            '"market_value": __MV__, "sensitivities": {"eq": __SENS__}}]}'
        )

        def body(mv, sens="1.0"):
            return template.replace("__MV__", mv).replace("__SENS__", sens).encode(
                "utf-8"
            )

        for token in ("true", "false"):
            with self.assertRaises(InvalidInput, msg=token):
                self.service.portfolio_aggregate(body(token))
            with self.assertRaises(InvalidInput, msg="sens " + token):
                self.service.portfolio_aggregate(body("1.0", token))
        # Booleans are also rejected as fx rates.
        with self.assertRaises(InvalidInput):
            self.service.portfolio_aggregate(
                body("1.0").replace(b'"EUR": 1.1', b'"EUR": true')
            )
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.assertRaises(InvalidInput, msg=token):
                self.service.portfolio_aggregate(body(token))
        huge = {
            "reporting_currency": "USD",
            "fx_rates": {"EUR": 1.1},
            "positions": [
                {
                    "id": "p1",
                    "book": "b",
                    "currency": "EUR",
                    "market_value": 10**400,
                    "sensitivities": {"eq": 1.0},
                }
            ],
        }
        with self.assertRaises(InvalidInput):
            self.service.portfolio_aggregate(json.dumps(huge).encode("utf-8"))

    def test_duplicate_position_id(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(
                positions=[
                    {
                        "id": "same",
                        "book": "b",
                        "currency": "USD",
                        "market_value": 1.0,
                        "sensitivities": {"eq": 1.0},
                    },
                    {
                        "id": "same",
                        "book": "b",
                        "currency": "USD",
                        "market_value": 2.0,
                        "sensitivities": {"eq": 2.0},
                    },
                ]
            )
        self.assertEqual(ctx.exception.code, "duplicate_position")

    def test_missing_fx_rate_for_referenced_currency(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.service.portfolio_aggregate(
                json.dumps(
                    {
                        "reporting_currency": "USD",
                        "fx_rates": {"JPY": 0.01},
                        "positions": [
                            {
                                "id": "p1",
                                "book": "b",
                                "currency": "EUR",
                                "market_value": 1.0,
                                "sensitivities": {"eq": 1.0},
                            }
                        ],
                    }
                ).encode("utf-8")
            )
        self.assertEqual(ctx.exception.code, "missing_fx_rate")

    def test_referenced_fx_rate_must_be_finite_positive(self):
        for rate in (0.0, -1.0, "1.1", True):
            with self.assertRaises(InvalidInput, msg=repr(rate)):
                self.call(fx_rates={"EUR": rate, "JPY": 0.01})

    def test_reporting_currency_explicit_rate_must_equal_one(self):
        for rate in (0.999, 1.0001, -1.0, 0.0):
            with self.assertRaises(InvalidInput, msg=repr(rate)):
                self.call(fx_rates={"EUR": 1.1, "JPY": 0.01, "USD": rate})

    def test_non_finite_computation_fails(self):
        with self.assertRaises(InvalidInput):
            self.service.portfolio_aggregate(
                json.dumps(
                    {
                        "reporting_currency": "USD",
                        "fx_rates": {"EUR": 1e308},
                        "positions": [
                            {
                                "id": "p1",
                                "book": "b",
                                "currency": "EUR",
                                "market_value": 1e308,
                                "sensitivities": {"eq": 1.0},
                            }
                        ],
                    }
                ).encode("utf-8")
            )

    # ---- size limits ----

    def test_positions_boundary(self):
        def build(n):
            return {
                "reporting_currency": "USD",
                "fx_rates": {},
                "positions": [
                    {
                        "id": f"p{i}",
                        "book": "b",
                        "currency": "USD",
                        "market_value": 1.0,
                        "sensitivities": {"eq": 1.0},
                    }
                    for i in range(n)
                ],
            }

        result = self.service.portfolio_aggregate(
            json.dumps(build(10000)).encode("utf-8")
        )
        self.assertEqual(result["position_count"], 10000)
        with self.assertRaises(RequestTooLarge):
            self.service.portfolio_aggregate(
                json.dumps(build(10001)).encode("utf-8")
            )

    def test_fx_rates_boundary(self):
        def build(n):
            # n entries in total: n-1 unreferenced currencies plus USD.
            rates = {f"C{i}": 1.0 + i * 1e-12 for i in range(n - 1)}
            rates["USD"] = 1.0
            return {
                "reporting_currency": "USD",
                "fx_rates": rates,
                "positions": [
                    {
                        "id": "p1",
                        "book": "b",
                        "currency": "USD",
                        "market_value": 1.0,
                        "sensitivities": {"eq": 1.0},
                    }
                ],
            }

        self.service.portfolio_aggregate(
            json.dumps(build(1000)).encode("utf-8")
        )
        with self.assertRaises(RequestTooLarge):
            self.service.portfolio_aggregate(
                json.dumps(build(1001)).encode("utf-8")
            )

    def test_sensitivity_entries_boundary(self):
        def build(positions_n, factors_n):
            factors = [f"f{i}" for i in range(factors_n)]
            return {
                "reporting_currency": "USD",
                "fx_rates": {},
                "positions": [
                    {
                        "id": f"p{i}",
                        "book": "b",
                        "currency": "USD",
                        "market_value": 1.0,
                        "sensitivities": {f: 1.0 for f in factors},
                    }
                    for i in range(positions_n)
                ],
            }

        self.service.portfolio_aggregate(
            json.dumps(build(5000, 200)).encode("utf-8")
        )
        with self.assertRaises(RequestTooLarge):
            self.service.portfolio_aggregate(
                json.dumps(build(5001, 200)).encode("utf-8")
            )


class PortfolioAggregateHttpTest(unittest.TestCase):
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

    def post(self, body, path="/portfolio-risk/aggregate"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_aggregate_route_ok(self):
        status, payload = self.post(request_body())
        self.assertEqual(status, 200)
        self.assertEqual(payload["reporting_currency"], "USD")
        self.assertEqual(payload["position_count"], 3)

    def test_invalid_json_returns_400_error_object(self):
        status, payload = self.post(b"nonsense")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")
        self.assertNotIn("position_count", payload)

    def test_missing_fx_rate_returns_422(self):
        body = json.dumps(
            {
                "reporting_currency": "USD",
                "fx_rates": {},
                "positions": [
                    {
                        "id": "p1",
                        "book": "b",
                        "currency": "EUR",
                        "market_value": 1.0,
                        "sensitivities": {"eq": 1.0},
                    }
                ],
            }
        ).encode("utf-8")
        status, payload = self.post(body)
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "missing_fx_rate")

    def test_too_large_returns_413(self):
        body = json.dumps(
            {
                "reporting_currency": "USD",
                "fx_rates": {},
                "positions": [
                    {
                        "id": f"p{i}",
                        "book": "b",
                        "currency": "USD",
                        "market_value": 1.0,
                        "sensitivities": {"eq": 1.0},
                    }
                    for i in range(10001)
                ],
            }
        ).encode("utf-8")
        status, payload = self.post(body)
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "request_too_large")

    def test_health_and_unknown_route_unchanged(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/healthz")
        response = conn.getresponse()
        health = json.loads(response.read().decode("utf-8"))
        conn.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(health["status"], "ok")

        status, payload = self.post(b"{}", path="/no-such-route")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
