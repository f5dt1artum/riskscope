import http.client
import json
import math
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


PATH = "/market-risk/var-backtest-validation"


def obs(date, var=1.0, pnl=0.0):
    return {"date": date, "var": var, "realized_pnl": pnl}


def breach_rows(flags, var=1.0):
    return [
        obs(f"d{i:02d}", var=var, pnl=(-2.0 if flag else 0.0))
        for i, flag in enumerate(flags)
    ]


def request_body(**overrides):
    body = {
        "confidence": 0.95,
        "observations": [
            {"date": "2024-01-01", "var": 10.0, "realized_pnl": 5.0},
            {"date": "2024-01-02", "var": 10.0, "realized_pnl": -15.0},
            {"date": "2024-01-03", "var": 10.0, "realized_pnl": -10.0},
            {"date": "2024-01-04", "var": 1.0, "realized_pnl": -1.0000001},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


def christoffersen(flags, confidence, significance=0.05):
    """Independent double-precision reference for the new statistics."""
    n = len(flags)
    x = sum(flags)
    n00 = n01 = n10 = n11 = 0
    for previous, current in zip(flags, flags[1:]):
        if not previous and not current:
            n00 += 1
        elif not previous and current:
            n01 += 1
        elif previous and not current:
            n10 += 1
        else:
            n11 += 1

    def term(count, probability):
        return 0.0 if count == 0 else count * math.log(probability)

    q = (n01 + n11) / (n - 1)
    ln_l0 = term(n00 + n10, 1.0 - q) + term(n01 + n11, q)
    q0 = n01 / (n00 + n01) if n00 + n01 > 0 else None
    q1 = n11 / (n10 + n11) if n10 + n11 > 0 else None
    ln_l1 = 0.0
    if q0 is not None:
        ln_l1 += term(n00, 1.0 - q0) + term(n01, q0)
    if q1 is not None:
        ln_l1 += term(n10, 1.0 - q1) + term(n11, q1)
    lr_ind = max(2.0 * (ln_l1 - ln_l0), 0.0)
    ind_pv = math.erfc(math.sqrt(lr_ind / 2.0))

    # Kupiec POF, same formula the frozen baseline uses.
    p = 1.0 - confidence
    qk = x / n
    lr_kup = max(
        -2.0
        * (
            term(n - x, 1.0 - p) + term(x, p)
            - term(n - x, 1.0 - qk) - term(x, qk)
        ),
        0.0,
    )
    cc_lr = lr_kup + lr_ind
    cc_pv = math.exp(-cc_lr / 2.0)

    return {
        "counts": (n00, n01, n10, n11),
        "q": q,
        "q0": q0,
        "q1": q1,
        "lr_ind": lr_ind,
        "ind_pv": ind_pv,
        "ind_accepted": ind_pv >= significance,
        "cc_lr": cc_lr,
        "cc_pv": cc_pv,
        "cc_accepted": cc_pv >= significance,
    }


class VarBacktestValidationTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.var_backtest_validation(request_body(**overrides))

    def assert_matches_reference(self, flags, confidence=0.95, **body_overrides):
        result = self.service.var_backtest_validation(
            request_body(
                confidence=confidence, observations=breach_rows(flags), **body_overrides
            )
        )
        expected = christoffersen(flags, confidence)
        ind, cc = result["independence"], result["conditional_coverage"]
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            expected["counts"],
        )
        self.assertAlmostEqual(ind["q"], expected["q"], delta=1e-12)
        if expected["q0"] is None:
            self.assertIsNone(ind["q0"])
        else:
            self.assertAlmostEqual(ind["q0"], expected["q0"], delta=1e-12)
        if expected["q1"] is None:
            self.assertIsNone(ind["q1"])
        else:
            self.assertAlmostEqual(ind["q1"], expected["q1"], delta=1e-12)
        self.assertTrue(
            math.isclose(ind["lr_statistic"], expected["lr_ind"], abs_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(ind["p_value"], expected["ind_pv"], abs_tol=1e-12)
        )
        self.assertEqual(ind["accepted"], expected["ind_accepted"])
        self.assertTrue(math.isclose(cc["lr_statistic"], expected["cc_lr"], abs_tol=1e-12))
        self.assertTrue(
            math.isclose(cc["p_value"], expected["cc_pv"], rel_tol=1e-12, abs_tol=1e-12)
        )
        self.assertEqual(cc["accepted"], expected["cc_accepted"])
        return result

    # ---- response shape ----

    def test_response_extends_var_backtest_response(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["significance"], 0.05)
        self.assertEqual(result["observation_count"], 4)
        self.assertEqual(result["breach_count"], 2)
        self.assertEqual(result["breach_rate"], 0.5)
        self.assertEqual(len(result["observations"]), 4)
        self.assertIn("kupiec", result)
        for key in ("n00", "n01", "n10", "n11"):
            self.assertIn(key, result)
            self.assertIsInstance(result[key], int)
        for name, extra_keys in (
            ("independence", ("q", "q0", "q1")),
            ("conditional_coverage", ()),
        ):
            section = result[name]
            self.assertEqual(
                set(section), {"lr_statistic", "p_value", "accepted", *extra_keys}
            )
            self.assertIsInstance(section["lr_statistic"], float)
            self.assertIsInstance(section["p_value"], float)
            self.assertIsInstance(section["accepted"], bool)

    def test_detail_rows_unchanged_and_in_input_order(self):
        result = self.call()
        self.assertEqual(
            [row["date"] for row in result["observations"]],
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        )
        self.assertEqual(
            result["observations"][0],
            {"date": "2024-01-01", "var": 10.0, "realized_pnl": 5.0,
             "loss": -5.0, "breach": False},
        )
        # Equality loss == var does not breach; strict excess does.
        self.assertFalse(result["observations"][2]["breach"])
        self.assertTrue(result["observations"][3]["breach"])

    def test_kupiec_section_matches_standalone_endpoint(self):
        standalone = self.service.var_backtest(request_body())
        result = self.call()
        self.assertEqual(result["kupiec"], standalone["kupiec"])
        self.assertEqual(result["observations"], standalone["observations"])
        self.assertEqual(result["breach_count"], standalone["breach_count"])
        self.assertEqual(result["breach_rate"], standalone["breach_rate"])

    # ---- transition counting ----

    def test_transition_counts_default_case(self):
        # breach flags: [False, True, False, True]
        # transitions: 0->1, 1->0, 0->1
        result = self.call()
        self.assertEqual((result["n00"], result["n01"], result["n10"], result["n11"]),
                         (0, 2, 1, 0))
        self.assertAlmostEqual(result["independence"]["q"], 2 / 3)
        self.assertEqual(result["independence"]["q0"], 1.0)
        self.assertEqual(result["independence"]["q1"], 0.0)

    def test_transition_counts_use_input_order_not_date_order(self):
        # Deliberately unordered dates; counts follow the input sequence.
        observations = [
            obs("2024-03-01", pnl=-2.0),  # breach
            obs("2024-01-01", pnl=0.0),   # no breach
            obs("2024-02-01", pnl=-2.0),  # breach
        ]
        result = self.service.var_backtest_validation(
            request_body(observations=observations)
        )
        # 1->0, 0->1
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            (0, 1, 1, 0),
        )
        self.assertEqual(
            [row["date"] for row in result["observations"]],
            ["2024-03-01", "2024-01-01", "2024-02-01"],
        )

    def test_no_breaches_q1_is_null(self):
        result = self.assert_matches_reference([0, 0, 0, 0, 0])
        ind = result["independence"]
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            (4, 0, 0, 0),
        )
        self.assertEqual(ind["q"], 0.0)
        self.assertEqual(ind["q0"], 0.0)
        self.assertIsNone(ind["q1"])
        self.assertEqual(ind["lr_statistic"], 0.0)
        self.assertEqual(ind["p_value"], 1.0)
        self.assertTrue(ind["accepted"])

    def test_all_breaches_q0_is_null(self):
        result = self.assert_matches_reference([1, 1, 1, 1])
        ind = result["independence"]
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            (0, 0, 0, 3),
        )
        self.assertEqual(ind["q"], 1.0)
        self.assertIsNone(ind["q0"])
        self.assertEqual(ind["q1"], 1.0)
        self.assertEqual(ind["lr_statistic"], 0.0)
        self.assertEqual(ind["p_value"], 1.0)
        # Independence is trivially satisfied; unconditional coverage fails.
        self.assertTrue(ind["accepted"])
        self.assertFalse(result["conditional_coverage"]["accepted"])

    def test_q1_null_when_state_one_never_a_predecessor(self):
        # Single breach on the final day: state 1 is never a predecessor.
        result = self.assert_matches_reference([0, 0, 0, 1])
        ind = result["independence"]
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            (2, 1, 0, 0),
        )
        self.assertIsNone(ind["q1"])
        self.assertAlmostEqual(ind["q0"], 1 / 3)
        # With q == q0 the two models coincide: LRind is exactly zero.
        self.assertEqual(ind["lr_statistic"], 0.0)
        self.assertEqual(ind["p_value"], 1.0)

    def test_q0_null_when_state_zero_never_a_predecessor(self):
        # Breach on every day but the last: state 0 is never a predecessor.
        result = self.assert_matches_reference([1, 1, 1, 0])
        ind = result["independence"]
        self.assertEqual(
            (result["n00"], result["n01"], result["n10"], result["n11"]),
            (0, 0, 1, 2),
        )
        self.assertIsNone(ind["q0"])
        self.assertEqual(ind["q1"], 2 / 3)

    # ---- statistics vs the independent reference ----

    def test_clustered_breaches_do_not_reject_independence(self):
        flags = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
        result = self.assert_matches_reference(flags)
        # Clustering means the conditional breach probabilities are close
        # to each other; the independence statistic stays small.
        self.assertGreater(result["independence"]["p_value"], 0.05)
        self.assertTrue(result["independence"]["accepted"])
        # Conditional coverage still rejects the 5% tail (5/10 breaches).
        self.assertFalse(result["conditional_coverage"]["accepted"])

    def test_alternating_breaches_reject_independence(self):
        flags = [1, 0, 1, 0, 1, 0]
        result = self.assert_matches_reference(flags)
        self.assertLess(result["independence"]["p_value"], 0.05)
        self.assertFalse(result["independence"]["accepted"])

    def test_conditional_coverage_statistic_is_sum(self):
        flags = [0, 1, 0, 1, 0, 0, 1, 0]
        result = self.assert_matches_reference(flags)
        self.assertAlmostEqual(
            result["conditional_coverage"]["lr_statistic"],
            result["kupiec"]["lr_statistic"]
            + result["independence"]["lr_statistic"],
            delta=1e-12,
        )
        expected_pv = math.exp(
            -result["conditional_coverage"]["lr_statistic"] / 2.0
        )
        self.assertAlmostEqual(
            result["conditional_coverage"]["p_value"], expected_pv, delta=1e-12
        )

    def test_accepted_boundary_is_p_value_equals_significance(self):
        flags = [0, 1, 0, 1]
        base = self.assert_matches_reference(flags)
        pv = base["independence"]["p_value"]
        below = self.service.var_backtest_validation(
            request_body(observations=breach_rows(flags), significance=pv + 1e-9)
        )
        equal = self.service.var_backtest_validation(
            request_body(observations=breach_rows(flags), significance=pv)
        )
        self.assertFalse(below["independence"]["accepted"])
        self.assertTrue(equal["independence"]["accepted"])

        cc_pv = base["conditional_coverage"]["p_value"]
        cc_below = self.service.var_backtest_validation(
            request_body(observations=breach_rows(flags), significance=cc_pv + 1e-12)
        )
        cc_equal = self.service.var_backtest_validation(
            request_body(observations=breach_rows(flags), significance=cc_pv)
        )
        self.assertFalse(cc_below["conditional_coverage"]["accepted"])
        self.assertTrue(cc_equal["conditional_coverage"]["accepted"])

    def test_two_observations_all_transition_cells(self):
        # n-1 == 1 transitions; each of the four cells is visited alone.
        for flags, counts in (
            ([0, 0], (1, 0, 0, 0)),
            ([0, 1], (0, 1, 0, 0)),
            ([1, 0], (0, 0, 1, 0)),
            ([1, 1], (0, 0, 0, 1)),
        ):
            with self.subTest(flags=flags):
                result = self.assert_matches_reference(flags)
                self.assertEqual(
                    (result["n00"], result["n01"], result["n10"], result["n11"]),
                    counts,
                )

    def test_exactly_10000_observations_allowed(self):
        observations = [
            obs(f"d{i:05d}", pnl=(-2.0 if i % 1000 == 0 else 0.0))
            for i in range(10000)
        ]
        result = self.service.var_backtest_validation(
            request_body(observations=observations)
        )
        self.assertEqual(result["observation_count"], 10000)
        self.assertEqual(
            result["n00"] + result["n01"] + result["n10"] + result["n11"], 9999
        )
        self.assertTrue(math.isfinite(result["independence"]["lr_statistic"]))
        self.assertTrue(math.isfinite(result["conditional_coverage"]["p_value"]))

    def test_result_is_standard_json_serializable(self):
        result = self.call()
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)
        roundtrip = json.loads(encoded)
        self.assertEqual(roundtrip["n11"], 0)
        # Null q-values round-trip as JSON null.
        null_case = self.service.var_backtest_validation(
            request_body(observations=breach_rows([0, 0, 0]))
        )
        self.assertIsNone(json.loads(json.dumps(null_case))["independence"]["q1"])

    # ---- error semantics mirror var-backtest ----

    def test_malformed_json(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.var_backtest_validation(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_top_level_non_object(self):
        for body in (b"[]", b"42", b"null", b'"x"'):
            with self.subTest(body=body):
                with self.assertRaises(InvalidRequest):
                    self.service.var_backtest_validation(body)

    def test_field_type_and_range_errors(self):
        with self.assertRaises(InvalidInput):
            self.call(confidence="0.95")
        with self.assertRaises(InvalidInput):
            self.call(confidence=1.0)
        with self.assertRaises(InvalidInput):
            self.call(significance=0)
        with self.assertRaises(InvalidInput):
            self.call(currency="")
        with self.assertRaises(InvalidInput):
            self.call(observations=[obs("a", var=-0.01)])
        with self.assertRaises(InvalidInput):
            self.call(observations=[{"date": "a", "var": 1.0}])
        with self.assertRaises(InvalidInput):
            self.call(observations="nope")
        with self.assertRaises(InvalidInput):
            self.call(observations=[obs("a"), ])
        with self.assertRaises(InvalidInput):
            self.call(observations=[obs("a"), "x"])

    def test_nan_and_infinity_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                body = (
                    b'{"confidence":0.95,"observations":['
                    b'{"date":"a","var":1.0,"realized_pnl":' + token.encode() + b'},'
                    b'{"date":"b","var":1.0,"realized_pnl":0.0}]}'
                )
                with self.assertRaises(InvalidInput):
                    self.service.var_backtest_validation(body)

    def test_duplicate_observation(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(observations=[obs("d", pnl=-2.0), obs("d")])
        self.assertEqual(ctx.exception.code, "duplicate_observation")

    def test_too_many_observations(self):
        observations = [obs(f"d{i}") for i in range(10001)]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(observations=observations)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")


class VarBacktestValidationHttpTest(unittest.TestCase):
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

    def post(self, body, path=PATH):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        conn.close()
        return response.status, payload

    def test_route_ok(self):
        status, payload = self.post(request_body())
        self.assertEqual(status, 200)
        self.assertEqual(
            (payload["n00"], payload["n01"], payload["n10"], payload["n11"]),
            (0, 2, 1, 0),
        )
        self.assertEqual(
            set(payload["independence"]),
            {"lr_statistic", "p_value", "accepted", "q", "q0", "q1"},
        )
        self.assertEqual(
            set(payload["conditional_coverage"]),
            {"lr_statistic", "p_value", "accepted"},
        )

    def test_original_route_unchanged(self):
        status, payload = self.post(
            request_body(), path="/market-risk/var-backtest"
        )
        self.assertEqual(status, 200)
        self.assertNotIn("independence", payload)
        self.assertNotIn("n00", payload)
        self.assertIn("kupiec", payload)

    def test_invalid_json_returns_400_without_partial_results(self):
        status, payload = self.post(b"nonsense")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_request")
        self.assertNotIn("independence", payload)

    def test_invalid_input_returns_422(self):
        status, payload = self.post(
            json.dumps({"confidence": 2.0, "observations": [obs("a"), obs("b")]}).encode()
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "invalid_input")
        self.assertNotIn("n00", payload)

    def test_duplicate_returns_422(self):
        body = json.dumps(
            {"confidence": 0.95, "observations": [obs("d"), obs("d")]}
        ).encode()
        status, payload = self.post(body)
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "duplicate_observation")

    def test_too_large_returns_413(self):
        body = json.dumps(
            {"confidence": 0.95,
             "observations": [obs(f"d{i}") for i in range(10001)]}
        ).encode()
        status, payload = self.post(body)
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "request_too_large")


if __name__ == "__main__":
    unittest.main()
