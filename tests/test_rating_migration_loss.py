import json
import unittest

from riskscope.service import (
    InvalidInput,
    InvalidRequest,
    RequestTooLarge,
    Service,
)


def naive_matrix_multiply(a, b):
    n = len(a)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a[i][k] * b[k][j]
            result[i][j] = total
    return result


def request_body(**overrides):
    body = {
        "horizon": 2,
        "ratings": ["A", "B", "D"],
        "default_rating": "D",
        "transition_matrix": [
            [0.9, 0.09, 0.01],
            [0.1, 0.8, 0.1],
            [0.0, 0.0, 1.0],
        ],
        "counterparties": [
            {"id": "cp-1", "rating": "A", "ead": 1_000_000.0, "lgd": 0.4},
            {"id": "cp-2", "rating": "B", "ead": 500_000.0, "lgd": 0.5},
            {"id": "cp-3", "rating": "A", "ead": 250_000.0, "lgd": 1.0},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


class RatingMigrationLossTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.rating_migration_loss(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_values(self):
        result = self.call()
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["horizon"], 2)
        self.assertEqual(result["ratings"], ["A", "B", "D"])
        self.assertEqual(result["default_rating"], "D")
        self.assertEqual(
            set(result),
            {
                "currency",
                "horizon",
                "ratings",
                "default_rating",
                "horizon_transition_matrix",
                "counterparties",
                "rating_summaries",
                "portfolio_totals",
            },
        )

        matrix = result["horizon_transition_matrix"]
        self.assertEqual(len(matrix), 3)
        for row in matrix:
            self.assertEqual(len(row), 3)
        # P^2, computed by hand from the single-period matrix.
        self.assertAlmostEqual(matrix[0][0], 0.819)
        self.assertAlmostEqual(matrix[0][1], 0.153)
        self.assertAlmostEqual(matrix[0][2], 0.028)
        self.assertAlmostEqual(matrix[1][0], 0.17)
        self.assertAlmostEqual(matrix[1][1], 0.649)
        self.assertAlmostEqual(matrix[1][2], 0.181)
        self.assertEqual(matrix[2], [0.0, 0.0, 1.0])

        counterparties = result["counterparties"]
        self.assertEqual([c["id"] for c in counterparties], ["cp-1", "cp-2", "cp-3"])
        self.assertEqual(
            set(counterparties[0]),
            {
                "id",
                "rating",
                "horizon_probabilities",
                "cumulative_pd",
                "ead",
                "lgd",
                "expected_loss",
            },
        )
        cp1 = counterparties[0]
        self.assertEqual(cp1["rating"], "A")
        self.assertEqual(cp1["horizon_probabilities"], matrix[0])
        self.assertEqual(cp1["cumulative_pd"], matrix[0][2])
        self.assertEqual(cp1["ead"], 1_000_000.0)
        self.assertEqual(cp1["lgd"], 0.4)
        self.assertEqual(cp1["expected_loss"], 1_000_000.0 * 0.4 * matrix[0][2])
        self.assertAlmostEqual(cp1["expected_loss"], 11_200.0)

        cp2 = counterparties[1]
        self.assertEqual(cp2["horizon_probabilities"], matrix[1])
        self.assertAlmostEqual(cp2["expected_loss"], 45_250.0)

        cp3 = counterparties[2]
        self.assertEqual(cp3["horizon_probabilities"], matrix[0])
        self.assertAlmostEqual(cp3["expected_loss"], 7_000.0)

    def test_horizon_one_returns_input_matrix(self):
        result = self.call(horizon=1)
        self.assertEqual(
            result["horizon_transition_matrix"],
            [
                [0.9, 0.09, 0.01],
                [0.1, 0.8, 0.1],
                [0.0, 0.0, 1.0],
            ],
        )
        self.assertAlmostEqual(
            result["counterparties"][0]["cumulative_pd"], 0.01
        )

    def test_horizon_matches_naive_power(self):
        matrix = [
            [0.9, 0.09, 0.01],
            [0.1, 0.8, 0.1],
            [0.0, 0.0, 1.0],
        ]
        expected = [row[:] for row in matrix]
        for _ in range(4):
            expected = naive_matrix_multiply(expected, matrix)
        result = self.call(horizon=5)
        self.assertEqual(result["horizon_transition_matrix"], expected)
        # The default row stays exactly absorbing at every horizon.
        self.assertEqual(result["horizon_transition_matrix"][2], [0.0, 0.0, 1.0])

    def test_currency_default_and_echo(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_rating_summaries_and_portfolio_totals(self):
        result = self.call(
            ratings=["A", "B", "C", "D"],
            transition_matrix=[
                [0.9, 0.09, 0.0, 0.01],
                [0.1, 0.8, 0.0, 0.1],
                [0.2, 0.2, 0.5, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        summaries = result["rating_summaries"]
        # The default rating is excluded; the empty C group keeps zeros.
        self.assertEqual([s["rating"] for s in summaries], ["A", "B", "C"])
        self.assertEqual(
            set(summaries[0]),
            {
                "rating",
                "counterparty_count",
                "ead",
                "expected_defaulted_exposure",
                "expected_loss",
            },
        )
        by_rating = {s["rating"]: s for s in summaries}
        self.assertEqual(by_rating["A"]["counterparty_count"], 2)
        self.assertEqual(by_rating["B"]["counterparty_count"], 1)
        self.assertEqual(
            by_rating["C"],
            {
                "rating": "C",
                "counterparty_count": 0,
                "ead": 0.0,
                "expected_defaulted_exposure": 0.0,
                "expected_loss": 0.0,
            },
        )

        counterparties = result["counterparties"]
        a_rows = [c for c in counterparties if c["rating"] == "A"]
        self.assertEqual(
            by_rating["A"]["ead"], sum(c["ead"] for c in a_rows)
        )
        self.assertEqual(
            by_rating["A"]["expected_loss"],
            sum(c["expected_loss"] for c in a_rows),
        )
        self.assertEqual(
            by_rating["A"]["expected_defaulted_exposure"],
            sum(c["ead"] * c["cumulative_pd"] for c in a_rows),
        )

        totals = result["portfolio_totals"]
        self.assertEqual(
            set(totals),
            {
                "counterparty_count",
                "ead",
                "expected_defaulted_exposure",
                "expected_loss",
            },
        )
        self.assertEqual(totals["counterparty_count"], 3)
        self.assertEqual(
            totals["ead"], sum(s["ead"] for s in summaries)
        )
        self.assertEqual(
            totals["expected_loss"], sum(s["expected_loss"] for s in summaries)
        )
        self.assertEqual(
            totals["expected_defaulted_exposure"],
            sum(s["expected_defaulted_exposure"] for s in summaries),
        )
        # The portfolio totals equal the counterparty detail sums.
        self.assertAlmostEqual(
            totals["expected_loss"],
            sum(c["expected_loss"] for c in counterparties),
        )

    def test_probabilities_snap_to_zero_and_one(self):
        tiny = 5e-13
        result = self.call(
            horizon=1,
            transition_matrix=[
                [1.0 - tiny, tiny, 0.0],
                [tiny, 1.0 - 2 * tiny, tiny],
                [0.0, 0.0, 1.0],
            ],
        )
        matrix = result["horizon_transition_matrix"]
        self.assertEqual(matrix[0], [1.0, 0.0, 0.0])
        self.assertEqual(matrix[1], [0.0, 1.0, 0.0])
        self.assertEqual(matrix[2], [0.0, 0.0, 1.0])
        self.assertEqual(result["counterparties"][0]["cumulative_pd"], 0.0)

    def test_default_rating_need_not_be_last(self):
        result = self.call(
            ratings=["D", "A", "B"],
            default_rating="D",
            transition_matrix=[
                [1.0, 0.0, 0.0],
                [0.01, 0.9, 0.09],
                [0.1, 0.1, 0.8],
            ],
            counterparties=[
                {"id": "cp-1", "rating": "A", "ead": 100.0, "lgd": 1.0},
            ],
        )
        matrix = result["horizon_transition_matrix"]
        self.assertEqual(matrix[0], [1.0, 0.0, 0.0])
        self.assertEqual(
            result["counterparties"][0]["cumulative_pd"], matrix[1][0]
        )
        self.assertEqual(
            [s["rating"] for s in result["rating_summaries"]], ["A", "B"]
        )

    def test_boundaries_are_allowed(self):
        ratings = [f"r{i}" for i in range(100)]
        matrix = [[0.0] * 100 for _ in range(100)]
        for i in range(99):
            matrix[i][i] = 0.5
            matrix[i][99] = 0.5
        matrix[99][99] = 1.0
        result = self.call(
            horizon=50,
            ratings=ratings,
            default_rating="r99",
            transition_matrix=matrix,
            counterparties=[
                {"id": "cp-1", "rating": "r0", "ead": 1.0, "lgd": 1.0}
            ],
        )
        self.assertEqual(result["horizon"], 50)
        self.assertEqual(len(result["ratings"]), 100)
        self.assertAlmostEqual(
            result["counterparties"][0]["cumulative_pd"], 1.0 - 0.5**50
        )

        many = [
            {"id": f"cp-{i}", "rating": "A", "ead": 1.0, "lgd": 0.5}
            for i in range(10000)
        ]
        result = self.call(counterparties=many)
        self.assertEqual(len(result["counterparties"]), 10000)
        self.assertEqual(result["portfolio_totals"]["counterparty_count"], 10000)

    # ---- error semantics ----

    def assert_error(self, exc_type, code, **overrides):
        with self.assertRaises(exc_type) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)
        return ctx.exception

    def test_invalid_json_is_invalid_request(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.rating_migration_loss(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_non_object_is_invalid_request(self):
        with self.assertRaises(InvalidRequest):
            self.service.rating_migration_loss(b"[1, 2, 3]")

    def test_currency_validation(self):
        self.assert_error(InvalidInput, "invalid_input", currency="")
        self.assert_error(InvalidInput, "invalid_input", currency=1)

    def test_horizon_validation(self):
        for bad in (None, 0, -1, 1.5, True, "2"):
            self.assert_error(InvalidInput, "invalid_input", horizon=bad)
        exc = self.assert_error(
            RequestTooLarge, "request_too_large", horizon=51
        )
        self.assertEqual(exc.status, 413)

    def test_ratings_validation(self):
        self.assert_error(InvalidInput, "invalid_input", ratings=None)
        self.assert_error(InvalidInput, "invalid_input", ratings=[])
        self.assert_error(InvalidInput, "invalid_input", ratings=["A", 1, "D"])
        self.assert_error(InvalidInput, "invalid_input", ratings=["A", "", "D"])
        self.assert_error(
            InvalidInput, "duplicate_rating", ratings=["A", "A", "D"]
        )
        self.assert_error(
            RequestTooLarge,
            "request_too_large",
            ratings=[f"r{i}" for i in range(101)],
        )

    def test_default_rating_validation(self):
        self.assert_error(InvalidInput, "invalid_input", default_rating=None)
        self.assert_error(InvalidInput, "invalid_input", default_rating="")
        self.assert_error(InvalidInput, "invalid_input", default_rating="X")

    def test_matrix_validation(self):
        self.assert_error(InvalidInput, "invalid_input", transition_matrix=None)
        self.assert_error(
            InvalidInput, "invalid_input", transition_matrix=[[1.0, 0.0, 0.0]]
        )
        self.assert_error(
            InvalidInput,
            "invalid_input",
            transition_matrix=[
                [0.9, 0.09, 0.01],
                [0.1, 0.8],
                [0.0, 0.0, 1.0],
            ],
        )
        # Negative entry.
        self.assert_error(
            InvalidInput,
            "invalid_input",
            transition_matrix=[
                [1.1, -0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.0, 0.0, 1.0],
            ],
        )
        # Row not summing to 1.
        self.assert_error(
            InvalidInput,
            "invalid_input",
            transition_matrix=[
                [0.9, 0.09, 0.0],
                [0.1, 0.8, 0.1],
                [0.0, 0.0, 1.0],
            ],
        )
        # Default state must be absorbing.
        self.assert_error(
            InvalidInput,
            "invalid_input",
            transition_matrix=[
                [0.9, 0.09, 0.01],
                [0.1, 0.8, 0.1],
                [0.0, 0.5, 0.5],
            ],
        )
        self.assert_error(
            InvalidInput,
            "invalid_input",
            transition_matrix=[
                [0.9, 0.09, 0.01],
                [0.1, 0.8, 0.1],
                [0.0, 0.0, 0.0],
            ],
        )
        # Non-numeric and boolean entries.
        self.assert_error(
            InvalidInput,
            "invalid_input",
            transition_matrix=[
                [0.9, 0.09, "0.01"],
                [0.1, 0.8, 0.1],
                [0.0, 0.0, 1.0],
            ],
        )
        self.assert_error(
            InvalidInput,
            "invalid_input",
            transition_matrix=[
                [0.9, 0.09, True],
                [0.1, 0.8, 0.1],
                [0.0, 0.0, 1.0],
            ],
        )

    def test_nan_and_infinity_rejected(self):
        with self.assertRaises(InvalidInput):
            self.service.rating_migration_loss(
                request_body().replace(b"0.9", b"NaN", 1)
            )
        with self.assertRaises(InvalidInput):
            self.service.rating_migration_loss(
                request_body().replace(b"1000000.0", b"Infinity", 1)
            )

    def test_counterparty_validation(self):
        self.assert_error(InvalidInput, "invalid_input", counterparties=None)
        self.assert_error(InvalidInput, "invalid_input", counterparties=[])
        self.assert_error(
            InvalidInput,
            "invalid_input",
            counterparties=[{"id": "c", "rating": "A", "ead": 1.0, "lgd": 0.5}, "x"],
        )
        self.assert_error(
            InvalidInput,
            "invalid_input",
            counterparties=[{"id": "", "rating": "A", "ead": 1.0, "lgd": 0.5}],
        )
        self.assert_error(
            InvalidInput,
            "invalid_input",
            counterparties=[{"id": "c", "rating": "X", "ead": 1.0, "lgd": 0.5}],
        )
        # A counterparty already in the default state is rejected.
        self.assert_error(
            InvalidInput,
            "invalid_input",
            counterparties=[{"id": "c", "rating": "D", "ead": 1.0, "lgd": 0.5}],
        )
        self.assert_error(
            InvalidInput,
            "invalid_input",
            counterparties=[{"id": "c", "rating": "A", "ead": -1.0, "lgd": 0.5}],
        )
        self.assert_error(
            InvalidInput,
            "invalid_input",
            counterparties=[{"id": "c", "rating": "A", "ead": 1.0, "lgd": 1.5}],
        )
        self.assert_error(
            InvalidInput,
            "invalid_input",
            counterparties=[{"id": "c", "rating": "A", "ead": 1.0, "lgd": -0.5}],
        )
        self.assert_error(
            InvalidInput,
            "invalid_input",
            counterparties=[{"id": "c", "rating": "A", "ead": True, "lgd": 0.5}],
        )
        self.assert_error(
            InvalidInput,
            "invalid_input",
            counterparties=[
                {"id": "c", "rating": "A", "ead": 10**400, "lgd": 0.5}
            ],
        )
        self.assert_error(
            InvalidInput,
            "duplicate_counterparty",
            counterparties=[
                {"id": "c", "rating": "A", "ead": 1.0, "lgd": 0.5},
                {"id": "c", "rating": "B", "ead": 2.0, "lgd": 0.5},
            ],
        )
        self.assert_error(
            RequestTooLarge,
            "request_too_large",
            counterparties=[
                {"id": f"cp-{i}", "rating": "A", "ead": 1.0, "lgd": 0.5}
                for i in range(10001)
            ],
        )

    def test_non_finite_computation_rejected(self):
        self.assert_error(
            InvalidInput,
            "invalid_input",
            ratings=["A", "D"],
            transition_matrix=[[0.0, 1.0], [0.0, 1.0]],
            counterparties=[
                {"id": "c1", "rating": "A", "ead": 1e308, "lgd": 1.0},
                {"id": "c2", "rating": "A", "ead": 1e308, "lgd": 1.0},
            ],
        )

    def test_error_carries_status(self):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(horizon=0)
        self.assertEqual(ctx.exception.status, 422)
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(horizon=51)
        self.assertEqual(ctx.exception.status, 413)


if __name__ == "__main__":
    unittest.main()
