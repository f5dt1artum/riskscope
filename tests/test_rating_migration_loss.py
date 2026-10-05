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
        "horizon": 2,
        "ratings": ["A", "B", "D"],
        "default_rating": "D",
        "transition_matrix": [
            [0.9, 0.09, 0.01],
            [0.05, 0.9, 0.05],
            [0.0, 0.0, 1.0],
        ],
        "counterparties": [
            {"id": "cp-1", "rating": "A", "ead": 1000.0, "lgd": 0.5},
            {"id": "cp-2", "rating": "B", "ead": 2000.0, "lgd": 0.25},
            {"id": "cp-3", "rating": "A", "ead": 500.0, "lgd": 1.0},
        ],
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


def matrix_power(matrix, horizon):
    """Naive sequential multiplication: ((M × M) × ...) × M."""
    result = [row[:] for row in matrix]
    for _ in range(horizon - 1):
        result = [
            [
                sum(row[k] * matrix[k][j] for k in range(len(matrix)))
                for j in range(len(matrix))
            ]
            for row in result
        ]
    return result


class RatingMigrationLossTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def call(self, **overrides):
        return self.service.rating_migration_loss(request_body(**overrides))

    # ---- success shapes and values ----

    def test_basic_shapes_and_echoes(self):
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
                "rating_aggregates",
                "portfolio_totals",
            },
        )

    def test_horizon_matrix_matches_independent_power(self):
        matrix = [
            [0.9, 0.09, 0.01],
            [0.05, 0.9, 0.05],
            [0.0, 0.0, 1.0],
        ]
        for horizon in (1, 2, 3, 7):
            result = self.call(horizon=horizon)
            expected = matrix_power(matrix, horizon)
            got = result["horizon_transition_matrix"]
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(got[i][j], expected[i][j], places=15)

    def test_horizon_one_returns_the_input_matrix(self):
        result = self.call(horizon=1)
        self.assertEqual(
            result["horizon_transition_matrix"],
            [
                [0.9, 0.09, 0.01],
                [0.05, 0.9, 0.05],
                [0.0, 0.0, 1.0],
            ],
        )

    def test_default_state_stays_absorbing(self):
        result = self.call(horizon=5)
        default_row = result["horizon_transition_matrix"][2]
        self.assertEqual(default_row, [0.0, 0.0, 1.0])

    def test_counterparty_details(self):
        result = self.call()
        counterparties = result["counterparties"]
        self.assertEqual([cp["id"] for cp in counterparties], ["cp-1", "cp-2", "cp-3"])
        self.assertEqual(
            set(counterparties[0]),
            {
                "id",
                "rating",
                "terminal_probabilities",
                "cumulative_pd",
                "ead",
                "lgd",
                "expected_loss",
            },
        )

        # Two-period cumulative default probabilities from the matrix.
        pd_a = 0.9 * 0.01 + 0.09 * 0.05 + 0.01 * 1.0
        pd_b = 0.05 * 0.01 + 0.9 * 0.05 + 0.05 * 1.0
        cp1, cp2, cp3 = counterparties
        self.assertEqual(cp1["rating"], "A")
        self.assertEqual(cp1["cumulative_pd"], pd_a)
        self.assertEqual(cp1["ead"], 1000.0)
        self.assertEqual(cp1["lgd"], 0.5)
        self.assertEqual(cp1["expected_loss"], 1000.0 * 0.5 * pd_a)
        self.assertEqual(cp2["rating"], "B")
        self.assertEqual(cp2["cumulative_pd"], pd_b)
        self.assertEqual(cp2["expected_loss"], 2000.0 * 0.25 * pd_b)
        self.assertEqual(cp3["rating"], "A")
        self.assertEqual(cp3["cumulative_pd"], pd_a)
        self.assertEqual(cp3["expected_loss"], 500.0 * 1.0 * pd_a)

        # Terminal probabilities cover every rating and match the matrix row.
        row_a = result["horizon_transition_matrix"][0]
        self.assertEqual(
            cp1["terminal_probabilities"],
            {"A": row_a[0], "B": row_a[1], "D": row_a[2]},
        )
        self.assertAlmostEqual(sum(cp1["terminal_probabilities"].values()), 1.0)

    def test_rating_aggregates_and_portfolio_totals(self):
        result = self.call()
        aggregates = result["rating_aggregates"]
        # Only the non-default ratings appear, in ratings order.
        self.assertEqual([row["rating"] for row in aggregates], ["A", "B"])
        self.assertEqual(
            set(aggregates[0]),
            {
                "rating",
                "counterparty_count",
                "ead",
                "expected_defaulted_exposure",
                "expected_loss",
            },
        )

        pd_a = 0.9 * 0.01 + 0.09 * 0.05 + 0.01 * 1.0
        pd_b = 0.05 * 0.01 + 0.9 * 0.05 + 0.05 * 1.0
        agg_a, agg_b = aggregates
        self.assertEqual(agg_a["counterparty_count"], 2)
        self.assertEqual(agg_a["ead"], 1500.0)
        self.assertEqual(
            agg_a["expected_defaulted_exposure"],
            1000.0 * pd_a + 500.0 * pd_a,
        )
        self.assertEqual(
            agg_a["expected_loss"],
            1000.0 * 0.5 * pd_a + 500.0 * 1.0 * pd_a,
        )
        self.assertEqual(agg_b["counterparty_count"], 1)
        self.assertEqual(agg_b["ead"], 2000.0)
        self.assertEqual(agg_b["expected_defaulted_exposure"], 2000.0 * pd_b)
        self.assertEqual(agg_b["expected_loss"], 2000.0 * 0.25 * pd_b)

        totals = result["portfolio_totals"]
        self.assertEqual(totals["counterparty_count"], 3)
        self.assertEqual(totals["ead"], agg_a["ead"] + agg_b["ead"])
        self.assertEqual(
            totals["expected_defaulted_exposure"],
            agg_a["expected_defaulted_exposure"]
            + agg_b["expected_defaulted_exposure"],
        )
        self.assertEqual(
            totals["expected_loss"],
            agg_a["expected_loss"] + agg_b["expected_loss"],
        )

    def test_aggregates_equal_detail_sums(self):
        result = self.call()
        details = result["counterparties"]
        for aggregate in result["rating_aggregates"]:
            owned = [cp for cp in details if cp["rating"] == aggregate["rating"]]
            self.assertEqual(aggregate["counterparty_count"], len(owned))
            self.assertEqual(aggregate["ead"], sum(cp["ead"] for cp in owned))
            self.assertEqual(
                aggregate["expected_loss"],
                sum(cp["expected_loss"] for cp in owned),
            )
            self.assertEqual(
                aggregate["expected_defaulted_exposure"],
                sum(cp["ead"] * cp["cumulative_pd"] for cp in owned),
            )
        totals = result["portfolio_totals"]
        self.assertEqual(totals["ead"], sum(cp["ead"] for cp in details))
        self.assertEqual(
            totals["expected_loss"], sum(cp["expected_loss"] for cp in details)
        )

    def test_empty_rating_group_keeps_zeros(self):
        result = self.call(ratings=["A", "B", "C", "D"], transition_matrix=[
            [0.9, 0.09, 0.0, 0.01],
            [0.05, 0.9, 0.0, 0.05],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ])
        aggregates = result["rating_aggregates"]
        self.assertEqual([row["rating"] for row in aggregates], ["A", "B", "C"])
        empty = aggregates[2]
        self.assertEqual(empty["counterparty_count"], 0)
        self.assertEqual(empty["ead"], 0.0)
        self.assertEqual(empty["expected_defaulted_exposure"], 0.0)
        self.assertEqual(empty["expected_loss"], 0.0)

    def test_currency_echo_and_default(self):
        self.assertEqual(self.call()["currency"], "USD")
        self.assertEqual(self.call(currency="EUR")["currency"], "EUR")

    def test_integer_numbers_are_accepted(self):
        result = self.call(
            horizon=2,
            transition_matrix=[
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            counterparties=[{"id": "c", "rating": "A", "ead": 1000, "lgd": 1}],
        )
        counterparty = result["counterparties"][0]
        self.assertEqual(counterparty["ead"], 1000.0)
        self.assertEqual(counterparty["lgd"], 1.0)
        self.assertEqual(counterparty["cumulative_pd"], 0.0)
        self.assertEqual(counterparty["expected_loss"], 0.0)

    def test_extra_fields_are_ignored(self):
        result = self.call(
            unexpected="top-level",
            counterparties=[
                {"id": "c", "rating": "A", "ead": 10.0, "lgd": 0.5, "note": "x"}
            ],
        )
        self.assertEqual(len(result["counterparties"]), 1)
        self.assertEqual(result["counterparties"][0]["id"], "c")

    def test_probabilities_near_boundaries_are_snapped(self):
        result = self.call(
            horizon=1,
            transition_matrix=[
                [1.0 - 1e-13, 1e-13, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        self.assertEqual(
            result["horizon_transition_matrix"][0], [1.0, 0.0, 0.0]
        )
        self.assertEqual(result["counterparties"][0]["cumulative_pd"], 0.0)

    def test_extreme_horizon_and_size_boundaries(self):
        ratings = [f"r-{i}" for i in range(99)] + ["D"]
        matrix = []
        for i in range(99):
            row = [0.0] * 100
            row[i] = 0.99
            row[-1] = 0.01
            matrix.append(row)
        matrix.append([0.0] * 99 + [1.0])
        result = self.call(
            horizon=50,
            ratings=ratings,
            transition_matrix=matrix,
            counterparties=[{"id": "c", "rating": "r-0", "ead": 1.0, "lgd": 1.0}],
        )
        self.assertEqual(len(result["horizon_transition_matrix"]), 100)
        self.assertAlmostEqual(
            result["counterparties"][0]["cumulative_pd"],
            1.0 - 0.99 ** 50,
            places=15,
        )

    # ---- 400 invalid_request ----

    def test_unparseable_json_is_invalid_request(self):
        with self.assertRaises(InvalidRequest) as ctx:
            self.service.rating_migration_loss(b"{not json")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_non_object_payload_is_invalid_request(self):
        for raw in (b"[1, 2]", b"42", b'"text"', b"null"):
            with self.assertRaises(InvalidRequest):
                self.service.rating_migration_loss(raw)

    # ---- 422 invalid_input: structure, types, ranges ----

    def test_currency_must_be_nonempty_string(self):
        for bad in ("", 3, None, True):
            with self.assertRaises(InvalidInput):
                self.call(currency=bad)

    def test_horizon_validation(self):
        for bad in (0, -1, 51, 100, 2.5, 1.0, True, "2", None, 10**400):
            with self.assertRaises(InvalidInput, msg=repr(bad)):
                self.call(horizon=bad)

    def test_ratings_validation(self):
        for bad in (
            None,
            [],
            "not-a-list",
            [""],
            ["A", ""],
            ["A", 3],
            ["A", None],
            ["A", True],
            ["A"],
        ):
            with self.assertRaises(InvalidInput, msg=repr(bad)):
                self.call(ratings=bad)

    def test_default_rating_validation(self):
        for bad in (None, "", 3, True, "ghost"):
            with self.assertRaises(InvalidInput, msg=repr(bad)):
                self.call(default_rating=bad)

    def test_transition_matrix_shape_validation(self):
        bad_matrices = [
            None,
            "not-a-list",
            [],
            [[0.9, 0.09, 0.01]],                       # too few rows
            [[0.9, 0.09, 0.01]] * 4,                   # too many rows
            [[0.9, 0.09], [0.05, 0.9], [0.0, 0.0]],    # too few columns
            [[0.9, 0.09, 0.01, 0.0]] * 3,              # too many columns
            ["not-a-row"] * 3,
        ]
        for matrix in bad_matrices:
            with self.assertRaises(InvalidInput, msg=repr(matrix)):
                self.call(transition_matrix=matrix)

    def test_transition_matrix_value_validation(self):
        bad_values = [-0.1, True, "0.5", None, 10**400]
        for bad in bad_values:
            matrix = [
                [0.9, 0.09, 0.01],
                [0.05, 0.9, 0.05],
                [0.0, 0.0, 1.0],
            ]
            matrix[0][0] = bad
            with self.assertRaises(InvalidInput, msg=repr(bad)):
                self.call(transition_matrix=matrix)

    def test_transition_matrix_row_sum_validation(self):
        for bad_row in (
            [0.9, 0.09, 0.02],       # sums to 1.01
            [0.9, 0.09, 0.0],        # sums to 0.99
            [0.9, 0.09, 0.01 + 1e-9],
        ):
            with self.assertRaises(InvalidInput, msg=repr(bad_row)):
                self.call(transition_matrix=[
                    bad_row,
                    [0.05, 0.9, 0.05],
                    [0.0, 0.0, 1.0],
                ])

    def test_default_state_must_be_absorbing(self):
        bad_rows = (
            [0.0, 0.1, 0.9],     # leaves the default state
            [0.5, 0.0, 0.5],
            [0.0, 0.0, 0.99],
        )
        for row in bad_rows:
            with self.assertRaises(InvalidInput, msg=repr(row)):
                self.call(transition_matrix=[
                    [0.9, 0.09, 0.01],
                    [0.05, 0.9, 0.05],
                    row,
                ])

    def test_counterparty_field_validation(self):
        bad_counterparties = [
            [{"rating": "A", "ead": 1.0, "lgd": 0.5}],              # missing id
            [{"id": "", "rating": "A", "ead": 1.0, "lgd": 0.5}],    # empty id
            [{"id": "c", "ead": 1.0, "lgd": 0.5}],                  # missing rating
            [{"id": "c", "rating": "", "ead": 1.0, "lgd": 0.5}],    # empty rating
            [{"id": "c", "rating": "ghost", "ead": 1.0, "lgd": 0.5}],
            [{"id": "c", "rating": "D", "ead": 1.0, "lgd": 0.5}],   # defaulted
            [{"id": "c", "rating": "A", "lgd": 0.5}],               # missing ead
            [{"id": "c", "rating": "A", "ead": -1.0, "lgd": 0.5}],
            [{"id": "c", "rating": "A", "ead": True, "lgd": 0.5}],
            [{"id": "c", "rating": "A", "ead": "1", "lgd": 0.5}],
            [{"id": "c", "rating": "A", "ead": 10**400, "lgd": 0.5}],
            [{"id": "c", "rating": "A", "ead": 1.0}],               # missing lgd
            [{"id": "c", "rating": "A", "ead": 1.0, "lgd": -0.1}],
            [{"id": "c", "rating": "A", "ead": 1.0, "lgd": 1.1}],
            [{"id": "c", "rating": "A", "ead": 1.0, "lgd": False}],
            [{"id": "c", "rating": "A", "ead": 1.0, "lgd": "0.5"}],
            [{"id": "c", "rating": "A", "ead": 1.0, "lgd": 10**400}],
            ["not-an-object"],
        ]
        for counterparties in bad_counterparties:
            with self.assertRaises(InvalidInput, msg=repr(counterparties)):
                self.call(counterparties=counterparties)

    def test_counterparties_must_be_nonempty_list(self):
        for bad in (None, [], "not-a-list"):
            with self.assertRaises(InvalidInput):
                self.call(counterparties=bad)

    def test_nan_and_infinity_tokens_are_invalid_input(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            raw = request_body().replace(b"0.9", token.encode(), 1)
            with self.assertRaises(InvalidInput):
                self.service.rating_migration_loss(raw)

    def test_non_finite_computation_is_invalid_input(self):
        with self.assertRaises(InvalidInput):
            self.call(
                counterparties=[
                    {"id": "c-1", "rating": "B", "ead": 1e308, "lgd": 1.0},
                    {"id": "c-2", "rating": "B", "ead": 1e308, "lgd": 1.0},
                ],
            )

    # ---- 422 duplicate_* ----

    def assert_code(self, code, **overrides):
        with self.assertRaises(InvalidInput) as ctx:
            self.call(**overrides)
        self.assertEqual(ctx.exception.code, code)
        self.assertEqual(ctx.exception.status, 422)

    def test_duplicate_ids(self):
        self.assert_code("duplicate_rating", ratings=["A", "B", "A"])
        self.assert_code(
            "duplicate_counterparty",
            counterparties=[
                {"id": "c", "rating": "A", "ead": 1.0, "lgd": 0.5},
                {"id": "c", "rating": "B", "ead": 2.0, "lgd": 0.5},
            ],
        )

    # ---- 413 request_too_large ----

    def test_size_limits(self):
        ratings = [f"r-{i}" for i in range(101)]
        with self.assertRaises(RequestTooLarge) as ctx:
            self.call(ratings=ratings)
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(ctx.exception.code, "request_too_large")

        counterparties = [
            {"id": f"cp-{i}", "rating": "A", "ead": 1.0, "lgd": 0.5}
            for i in range(10001)
        ]
        with self.assertRaises(RequestTooLarge):
            self.call(counterparties=counterparties)

    def test_size_boundaries_are_allowed(self):
        counterparties = [
            {"id": f"cp-{i}", "rating": "A", "ead": 1.0, "lgd": 0.5}
            for i in range(10000)
        ]
        result = self.call(counterparties=counterparties)
        self.assertEqual(len(result["counterparties"]), 10000)
        self.assertEqual(result["portfolio_totals"]["counterparty_count"], 10000)


if __name__ == "__main__":
    unittest.main()
