"""In-memory VaR backtesting: coverage and independence diagnostics.

:func:`backtest_var` pairs a realized P&L series with same-frequency VaR
forecasts and validates the forecaster two ways: the Kupiec (1995)
unconditional-coverage likelihood-ratio test compares the breach rate
with the predicted tail probability, and the Christoffersen (1998)
independence test checks whether breaches cluster in time; their sum is
the conditional-coverage statistic. The function is pure and in-memory:
it reads its inputs once, never mutates them, produces no files, and
returns plain Python structures that are deterministic for identical
inputs.
"""

from __future__ import annotations

import math
from typing import Iterable

__all__ = ["backtest_var"]


def _log_term(count: int, probability: float) -> float:
    """One ``count * ln(probability)`` log-likelihood term.

    ``0 * ln 0`` is defined as 0 here, so boundary outcomes (no breach
    or every period breaching) stay well defined. A positive count
    paired with a zero probability is an infinite term and rejected.
    """
    if count == 0:
        return 0.0
    if probability <= 0.0 or probability > 1.0:
        raise ValueError("backtest computation produced a non-finite result")
    return count * math.log(probability)


def _settle(statistic: float) -> float:
    """Floor a likelihood-ratio statistic at zero.

    The statistic is a likelihood ratio and cannot be negative, so any
    negative remainder is floating-point rounding noise; the canonical
    ``+0.0`` is emitted rather than a signed zero. Non-finite results
    are rejected.
    """
    if not math.isfinite(statistic):
        raise ValueError("backtest computation produced a non-finite result")
    if statistic <= 0.0:
        return 0.0
    return statistic


def _coerce_number(value: object, name: str) -> float:
    """Narrow one element to a float, rejecting booleans and non-numbers."""
    if isinstance(value, bool):
        raise TypeError(f"{name} elements must be real numbers, not booleans")
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} elements must be real numbers, not strings")
    try:
        result = float(value)  # type: ignore[arg-type]
    except OverflowError:
        # An integer too large for float64 is not a usable finite number.
        raise ValueError(f"{name} elements must be finite") from None
    except (TypeError, ValueError):
        raise TypeError(
            f"{name} must be a one-dimensional iterable of real numbers"
        ) from None
    return result


def _coerce_sequence(values: Iterable[float], name: str) -> list[float]:
    """Materialize a one-dimensional numeric iterable into a new list.

    The caller's object is only read, never mutated; strings, bytes and
    non-iterables are not numeric sequences.
    """
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a one-dimensional iterable of real numbers")
    try:
        iterator = iter(values)
    except TypeError:
        raise TypeError(
            f"{name} must be a one-dimensional iterable of real numbers"
        ) from None
    return [_coerce_number(item, name) for item in iterator]


def backtest_var(
    realized_pnl: Iterable[float],
    var_forecast: Iterable[float],
    confidence_level: float = 0.99,
) -> dict:
    """Backtest VaR forecasts against realized P&L, all in memory.

    ``realized_pnl`` and ``var_forecast`` are equal-length, one-dimensional
    iterables of real numbers (plain Python sequences and numeric array
    objects alike), paired element by element in their original order —
    nothing is sorted, no missing values are filled, and the caller's
    objects are never modified. P&L follows the profit-positive /
    loss-negative convention and each VaR forecast must be a non-negative
    finite number. Period ``i`` breaches when
    ``realized_pnl[i] < -var_forecast[i]``; equality does not breach.

    ``confidence_level`` defaults to ``0.99`` and must lie in the open
    interval (0, 1). Booleans are never accepted as numbers.

    The result is a dict with ``confidence_level``, ``observation_count``,
    ``breach_count``, ``expected_breach_count`` (``n * (1 -
    confidence_level)``), ``breach_rate``, the per-period boolean
    ``breaches`` sequence, the zero-based ``breach_indices``, the four
    first-order Markov transition counts ``n00``/``n01``/``n10``/``n11``
    over adjacent breach states, and three test objects:

    - ``kupiec``: unconditional-coverage likelihood ratio and its
      chi-square (1 dof) p-value ``erfc(sqrt(lr / 2))``;
    - ``independence``: Christoffersen independence likelihood ratio and
      its chi-square (1 dof) p-value;
    - ``conditional_coverage``: the sum of the two statistics above and
      its chi-square (2 dof) p-value ``exp(-lr / 2)``.

    Log-likelihoods use natural logarithms with ``0 * ln 0 == 0``;
    rounding-scale negative statistics are floored at zero, so every
    defined statistic is finite and non-negative and every p-value lies
    in [0, 1], including the zero-breach and all-breach boundaries. With
    a single observation there are no state transitions, so the
    independence and conditional-coverage statistics and p-values are
    ``None`` while the Kupiec coverage test is still reported.

    Raises ``TypeError`` when an input is not a one-dimensional iterable
    of real numbers (booleans included), and ``ValueError`` when
    ``confidence_level`` lies outside (0, 1), the sequences are empty or
    differ in length, any value is NaN or infinite, or any VaR forecast
    is negative.
    """
    confidence = _coerce_number(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie in the open interval (0, 1)")

    pnl = _coerce_sequence(realized_pnl, "realized_pnl")
    var = _coerce_sequence(var_forecast, "var_forecast")

    if len(pnl) != len(var):
        raise ValueError("realized_pnl and var_forecast must have equal lengths")
    n = len(pnl)
    if n == 0:
        raise ValueError("realized_pnl and var_forecast must not be empty")

    for value in pnl:
        if not math.isfinite(value):
            raise ValueError("realized_pnl elements must be finite")
    for value in var:
        if not math.isfinite(value):
            raise ValueError("var_forecast elements must be finite")
        if value < 0.0:
            raise ValueError("var_forecast elements must be non-negative")

    # Strict inequality: equality with the VaR forecast does not breach.
    breaches = [pnl[i] < -var[i] for i in range(n)]
    x = sum(breaches)

    # Kupiec (1995) proportion-of-failures unconditional-coverage test.
    p = 1.0 - confidence
    q = x / n
    constrained = _log_term(n - x, 1.0 - p) + _log_term(x, p)
    unconstrained = _log_term(n - x, 1.0 - q) + _log_term(x, q)
    lr_uc = _settle(-2.0 * (constrained - unconstrained))
    p_uc = math.erfc(math.sqrt(lr_uc / 2.0))
    if not math.isfinite(p_uc):
        raise ValueError("backtest computation produced a non-finite result")

    # Christoffersen (1998) first-order Markov transition counts over
    # adjacent breach states in input order: n_ij counts a transition
    # from state i in one period to state j in the next.
    n00 = n01 = n10 = n11 = 0
    previous = breaches[0]
    for current in breaches[1:]:
        if not previous and not current:
            n00 += 1
        elif not previous and current:
            n01 += 1
        elif previous and not current:
            n10 += 1
        else:
            n11 += 1
        previous = current

    lr_ind: float | None
    p_ind: float | None
    lr_cc: float | None
    p_cc: float | None
    if n < 2:
        # A single observation has no state transitions, so the
        # independence and conditional-coverage tests are undefined.
        lr_ind = p_ind = lr_cc = p_cc = None
    else:
        q_ind = (n01 + n11) / (n - 1)
        ln_l0 = _log_term(n00 + n10, 1.0 - q_ind) + _log_term(n01 + n11, q_ind)
        # A predecessor state that never occurs contributes nothing to
        # the alternative likelihood (the same zero-probability
        # convention as the 0 * ln 0 terms).
        ln_l1 = 0.0
        if n00 + n01 > 0:
            q0 = n01 / (n00 + n01)
            ln_l1 += _log_term(n00, 1.0 - q0) + _log_term(n01, q0)
        if n10 + n11 > 0:
            q1 = n11 / (n10 + n11)
            ln_l1 += _log_term(n10, 1.0 - q1) + _log_term(n11, q1)
        lr_ind = _settle(2.0 * (ln_l1 - ln_l0))
        p_ind = math.erfc(math.sqrt(lr_ind / 2.0))
        lr_cc = _settle(lr_uc + lr_ind)
        p_cc = math.exp(-lr_cc / 2.0)
        if not math.isfinite(p_ind) or not math.isfinite(p_cc):
            raise ValueError("backtest computation produced a non-finite result")

    return {
        "confidence_level": confidence,
        "observation_count": n,
        "breach_count": x,
        "expected_breach_count": n * p,
        "breach_rate": x / n,
        "breaches": breaches,
        "breach_indices": [i for i, breach in enumerate(breaches) if breach],
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "kupiec": {"lr_statistic": lr_uc, "p_value": p_uc},
        "independence": {"lr_statistic": lr_ind, "p_value": p_ind},
        "conditional_coverage": {"lr_statistic": lr_cc, "p_value": p_cc},
    }
