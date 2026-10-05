"""In-memory VaR backtesting: coverage and exception-clustering tests.

:func:`backtest_var` is the in-memory counterpart of the HTTP VaR
backtest endpoints. It pairs a realized P&L series with the matching
VaR forecasts period by period and reports the Kupiec (1995)
unconditional-coverage test, the Christoffersen (1998) independence
test and the combined conditional-coverage test. The function is pure:
it reads its inputs, computes in memory and returns a plain dict; no
files are written and no service state is touched.
"""

from __future__ import annotations

import math

__all__ = ["backtest_var"]


def _as_float_sequence(values: object, name: str) -> list[float]:
    """Narrow a one-dimensional iterable of real numbers to floats.

    Booleans and text are rejected even though ``float`` would accept
    some of them: a bool is not a risk number and a string is not a
    numeric sequence. Elements that cannot be interpreted as a real
    number raise :class:`TypeError`; an integer too large to represent
    as a float is a value problem and raises :class:`ValueError`.
    """
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a one-dimensional iterable of real numbers")
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(
            f"{name} must be a one-dimensional iterable of real numbers"
        ) from None
    result: list[float] = []
    for item in items:
        if isinstance(item, bool) or isinstance(item, (str, bytes, bytearray)):
            raise TypeError(f"{name} elements must be real numbers, not {type(item).__name__}")
        try:
            number = float(item)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise TypeError(
                f"{name} elements must be interpretable as real numbers"
            ) from None
        except OverflowError:
            raise ValueError(f"{name} elements must be finite") from None
        result.append(number)
    return result


def _validate_confidence(confidence_level: object) -> float:
    """Narrow ``confidence_level`` to a float strictly inside (0, 1)."""
    if isinstance(confidence_level, bool) or isinstance(
        confidence_level, (str, bytes, bytearray)
    ):
        raise TypeError("confidence_level must be a real number")
    try:
        confidence = float(confidence_level)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TypeError("confidence_level must be a real number") from None
    except OverflowError:
        raise ValueError("confidence_level must lie in the open interval (0, 1)") from None
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie in the open interval (0, 1)")
    return confidence


def _log_term(count: int, probability: float) -> float:
    """One ``count * ln(probability)`` log-likelihood term.

    ``0 * ln 0`` is defined as 0, so boundary outcomes (no breach or
    every period breaching) stay well defined. Whenever ``count`` is
    positive the paired probability is positive by construction, so the
    term is always finite.
    """
    if count == 0:
        return 0.0
    return count * math.log(probability)


def backtest_var(realized_pnl, var_forecast, confidence_level: float = 0.99) -> dict:
    """Backtest VaR forecasts against realized P&L, in memory.

    ``realized_pnl`` and ``var_forecast`` are equal-length, non-empty,
    one-dimensional iterables of real numbers (plain Python sequences
    and numeric array objects alike); profits are positive and losses
    negative, and every VaR forecast must be a non-negative finite
    number. ``confidence_level`` defaults to ``0.99`` and must lie in
    the open interval (0, 1). Booleans are not accepted as numbers.

    Period ``i`` breaches when ``realized_pnl[i] < -var_forecast[i]``
    (equality does not breach). Inputs are paired in their original
    order — never sorted, never padded — and caller objects are not
    modified.

    Returns a dict with ``confidence_level``, ``observation_count``,
    ``breach_count``, ``expected_breach_count``, ``breach_rate``, the
    per-period ``breaches`` boolean list, the zero-based
    ``breach_positions``, and three test sections. ``kupiec`` holds the
    unconditional-coverage likelihood-ratio statistic and its
    chi-square(1) p-value ``erfc(sqrt(lr/2))``. ``independence`` holds
    the Christoffersen first-order Markov likelihood-ratio statistic
    over adjacent breach states and its chi-square(1) p-value.
    ``conditional_coverage`` holds the sum of the two statistics and
    its chi-square(2) p-value ``exp(-lr/2)``. With a single observation
    there are no transitions, so the independence and
    conditional-coverage statistics and p-values are ``None`` while the
    coverage test still returns. All defined statistics are finite and
    non-negative (a tiny negative remainder from floating-point
    rounding is floored at zero) and every p-value lies in [0, 1].

    Raises :class:`ValueError` when ``confidence_level`` is outside
    (0, 1), a sequence is empty, the lengths differ, a value is NaN or
    infinite, or a VaR forecast is negative. Raises :class:`TypeError`
    when an input is not a one-dimensional iterable of real numbers or
    an element cannot be interpreted as one.
    """
    confidence = _validate_confidence(confidence_level)
    pnl = _as_float_sequence(realized_pnl, "realized_pnl")
    var = _as_float_sequence(var_forecast, "var_forecast")

    if len(pnl) != len(var):
        raise ValueError("realized_pnl and var_forecast must have the same length")
    n = len(pnl)
    if n == 0:
        raise ValueError("realized_pnl and var_forecast must not be empty")
    for value in pnl:
        if not math.isfinite(value):
            raise ValueError("realized_pnl must contain only finite values")
    for value in var:
        if not math.isfinite(value):
            raise ValueError("var_forecast must contain only finite values")
        if value < 0.0:
            raise ValueError("var_forecast must be non-negative")

    breaches = [p < -v for p, v in zip(pnl, var)]
    x = sum(breaches)

    # Kupiec (1995) proportion-of-failures unconditional-coverage test.
    p = 1.0 - confidence
    q = x / n
    constrained = _log_term(n - x, 1.0 - p) + _log_term(x, p)
    unconstrained = _log_term(n - x, 1.0 - q) + _log_term(x, q)
    # The unconstrained likelihood is maximal, so the statistic is
    # non-negative; a small negative remainder is rounding noise (e.g.
    # when the breach rate equals the predicted tail probability).
    kupiec_lr = -2.0 * (constrained - unconstrained)
    if kupiec_lr <= 0.0:
        # Also canonicalizes a signed zero to +0.0.
        kupiec_lr = 0.0
    kupiec_p_value = math.erfc(math.sqrt(kupiec_lr / 2.0))

    ind_lr: float | None
    ind_p_value: float | None
    cc_lr: float | None
    cc_p_value: float | None
    if n < 2:
        # A single observation has no state transitions, so the
        # independence and conditional-coverage tests are undefined.
        ind_lr = ind_p_value = cc_lr = cc_p_value = None
    else:
        # Christoffersen (1998) first-order Markov transition counts over
        # adjacent breach states in input order: n_ij counts a transition
        # from state i on one period to state j on the next.
        n00 = n01 = n10 = n11 = 0
        for previous, current in zip(breaches, breaches[1:]):
            if not previous and not current:
                n00 += 1
            elif not previous and current:
                n01 += 1
            elif previous and not current:
                n10 += 1
            else:
                n11 += 1

        q_ind = (n01 + n11) / (n - 1)
        ln_l0 = _log_term(n00 + n10, 1.0 - q_ind) + _log_term(n01 + n11, q_ind)

        # A predecessor state that never occurs leaves its conditional
        # breach probability undefined; its likelihood contribution is 0.
        ln_l1 = 0.0
        if n00 + n01 > 0:
            q0 = n01 / (n00 + n01)
            ln_l1 += _log_term(n00, 1.0 - q0) + _log_term(n01, q0)
        if n10 + n11 > 0:
            q1 = n11 / (n10 + n11)
            ln_l1 += _log_term(n10, 1.0 - q1) + _log_term(n11, q1)

        # The constrained model nests in the two-parameter alternative,
        # so the statistic cannot be negative; a tiny negative remainder
        # is floating-point noise around zero.
        ind_lr = 2.0 * (ln_l1 - ln_l0)
        if ind_lr <= 0.0:
            ind_lr = 0.0
        ind_p_value = math.erfc(math.sqrt(ind_lr / 2.0))

        # Conditional coverage combines the two statistics and is
        # chi-square with two degrees of freedom, whose survival
        # function is exp(-lr/2).
        cc_lr = kupiec_lr + ind_lr
        cc_p_value = math.exp(-cc_lr / 2.0)

    return {
        "confidence_level": confidence,
        "observation_count": n,
        "breach_count": x,
        "expected_breach_count": n * p,
        "breach_rate": q,
        "breaches": breaches,
        "breach_positions": [i for i, breach in enumerate(breaches) if breach],
        "kupiec": {"lr_statistic": kupiec_lr, "p_value": kupiec_p_value},
        "independence": {"lr_statistic": ind_lr, "p_value": ind_p_value},
        "conditional_coverage": {"lr_statistic": cc_lr, "p_value": cc_p_value},
    }
