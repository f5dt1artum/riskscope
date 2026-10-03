"""Core service surface for RiskScope.

The frozen baseline reports process health; historical-simulation VaR and
expected shortfall now live behind :meth:`Service.historical_var` and batch
sensitivity stress tests behind :meth:`Service.stress_test`. Counterparty
credit exposure and expected loss live behind
:meth:`Service.counterparty_exposure`. Liquidity gap analysis by maturity
bucket lives behind :meth:`Service.liquidity_gap`. Sample covariance
estimation from synchronized factor returns lives behind
:meth:`Service.covariance_estimate`. Zero-mean Delta-Normal parametric VaR
lives behind :meth:`Service.parametric_var`. The public surface stays
backward compatible.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, ROUND_CEILING

from . import __version__

MAX_OBSERVATIONS = 10000
MAX_SCENARIOS = 1000
MAX_SCENARIO_POSITION_PAIRS = 100000
MAX_TRADES = 10000
MAX_NETTING_SETS = 1000
MAX_FACTORS = 100
MAX_FACTOR_OBSERVATION_PAIRS = 1000000
MAX_POSITIONS = 10000
MAX_FACTOR_POSITION_PAIRS = 1000000

# Relative tolerance for structural covariance checks: entries this close
# are treated as symmetric, and a quadratic form this close to zero as zero.
COVARIANCE_REL_TOL = 1e-12


class ServiceError(Exception):
    """Error carrying the HTTP status and machine-readable code."""

    status = 400
    code = "invalid_request"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequest(ServiceError):
    status = 400
    code = "invalid_request"


class InvalidInput(ServiceError):
    status = 422
    code = "invalid_input"

    def __init__(self, code: str = "invalid_input", message: str | None = None) -> None:
        super().__init__(message or f"input validation failed: {code}")
        self.code = code


class RequestTooLarge(ServiceError):
    status = 413
    code = "request_too_large"


class _JsonConstant(ValueError):
    """Raised for NaN / Infinity tokens so they surface as invalid_input."""


def _reject_constant(value: str) -> None:
    raise _JsonConstant(value)


def _is_finite_number(value: object) -> bool:
    # bool is a subclass of int but is not a risk number.
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)  # type: ignore[arg-type]
    )


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and len(value) > 0


def _reconcile_totals(values: list[float], target: float) -> None:
    """Nudge a factor bin so its ``sum`` equals the scenario ``target`` loss.

    The two contribution partitions accumulate the same element losses in
    different groupings, so their totals can differ by an ulp or two. The
    position partition defines ``target``; here the factor partition is
    aligned onto it with the smallest reachable correction. Only nonzero
    entries move, so genuine zero contributions stay exactly zero. Where the
    target is unreachable in float64 the residual is absorbed into the
    largest bin, leaving a gap of at most a few ulps.
    """
    if sum(values) == target:
        return
    nonzero = [i for i, value in enumerate(values) if value != 0.0]
    if not nonzero:
        return
    natural = sum(values)
    target_step = math.ulp(abs(target)) if target != 0.0 else math.ulp(1.0)
    largest = max(nonzero, key=lambda i: abs(values[i]))

    def try_anchor(index: int) -> bool:
        original = values[index]
        values[index] = original + (target - natural)
        if sum(values) == target:
            return True
        # Exhaustive ulp probes are cheap only while the bin count is small;
        # real risk books reference at most a handful of factors.
        if len(values) <= 64:
            for step in {target_step, math.ulp(abs(original))}:
                for distance in range(1, 9):
                    values[index] = original + distance * step
                    if sum(values) == target:
                        return True
                    values[index] = original - distance * step
                    if sum(values) == target:
                        return True
        values[index] = original
        return False

    if try_anchor(largest) or try_anchor(nonzero[-1]):
        return
    # Unreachable in float64: absorb the residual into the largest bin, the
    # best one-ulp approximation; every other entry stays untouched.
    values[largest] = values[largest] + (target - sum(values))


def _as_float(value: object) -> float:
    """Narrow a JSON number to a finite float or raise invalid_input.

    JSON integers have arbitrary precision, so a literal like 10**400 parses
    fine but cannot be represented as a risk number. Booleans are rejected
    even though ``bool`` is an ``int`` subclass.
    """
    if isinstance(value, bool):
        raise InvalidInput()
    try:
        result = float(value)  # type: ignore[arg-type]
    except (OverflowError, TypeError, ValueError):
        raise InvalidInput()
    if not math.isfinite(result):
        raise InvalidInput()
    return result


def _strict_float(value: object) -> float:
    """Narrow to a finite float demanding a JSON number type.

    Unlike :func:`_as_float`, numeric-looking strings are rejected (only
    ``int``/``float`` pass), and oversized integers surface cleanly even
    though :func:`math.isfinite` raises ``OverflowError`` on them.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInput()
    try:
        result = float(value)
    except OverflowError:
        raise InvalidInput()
    if not math.isfinite(result):
        raise InvalidInput()
    return result


def _strict_int(value: object) -> int:
    """Narrow to a JSON integer, rejecting booleans and floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInput()
    return value


def _strict_int(value: object) -> int:
    """Narrow to a JSON integer, rejecting booleans and floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInput()
    return value


# Acklam's rational approximation to the standard normal inverse CDF. It is
# accurate to roughly 1e-9 across the whole (0, 1) range, plenty for a risk
# metric, and relies only on :mod:`math` so no extra dependency is needed.
# The tail numerator/denominator coefficients (c/d in the reference)...
_NORM_TAIL_NUM = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_NORM_TAIL_DEN = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
# ...and the central numerator/denominator coefficients (a/b).
_NORM_CENTER_NUM = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_NORM_CENTER_DEN = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)


def normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF for ``p`` strictly in (0, 1)."""
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        numerator = (
            ((((_NORM_TAIL_NUM[0] * q + _NORM_TAIL_NUM[1]) * q + _NORM_TAIL_NUM[2])
              * q + _NORM_TAIL_NUM[3]) * q + _NORM_TAIL_NUM[4]) * q
            + _NORM_TAIL_NUM[5]
        )
        denominator = (
            (((_NORM_TAIL_DEN[0] * q + _NORM_TAIL_DEN[1]) * q + _NORM_TAIL_DEN[2])
             * q + _NORM_TAIL_DEN[3]) * q + 1.0
        )
        x = numerator / denominator
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        numerator = (
            ((((_NORM_CENTER_NUM[0] * r + _NORM_CENTER_NUM[1]) * r
               + _NORM_CENTER_NUM[2]) * r + _NORM_CENTER_NUM[3]) * r
              + _NORM_CENTER_NUM[4]) * r + _NORM_CENTER_NUM[5]
        ) * q
        denominator = (
            ((((_NORM_CENTER_DEN[0] * r + _NORM_CENTER_DEN[1]) * r
               + _NORM_CENTER_DEN[2]) * r + _NORM_CENTER_DEN[3]) * r
              + _NORM_CENTER_DEN[4]) * r + 1.0
        )
        x = numerator / denominator
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        numerator = (
            ((((_NORM_TAIL_NUM[0] * q + _NORM_TAIL_NUM[1]) * q + _NORM_TAIL_NUM[2])
              * q + _NORM_TAIL_NUM[3]) * q + _NORM_TAIL_NUM[4]) * q
            + _NORM_TAIL_NUM[5]
        )
        denominator = (
            (((_NORM_TAIL_DEN[0] * q + _NORM_TAIL_DEN[1]) * q + _NORM_TAIL_DEN[2])
             * q + _NORM_TAIL_DEN[3]) * q + 1.0
        )
        x = -numerator / denominator
    # One Halley step against the CDF (via erfc) takes the ~1e-9 rational
    # approximation to machine precision.
    error = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    correction = error * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - correction / (1.0 + 0.5 * x * correction)


def normal_density(z: float) -> float:
    """Standard-normal probability density at ``z``."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


class Service:
    """Risk engine service."""

    name = "riskscope"
    version = __version__

    def health(self) -> dict[str, str]:
        return {"status": "ok", "service": self.name, "version": self.version}

    def historical_var(self, raw: bytes | str) -> dict:
        """Validate a historical VaR request and compute the result.

        ``raw`` is the unparsed request body. Parse failures and non-object
        payloads raise :class:`InvalidRequest`; everything else that is
        well-formed JSON but semantically wrong raises :class:`InvalidInput`
        or :class:`RequestTooLarge`.
        """
        return self._historical_var(self._load_object(raw))

    def stress_test(self, raw: bytes | str) -> dict:
        """Validate a batch sensitivity stress-test request and run it.

        Each scenario shocks the whole position book: a position's loss on a
        factor is ``-sensitivity * shock``. Missing referenced factors are
        zero shocks; extra scenario factors are ignored. Parse failures and
        non-object payloads raise :class:`InvalidRequest`; semantic problems
        raise :class:`InvalidInput` or :class:`RequestTooLarge`.
        """
        return self._stress_test(self._load_object(raw))

    def var_backtest(self, raw: bytes | str) -> dict:
        """Validate a VaR backtesting request and run the Kupiec POF test.

        Each observation pairs a daily VaR forecast with realized P&L; a
        breach is ``-realized_pnl > var`` (equality does not breach). The
        Kupiec unconditional-coverage likelihood-ratio statistic compares
        the breach rate with the predicted tail probability ``1 -
        confidence``. Parse failures and non-object payloads raise
        :class:`InvalidRequest`; semantic problems raise
        :class:`InvalidInput` or :class:`RequestTooLarge`.
        """
        return self._var_backtest(self._load_object(raw))

    def covariance_estimate(self, raw: bytes | str) -> dict:
        """Validate a covariance-estimation request and compute it.

        The request declares the risk ``factors`` (their order fixes every
        output vector and matrix axis) and synchronized ``observations`` of
        ``factor_returns``. The response echoes ``factors`` and reports
        ``observation_count``, the sample ``means`` and ``volatilities``
        per factor, and the sample ``covariance_matrix`` and
        ``correlation_matrix`` (row/column order follows ``factors``).
        Parse failures and non-object payloads raise
        :class:`InvalidRequest`; semantic problems raise
        :class:`InvalidInput` or :class:`RequestTooLarge`.
        """
        return self._covariance_estimate(self._load_object(raw))

    def parametric_var(self, raw: bytes | str) -> dict:
        """Validate a parametric (zero-mean Delta-Normal) VaR request.

        The request declares the risk ``factors`` (their order fixes the
        sensitivity vector and the covariance axes), per-position
        ``sensitivities`` (missing factors contribute zero), and a
        ``covariance_matrix`` aligned with the factors. The aggregated
        sensitivity vector ``s`` drives ``variance = sᵀΣs``,
        ``volatility = sqrt(variance)``, ``var = z·volatility`` and
        ``expected_shortfall = φ(z)·volatility/(1-confidence)`` for the
        standard-normal quantile ``z`` and density ``φ``. Parse failures
        and non-object payloads raise :class:`InvalidRequest`; semantic
        problems raise :class:`InvalidInput` or :class:`RequestTooLarge`.
        """
        return self._parametric_var(self._load_object(raw))

    def counterparty_exposure(self, raw: bytes | str) -> dict:
        """Validate a counterparty credit exposure request and compute it.

        Trades roll up into their netting set: ``gross_exposure`` sums the
        positive mark-to-markets, ``net_mtm`` sums all mark-to-markets, and
        ``potential_future_exposure`` sums the add-ons. ``exposure_at_default``
        floors the collateral-adjusted net exposure at zero, and
        ``expected_loss`` is ``exposure_at_default * pd * lgd`` for the
        netting set's counterparty. Parse failures and non-object payloads
        raise :class:`InvalidRequest`; semantic problems raise
        :class:`InvalidInput` or :class:`RequestTooLarge`.
        """
        return self._counterparty_exposure(self._load_object(raw))

    def liquidity_gap(self, raw: bytes | str) -> dict:
        """Validate a liquidity gap request and bucket the cashflows.

        Each cashflow lands in the first bucket whose day is not smaller
        than the cashflow's day; a liquid asset contributes its
        haircut-discounted value from the first bucket whose day is not
        smaller than its ``available_day``. Per bucket the response reports
        the net and cumulative net cashflow, the cumulative available
        liquidity, the surplus, and the required funding; the earliest
        negative-surplus bucket is singled out. Parse failures and
        non-object payloads raise :class:`InvalidRequest`; semantic problems
        raise :class:`InvalidInput`.
        """
        return self._liquidity_gap(self._load_object(raw))

    def _load_object(self, raw: bytes | str) -> dict:
        try:
            payload = json.loads(raw, parse_constant=_reject_constant)
        except _JsonConstant:
            raise InvalidInput(message="NaN and Infinity are not allowed")
        except (TypeError, ValueError):
            raise InvalidRequest("request body must be valid JSON")
        if not isinstance(payload, dict):
            raise InvalidRequest("request body must be a JSON object")
        return payload

    def _historical_var(self, payload: dict) -> dict:
        currency = payload.get("currency", "USD")
        if not isinstance(currency, str):
            raise InvalidInput()

        confidence = payload.get("confidence")
        if not _is_finite_number(confidence) or not 0 < confidence < 1:
            raise InvalidInput()

        positions_raw = payload.get("positions")
        if not isinstance(positions_raw, list) or len(positions_raw) == 0:
            raise InvalidInput()
        positions: list[tuple[str, dict]] = []
        for position in positions_raw:
            if not isinstance(position, dict):
                raise InvalidInput()
            position_id = position.get("id")
            if not _is_nonempty_str(position_id):
                raise InvalidInput()
            sensitivities = position.get("sensitivities")
            if not isinstance(sensitivities, dict) or len(sensitivities) == 0:
                raise InvalidInput()
            for value in sensitivities.values():
                if not _is_finite_number(value):
                    raise InvalidInput()
            positions.append((position_id, sensitivities))

        observations_raw = payload.get("observations")
        if not isinstance(observations_raw, list):
            raise InvalidInput()
        if len(observations_raw) > MAX_OBSERVATIONS:
            raise RequestTooLarge(f"at most {MAX_OBSERVATIONS} observations are allowed")
        if len(observations_raw) < 2:
            raise InvalidInput()
        observations: list[tuple[str, dict]] = []
        for observation in observations_raw:
            if not isinstance(observation, dict):
                raise InvalidInput()
            date = observation.get("date")
            if not _is_nonempty_str(date):
                raise InvalidInput()
            factor_returns = observation.get("factor_returns")
            if not isinstance(factor_returns, dict) or len(factor_returns) == 0:
                raise InvalidInput()
            for value in factor_returns.values():
                if not _is_finite_number(value):
                    raise InvalidInput()
            observations.append((date, factor_returns))

        seen_ids: set[str] = set()
        for position_id, _ in positions:
            if position_id in seen_ids:
                raise InvalidInput("duplicate_position", "position ids must be unique")
            seen_ids.add(position_id)

        seen_dates: set[str] = set()
        for date, _ in observations:
            if date in seen_dates:
                raise InvalidInput("duplicate_observation", "observation dates must be unique")
            seen_dates.add(date)

        # Aggregate per-factor sensitivity across positions (input order).
        aggregate: dict[str, float] = {}
        for _, sensitivities in positions:
            for factor, value in sensitivities.items():
                aggregate[factor] = aggregate.get(factor, 0.0) + _as_float(value)
        factors = list(aggregate)

        for date, factor_returns in observations:
            for factor in factors:
                if factor not in factor_returns:
                    raise InvalidInput(
                        "missing_factor",
                        f"observation {date!r} lacks factor {factor!r}",
                    )

        losses: list[float] = []
        factor_losses: list[dict[str, float]] = []
        for date, factor_returns in observations:
            per_factor: dict[str, float] = {}
            for factor in factors:
                factor_loss = -aggregate[factor] * _as_float(factor_returns[factor])
                if not math.isfinite(factor_loss):
                    raise InvalidInput()
                per_factor[factor] = factor_loss
            total = sum(per_factor.values())
            if not math.isfinite(total):
                raise InvalidInput()
            factor_losses.append(per_factor)
            losses.append(total)

        count = len(observations)
        confidence_decimal = Decimal(str(confidence))
        var_index = (
            int((confidence_decimal * Decimal(count)).to_integral_value(rounding=ROUND_CEILING))
            - 1
        )
        tail_count = max(
            1,
            int(
                ((Decimal(1) - confidence_decimal) * Decimal(count)).to_integral_value(
                    rounding=ROUND_CEILING
                )
            ),
        )

        # Worst losses first; ties keep the original observation order.
        tail_indices = sorted(range(count), key=lambda i: (-losses[i], i))[:tail_count]

        contributions = {
            factor: sum(factor_losses[i][factor] for i in tail_indices) / tail_count
            for factor in factors
        }
        # Sum of per-factor tail averages == average tail loss, so ES is
        # computed from the contributions to keep the identity exact.
        expected_shortfall = sum(contributions.values())

        return {
            "currency": currency,
            "var": sorted(losses)[var_index],
            "expected_shortfall": expected_shortfall,
            "losses": [
                {"date": observations[i][0], "loss": losses[i]} for i in range(count)
            ],
            "factor_expected_shortfall_contributions": contributions,
        }

    @staticmethod
    def _kupiec_term(count: int, probability: float) -> float:
        """One ``count * ln(probability)`` log-likelihood term.

        ``0 * ln 0`` is defined as 0 here, so boundary outcomes (no breach
        or every day breaching) stay well defined. A positive count paired
        with a zero probability is an infinite (non-finite) term.
        """
        if count == 0:
            return 0.0
        if probability <= 0.0 or probability > 1.0:
            raise InvalidInput(message="backtest computation produced a non-finite result")
        return count * math.log(probability)

    def _var_backtest(self, payload: dict) -> dict:
        currency = payload.get("currency", "USD")
        if not isinstance(currency, str) or len(currency) == 0:
            raise InvalidInput()

        confidence = _strict_float(payload.get("confidence"))
        if not 0 < confidence < 1:
            raise InvalidInput()

        significance = _strict_float(payload.get("significance", 0.05))
        if not 0 < significance < 1:
            raise InvalidInput()

        observations_raw = payload.get("observations")
        if not isinstance(observations_raw, list):
            raise InvalidInput()
        if len(observations_raw) > MAX_OBSERVATIONS:
            raise RequestTooLarge(f"at most {MAX_OBSERVATIONS} observations are allowed")
        if len(observations_raw) < 2:
            raise InvalidInput()

        rows: list[dict] = []
        dates: list[str] = []
        for observation in observations_raw:
            if not isinstance(observation, dict):
                raise InvalidInput()
            date = observation.get("date")
            if not _is_nonempty_str(date):
                raise InvalidInput()
            var = _strict_float(observation.get("var"))
            if var < 0.0:
                raise InvalidInput()
            realized_pnl = _strict_float(observation.get("realized_pnl"))
            loss = -realized_pnl
            if not math.isfinite(loss):
                raise InvalidInput()
            dates.append(date)
            rows.append(
                {
                    "date": date,
                    "var": var,
                    "realized_pnl": realized_pnl,
                    "loss": loss,
                    "breach": loss > var,
                }
            )

        seen_dates: set[str] = set()
        for date in dates:
            if date in seen_dates:
                raise InvalidInput("duplicate_observation", "observation dates must be unique")
            seen_dates.add(date)

        n = len(rows)
        x = sum(1 for row in rows if row["breach"])
        breach_rate = x / n

        # Kupiec (1995) proportion-of-failures unconditional-coverage test.
        p = 1.0 - confidence
        q = x / n
        constrained = self._kupiec_term(n - x, 1.0 - p) + self._kupiec_term(x, p)
        unconstrained = self._kupiec_term(n - x, 1.0 - q) + self._kupiec_term(x, q)
        # The unconstrained likelihood is maximal, so the statistic is
        # non-negative; a small negative remainder is rounding noise (e.g.
        # when the breach rate equals the predicted tail probability).
        lr_statistic = -2.0 * (constrained - unconstrained)
        if lr_statistic < 0.0:
            lr_statistic = 0.0
        elif lr_statistic == 0.0:
            # Emit a canonical +0.0 rather than a signed zero.
            lr_statistic = 0.0
        if not math.isfinite(lr_statistic):
            raise InvalidInput(message="backtest computation overflowed")
        p_value = math.erfc(math.sqrt(lr_statistic / 2.0))
        if not math.isfinite(p_value):
            raise InvalidInput(message="backtest computation overflowed")

        return {
            "currency": currency,
            "confidence": confidence,
            "significance": significance,
            "observations": rows,
            "observation_count": n,
            "breach_count": x,
            "breach_rate": breach_rate,
            "kupiec": {
                "lr_statistic": lr_statistic,
                "p_value": p_value,
                "accepted": p_value >= significance,
            },
        }

    def _covariance_estimate(self, payload: dict) -> dict:
        factors_raw = payload.get("factors")
        if not isinstance(factors_raw, list) or len(factors_raw) == 0:
            raise InvalidInput()
        if len(factors_raw) > MAX_FACTORS:
            raise RequestTooLarge(f"at most {MAX_FACTORS} factors are allowed")
        factors: list[str] = []
        for factor in factors_raw:
            if not _is_nonempty_str(factor):
                raise InvalidInput()
            factors.append(factor)
        seen_factors: set[str] = set()
        for factor in factors:
            if factor in seen_factors:
                raise InvalidInput("duplicate_factor", "factor names must be unique")
            seen_factors.add(factor)

        observations_raw = payload.get("observations")
        if not isinstance(observations_raw, list):
            raise InvalidInput()
        if len(observations_raw) > MAX_OBSERVATIONS:
            raise RequestTooLarge(f"at most {MAX_OBSERVATIONS} observations are allowed")
        if len(observations_raw) < 2:
            raise InvalidInput()
        if len(factors) * len(observations_raw) > MAX_FACTOR_OBSERVATION_PAIRS:
            raise RequestTooLarge(
                "factors times observations must not exceed "
                f"{MAX_FACTOR_OBSERVATION_PAIRS}"
            )

        dates: list[str] = []
        returns: list[list[float]] = []
        for observation in observations_raw:
            if not isinstance(observation, dict):
                raise InvalidInput()
            date = observation.get("date")
            if not _is_nonempty_str(date):
                raise InvalidInput()
            factor_returns = observation.get("factor_returns")
            if not isinstance(factor_returns, dict):
                raise InvalidInput()
            # Only declared factors are read; extra factors and any other
            # observation fields are ignored.
            row: list[float] = []
            for factor in factors:
                if factor not in factor_returns:
                    raise InvalidInput(
                        "missing_factor",
                        f"observation {date!r} lacks factor {factor!r}",
                    )
                row.append(_strict_float(factor_returns[factor]))
            dates.append(date)
            returns.append(row)

        seen_dates: set[str] = set()
        for date in dates:
            if date in seen_dates:
                raise InvalidInput(
                    "duplicate_observation", "observation dates must be unique"
                )
            seen_dates.add(date)

        n = len(returns)
        k = len(factors)

        # Arithmetic means per factor, summed in observation (input) order;
        # observations are never re-sorted by date.
        means: list[float] = []
        for j in range(k):
            mean = sum(row[j] for row in returns) / n
            if not math.isfinite(mean):
                raise InvalidInput(
                    message="covariance computation produced a non-finite result"
                )
            means.append(mean)

        # Sample covariance: sum of demeaned cross-products over (n - 1).
        # Each pair is computed once and mirrored, so the matrix is exactly
        # symmetric.
        covariance = [[0.0] * k for _ in range(k)]
        for i in range(k):
            for j in range(i, k):
                total = sum(
                    (row[i] - means[i]) * (row[j] - means[j]) for row in returns
                )
                value = total / (n - 1)
                if not math.isfinite(value):
                    raise InvalidInput(
                        message="covariance computation produced a non-finite result"
                    )
                covariance[i][j] = value
                covariance[j][i] = value

        # Volatility is the non-negative square root of the variance; the
        # diagonal is a sum of squares, so it is never negative.
        volatilities: list[float] = []
        for i in range(k):
            volatility = math.sqrt(covariance[i][i])
            if not math.isfinite(volatility):
                raise InvalidInput(
                    message="covariance computation produced a non-finite result"
                )
            volatilities.append(volatility)

        correlation = [[0.0] * k for _ in range(k)]
        for i in range(k):
            for j in range(i, k):
                if volatilities[i] == 0.0 or volatilities[j] == 0.0:
                    # A constant factor co-moves with nothing, but is
                    # perfectly correlated with itself.
                    value = 1.0 if i == j else 0.0
                else:
                    denominator = volatilities[i] * volatilities[j]
                    if denominator == 0.0:
                        raise InvalidInput(
                            message="covariance computation produced a non-finite result"
                        )
                    value = covariance[i][j] / denominator
                if not math.isfinite(value):
                    raise InvalidInput(
                        message="covariance computation produced a non-finite result"
                    )
                correlation[i][j] = value
                correlation[j][i] = value

        return {
            "factors": factors,
            "observation_count": n,
            "means": means,
            "volatilities": volatilities,
            "covariance_matrix": covariance,
            "correlation_matrix": correlation,
        }

    def _parametric_var(self, payload: dict) -> dict:
        currency = payload.get("currency", "USD")
        if not _is_nonempty_str(currency):
            raise InvalidInput()

        confidence = _strict_float(payload.get("confidence"))
        if not 0 < confidence < 1:
            raise InvalidInput()

        factors_raw = payload.get("factors")
        if not isinstance(factors_raw, list) or len(factors_raw) == 0:
            raise InvalidInput()
        if len(factors_raw) > MAX_FACTORS:
            raise RequestTooLarge(f"at most {MAX_FACTORS} factors are allowed")
        factors: list[str] = []
        for factor in factors_raw:
            if not _is_nonempty_str(factor):
                raise InvalidInput()
            factors.append(factor)
        seen_factors: set[str] = set()
        for factor in factors:
            if factor in seen_factors:
                raise InvalidInput("duplicate_factor", "factor names must be unique")
            seen_factors.add(factor)

        positions_raw = payload.get("positions")
        if not isinstance(positions_raw, list) or len(positions_raw) == 0:
            raise InvalidInput()
        if len(positions_raw) > MAX_POSITIONS:
            raise RequestTooLarge(f"at most {MAX_POSITIONS} positions are allowed")
        if len(factors) * len(positions_raw) > MAX_FACTOR_POSITION_PAIRS:
            raise RequestTooLarge(
                "factors times positions must not exceed "
                f"{MAX_FACTOR_POSITION_PAIRS}"
            )
        positions: list[tuple[str, dict[str, float]]] = []
        for position in positions_raw:
            if not isinstance(position, dict):
                raise InvalidInput()
            position_id = position.get("id")
            if not _is_nonempty_str(position_id):
                raise InvalidInput()
            sensitivities_raw = position.get("sensitivities")
            if not isinstance(sensitivities_raw, dict):
                raise InvalidInput()
            # An empty sensitivities object is allowed: the position simply
            # contributes zero on every factor.
            sensitivities = {
                factor: _strict_float(value)
                for factor, value in sensitivities_raw.items()
            }
            positions.append((position_id, sensitivities))

        seen_position_ids: set[str] = set()
        for position_id, _ in positions:
            if position_id in seen_position_ids:
                raise InvalidInput("duplicate_position", "position ids must be unique")
            seen_position_ids.add(position_id)

        n = len(factors)
        covariance_raw = payload.get("covariance_matrix")
        # Shape and element finiteness are ordinary input problems; only
        # asymmetry and non-semidefiniteness earn the invalid_covariance code.
        if not isinstance(covariance_raw, list) or len(covariance_raw) != n:
            raise InvalidInput(
                message="covariance_matrix must be an n x n matrix aligned with factors"
            )
        covariance: list[list[float]] = []
        for row in covariance_raw:
            if not isinstance(row, list) or len(row) != n:
                raise InvalidInput(
                    message="covariance_matrix must be an n x n matrix aligned with factors"
                )
            covariance.append([_strict_float(value) for value in row])

        # Sensitivities may only reference declared factors; missing entries
        # are aggregated as zero. This runs before the structural covariance
        # checks so an unknown factor is reported even when the matrix is also
        # bad, matching the documented error ordering.
        aggregate = [0.0] * n
        for _, sensitivities in positions:
            for factor in sensitivities:
                if factor not in seen_factors:
                    raise InvalidInput(
                        "unknown_factor",
                        f"sensitivity references unknown factor {factor!r}",
                    )
        for _, sensitivities in positions:
            for j, factor in enumerate(factors):
                if factor in sensitivities:
                    aggregate[j] += sensitivities[factor]
        for value in aggregate:
            if not math.isfinite(value):
                raise InvalidInput(
                    message="parametric VaR computation produced a non-finite result"
                )

        matrix_scale = 1.0
        for row in covariance:
            for value in row:
                matrix_scale = max(matrix_scale, abs(value))
        tolerance = COVARIANCE_REL_TOL * matrix_scale

        # Symmetry within the relative tolerance. A matrix this close to
        # symmetric is canonicalized onto its symmetric part so the same
        # tolerance governs the semidefiniteness check and every later dot
        # product.
        sym = [row[:] for row in covariance]
        for i in range(n):
            for j in range(i + 1, n):
                upper = covariance[i][j]
                lower = covariance[j][i]
                if abs(upper - lower) > tolerance:
                    raise InvalidInput(
                        "invalid_covariance",
                        "covariance_matrix must be symmetric",
                    )
                averaged = 0.5 * (upper + lower)
                sym[i][j] = averaged
                sym[j][i] = averaged

        # Positive semidefiniteness through LDLᵀ without pivoting. A pivot
        # within the tolerance of zero is accepted only when its whole cross
        # row is also within tolerance; the factor is then constant on the
        # preceding space, so the pivot is skipped and the untouched block
        # (the exact Schur complement when the cross row is zero) is tested
        # next. This keeps diag(0, 1) valid while rejecting [[0,1],[1,0]].
        schur = [row[:] for row in sym]
        for k in range(n):
            pivot = schur[k][k]
            if pivot < -tolerance:
                raise InvalidInput(
                    "invalid_covariance",
                    "covariance_matrix must be positive semidefinite",
                )
            if pivot <= tolerance:
                for i in range(k + 1, n):
                    if abs(schur[i][k]) > tolerance:
                        raise InvalidInput(
                            "invalid_covariance",
                            "covariance_matrix must be positive semidefinite",
                        )
                continue
            for i in range(k + 1, n):
                factor_ratio = schur[i][k] / pivot
                for j in range(k + 1, n):
                    schur[i][j] -= factor_ratio * schur[k][j]

        # variance = sᵀΣs, summed from the symmetric part. The absolute sum
        # bounds the magnitude any rounding slip could reach, so a negative
        # result inside the same relative tolerance is numerical noise.
        variance = 0.0
        absolute_scale = 0.0
        for i in range(n):
            diagonal_term = aggregate[i] * aggregate[i] * sym[i][i]
            variance += diagonal_term
            absolute_scale += abs(diagonal_term)
            for j in range(i + 1, n):
                cross_term = 2.0 * aggregate[i] * aggregate[j] * sym[i][j]
                variance += cross_term
                absolute_scale += abs(cross_term)
        if not math.isfinite(variance):
            raise InvalidInput(
                message="parametric VaR computation produced a non-finite result"
            )
        if variance < 0.0:
            if variance >= -COVARIANCE_REL_TOL * max(1.0, absolute_scale):
                variance = 0.0
            else:
                raise InvalidInput(
                    "invalid_covariance",
                    "covariance_matrix must be positive semidefinite",
                )

        volatility = math.sqrt(variance)
        if volatility == 0.0:
            # A degenerate book has no loss distribution: both tail metrics
            # are exactly zero regardless of the confidence level.
            var = 0.0
            expected_shortfall = 0.0
        else:
            z = normal_quantile(confidence)
            density = normal_density(z)
            var = z * volatility
            expected_shortfall = density * volatility / (1.0 - confidence)
            for value in (z, density, var, expected_shortfall):
                if not math.isfinite(value):
                    raise InvalidInput(
                        message="parametric VaR computation produced a non-finite result"
                    )

        return {
            "currency": currency,
            "confidence": confidence,
            "factors": factors,
            "aggregate_sensitivities": aggregate,
            "variance": variance,
            "volatility": volatility,
            "var": var,
            "expected_shortfall": expected_shortfall,
        }

    def _stress_test(self, payload: dict) -> dict:
        currency = payload.get("currency", "USD")
        if not isinstance(currency, str) or len(currency) == 0:
            raise InvalidInput()

        positions_raw = payload.get("positions")
        if not isinstance(positions_raw, list) or len(positions_raw) == 0:
            raise InvalidInput()
        positions: list[tuple[str, dict[str, float]]] = []
        for position in positions_raw:
            if not isinstance(position, dict):
                raise InvalidInput()
            position_id = position.get("id")
            if not _is_nonempty_str(position_id):
                raise InvalidInput()
            sensitivities_raw = position.get("sensitivities")
            if not isinstance(sensitivities_raw, dict) or len(sensitivities_raw) == 0:
                raise InvalidInput()
            # Narrow once so multiplication can never hit an oversized int.
            sensitivities = {
                factor: _as_float(value)
                for factor, value in sensitivities_raw.items()
            }
            positions.append((position_id, sensitivities))

        scenarios_raw = payload.get("scenarios")
        if not isinstance(scenarios_raw, list) or len(scenarios_raw) == 0:
            raise InvalidInput()
        scenarios: list[tuple[str, dict[str, float]]] = []
        for scenario in scenarios_raw:
            if not isinstance(scenario, dict):
                raise InvalidInput()
            scenario_id = scenario.get("id")
            if not _is_nonempty_str(scenario_id):
                raise InvalidInput()
            shocks_raw = scenario.get("factor_shocks")
            if not isinstance(shocks_raw, dict) or len(shocks_raw) == 0:
                raise InvalidInput()
            shocks = {factor: _as_float(value) for factor, value in shocks_raw.items()}
            scenarios.append((scenario_id, shocks))

        if len(scenarios) > MAX_SCENARIOS:
            raise RequestTooLarge(f"at most {MAX_SCENARIOS} scenarios are allowed")
        if len(scenarios) * len(positions) > MAX_SCENARIO_POSITION_PAIRS:
            raise RequestTooLarge(
                f"scenarios times positions must not exceed {MAX_SCENARIO_POSITION_PAIRS}"
            )

        seen_position_ids: set[str] = set()
        for position_id, _ in positions:
            if position_id in seen_position_ids:
                raise InvalidInput("duplicate_position", "position ids must be unique")
            seen_position_ids.add(position_id)

        seen_scenario_ids: set[str] = set()
        for scenario_id, _ in scenarios:
            if scenario_id in seen_scenario_ids:
                raise InvalidInput("duplicate_scenario", "scenario ids must be unique")
            seen_scenario_ids.add(scenario_id)

        results: list[dict] = []
        # Every position and every factor it references stays visible, with
        # zero contributions when a scenario leaves it unshocked. The wire
        # format sorts keys, so totals are folded in sorted-key order to keep
        # ``sum(contributions.values()) == loss`` exact after JSON round-trip.
        position_ids = [position_id for position_id, _ in positions]
        sorted_position_ids = sorted(position_ids)
        factor_order: list[str] = []
        seen_factors: set[str] = set()
        for _, sensitivities in positions:
            for factor in sensitivities:
                if factor not in seen_factors:
                    seen_factors.add(factor)
                    factor_order.append(factor)
        sorted_factors = sorted(factor_order)
        factor_index = {factor: i for i, factor in enumerate(sorted_factors)}

        worst_index = 0
        worst_loss = 0.0
        have_worst = False
        for scenario_index, (scenario_id, shocks) in enumerate(scenarios):
            position_contributions = {position_id: 0.0 for position_id in position_ids}
            factor_subtotals = [0.0 for _ in sorted_factors]
            for position_id, sensitivities in positions:
                subtotal = 0.0
                for factor, sensitivity in sensitivities.items():
                    # A referenced factor absent from the scenario is a zero
                    # shock; factors only the scenario names are ignored.
                    factor_loss = -sensitivity * shocks.get(factor, 0.0)
                    if not math.isfinite(factor_loss):
                        raise InvalidInput(message="stress computation overflowed")
                    subtotal += factor_loss
                    factor_subtotals[factor_index[factor]] += factor_loss
                    if not math.isfinite(subtotal) or not math.isfinite(
                        factor_subtotals[factor_index[factor]]
                    ):
                        raise InvalidInput(message="stress computation overflowed")
                position_contributions[position_id] = subtotal

            # Fold in the same sorted-key order the JSON body is delivered in.
            total = sum(position_contributions[pid] for pid in sorted_position_ids)
            if not math.isfinite(total) or not math.isfinite(sum(factor_subtotals)):
                raise InvalidInput(message="stress computation overflowed")
            _reconcile_totals(factor_subtotals, total)

            results.append(
                {
                    "id": scenario_id,
                    "loss": total,
                    "position_loss_contributions": position_contributions,
                    "factor_loss_contributions": {
                        factor: factor_subtotals[factor_index[factor]]
                        for factor in factor_order
                    },
                }
            )
            # Strictly greater keeps the earliest scenario on a tie.
            if not have_worst or total > worst_loss:
                have_worst = True
                worst_loss = total
                worst_index = scenario_index

        return {
            "currency": currency,
            "results": results,
            "worst_scenario": {
                "id": scenarios[worst_index][0],
                "loss": worst_loss,
            },
        }

    def _counterparty_exposure(self, payload: dict) -> dict:
        currency = payload.get("currency", "USD")
        if not _is_nonempty_str(currency):
            raise InvalidInput()

        counterparties_raw = payload.get("counterparties")
        if not isinstance(counterparties_raw, list) or len(counterparties_raw) == 0:
            raise InvalidInput()
        counterparties: list[dict] = []
        for counterparty in counterparties_raw:
            if not isinstance(counterparty, dict):
                raise InvalidInput()
            counterparty_id = counterparty.get("id")
            if not _is_nonempty_str(counterparty_id):
                raise InvalidInput()
            pd = _strict_float(counterparty.get("pd"))
            lgd = _strict_float(counterparty.get("lgd"))
            if not 0.0 <= pd <= 1.0 or not 0.0 <= lgd <= 1.0:
                raise InvalidInput()
            counterparties.append({"id": counterparty_id, "pd": pd, "lgd": lgd})

        netting_sets_raw = payload.get("netting_sets")
        if not isinstance(netting_sets_raw, list) or len(netting_sets_raw) == 0:
            raise InvalidInput()
        if len(netting_sets_raw) > MAX_NETTING_SETS:
            raise RequestTooLarge(f"at most {MAX_NETTING_SETS} netting sets are allowed")
        netting_sets: list[dict] = []
        for netting_set in netting_sets_raw:
            if not isinstance(netting_set, dict):
                raise InvalidInput()
            netting_set_id = netting_set.get("id")
            if not _is_nonempty_str(netting_set_id):
                raise InvalidInput()
            counterparty_id = netting_set.get("counterparty_id")
            if not _is_nonempty_str(counterparty_id):
                raise InvalidInput()
            collateral = _strict_float(netting_set.get("collateral"))
            if collateral < 0.0:
                raise InvalidInput()
            netting_sets.append(
                {
                    "id": netting_set_id,
                    "counterparty_id": counterparty_id,
                    "collateral": collateral,
                }
            )

        trades_raw = payload.get("trades")
        if not isinstance(trades_raw, list) or len(trades_raw) == 0:
            raise InvalidInput()
        if len(trades_raw) > MAX_TRADES:
            raise RequestTooLarge(f"at most {MAX_TRADES} trades are allowed")
        trades: list[dict] = []
        for trade in trades_raw:
            if not isinstance(trade, dict):
                raise InvalidInput()
            trade_id = trade.get("id")
            if not _is_nonempty_str(trade_id):
                raise InvalidInput()
            netting_set_id = trade.get("netting_set_id")
            if not _is_nonempty_str(netting_set_id):
                raise InvalidInput()
            mtm = _strict_float(trade.get("mtm"))
            add_on = _strict_float(trade.get("add_on"))
            if add_on < 0.0:
                raise InvalidInput()
            trades.append(
                {
                    "id": trade_id,
                    "netting_set_id": netting_set_id,
                    "mtm": mtm,
                    "add_on": add_on,
                }
            )

        seen_counterparty_ids: set[str] = set()
        for counterparty in counterparties:
            if counterparty["id"] in seen_counterparty_ids:
                raise InvalidInput(
                    "duplicate_counterparty", "counterparty ids must be unique"
                )
            seen_counterparty_ids.add(counterparty["id"])

        seen_netting_set_ids: set[str] = set()
        for netting_set in netting_sets:
            if netting_set["id"] in seen_netting_set_ids:
                raise InvalidInput(
                    "duplicate_netting_set", "netting set ids must be unique"
                )
            seen_netting_set_ids.add(netting_set["id"])

        seen_trade_ids: set[str] = set()
        for trade in trades:
            if trade["id"] in seen_trade_ids:
                raise InvalidInput("duplicate_trade", "trade ids must be unique")
            seen_trade_ids.add(trade["id"])

        counterparty_by_id = {cp["id"]: cp for cp in counterparties}
        for netting_set in netting_sets:
            if netting_set["counterparty_id"] not in counterparty_by_id:
                raise InvalidInput(
                    message="netting set references an unknown counterparty"
                )
        netting_set_ids = {ns["id"] for ns in netting_sets}
        for trade in trades:
            if trade["netting_set_id"] not in netting_set_ids:
                raise InvalidInput(message="trade references an unknown netting set")

        amount_keys = (
            "gross_exposure",
            "net_mtm",
            "potential_future_exposure",
            "collateral",
            "exposure_at_default",
            "expected_loss",
        )

        netting_set_metrics: list[dict] = []
        for netting_set in netting_sets:
            gross_exposure = 0.0
            net_mtm = 0.0
            potential_future_exposure = 0.0
            trade_count = 0
            for trade in trades:
                if trade["netting_set_id"] != netting_set["id"]:
                    continue
                trade_count += 1
                gross_exposure += max(trade["mtm"], 0.0)
                net_mtm += trade["mtm"]
                potential_future_exposure += trade["add_on"]
            # Over-collateralization floors the exposure at zero; it never
            # turns negative.
            exposure_at_default = max(
                net_mtm + potential_future_exposure - netting_set["collateral"], 0.0
            )
            counterparty = counterparty_by_id[netting_set["counterparty_id"]]
            expected_loss = (
                exposure_at_default * counterparty["pd"] * counterparty["lgd"]
            )
            for value in (
                gross_exposure,
                net_mtm,
                potential_future_exposure,
                exposure_at_default,
                expected_loss,
            ):
                if not math.isfinite(value):
                    raise InvalidInput(
                        message="exposure computation produced a non-finite result"
                    )
            netting_set_metrics.append(
                {
                    "id": netting_set["id"],
                    "counterparty_id": netting_set["counterparty_id"],
                    "trade_count": trade_count,
                    "gross_exposure": gross_exposure,
                    "net_mtm": net_mtm,
                    "potential_future_exposure": potential_future_exposure,
                    "collateral": netting_set["collateral"],
                    "exposure_at_default": exposure_at_default,
                    "expected_loss": expected_loss,
                }
            )

        # Aggregate in input order at both levels so the portfolio totals are
        # exactly the sequential sum of the counterparty details.
        counterparty_metrics: list[dict] = []
        for counterparty in counterparties:
            totals = {key: 0.0 for key in amount_keys}
            for metrics in netting_set_metrics:
                if metrics["counterparty_id"] != counterparty["id"]:
                    continue
                for key in amount_keys:
                    totals[key] += metrics[key]
            for value in totals.values():
                if not math.isfinite(value):
                    raise InvalidInput(
                        message="exposure computation produced a non-finite result"
                    )
            counterparty_metrics.append({"id": counterparty["id"], **totals})

        portfolio_totals = {key: 0.0 for key in amount_keys}
        for metrics in counterparty_metrics:
            for key in amount_keys:
                portfolio_totals[key] += metrics[key]
        for value in portfolio_totals.values():
            if not math.isfinite(value):
                raise InvalidInput(
                    message="exposure computation produced a non-finite result"
                )

        return {
            "currency": currency,
            "netting_sets": netting_set_metrics,
            "counterparties": counterparty_metrics,
            "portfolio_totals": portfolio_totals,
        }

    def _liquidity_gap(self, payload: dict) -> dict:
        currency = payload.get("currency", "USD")
        if not _is_nonempty_str(currency):
            raise InvalidInput()

        buckets_raw = payload.get("buckets")
        if not isinstance(buckets_raw, list) or len(buckets_raw) == 0:
            raise InvalidInput()
        bucket_days: list[int] = []
        for bucket in buckets_raw:
            day = _strict_int(bucket)
            if day <= 0:
                raise InvalidInput()
            bucket_days.append(day)
        # Strictly increasing terms; duplicates cannot satisfy that.
        for earlier, later in zip(bucket_days, bucket_days[1:]):
            if later <= earlier:
                raise InvalidInput(
                    message="buckets must be strictly increasing"
                )

        cashflows_raw = payload.get("cashflows")
        if not isinstance(cashflows_raw, list) or len(cashflows_raw) == 0:
            raise InvalidInput()
        cashflows: list[dict] = []
        for cashflow in cashflows_raw:
            if not isinstance(cashflow, dict):
                raise InvalidInput()
            cashflow_id = cashflow.get("id")
            if not _is_nonempty_str(cashflow_id):
                raise InvalidInput()
            day = _strict_int(cashflow.get("day"))
            if day <= 0:
                raise InvalidInput()
            amount = _strict_float(cashflow.get("amount"))
            cashflows.append({"id": cashflow_id, "day": day, "amount": amount})

        assets_raw = payload.get("liquid_assets", [])
        if not isinstance(assets_raw, list):
            raise InvalidInput()
        assets: list[dict] = []
        for asset in assets_raw:
            if not isinstance(asset, dict):
                raise InvalidInput()
            asset_id = asset.get("id")
            if not _is_nonempty_str(asset_id):
                raise InvalidInput()
            market_value = _strict_float(asset.get("market_value"))
            if market_value < 0.0:
                raise InvalidInput()
            haircut = _strict_float(asset.get("haircut"))
            if not 0.0 <= haircut <= 1.0:
                raise InvalidInput()
            available_day = _strict_int(asset.get("available_day"))
            if available_day < 0:
                raise InvalidInput()
            assets.append(
                {
                    "id": asset_id,
                    "market_value": market_value,
                    "haircut": haircut,
                    "available_day": available_day,
                }
            )

        seen_cashflow_ids: set[str] = set()
        for cashflow in cashflows:
            if cashflow["id"] in seen_cashflow_ids:
                raise InvalidInput(
                    "duplicate_cashflow", "cashflow ids must be unique"
                )
            seen_cashflow_ids.add(cashflow["id"])

        seen_asset_ids: set[str] = set()
        for asset in assets:
            if asset["id"] in seen_asset_ids:
                raise InvalidInput("duplicate_asset", "asset ids must be unique")
            seen_asset_ids.add(asset["id"])

        final_day = bucket_days[-1]

        # A cashflow belongs to the first bucket not earlier than its day;
        # anything past the final bucket has no home and is rejected.
        net_cashflows = [0.0 for _ in bucket_days]
        for cashflow in cashflows:
            if cashflow["day"] > final_day:
                raise InvalidInput(
                    message="cashflow day exceeds the final bucket"
                )
            for index, bucket_day in enumerate(bucket_days):
                if bucket_day >= cashflow["day"]:
                    net_cashflows[index] += cashflow["amount"]
                    break

        # An asset is usable from the first bucket not earlier than its
        # available_day; one maturing past the final bucket simply never
        # contributes.
        liquidity_steps = [0.0 for _ in bucket_days]
        for asset in assets:
            discounted = asset["market_value"] * (1.0 - asset["haircut"])
            if not math.isfinite(discounted):
                raise InvalidInput(
                    message="liquidity computation produced a non-finite result"
                )
            for index, bucket_day in enumerate(bucket_days):
                if bucket_day >= asset["available_day"]:
                    liquidity_steps[index] += discounted
                    break

        buckets: list[dict] = []
        cumulative_net_cashflow = 0.0
        available_liquidity = 0.0
        earliest_shortfall: dict | None = None
        for index, bucket_day in enumerate(bucket_days):
            net_cashflow = net_cashflows[index]
            cumulative_net_cashflow += net_cashflow
            available_liquidity += liquidity_steps[index]
            surplus = cumulative_net_cashflow + available_liquidity
            required_funding = max(-surplus, 0.0)
            for value in (
                net_cashflow,
                cumulative_net_cashflow,
                available_liquidity,
                surplus,
                required_funding,
            ):
                if not math.isfinite(value):
                    raise InvalidInput(
                        message="liquidity computation produced a non-finite result"
                    )
            buckets.append(
                {
                    "day": bucket_day,
                    "net_cashflow": net_cashflow,
                    "cumulative_net_cashflow": cumulative_net_cashflow,
                    "available_liquidity": available_liquidity,
                    "surplus": surplus,
                    "required_funding": required_funding,
                }
            )
            # The first negative-surplus bucket wins; later ties never
            # replace it.
            if earliest_shortfall is None and surplus < 0.0:
                earliest_shortfall = {
                    "day": bucket_day,
                    "required_funding": required_funding,
                }

        return {
            "currency": currency,
            "buckets": buckets,
            "earliest_shortfall": earliest_shortfall,
        }
