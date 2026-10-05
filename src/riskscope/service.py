"""Core service surface for RiskScope.

The frozen baseline reports process health; historical-simulation VaR and
expected shortfall now live behind :meth:`Service.historical_var` and batch
sensitivity stress tests behind :meth:`Service.stress_test`. Counterparty
credit exposure and expected loss live behind
:meth:`Service.counterparty_exposure`. Liquidity gap analysis by maturity
bucket lives behind :meth:`Service.liquidity_gap`. Sample covariance
estimation from synchronized factor returns lives behind
:meth:`Service.covariance_estimate`, with the RiskMetrics-style
exponentially weighted variant behind
:meth:`Service.ewma_covariance_estimate`. Zero-mean Delta-Normal
(parametric) VaR and expected shortfall live behind
:meth:`Service.parametric_var`, with their factor- and position-level
attribution behind :meth:`Service.parametric_var_attribution`.
Cross-book, multi-currency position
aggregation with FX conversion lives behind
:meth:`Service.portfolio_aggregate`. Declarative limit monitoring
against caller-supplied measurements lives behind
:meth:`Service.limit_check`. Multi-period expected credit loss from
term-structure default probabilities and exposure forecasts lives
behind :meth:`Service.expected_loss_schedule`. Foundation-IRB capital
requirements under the Basel corporate correlation and maturity
adjustment live behind :meth:`Service.irb_capital`. Discounted
cashflow valuation off a zero-rate curve, with key-rate sensitivities,
lives behind :meth:`Service.discounted_cashflow`. Term-default
probabilities derived from a single-period rating transition matrix,
together with the resulting expected losses, live behind
:meth:`Service.rating_migration_loss`. The public surface
stays backward compatible.
"""

from __future__ import annotations

import bisect
import json
import math
from decimal import Decimal, ROUND_CEILING
from statistics import NormalDist

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
MAX_FX_RATES = 1000
MAX_SENSITIVITY_ENTRIES = 1000000
MAX_LIMITS = 10000
MAX_MEASUREMENTS = 10000
MAX_SCHEDULE_COUNTERPARTIES = 1000
MAX_SCHEDULE_FACILITIES = 10000
MAX_SCHEDULE_PERIOD_FACILITY_PAIRS = 1000000
MAX_IRB_COUNTERPARTIES = 1000
MAX_IRB_FACILITIES = 10000
MAX_CURVE_POINTS = 100
MAX_CASHFLOWS = 100000
MAX_MIGRATION_RATINGS = 100
MAX_MIGRATION_HORIZON = 50
MAX_MIGRATION_COUNTERPARTIES = 10000

# Relative tolerance for covariance-matrix symmetry, positive
# semidefiniteness, and the tiny negative variance that floating-point
# rounding can leave behind.
_COVARIANCE_RTOL = 1e-12

# Absolute tolerance for transition-matrix row sums, the absorbing
# default row, and the probability snapping applied to results.
_MIGRATION_TOLERANCE = 1e-12

_STANDARD_NORMAL = NormalDist()


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


def _strict_day(value: object) -> int:
    """Narrow to a positive JSON integer day, rejecting oversized ints.

    A day is turned into a year fraction with ``day / 365``, so an
    integer too large to represent as a float is not a usable day.
    """
    day = _strict_int(value)
    if day <= 0:
        raise InvalidInput()
    try:
        float(day)
    except OverflowError:
        raise InvalidInput()
    return day


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

    def var_backtest_validation(self, raw: bytes | str) -> dict:
        """Validate a VaR backtest request and run the Christoffersen tests.

        The request semantics, defaults and breach decision are identical
        to :meth:`var_backtest`; on top of the Kupiec result the response
        reports the four first-order Markov transition counts (``n00``,
        ``n01``, ``n10``, ``n11``) and two tests: ``independence`` (a
        likelihood-ratio test of clustering, chi-square with one degree of
        freedom) and ``conditional_coverage`` (Kupiec plus independence,
        chi-square with two degrees of freedom). A predecessor state that
        never occurs leaves ``q0`` or ``q1`` null. Parse failures and
        non-object payloads raise :class:`InvalidRequest`; semantic
        problems raise :class:`InvalidInput` or :class:`RequestTooLarge`.
        """
        return self._var_backtest_validation(self._load_object(raw))

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

    def ewma_covariance_estimate(self, raw: bytes | str) -> dict:
        """Validate an EWMA covariance-estimation request and compute it.

        The request declares the risk ``factors`` (their order fixes every
        output vector and matrix axis), time-ordered ``observations`` of
        ``factor_returns`` (input order is the recursion order; they are
        never re-sorted by date) and an optional ``decay`` defaulting to
        ``0.94`` that must lie strictly inside ``(0, 1)``. A zero-mean
        RiskMetrics recursion starts from the first outer product
        ``Σ₁ = r₁r₁ᵀ`` and applies ``Σₜ = decay·Σₜ₋₁ + (1 - decay)·rₜrₜᵀ``.
        The response echoes ``factors`` and ``decay`` and reports
        ``observation_count``, ``volatilities`` (the non-negative square
        roots of the diagonal), ``covariance_matrix`` and
        ``correlation_matrix`` (row/column order follows ``factors``); the
        covariance matrix is exactly symmetric and can be passed straight
        to :meth:`parametric_var`. Parse failures and non-object payloads
        raise :class:`InvalidRequest`; semantic problems raise
        :class:`InvalidInput` or :class:`RequestTooLarge`.
        """
        return self._ewma_covariance_estimate(self._load_object(raw))

    def parametric_var(self, raw: bytes | str) -> dict:
        """Validate a parametric VaR request and compute the result.

        Zero-mean Delta-Normal method: the aggregate sensitivity vector
        ``s`` (in ``factors`` order) combines with the supplied covariance
        matrix ``Σ`` into ``variance = sᵀΣs``, ``volatility = sqrt(variance)``,
        ``var = z × volatility`` and ``expected_shortfall = φ(z) ×
        volatility / (1 - confidence)``, where ``z`` and ``φ`` are the
        standard normal quantile and density. A zero volatility yields
        ``0.0`` for both tail metrics. Parse failures and non-object
        payloads raise :class:`InvalidRequest`; an asymmetric or
        non-positive-semidefinite matrix raises :class:`InvalidInput` with
        code ``invalid_covariance``; other semantic problems raise
        :class:`InvalidInput` or :class:`RequestTooLarge`.
        """
        return self._parametric_var(self._load_object(raw))

    def parametric_var_attribution(self, raw: bytes | str) -> dict:
        """Validate a parametric VaR attribution request and compute it.

        The request semantics, defaults, factor/position ordering,
        covariance validation and size limits are identical to
        :meth:`parametric_var`. Alongside the unchanged portfolio metrics
        the response partitions the variance, VaR and expected shortfall
        by factor (``factor_attributions``) and by position
        (``position_attributions``): with the covariance loading ``c =
        Σs`` and portfolio volatility ``σ``, each layer reports
        ``variance_contribution = x·c`` (``x = s`` for a factor, the
        position sensitivity vector for a position), ``component_var =
        z × variance_contribution / σ`` and
        ``component_expected_shortfall = φ(z) × variance_contribution /
        ((1 - confidence) × σ)``. Contributions may be negative and zero
        entries are kept; the two layers each sum exactly to the matching
        portfolio value. At zero volatility the covariance loading is
        still returned from the matrix product but every contribution is
        ``0.0``. Parse failures and non-object payloads raise
        :class:`InvalidRequest`; an asymmetric or non-positive-semidefinite
        matrix raises :class:`InvalidInput` with code
        ``invalid_covariance``; a non-finite attribution result raises
        :class:`InvalidInput`; other semantic problems raise
        :class:`InvalidInput` or :class:`RequestTooLarge`.
        """
        return self._parametric_var_attribution(self._load_object(raw))

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

    def portfolio_aggregate(self, raw: bytes | str) -> dict:
        """Validate a cross-book, multi-currency aggregation request and run it.

        Every position's market value and sensitivities are converted into the
        reporting currency with its currency's FX rate, then rolled up by
        currency and by book. Parse failures and non-object payloads raise
        :class:`InvalidRequest`; semantic problems raise :class:`InvalidInput`,
        :class:`RequestTooLarge`, ``duplicate_position`` or
        ``missing_fx_rate``.
        """
        return self._portfolio_aggregate(self._load_object(raw))

    def limit_check(self, raw: bytes | str) -> dict:
        """Validate a limit-monitoring request and evaluate each limit.

        The caller computes the metrics; this check only compares each
        declared limit with the measurement referencing it. A limit with a
        measurement reports ``value``, ``utilization = value / limit`` and
        ``headroom = limit - value`` and is classified ``breach``
        (``value >= limit``), ``warning`` (``value >= limit *
        warning_ratio``) or ``ok``; a limit without a measurement reports
        nulls and ``no_data``. No history is kept and no other engine is
        consulted. Parse failures and non-object payloads raise
        :class:`InvalidRequest`; semantic problems raise
        :class:`InvalidInput` (including the ``duplicate_limit``,
        ``duplicate_measurement`` and ``unknown_limit`` codes) or
        :class:`RequestTooLarge`.
        """
        return self._limit_check(self._load_object(raw))

    def expected_loss_schedule(self, raw: bytes | str) -> dict:
        """Validate a multi-period expected-loss request and compute it.

        Each counterparty declares a term structure of ``cumulative_pd``
        aligned with ``periods``; each facility declares an ``ead``
        forecast on the same grid. The marginal default probability of
        the first period is the first cumulative value, later periods
        take the difference of adjacent cumulative values, and a
        facility's per-period ``discounted_expected_loss`` is
        ``ead * lgd * marginal_pd * discount_factor``. Facilities roll
        up into their counterparty and the counterparties into the
        portfolio total, all in input order. Parse failures and
        non-object payloads raise :class:`InvalidRequest`; semantic
        problems raise :class:`InvalidInput` (including the
        ``duplicate_counterparty``, ``duplicate_facility`` and
        ``unknown_counterparty`` codes) or :class:`RequestTooLarge`.
        """
        return self._expected_loss_schedule(self._load_object(raw))

    def irb_capital(self, raw: bytes | str) -> dict:
        """Validate a foundation-IRB capital request and compute it.

        Each facility borrows the PD of its counterparty. With the Basel
        corporate asset correlation ``R`` and maturity adjustment ``MA``,
        the capital requirement per unit exposure is

        ``K = lgd * (Phi((Phi^-1(PD) + sqrt(R) * Phi^-1(0.999)) /
        sqrt(1 - R)) - PD) * MA``;

        ``capital_requirement = ead * K`` and
        ``risk_weighted_assets = 12.5 * capital_requirement``. Facilities
        roll up into their counterparty and the counterparties into the
        portfolio total, all in input order. Parse failures and
        non-object payloads raise :class:`InvalidRequest`; semantic
        problems raise :class:`InvalidInput` (including the
        ``duplicate_counterparty``, ``duplicate_facility`` and
        ``unknown_counterparty`` codes) or :class:`RequestTooLarge`.
        """
        return self._irb_capital(self._load_object(raw))

    def discounted_cashflow(self, raw: bytes | str) -> dict:
        """Validate a discounted-cashflow request and value the positions.

        The zero curve is sorted ascending by ``day``; a cashflow landing
        exactly on a node takes that node's ``zero_rate``, one between
        adjacent nodes takes the day-linear interpolation, and one outside
        the first/last node fails the whole request. With ``t = day /
        365`` each cashflow contributes ``amount * exp(-zero_rate * t)``
        to the present value and ``amount * exp(-zero_rate * t) * t *
        0.0001`` times the node's interpolation weight to each curve
        node's key-rate sensitivity; ``dv01`` is the sum of the node
        sensitivities at that level. Parse failures and non-object
        payloads raise :class:`InvalidRequest`; semantic problems raise
        :class:`InvalidInput` (including the ``duplicate_curve_point``,
        ``duplicate_position`` and ``curve_out_of_range`` codes) or
        :class:`RequestTooLarge`.
        """
        return self._discounted_cashflow(self._load_object(raw))

    def rating_migration_loss(self, raw: bytes | str) -> dict:
        """Validate a rating-migration request and compute the losses.

        The single-period ``transition_matrix`` over the ordered
        ``ratings`` is applied ``horizon`` times, so row ``i`` of the
        horizon matrix holds the probabilities that a name rated
        ``ratings[i]`` today sits in each rating at the horizon; the
        probability of landing in ``default_rating`` is the cumulative
        PD. Each counterparty's ``expected_loss`` is
        ``ead * lgd * cumulative_pd``. The response echoes ``currency``,
        ``horizon``, ``ratings`` and ``default_rating``, returns the
        horizon transition matrix, the per-counterparty details in input
        order, a per-rating summary over the non-default initial ratings
        in ``ratings`` order, and ``portfolio_totals``. Parse failures
        and non-object payloads raise :class:`InvalidRequest`; semantic
        problems raise :class:`InvalidInput` (including the
        ``duplicate_rating`` and ``duplicate_counterparty`` codes) or
        :class:`RequestTooLarge`.
        """
        return self._rating_migration_loss(self._load_object(raw))

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

    def _parse_var_backtest(self, payload: dict) -> tuple[str, float, float, list[dict]]:
        """Validate a VaR backtest request into its shared inputs.

        Both the Kupiec backtest and the Christoffersen validation route
        accept exactly this request shape; the breach indicator on each
        row follows the frozen ``-realized_pnl > var`` rule (equality does
        not breach).
        """
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

        return currency, confidence, significance, rows

    def _run_var_backtest(
        self,
        currency: str,
        confidence: float,
        significance: float,
        rows: list[dict],
    ) -> dict:
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

    def _var_backtest(self, payload: dict) -> dict:
        currency, confidence, significance, rows = self._parse_var_backtest(payload)
        return self._run_var_backtest(currency, confidence, significance, rows)

    @staticmethod
    def _markov_term(count: int, probability: float) -> float:
        """One ``count * ln(probability)`` transition log-likelihood term.

        ``0 * ln 0`` is defined as 0 here. A transition that never leaves a
        given predecessor state leaves its conditional probability
        undefined (handled by the caller); any other positive count paired
        with a probability outside (0, 1] is a non-finite term.
        """
        if count == 0:
            return 0.0
        if probability <= 0.0 or probability > 1.0:
            raise InvalidInput(
                message="backtest validation produced a non-finite result"
            )
        return count * math.log(probability)

    def _run_var_backtest_validation(
        self,
        currency: str,
        confidence: float,
        significance: float,
        rows: list[dict],
    ) -> dict:
        result = self._run_var_backtest(currency, confidence, significance, rows)
        n = result["observation_count"]

        # Christoffersen (1998) first-order Markov transition counts over
        # adjacent breach states in observation order: n_ij counts a
        # transition from state i on one day to state j on the next.
        n00 = n01 = n10 = n11 = 0
        previous = rows[0]["breach"]
        for row in rows[1:]:
            current = row["breach"]
            if not previous and not current:
                n00 += 1
            elif not previous and current:
                n01 += 1
            elif previous and not current:
                n10 += 1
            else:
                n11 += 1
            previous = current

        q = (n01 + n11) / (n - 1)
        ln_l0 = (
            self._markov_term(n00 + n10, 1.0 - q)
            + self._markov_term(n01 + n11, q)
        )

        # A predecessor state that never occurs leaves its conditional
        # breach probability undefined; its likelihood contribution is 0.
        q0: float | None
        q1: float | None
        ln_l1 = 0.0
        if n00 + n01 > 0:
            q0 = n01 / (n00 + n01)
            ln_l1 += self._markov_term(n00, 1.0 - q0) + self._markov_term(n01, q0)
        else:
            q0 = None
        if n10 + n11 > 0:
            q1 = n11 / (n10 + n11)
            ln_l1 += self._markov_term(n10, 1.0 - q1) + self._markov_term(n11, q1)
        else:
            q1 = None

        if not math.isfinite(ln_l0) or not math.isfinite(ln_l1):
            raise InvalidInput(
                message="backtest validation produced a non-finite result"
            )

        # The constrained model nests in the two-parameter alternative, so
        # the statistic cannot be negative; a tiny negative remainder is
        # floating-point noise around zero.
        lr_ind = 2.0 * (ln_l1 - ln_l0)
        if lr_ind < 0.0:
            lr_ind = 0.0
        if not math.isfinite(lr_ind):
            raise InvalidInput(
                message="backtest validation produced a non-finite result"
            )
        ind_p_value = math.erfc(math.sqrt(lr_ind / 2.0))
        if not math.isfinite(ind_p_value):
            raise InvalidInput(
                message="backtest validation produced a non-finite result"
            )

        # Conditional coverage combines Kupiec's unconditional-coverage
        # statistic with the independence statistic and is chi-square with
        # two degrees of freedom, whose survival function is exp(-lr/2).
        cc_lr = result["kupiec"]["lr_statistic"] + lr_ind
        if not math.isfinite(cc_lr):
            raise InvalidInput(
                message="backtest validation produced a non-finite result"
            )
        cc_p_value = math.exp(-cc_lr / 2.0)
        if not math.isfinite(cc_p_value):
            raise InvalidInput(
                message="backtest validation produced a non-finite result"
            )

        result.update(
            {
                "n00": n00,
                "n01": n01,
                "n10": n10,
                "n11": n11,
                "independence": {
                    "lr_statistic": lr_ind,
                    "p_value": ind_p_value,
                    "accepted": ind_p_value >= significance,
                    "q": q,
                    "q0": q0,
                    "q1": q1,
                },
                "conditional_coverage": {
                    "lr_statistic": cc_lr,
                    "p_value": cc_p_value,
                    "accepted": cc_p_value >= significance,
                },
            }
        )
        return result

    def _var_backtest_validation(self, payload: dict) -> dict:
        currency, confidence, significance, rows = self._parse_var_backtest(payload)
        return self._run_var_backtest_validation(
            currency, confidence, significance, rows
        )

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

    def _ewma_covariance_estimate(self, payload: dict) -> dict:
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

        # RiskMetrics classic daily decay; the boundary values 0 and 1 are
        # both rejected (the recursion needs weights strictly on both terms).
        decay = _strict_float(payload.get("decay", 0.94))
        if not 0.0 < decay < 1.0:
            raise InvalidInput()

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
        innovation_weight = 1.0 - decay

        # Zero-mean RiskMetrics recursion. Σ₁ = r₁r₁ᵀ seeds the estimate,
        # then Σₜ = decay·Σₜ₋₁ + (1-decay)·rₜrₜᵀ in observation (input)
        # order; observations are never re-sorted by date. Each (i, j)
        # scalar chain is reduced independently and mirrored onto both
        # triangle halves, so the finished matrix is exactly symmetric.
        columns = [[row[j] for row in returns] for j in range(k)]
        covariance = [[0.0] * k for _ in range(k)]
        for i in range(k):
            column_i = columns[i]
            for j in range(i, k):
                column_j = columns[j]
                value = column_i[0] * column_j[0]
                for t in range(1, n):
                    value = (
                        decay * value
                        + innovation_weight * column_i[t] * column_j[t]
                    )
                if not math.isfinite(value):
                    raise InvalidInput(
                        message="covariance computation produced a non-finite result"
                    )
                covariance[i][j] = value
                covariance[j][i] = value

        # Volatility is the non-negative square root of the diagonal; the
        # EWMA variance is a non-negative combination of squares, so the
        # diagonal never goes negative.
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
                    # A zero-volatility factor co-moves with nothing, but is
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
            "decay": decay,
            "observation_count": n,
            "covariance_matrix": covariance,
            "volatilities": volatilities,
            "correlation_matrix": correlation,
        }

    @staticmethod
    def _check_covariance(matrix: list[list[float]]) -> None:
        """Enforce symmetry and positive semidefiniteness of ``matrix``.

        Both properties hold within a ``1e-12`` relative tolerance, so a
        matrix assembled from independently rounded estimates is not
        rejected on noise alone. Violations raise :class:`InvalidInput`
        with code ``invalid_covariance``.
        """
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = matrix[i][j], matrix[j][i]
                if abs(a - b) > _COVARIANCE_RTOL * max(abs(a), abs(b)):
                    raise InvalidInput(
                        "invalid_covariance", "covariance matrix must be symmetric"
                    )

        # Semidefinite Cholesky: a materially negative pivot, or a
        # materially nonzero column below a zero pivot, means the matrix
        # is not positive semidefinite. The scale comes from the largest
        # diagonal magnitude, so the tolerance is relative.
        scale = max(abs(matrix[i][i]) for i in range(n))
        tol = _COVARIANCE_RTOL * scale
        lower = [[0.0] * n for _ in range(n)]
        for j in range(n):
            pivot = matrix[j][j] - sum(lower[j][k] ** 2 for k in range(j))
            if pivot < -tol:
                raise InvalidInput(
                    "invalid_covariance",
                    "covariance matrix must be positive semidefinite",
                )
            if pivot <= 0.0:
                for i in range(j + 1, n):
                    residual = matrix[i][j] - sum(
                        lower[i][k] * lower[j][k] for k in range(j)
                    )
                    if abs(residual) > tol:
                        raise InvalidInput(
                            "invalid_covariance",
                            "covariance matrix must be positive semidefinite",
                        )
                continue
            lower[j][j] = math.sqrt(pivot)
            for i in range(j + 1, n):
                lower[i][j] = (
                    matrix[i][j]
                    - sum(lower[i][k] * lower[j][k] for k in range(j))
                ) / lower[j][j]

    def _parse_parametric_var(self, payload: dict) -> dict:
        """Validate the shared parametric-VaR request shape.

        Both :meth:`_parametric_var` and
        :meth:`_parametric_var_attribution` accept exactly this shape, so
        the defaults, ordering, covariance checks and size caps live here
        once. Returns the narrowed and validated inputs together with the
        aggregate sensitivity vector; no tail metrics are computed.
        """
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
            # Narrow once so aggregation can never hit an oversized int.
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

        declared = set(factors)
        for position_id, sensitivities in positions:
            for factor in sensitivities:
                if factor not in declared:
                    raise InvalidInput(
                        "unknown_factor",
                        f"position {position_id!r} references undeclared factor "
                        f"{factor!r}",
                    )

        matrix_raw = payload.get("covariance_matrix")
        n = len(factors)
        if not isinstance(matrix_raw, list) or len(matrix_raw) != n:
            raise InvalidInput()
        matrix: list[list[float]] = []
        for row in matrix_raw:
            if not isinstance(row, list) or len(row) != n:
                raise InvalidInput()
            matrix.append([_strict_float(value) for value in row])
        self._check_covariance(matrix)

        # Aggregate per-factor sensitivity across positions (input order);
        # a factor a position does not reference contributes zero.
        factor_index = {factor: j for j, factor in enumerate(factors)}
        aggregate = [0.0] * n
        for _, sensitivities in positions:
            for factor, value in sensitivities.items():
                aggregate[factor_index[factor]] += value
        for value in aggregate:
            if not math.isfinite(value):
                raise InvalidInput(
                    message="parametric computation produced a non-finite result"
                )

        return {
            "currency": currency,
            "confidence": confidence,
            "factors": factors,
            "positions": positions,
            "factor_index": factor_index,
            "matrix": matrix,
            "aggregate": aggregate,
        }

    @staticmethod
    def _parametric_loading_and_variance(
        aggregate: list[float], matrix: list[list[float]]
    ) -> tuple[list[float], float, float]:
        """Return ``(c = Σs, variance = sᵀc, abs_scale)``.

        ``abs_scale`` is the matching sum of absolute terms, the yardstick
        for deciding whether a negative variance is mere rounding noise.
        The accumulation runs row by row in a fixed order so callers that
        partition ``variance`` reproduce these exact addends.
        """
        n = len(aggregate)
        loading = [0.0] * n
        variance = 0.0
        abs_scale = 0.0
        for i in range(n):
            row_total = 0.0
            abs_row_total = 0.0
            for j in range(n):
                row_total += matrix[i][j] * aggregate[j]
                abs_row_total += abs(matrix[i][j]) * abs(aggregate[j])
            loading[i] = row_total
            variance += aggregate[i] * row_total
            abs_scale += abs(aggregate[i]) * abs_row_total
        return loading, variance, abs_scale

    @staticmethod
    def _settle_variance(variance: float, abs_scale: float) -> float:
        """Floor noise-scale negative variances at zero or reject them."""
        if not math.isfinite(variance):
            raise InvalidInput(
                message="parametric computation produced a non-finite result"
            )
        if variance < 0.0:
            if variance < -_COVARIANCE_RTOL * abs_scale:
                raise InvalidInput(
                    message="parametric computation produced a negative variance"
                )
            variance = 0.0
        elif variance == 0.0:
            # Emit a canonical +0.0 rather than a signed zero.
            variance = 0.0
        return variance

    def _parametric_var(self, payload: dict) -> dict:
        parsed = self._parse_parametric_var(payload)
        currency = parsed["currency"]
        confidence = parsed["confidence"]
        factors = parsed["factors"]
        matrix = parsed["matrix"]
        aggregate = parsed["aggregate"]

        _, variance, abs_scale = self._parametric_loading_and_variance(
            aggregate, matrix
        )
        variance = self._settle_variance(variance, abs_scale)

        volatility = math.sqrt(variance)
        if volatility == 0.0:
            var = 0.0
            expected_shortfall = 0.0
        else:
            z = _STANDARD_NORMAL.inv_cdf(confidence)
            density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            var = z * volatility
            expected_shortfall = density * volatility / (1.0 - confidence)
            if not math.isfinite(var) or not math.isfinite(expected_shortfall):
                raise InvalidInput(
                    message="parametric computation produced a non-finite result"
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

    def _parametric_var_attribution(self, payload: dict) -> dict:
        parsed = self._parse_parametric_var(payload)
        currency = parsed["currency"]
        confidence = parsed["confidence"]
        factors = parsed["factors"]
        positions = parsed["positions"]
        matrix = parsed["matrix"]
        aggregate = parsed["aggregate"]
        n = len(factors)

        loading, variance, abs_scale = self._parametric_loading_and_variance(
            aggregate, matrix
        )
        for value in loading:
            if not math.isfinite(value):
                raise InvalidInput(
                    message="parametric attribution produced a non-finite result"
                )
        variance = self._settle_variance(variance, abs_scale)

        volatility = math.sqrt(variance)
        if volatility == 0.0:
            # The covariance loading is still the matrix product, but every
            # contribution is exactly zero at zero volatility.
            var = 0.0
            expected_shortfall = 0.0
            factor_attributions = [
                {
                    "factor": factors[i],
                    "aggregate_sensitivity": aggregate[i],
                    "covariance_loading": loading[i],
                    "variance_contribution": 0.0,
                    "component_var": 0.0,
                    "component_expected_shortfall": 0.0,
                }
                for i in range(n)
            ]
            position_attributions = [
                {
                    "id": position_id,
                    "variance_contribution": 0.0,
                    "component_var": 0.0,
                    "component_expected_shortfall": 0.0,
                }
                for position_id, _ in positions
            ]
        else:
            z = _STANDARD_NORMAL.inv_cdf(confidence)
            density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            var = z * volatility
            expected_shortfall = density * volatility / (1.0 - confidence)
            if not math.isfinite(var) or not math.isfinite(expected_shortfall):
                raise InvalidInput(
                    message="parametric attribution produced a non-finite result"
                )

            # The two layers share one tail denominator per metric:
            # component VaR is ``z × contribution / σ`` and component ES is
            # ``φ(z) × contribution / ((1 - confidence) × σ)``. The products
            # are taken in the formula's order so the reported components are
            # the direct per-bin evaluation of those expressions.
            tail_scale = (1.0 - confidence) * volatility
            if tail_scale <= 0.0:
                raise InvalidInput(
                    message="parametric attribution produced a non-finite result"
                )

            # Factor layer: ``variance_contribution = s_i × c_i``. These are
            # exactly the addends accumulated into ``variance``, so their
            # sum matches the portfolio variance before reconciliation.
            factor_variance = [aggregate[i] * loading[i] for i in range(n)]

            # Position layer: ``variance_contribution = pᵀc`` for the
            # position's own sensitivity vector (factor input order, with
            # unreferenced factors zero). The position partition regroups
            # the same products, so its natural total can trail the
            # portfolio variance by an ulp or two.
            position_variance: list[float] = []
            for _, sensitivities in positions:
                contribution = 0.0
                for i, factor in enumerate(factors):
                    value = sensitivities.get(factor)
                    if value is not None:
                        contribution += value * loading[i]
                if not math.isfinite(contribution):
                    raise InvalidInput(
                        message="parametric attribution produced a non-finite result"
                    )
                position_variance.append(contribution)

            # Force each variance partition to equal the portfolio variance
            # exactly, then derive the component metrics from the aligned
            # contributions and align those sums onto VaR and ES too.
            _reconcile_totals(factor_variance, variance)
            _reconcile_totals(position_variance, variance)

            factor_var = [z * value / volatility for value in factor_variance]
            factor_es = [
                density * value / tail_scale for value in factor_variance
            ]
            position_var = [z * value / volatility for value in position_variance]
            position_es = [
                density * value / tail_scale for value in position_variance
            ]
            for values in (factor_var, factor_es, position_var, position_es):
                if any(not math.isfinite(value) for value in values):
                    raise InvalidInput(
                        message="parametric attribution produced a non-finite result"
                    )
                if not math.isfinite(sum(values)):
                    raise InvalidInput(
                        message="parametric attribution produced a non-finite result"
                    )
            _reconcile_totals(factor_var, var)
            _reconcile_totals(factor_es, expected_shortfall)
            _reconcile_totals(position_var, var)
            _reconcile_totals(position_es, expected_shortfall)

            factor_attributions = [
                {
                    "factor": factors[i],
                    "aggregate_sensitivity": aggregate[i],
                    "covariance_loading": loading[i],
                    "variance_contribution": factor_variance[i],
                    "component_var": factor_var[i],
                    "component_expected_shortfall": factor_es[i],
                }
                for i in range(n)
            ]
            position_attributions = [
                {
                    "id": positions[k][0],
                    "variance_contribution": position_variance[k],
                    "component_var": position_var[k],
                    "component_expected_shortfall": position_es[k],
                }
                for k in range(len(positions))
            ]

        return {
            "currency": currency,
            "confidence": confidence,
            "factors": factors,
            "aggregate_sensitivities": aggregate,
            "variance": variance,
            "volatility": volatility,
            "var": var,
            "expected_shortfall": expected_shortfall,
            "factor_attributions": factor_attributions,
            "position_attributions": position_attributions,
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

    def _portfolio_aggregate(self, payload: dict) -> dict:
        reporting_currency = payload.get("reporting_currency")
        if not _is_nonempty_str(reporting_currency):
            raise InvalidInput()

        fx_rates_raw = payload.get("fx_rates")
        if not isinstance(fx_rates_raw, dict):
            raise InvalidInput()
        if len(fx_rates_raw) > MAX_FX_RATES:
            raise RequestTooLarge(f"at most {MAX_FX_RATES} fx rates are allowed")

        positions_raw = payload.get("positions")
        if not isinstance(positions_raw, list) or len(positions_raw) == 0:
            raise InvalidInput()
        if len(positions_raw) > MAX_POSITIONS:
            raise RequestTooLarge(f"at most {MAX_POSITIONS} positions are allowed")

        positions: list[dict] = []
        sensitivity_entry_count = 0
        for position in positions_raw:
            if not isinstance(position, dict):
                raise InvalidInput()
            position_id = position.get("id")
            if not _is_nonempty_str(position_id):
                raise InvalidInput()
            book = position.get("book")
            if not _is_nonempty_str(book):
                raise InvalidInput()
            currency = position.get("currency")
            if not _is_nonempty_str(currency):
                raise InvalidInput()
            market_value = _strict_float(position.get("market_value"))
            sensitivities_raw = position.get("sensitivities")
            if not isinstance(sensitivities_raw, dict) or len(sensitivities_raw) == 0:
                raise InvalidInput()
            sensitivities: dict[str, float] = {}
            for factor, value in sensitivities_raw.items():
                if not _is_nonempty_str(factor):
                    raise InvalidInput()
                sensitivities[factor] = _strict_float(value)
            sensitivity_entry_count += len(sensitivities)
            positions.append(
                {
                    "id": position_id,
                    "book": book,
                    "currency": currency,
                    "market_value": market_value,
                    "sensitivities": sensitivities,
                }
            )

        if sensitivity_entry_count > MAX_SENSITIVITY_ENTRIES:
            raise RequestTooLarge(
                "sensitivity entries must not exceed "
                f"{MAX_SENSITIVITY_ENTRIES}"
            )

        seen_position_ids: set[str] = set()
        for position in positions:
            if position["id"] in seen_position_ids:
                raise InvalidInput("duplicate_position", "position ids must be unique")
            seen_position_ids.add(position["id"])

        # Resolve the FX rate of every referenced currency in first-appearance
        # order. The reporting currency always converts at 1.0, including when
        # an explicit rate is supplied (it must equal 1); rates for currencies
        # no position references are never inspected.
        used_rates: dict[str, float] = {}
        for position in positions:
            currency = position["currency"]
            if currency in used_rates:
                continue
            if currency == reporting_currency:
                rate = fx_rates_raw.get(currency, 1.0)
                rate = _strict_float(rate)
                if rate != 1.0:
                    raise InvalidInput(
                        message="reporting currency fx rate must be 1"
                    )
                used_rates[currency] = 1.0
            else:
                if currency not in fx_rates_raw:
                    raise InvalidInput(
                        "missing_fx_rate",
                        f"missing fx rate for currency {currency!r}",
                    )
                rate = _strict_float(fx_rates_raw[currency])
                if rate <= 0.0:
                    raise InvalidInput()
                used_rates[currency] = rate

        # First-appearance orders: positions drive currencies, books and (with
        # each object's own key order) factors.
        currency_order: list[str] = []
        book_order: list[str] = []
        factor_order: list[str] = []
        seen_currencies: set[str] = set()
        seen_books: set[str] = set()
        seen_factors: set[str] = set()

        converted: list[dict] = []
        for position in positions:
            currency = position["currency"]
            book = position["book"]
            if currency not in seen_currencies:
                seen_currencies.add(currency)
                currency_order.append(currency)
            if book not in seen_books:
                seen_books.add(book)
                book_order.append(book)
            rate = used_rates[currency]
            converted_mv = position["market_value"] * rate
            if not math.isfinite(converted_mv):
                raise InvalidInput(
                    message="aggregation produced a non-finite result"
                )
            converted_sensitivities: dict[str, float] = {}
            for factor, value in position["sensitivities"].items():
                if factor not in seen_factors:
                    seen_factors.add(factor)
                    factor_order.append(factor)
                converted_value = value * rate
                if not math.isfinite(converted_value):
                    raise InvalidInput(
                        message="aggregation produced a non-finite result"
                    )
                converted_sensitivities[factor] = converted_value
            converted.append(
                {
                    "currency": currency,
                    "book": book,
                    "market_value": converted_mv,
                    "sensitivities": converted_sensitivities,
                }
            )

        # Accumulate the portfolio totals as the sequential position sum (input
        # order); the two partitions accumulate the same values grouped by
        # currency and by book.
        currency_acc = {
            currency: {"market_value": 0.0, "sensitivities": {}}
            for currency in currency_order
        }
        book_acc = {
            book: {"market_value": 0.0, "sensitivities": {}} for book in book_order
        }
        total_market_value = 0.0
        total_sensitivities: dict[str, float] = {}
        for entry in converted:
            total_market_value += entry["market_value"]
            for factor, value in entry["sensitivities"].items():
                total_sensitivities[factor] = total_sensitivities.get(factor, 0.0) + value
            for bucket in (
                currency_acc[entry["currency"]],
                book_acc[entry["book"]],
            ):
                bucket["market_value"] += entry["market_value"]
                for factor, value in entry["sensitivities"].items():
                    bucket["sensitivities"][factor] = (
                        bucket["sensitivities"].get(factor, 0.0) + value
                    )

        for value in [total_market_value, *total_sensitivities.values()]:
            if not math.isfinite(value):
                raise InvalidInput(
                    message="aggregation produced a non-finite result"
                )

        def render_details(order: list[str], acc: dict, label: str) -> list[dict]:
            details: list[dict] = []
            market_values: list[float] = []
            for key in order:
                bucket = acc[key]
                if not math.isfinite(bucket["market_value"]) or any(
                    not math.isfinite(value)
                    for value in bucket["sensitivities"].values()
                ):
                    raise InvalidInput(
                        message="aggregation produced a non-finite result"
                    )
                sensitivities = {
                    factor: bucket["sensitivities"].get(factor, 0.0)
                    for factor in factor_order
                }
                detail = {
                    label: key,
                    "position_count": 0,
                    "converted_market_value": bucket["market_value"],
                    "converted_sensitivities": sensitivities,
                }
                details.append(detail)
                market_values.append(bucket["market_value"])
            # Fold the grouped sums onto the sequential portfolio total so
            # each layer's totals equal the sum of its details exactly.
            _reconcile_totals(market_values, total_market_value)
            for index, key in enumerate(order):
                details[index]["converted_market_value"] = market_values[index]
            for factor in factor_order:
                factor_values = [
                    detail["converted_sensitivities"][factor] for detail in details
                ]
                _reconcile_totals(
                    factor_values, total_sensitivities.get(factor, 0.0)
                )
                for index in range(len(details)):
                    details[index]["converted_sensitivities"][factor] = (
                        factor_values[index]
                    )
            return details

        position_counts: dict[str, int] = {key: 0 for key in currency_order}
        book_counts: dict[str, int] = {key: 0 for key in book_order}
        for entry in converted:
            position_counts[entry["currency"]] += 1
            book_counts[entry["book"]] += 1

        currency_details = render_details(currency_order, currency_acc, "currency")
        book_details = render_details(book_order, book_acc, "book")
        for detail, key in zip(currency_details, currency_order):
            detail["position_count"] = position_counts[key]
            detail["fx_rate"] = used_rates[key]
        for detail, key in zip(book_details, book_order):
            detail["position_count"] = book_counts[key]

        return {
            "reporting_currency": reporting_currency,
            "position_count": len(positions),
            "currencies": currency_details,
            "books": book_details,
            "portfolio_totals": {
                "converted_market_value": total_market_value,
                "converted_sensitivities": {
                    factor: total_sensitivities.get(factor, 0.0)
                    for factor in factor_order
                },
            },
        }

    def _limit_check(self, payload: dict) -> dict:
        as_of = payload.get("as_of")
        if not _is_nonempty_str(as_of):
            raise InvalidInput()

        limits_raw = payload.get("limits")
        if not isinstance(limits_raw, list) or len(limits_raw) == 0:
            raise InvalidInput()
        if len(limits_raw) > MAX_LIMITS:
            raise RequestTooLarge(f"at most {MAX_LIMITS} limits are allowed")
        limits: list[dict] = []
        for item in limits_raw:
            if not isinstance(item, dict):
                raise InvalidInput()
            limit_id = item.get("id")
            if not _is_nonempty_str(limit_id):
                raise InvalidInput()
            scope = item.get("scope")
            if not isinstance(scope, dict):
                raise InvalidInput()
            scope_type = scope.get("type")
            scope_id = scope.get("id")
            if not _is_nonempty_str(scope_type) or not _is_nonempty_str(scope_id):
                raise InvalidInput()
            metric = item.get("metric")
            if not _is_nonempty_str(metric):
                raise InvalidInput()
            unit = item.get("unit")
            if not _is_nonempty_str(unit):
                raise InvalidInput()
            limit_value = _strict_float(item.get("limit"))
            if limit_value <= 0.0:
                raise InvalidInput()
            warning_ratio = _strict_float(item.get("warning_ratio", 0.8))
            if not 0.0 < warning_ratio < 1.0:
                raise InvalidInput()
            limits.append(
                {
                    "id": limit_id,
                    "scope": {"type": scope_type, "id": scope_id},
                    "metric": metric,
                    "unit": unit,
                    "limit": limit_value,
                    "warning_ratio": warning_ratio,
                }
            )

        seen_limit_ids: set[str] = set()
        for limit in limits:
            if limit["id"] in seen_limit_ids:
                raise InvalidInput("duplicate_limit", "limit ids must be unique")
            seen_limit_ids.add(limit["id"])

        measurements_raw = payload.get("measurements", [])
        if not isinstance(measurements_raw, list):
            raise InvalidInput()
        if len(measurements_raw) > MAX_MEASUREMENTS:
            raise RequestTooLarge(
                f"at most {MAX_MEASUREMENTS} measurements are allowed"
            )
        measurements: list[dict] = []
        for item in measurements_raw:
            if not isinstance(item, dict):
                raise InvalidInput()
            limit_id = item.get("limit_id")
            if not _is_nonempty_str(limit_id):
                raise InvalidInput()
            value = _strict_float(item.get("value"))
            if value < 0.0:
                raise InvalidInput()
            measurements.append({"limit_id": limit_id, "value": value})

        values_by_limit: dict[str, float] = {}
        for measurement in measurements:
            limit_id = measurement["limit_id"]
            if limit_id not in seen_limit_ids:
                raise InvalidInput(
                    "unknown_limit",
                    f"measurement references undeclared limit {limit_id!r}",
                )
            if limit_id in values_by_limit:
                raise InvalidInput(
                    "duplicate_measurement",
                    "at most one measurement per limit is allowed",
                )
            values_by_limit[limit_id] = measurement["value"]

        results: list[dict] = []
        alerts: list[dict] = []
        summary = {"ok": 0, "warning": 0, "breach": 0, "no_data": 0}
        for limit in limits:
            entry = dict(limit)
            if limit["id"] in values_by_limit:
                value = values_by_limit[limit["id"]]
                limit_value = limit["limit"]
                utilization = value / limit_value
                headroom = limit_value - value
                if not math.isfinite(utilization) or not math.isfinite(headroom):
                    raise InvalidInput(
                        message="limit check produced a non-finite result"
                    )
                if value >= limit_value:
                    status = "breach"
                elif value >= limit_value * limit["warning_ratio"]:
                    status = "warning"
                else:
                    status = "ok"
                entry.update(
                    {
                        "value": value,
                        "utilization": utilization,
                        "headroom": headroom,
                        "status": status,
                    }
                )
            else:
                status = "no_data"
                entry.update(
                    {
                        "value": None,
                        "utilization": None,
                        "headroom": None,
                        "status": status,
                    }
                )
            summary[status] += 1
            results.append(entry)
            if status in ("warning", "breach"):
                alerts.append(entry)

        if summary["breach"] > 0:
            overall_status = "breach"
        elif summary["warning"] > 0:
            overall_status = "warning"
        elif summary["no_data"] > 0:
            overall_status = "incomplete"
        else:
            overall_status = "ok"

        return {
            "as_of": as_of,
            "limits": results,
            "alerts": alerts,
            "summary": summary,
            "overall_status": overall_status,
        }

    def _expected_loss_schedule(self, payload: dict) -> dict:
        currency = payload.get("currency", "USD")
        if not _is_nonempty_str(currency):
            raise InvalidInput()

        periods_raw = payload.get("periods")
        if not isinstance(periods_raw, list) or len(periods_raw) == 0:
            raise InvalidInput()
        periods: list[int] = []
        for period in periods_raw:
            value = _strict_int(period)
            if value <= 0:
                raise InvalidInput()
            periods.append(value)
        # Strictly increasing terms; duplicates cannot satisfy that.
        for earlier, later in zip(periods, periods[1:]):
            if later <= earlier:
                raise InvalidInput(message="periods must be strictly increasing")
        period_count = len(periods)

        discount_factors_raw = payload.get("discount_factors")
        if not isinstance(discount_factors_raw, list) or len(discount_factors_raw) != period_count:
            raise InvalidInput()
        discount_factors: list[float] = []
        for value in discount_factors_raw:
            factor = _strict_float(value)
            if not 0.0 < factor <= 1.0:
                raise InvalidInput()
            discount_factors.append(factor)

        counterparties_raw = payload.get("counterparties")
        if not isinstance(counterparties_raw, list) or len(counterparties_raw) == 0:
            raise InvalidInput()
        if len(counterparties_raw) > MAX_SCHEDULE_COUNTERPARTIES:
            raise RequestTooLarge(
                f"at most {MAX_SCHEDULE_COUNTERPARTIES} counterparties are allowed"
            )
        counterparties: list[dict] = []
        for counterparty in counterparties_raw:
            if not isinstance(counterparty, dict):
                raise InvalidInput()
            counterparty_id = counterparty.get("id")
            if not _is_nonempty_str(counterparty_id):
                raise InvalidInput()
            cumulative_raw = counterparty.get("cumulative_pd")
            if not isinstance(cumulative_raw, list) or len(cumulative_raw) != period_count:
                raise InvalidInput()
            cumulative_pd: list[float] = []
            for value in cumulative_raw:
                pd = _strict_float(value)
                if not 0.0 <= pd <= 1.0:
                    raise InvalidInput()
                cumulative_pd.append(pd)
            # A cumulative default probability never falls with the term.
            for earlier, later in zip(cumulative_pd, cumulative_pd[1:]):
                if later < earlier:
                    raise InvalidInput(
                        message="cumulative_pd must not decrease across periods"
                    )
            counterparties.append({"id": counterparty_id, "cumulative_pd": cumulative_pd})

        facilities_raw = payload.get("facilities")
        if not isinstance(facilities_raw, list) or len(facilities_raw) == 0:
            raise InvalidInput()
        if len(facilities_raw) > MAX_SCHEDULE_FACILITIES:
            raise RequestTooLarge(
                f"at most {MAX_SCHEDULE_FACILITIES} facilities are allowed"
            )
        if period_count * len(facilities_raw) > MAX_SCHEDULE_PERIOD_FACILITY_PAIRS:
            raise RequestTooLarge(
                "periods times facilities must not exceed "
                f"{MAX_SCHEDULE_PERIOD_FACILITY_PAIRS}"
            )
        facilities: list[dict] = []
        for facility in facilities_raw:
            if not isinstance(facility, dict):
                raise InvalidInput()
            facility_id = facility.get("id")
            if not _is_nonempty_str(facility_id):
                raise InvalidInput()
            counterparty_id = facility.get("counterparty_id")
            if not _is_nonempty_str(counterparty_id):
                raise InvalidInput()
            lgd = _strict_float(facility.get("lgd"))
            if not 0.0 <= lgd <= 1.0:
                raise InvalidInput()
            ead_raw = facility.get("ead")
            if not isinstance(ead_raw, list) or len(ead_raw) != period_count:
                raise InvalidInput()
            ead: list[float] = []
            for value in ead_raw:
                amount = _strict_float(value)
                if amount < 0.0:
                    raise InvalidInput()
                ead.append(amount)
            facilities.append(
                {
                    "id": facility_id,
                    "counterparty_id": counterparty_id,
                    "lgd": lgd,
                    "ead": ead,
                }
            )

        seen_counterparty_ids: set[str] = set()
        for counterparty in counterparties:
            if counterparty["id"] in seen_counterparty_ids:
                raise InvalidInput(
                    "duplicate_counterparty", "counterparty ids must be unique"
                )
            seen_counterparty_ids.add(counterparty["id"])

        seen_facility_ids: set[str] = set()
        for facility in facilities:
            if facility["id"] in seen_facility_ids:
                raise InvalidInput("duplicate_facility", "facility ids must be unique")
            seen_facility_ids.add(facility["id"])

        counterparty_by_id = {cp["id"]: cp for cp in counterparties}
        for facility in facilities:
            if facility["counterparty_id"] not in counterparty_by_id:
                raise InvalidInput(
                    "unknown_counterparty",
                    "facility references an unknown counterparty "
                    f"{facility['counterparty_id']!r}",
                )

        def non_finite() -> InvalidInput:
            return InvalidInput(
                message="schedule computation produced a non-finite result"
            )

        # Marginal default probability per counterparty: the first period
        # takes the first cumulative value, later periods the difference
        # of adjacent cumulative values.
        marginal_pd: dict[str, list[float]] = {}
        for counterparty in counterparties:
            cumulative_pd = counterparty["cumulative_pd"]
            marginal = [cumulative_pd[0]]
            for index in range(1, period_count):
                marginal.append(cumulative_pd[index] - cumulative_pd[index - 1])
            marginal_pd[counterparty["id"]] = marginal

        facility_values: list[list[float]] = []
        facility_totals: list[float] = []
        facility_results: list[dict] = []
        for facility in facilities:
            marginal = marginal_pd[facility["counterparty_id"]]
            lgd = facility["lgd"]
            values: list[float] = []
            for index in range(period_count):
                value = (
                    facility["ead"][index]
                    * lgd
                    * marginal[index]
                    * discount_factors[index]
                )
                if not math.isfinite(value):
                    raise non_finite()
                values.append(value)
            total = 0.0
            for value in values:
                total += value
                if not math.isfinite(total):
                    raise non_finite()
            facility_values.append(values)
            facility_totals.append(total)
            facility_results.append(
                {
                    "id": facility["id"],
                    "counterparty_id": facility["counterparty_id"],
                    "contributions": [
                        {
                            "period": periods[index],
                            "discounted_expected_loss": values[index],
                        }
                        for index in range(period_count)
                    ],
                    "total_discounted_expected_loss": total,
                }
            )

        # Aggregate in input order at both levels so each layer's totals
        # are exactly the sequential sum of the layer below. The per-period
        # lists are folded onto those totals so each layer's total also
        # equals the sum of its own per-period contributions exactly.
        counterparty_results: list[dict] = []
        counterparty_period_values: list[list[float]] = []
        counterparty_totals: list[float] = []
        for counterparty in counterparties:
            owned = [
                index
                for index, facility in enumerate(facilities)
                if facility["counterparty_id"] == counterparty["id"]
            ]
            per_period = [0.0] * period_count
            for index in owned:
                for period_index in range(period_count):
                    per_period[period_index] += facility_values[index][period_index]
                    if not math.isfinite(per_period[period_index]):
                        raise non_finite()
            total = 0.0
            for index in owned:
                total += facility_totals[index]
                if not math.isfinite(total):
                    raise non_finite()
            _reconcile_totals(per_period, total)
            counterparty_period_values.append(per_period)
            counterparty_totals.append(total)
            counterparty_results.append(
                {
                    "id": counterparty["id"],
                    "contributions": [
                        {
                            "period": periods[index],
                            "discounted_expected_loss": per_period[index],
                        }
                        for index in range(period_count)
                    ],
                    "total_discounted_expected_loss": total,
                }
            )

        portfolio_per_period = [0.0] * period_count
        for per_period in counterparty_period_values:
            for period_index in range(period_count):
                portfolio_per_period[period_index] += per_period[period_index]
                if not math.isfinite(portfolio_per_period[period_index]):
                    raise non_finite()
        portfolio_total = 0.0
        for total in counterparty_totals:
            portfolio_total += total
            if not math.isfinite(portfolio_total):
                raise non_finite()
        _reconcile_totals(portfolio_per_period, portfolio_total)

        return {
            "currency": currency,
            "periods": periods,
            "facilities": facility_results,
            "counterparties": counterparty_results,
            "portfolio_total": {
                "contributions": [
                    {
                        "period": periods[index],
                        "discounted_expected_loss": portfolio_per_period[index],
                    }
                    for index in range(period_count)
                ],
                "total_discounted_expected_loss": portfolio_total,
            },
        }

    def _irb_capital(self, payload: dict) -> dict:
        currency = payload.get("currency", "USD")
        if not _is_nonempty_str(currency):
            raise InvalidInput()

        counterparties_raw = payload.get("counterparties")
        if not isinstance(counterparties_raw, list) or len(counterparties_raw) == 0:
            raise InvalidInput()
        if len(counterparties_raw) > MAX_IRB_COUNTERPARTIES:
            raise RequestTooLarge(
                f"at most {MAX_IRB_COUNTERPARTIES} counterparties are allowed"
            )
        counterparties: list[dict] = []
        for counterparty in counterparties_raw:
            if not isinstance(counterparty, dict):
                raise InvalidInput()
            counterparty_id = counterparty.get("id")
            if not _is_nonempty_str(counterparty_id):
                raise InvalidInput()
            pd = _strict_float(counterparty.get("pd"))
            if not 0.0 < pd < 1.0:
                raise InvalidInput()
            counterparties.append({"id": counterparty_id, "pd": pd})

        facilities_raw = payload.get("facilities")
        if not isinstance(facilities_raw, list) or len(facilities_raw) == 0:
            raise InvalidInput()
        if len(facilities_raw) > MAX_IRB_FACILITIES:
            raise RequestTooLarge(
                f"at most {MAX_IRB_FACILITIES} facilities are allowed"
            )
        facilities: list[dict] = []
        for facility in facilities_raw:
            if not isinstance(facility, dict):
                raise InvalidInput()
            facility_id = facility.get("id")
            if not _is_nonempty_str(facility_id):
                raise InvalidInput()
            counterparty_id = facility.get("counterparty_id")
            if not _is_nonempty_str(counterparty_id):
                raise InvalidInput()
            lgd = _strict_float(facility.get("lgd"))
            if not 0.0 <= lgd <= 1.0:
                raise InvalidInput()
            ead = _strict_float(facility.get("ead"))
            if ead < 0.0:
                raise InvalidInput()
            maturity = _strict_float(facility.get("maturity"))
            if not 1.0 <= maturity <= 5.0:
                raise InvalidInput()
            facilities.append(
                {
                    "id": facility_id,
                    "counterparty_id": counterparty_id,
                    "lgd": lgd,
                    "ead": ead,
                    "maturity": maturity,
                }
            )

        seen_counterparty_ids: set[str] = set()
        for counterparty in counterparties:
            if counterparty["id"] in seen_counterparty_ids:
                raise InvalidInput(
                    "duplicate_counterparty", "counterparty ids must be unique"
                )
            seen_counterparty_ids.add(counterparty["id"])

        seen_facility_ids: set[str] = set()
        for facility in facilities:
            if facility["id"] in seen_facility_ids:
                raise InvalidInput("duplicate_facility", "facility ids must be unique")
            seen_facility_ids.add(facility["id"])

        counterparty_by_id = {cp["id"]: cp for cp in counterparties}
        for facility in facilities:
            if facility["counterparty_id"] not in counterparty_by_id:
                raise InvalidInput(
                    "unknown_counterparty",
                    "facility references an unknown counterparty "
                    f"{facility['counterparty_id']!r}",
                )

        def non_finite() -> InvalidInput:
            return InvalidInput(
                message="IRB capital computation produced a non-finite result"
            )

        # Foundation-IRB uses a single non-defaulted corporate exposure
        # formula: the maturity adjustment scales the unexpected-loss gap
        # between the 99.9th-percentile conditional default loss and the
        # expected loss PD, and RWA is 12.5 times the capital requirement.
        z_999 = _STANDARD_NORMAL.inv_cdf(0.999)
        denom_50 = 1.0 - math.exp(-50.0)

        facility_results: list[dict] = []
        for facility in facilities:
            pd = counterparty_by_id[facility["counterparty_id"]]["pd"]
            a = (1.0 - math.exp(-50.0 * pd)) / denom_50
            correlation = 0.12 * a + 0.24 * (1.0 - a)
            b = (0.11852 - 0.05478 * math.log(pd)) ** 2
            maturity_adjustment = (
                1.0 + (facility["maturity"] - 2.5) * b
            ) / (1.0 - 1.5 * b)
            z_pd = _STANDARD_NORMAL.inv_cdf(pd)
            worst_case = _STANDARD_NORMAL.cdf(
                (z_pd + math.sqrt(correlation) * z_999) / math.sqrt(1.0 - correlation)
            )
            k = facility["lgd"] * (worst_case - pd) * maturity_adjustment
            capital_requirement = facility["ead"] * k
            risk_weighted_assets = 12.5 * capital_requirement
            for value in (
                a,
                correlation,
                b,
                maturity_adjustment,
                z_pd,
                worst_case,
                k,
                capital_requirement,
                risk_weighted_assets,
            ):
                if not math.isfinite(value):
                    raise non_finite()
            facility_results.append(
                {
                    "id": facility["id"],
                    "counterparty_id": facility["counterparty_id"],
                    "pd": pd,
                    "R": correlation,
                    "MA": maturity_adjustment,
                    "K": k,
                    "ead": facility["ead"],
                    "capital_requirement": capital_requirement,
                    "risk_weighted_assets": risk_weighted_assets,
                }
            )

        # Aggregate in counterparty input order so the portfolio totals are
        # exactly the sequential sum of the counterparty details; a
        # counterparty without facilities keeps explicit zero values.
        amount_keys = ("ead", "capital_requirement", "risk_weighted_assets")
        counterparty_results: list[dict] = []
        for counterparty in counterparties:
            totals = {key: 0.0 for key in amount_keys}
            for result in facility_results:
                if result["counterparty_id"] != counterparty["id"]:
                    continue
                for key in amount_keys:
                    totals[key] += result[key]
            for value in totals.values():
                if not math.isfinite(value):
                    raise non_finite()
            counterparty_results.append({"id": counterparty["id"], **totals})

        portfolio_totals = {key: 0.0 for key in amount_keys}
        for result in counterparty_results:
            for key in amount_keys:
                portfolio_totals[key] += result[key]
        for value in portfolio_totals.values():
            if not math.isfinite(value):
                raise non_finite()

        return {
            "currency": currency,
            "facilities": facility_results,
            "counterparties": counterparty_results,
            "portfolio_totals": portfolio_totals,
        }

    def _discounted_cashflow(self, payload: dict) -> dict:
        currency = payload.get("currency", "USD")
        if not _is_nonempty_str(currency):
            raise InvalidInput()

        curve_raw = payload.get("curve_points")
        if not isinstance(curve_raw, list):
            raise InvalidInput()
        if len(curve_raw) > MAX_CURVE_POINTS:
            raise RequestTooLarge(
                f"at most {MAX_CURVE_POINTS} curve points are allowed"
            )
        if len(curve_raw) < 2:
            raise InvalidInput()
        curve_points: list[dict] = []
        for point in curve_raw:
            if not isinstance(point, dict):
                raise InvalidInput()
            day = _strict_day(point.get("day"))
            zero_rate = _strict_float(point.get("zero_rate"))
            curve_points.append({"day": day, "zero_rate": zero_rate})

        positions_raw = payload.get("positions")
        if not isinstance(positions_raw, list) or len(positions_raw) == 0:
            raise InvalidInput()
        if len(positions_raw) > MAX_POSITIONS:
            raise RequestTooLarge(f"at most {MAX_POSITIONS} positions are allowed")
        positions: list[dict] = []
        cashflow_count = 0
        for position in positions_raw:
            if not isinstance(position, dict):
                raise InvalidInput()
            position_id = position.get("id")
            if not _is_nonempty_str(position_id):
                raise InvalidInput()
            cashflows_raw = position.get("cashflows")
            if not isinstance(cashflows_raw, list) or len(cashflows_raw) == 0:
                raise InvalidInput()
            cashflows: list[dict] = []
            for cashflow in cashflows_raw:
                if not isinstance(cashflow, dict):
                    raise InvalidInput()
                day = _strict_day(cashflow.get("day"))
                amount = _strict_float(cashflow.get("amount"))
                cashflows.append({"day": day, "amount": amount})
            cashflow_count += len(cashflows)
            positions.append({"id": position_id, "cashflows": cashflows})
        if cashflow_count > MAX_CASHFLOWS:
            raise RequestTooLarge(
                f"at most {MAX_CASHFLOWS} cashflows are allowed"
            )

        seen_days: set[int] = set()
        for point in curve_points:
            if point["day"] in seen_days:
                raise InvalidInput(
                    "duplicate_curve_point", "curve point days must be unique"
                )
            seen_days.add(point["day"])

        seen_position_ids: set[str] = set()
        for position in positions:
            if position["id"] in seen_position_ids:
                raise InvalidInput("duplicate_position", "position ids must be unique")
            seen_position_ids.add(position["id"])

        # The curve may arrive unordered; valuation and the echoed curve
        # both use ascending day order.
        curve_points.sort(key=lambda point: point["day"])
        node_days = [point["day"] for point in curve_points]
        node_rates = [point["zero_rate"] for point in curve_points]
        node_count = len(node_days)

        def non_finite() -> InvalidInput:
            return InvalidInput(
                message="discount computation produced a non-finite result"
            )

        def locate(day: int) -> tuple[float, list[tuple[int, float]]]:
            """Zero rate and per-node interpolation weights for ``day``."""
            if day < node_days[0] or day > node_days[-1]:
                raise InvalidInput(
                    "curve_out_of_range",
                    "cashflow day lies outside the curve",
                )
            index = bisect.bisect_left(node_days, day)
            if node_days[index] == day:
                return node_rates[index], [(index, 1.0)]
            left, right = index - 1, index
            span = node_days[right] - node_days[left]
            weight_left = (node_days[right] - day) / span
            weight_right = (day - node_days[left]) / span
            rate = (
                node_rates[left] * weight_left + node_rates[right] * weight_right
            )
            return rate, [(left, weight_left), (right, weight_right)]

        position_results: list[dict] = []
        position_node_sensitivities: list[list[float]] = []
        for position in positions:
            present_value = 0.0
            node_sensitivities = [0.0] * node_count
            for cashflow in position["cashflows"]:
                rate, weights = locate(cashflow["day"])
                t = cashflow["day"] / 365.0
                exponent = -rate * t
                if not math.isfinite(exponent):
                    raise non_finite()
                try:
                    discount = math.exp(exponent)
                except OverflowError:
                    raise non_finite()
                pv = cashflow["amount"] * discount
                if not math.isfinite(pv):
                    raise non_finite()
                base = pv * t * 0.0001
                if not math.isfinite(base):
                    raise non_finite()
                present_value += pv
                if not math.isfinite(present_value):
                    raise non_finite()
                for node_index, weight in weights:
                    contribution = base * weight
                    if not math.isfinite(contribution):
                        raise non_finite()
                    node_sensitivities[node_index] += contribution
                    if not math.isfinite(node_sensitivities[node_index]):
                        raise non_finite()
            # dv01 is the sum of this level's node sensitivities, folded
            # in ascending node order; zero entries stay visible.
            dv01 = 0.0
            for value in node_sensitivities:
                dv01 += value
                if not math.isfinite(dv01):
                    raise non_finite()
            position_node_sensitivities.append(node_sensitivities)
            position_results.append(
                {
                    "id": position["id"],
                    "present_value": present_value,
                    "dv01": dv01,
                    "key_rate_dv01": [
                        {"day": node_days[i], "key_rate_dv01": node_sensitivities[i]}
                        for i in range(node_count)
                    ],
                }
            )

        # Portfolio totals accumulate the position layer in input order.
        portfolio_present_value = 0.0
        portfolio_node_sensitivities = [0.0] * node_count
        for result, node_sensitivities in zip(
            position_results, position_node_sensitivities
        ):
            portfolio_present_value += result["present_value"]
            if not math.isfinite(portfolio_present_value):
                raise non_finite()
            for i in range(node_count):
                portfolio_node_sensitivities[i] += node_sensitivities[i]
                if not math.isfinite(portfolio_node_sensitivities[i]):
                    raise non_finite()
        portfolio_dv01 = 0.0
        for value in portfolio_node_sensitivities:
            portfolio_dv01 += value
            if not math.isfinite(portfolio_dv01):
                raise non_finite()

        return {
            "currency": currency,
            "curve_points": [
                {"day": node_days[i], "zero_rate": node_rates[i]}
                for i in range(node_count)
            ],
            "positions": position_results,
            "portfolio_totals": {
                "present_value": portfolio_present_value,
                "dv01": portfolio_dv01,
                "key_rate_dv01": [
                    {
                        "day": node_days[i],
                        "key_rate_dv01": portfolio_node_sensitivities[i],
                    }
                    for i in range(node_count)
                ],
            },
        }

    @staticmethod
    def _matrix_multiply(
        a: list[list[float]], b: list[list[float]]
    ) -> list[list[float]]:
        """Square matrix product ``a · b`` accumulating in column order.

        Each entry sums its products in ascending column index, exactly
        like the naive dot product; zero factors are skipped, which
        cannot change the sum because every entry is non-negative.
        """
        n = len(a)
        result = [[0.0] * n for _ in range(n)]
        for i in range(n):
            result_row = result[i]
            a_row = a[i]
            for k in range(n):
                factor = a_row[k]
                if factor == 0.0:
                    continue
                b_row = b[k]
                for j in range(n):
                    result_row[j] += factor * b_row[j]
        return result

    def _rating_migration_loss(self, payload: dict) -> dict:
        currency = payload.get("currency", "USD")
        if not _is_nonempty_str(currency):
            raise InvalidInput()

        horizon = _strict_int(payload.get("horizon"))
        if horizon > MAX_MIGRATION_HORIZON:
            raise RequestTooLarge(
                f"horizon must not exceed {MAX_MIGRATION_HORIZON}"
            )
        if horizon < 1:
            raise InvalidInput()

        ratings_raw = payload.get("ratings")
        if not isinstance(ratings_raw, list) or len(ratings_raw) == 0:
            raise InvalidInput()
        if len(ratings_raw) > MAX_MIGRATION_RATINGS:
            raise RequestTooLarge(
                f"at most {MAX_MIGRATION_RATINGS} ratings are allowed"
            )
        ratings: list[str] = []
        for rating in ratings_raw:
            if not _is_nonempty_str(rating):
                raise InvalidInput()
            ratings.append(rating)
        seen_ratings: set[str] = set()
        for rating in ratings:
            if rating in seen_ratings:
                raise InvalidInput("duplicate_rating", "rating names must be unique")
            seen_ratings.add(rating)
        rating_index = {rating: index for index, rating in enumerate(ratings)}
        n = len(ratings)

        default_rating = payload.get("default_rating")
        if not _is_nonempty_str(default_rating) or default_rating not in rating_index:
            raise InvalidInput()
        default_index = rating_index[default_rating]

        matrix_raw = payload.get("transition_matrix")
        if not isinstance(matrix_raw, list) or len(matrix_raw) != n:
            raise InvalidInput()
        matrix: list[list[float]] = []
        for row in matrix_raw:
            if not isinstance(row, list) or len(row) != n:
                raise InvalidInput()
            matrix.append([_strict_float(value) for value in row])
        for row in matrix:
            total = 0.0
            for value in row:
                if value < 0.0:
                    raise InvalidInput()
                total += value
            if abs(total - 1.0) > _MIGRATION_TOLERANCE:
                raise InvalidInput(
                    message="transition matrix rows must sum to 1"
                )
        # The default state is absorbing: it only stays in itself, with
        # probability 1. Noise within tolerance is canonicalized to an
        # exactly absorbing row so the horizon matrix keeps the property.
        default_row = matrix[default_index]
        for j, value in enumerate(default_row):
            if j == default_index:
                if abs(value - 1.0) > _MIGRATION_TOLERANCE:
                    raise InvalidInput(
                        message="default rating must be absorbing"
                    )
            elif value > _MIGRATION_TOLERANCE:
                raise InvalidInput(
                    message="default rating must be absorbing"
                )
        matrix[default_index] = [
            1.0 if j == default_index else 0.0 for j in range(n)
        ]

        counterparties_raw = payload.get("counterparties")
        if not isinstance(counterparties_raw, list) or len(counterparties_raw) == 0:
            raise InvalidInput()
        if len(counterparties_raw) > MAX_MIGRATION_COUNTERPARTIES:
            raise RequestTooLarge(
                f"at most {MAX_MIGRATION_COUNTERPARTIES} counterparties are allowed"
            )
        counterparties: list[dict] = []
        for counterparty in counterparties_raw:
            if not isinstance(counterparty, dict):
                raise InvalidInput()
            counterparty_id = counterparty.get("id")
            if not _is_nonempty_str(counterparty_id):
                raise InvalidInput()
            rating = counterparty.get("rating")
            if not _is_nonempty_str(rating) or rating not in rating_index:
                raise InvalidInput(
                    message="counterparty references an unknown rating"
                )
            if rating == default_rating:
                raise InvalidInput(
                    message="counterparty is already in the default rating"
                )
            ead = _strict_float(counterparty.get("ead"))
            if ead < 0.0:
                raise InvalidInput()
            lgd = _strict_float(counterparty.get("lgd"))
            if not 0.0 <= lgd <= 1.0:
                raise InvalidInput()
            counterparties.append(
                {
                    "id": counterparty_id,
                    "rating": rating,
                    "ead": ead,
                    "lgd": lgd,
                }
            )

        seen_counterparty_ids: set[str] = set()
        for counterparty in counterparties:
            if counterparty["id"] in seen_counterparty_ids:
                raise InvalidInput(
                    "duplicate_counterparty", "counterparty ids must be unique"
                )
            seen_counterparty_ids.add(counterparty["id"])

        def non_finite() -> InvalidInput:
            return InvalidInput(
                message="migration computation produced a non-finite result"
            )

        # Apply the single-period matrix horizon times: M_1 = P and
        # M_h = M_{h-1} · P, so row i of the horizon matrix is the
        # horizon distribution of a name starting in ratings[i].
        horizon_matrix = [row[:] for row in matrix]
        for _ in range(horizon - 1):
            horizon_matrix = self._matrix_multiply(horizon_matrix, matrix)
        for row in horizon_matrix:
            for value in row:
                if not math.isfinite(value):
                    raise non_finite()
        # Results are not rounded, but probabilities within tolerance of
        # 0 or 1 snap onto the boundary.
        for row in horizon_matrix:
            for j, value in enumerate(row):
                if abs(value) <= _MIGRATION_TOLERANCE:
                    row[j] = 0.0
                elif abs(value - 1.0) <= _MIGRATION_TOLERANCE:
                    row[j] = 1.0

        amount_keys = ("ead", "expected_defaulted_exposure", "expected_loss")
        summary_acc = {
            rating: {
                "counterparty_count": 0,
                "ead": 0.0,
                "expected_defaulted_exposure": 0.0,
                "expected_loss": 0.0,
            }
            for rating in ratings
        }
        counterparty_results: list[dict] = []
        for counterparty in counterparties:
            rating = counterparty["rating"]
            row = horizon_matrix[rating_index[rating]]
            cumulative_pd = row[default_index]
            ead = counterparty["ead"]
            lgd = counterparty["lgd"]
            expected_loss = ead * lgd * cumulative_pd
            defaulted_exposure = ead * cumulative_pd
            for value in (cumulative_pd, expected_loss, defaulted_exposure):
                if not math.isfinite(value):
                    raise non_finite()
            counterparty_results.append(
                {
                    "id": counterparty["id"],
                    "rating": rating,
                    "horizon_probabilities": list(row),
                    "cumulative_pd": cumulative_pd,
                    "ead": ead,
                    "lgd": lgd,
                    "expected_loss": expected_loss,
                }
            )
            acc = summary_acc[rating]
            acc["counterparty_count"] += 1
            acc["ead"] += ead
            acc["expected_defaulted_exposure"] += defaulted_exposure
            acc["expected_loss"] += expected_loss
            for key in amount_keys:
                if not math.isfinite(acc[key]):
                    raise non_finite()

        # One summary per non-default rating, in ratings order; ratings
        # no counterparty starts from keep explicit zero values.
        rating_summaries = [
            {"rating": rating, **summary_acc[rating]}
            for rating in ratings
            if rating != default_rating
        ]

        portfolio_totals = {
            "counterparty_count": 0,
            "ead": 0.0,
            "expected_defaulted_exposure": 0.0,
            "expected_loss": 0.0,
        }
        for summary in rating_summaries:
            portfolio_totals["counterparty_count"] += summary["counterparty_count"]
            for key in amount_keys:
                portfolio_totals[key] += summary[key]
                if not math.isfinite(portfolio_totals[key]):
                    raise non_finite()

        return {
            "currency": currency,
            "horizon": horizon,
            "ratings": ratings,
            "default_rating": default_rating,
            "horizon_transition_matrix": horizon_matrix,
            "counterparties": counterparty_results,
            "rating_summaries": rating_summaries,
            "portfolio_totals": portfolio_totals,
        }
