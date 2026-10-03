"""Core service surface for RiskScope.

The frozen baseline reports process health; historical-simulation VaR and
expected shortfall now live behind :meth:`Service.historical_var`, batch
sensitivity stress tests behind :meth:`Service.stress_test`, and VaR
backtesting behind :meth:`Service.var_backtest`. The public surface stays
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
    # bool is a subclass of int but is not a risk number; an oversized JSON
    # integer raises OverflowError when tested against the float64 domain.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)  # type: ignore[arg-type]
    except OverflowError:
        return False


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and len(value) > 0


def _count_log(count: int, probability: float) -> float:
    """Return ``count * log(probability)`` with the ``0 log 0`` convention.

    When the count is zero the constrained-probability logarithm can be
    ``log(0)``; that whole term is defined as zero rather than raising.
    """
    if count == 0:
        return 0.0
    return count * math.log(probability)


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
        """Validate a VaR backtest request and run the coverage check.

        Each observation pairs a daily VaR forecast with the realized P&L.
        A breach occurs when the realized loss ``-realized_pnl`` strictly
        exceeds the forecast ``var``. Breach counts feed the Kupiec
        unconditional-coverage likelihood-ratio test. Parse failures and
        non-object payloads raise :class:`InvalidRequest`; semantic problems
        raise :class:`InvalidInput` or :class:`RequestTooLarge`.
        """
        return self._var_backtest(self._load_object(raw))

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

    def _var_backtest(self, payload: dict) -> dict:
        currency = payload.get("currency", "USD")
        if not isinstance(currency, str) or len(currency) == 0:
            raise InvalidInput()

        confidence = payload.get("confidence")
        if not _is_finite_number(confidence) or not 0 < confidence < 1:
            raise InvalidInput()

        significance_raw = payload.get("significance", 0.05)
        if not _is_finite_number(significance_raw) or not 0 < significance_raw < 1:
            raise InvalidInput()
        significance = float(significance_raw)

        observations_raw = payload.get("observations")
        if not isinstance(observations_raw, list):
            raise InvalidInput()
        if len(observations_raw) > MAX_OBSERVATIONS:
            raise RequestTooLarge(f"at most {MAX_OBSERVATIONS} observations are allowed")
        if not 2 <= len(observations_raw):
            raise InvalidInput()

        observations: list[tuple[str, float, float]] = []
        seen_dates: set[str] = set()
        for observation in observations_raw:
            if not isinstance(observation, dict):
                raise InvalidInput()
            date = observation.get("date")
            if not _is_nonempty_str(date):
                raise InvalidInput()
            if date in seen_dates:
                raise InvalidInput("duplicate_observation", "observation dates must be unique")
            seen_dates.add(date)
            var_value_raw = observation.get("var")
            if not _is_finite_number(var_value_raw) or var_value_raw < 0:
                raise InvalidInput()
            realized_pnl_raw = observation.get("realized_pnl")
            if not _is_finite_number(realized_pnl_raw):
                raise InvalidInput()
            var_value = float(var_value_raw)
            realized_pnl = float(realized_pnl_raw)
            observations.append((date, var_value, realized_pnl))

        details: list[dict] = []
        breach_count = 0
        for date, var_value, realized_pnl in observations:
            loss = -realized_pnl
            if loss == 0.0:
                # Emit a canonical +0.0 rather than -0.0 in the wire body.
                loss = 0.0
            # Strictly greater: a loss equal to the VaR forecast is not a breach.
            breach = loss > var_value
            if breach:
                breach_count += 1
            details.append(
                {
                    "date": date,
                    "var": var_value,
                    "realized_pnl": realized_pnl,
                    "loss": loss,
                    "breach": breach,
                }
            )

        n = len(observations)
        x = breach_count
        p = 1.0 - float(confidence)
        observed_rate = x / n
        lr_statistic = -2.0 * (
            _count_log(n - x, 1.0 - p)
            + _count_log(x, p)
            - _count_log(n - x, 1.0 - observed_rate)
            - _count_log(x, observed_rate)
        )
        if not math.isfinite(lr_statistic):
            raise InvalidInput(message="backtest computation overflowed")
        # The statistic is theoretically non-negative; tiny negatives born of
        # floating-point rounding are clamped to zero.
        if lr_statistic < 0.0:
            lr_statistic = 0.0
        p_value = math.erfc(math.sqrt(lr_statistic / 2.0))
        if not math.isfinite(p_value):
            raise InvalidInput(message="backtest computation overflowed")

        return {
            "currency": currency,
            "confidence": float(confidence),
            "significance": significance,
            "observations": details,
            "observation_count": n,
            "breach_count": x,
            "breach_rate": x / n,
            "kupiec": {
                "lr_statistic": lr_statistic,
                "p_value": p_value,
                "accepted": p_value >= significance,
            },
        }
