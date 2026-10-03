"""Core service surface for RiskScope.

The frozen baseline reports process health; historical-simulation VaR and
expected shortfall now live behind :meth:`Service.historical_var` and batch
sensitivity stress testing behind :meth:`Service.stress_test`. The public
surface stays backward compatible.
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
    # bool is a subclass of int but is not a risk number.
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)  # type: ignore[arg-type]
    )


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and len(value) > 0


def _as_float(value: object) -> float:
    """Narrow a JSON number to a finite float or raise invalid_input.

    JSON integers have arbitrary precision, so a literal like 10**400 parses
    fine but cannot be represented as a risk number.
    """
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

    def _parse_object(self, raw: bytes | str) -> dict:
        """Parse ``raw`` to a JSON object.

        Parse failures and non-object payloads raise :class:`InvalidRequest`;
        NaN / Infinity tokens raise :class:`InvalidInput`.
        """
        try:
            payload = json.loads(raw, parse_constant=_reject_constant)
        except _JsonConstant:
            raise InvalidInput(message="NaN and Infinity are not allowed")
        except (TypeError, ValueError):
            raise InvalidRequest("request body must be valid JSON")
        if not isinstance(payload, dict):
            raise InvalidRequest("request body must be a JSON object")
        return payload

    def historical_var(self, raw: bytes | str) -> dict:
        """Validate a historical VaR request and compute the result.

        ``raw`` is the unparsed request body. Parse failures and non-object
        payloads raise :class:`InvalidRequest`; everything else that is
        well-formed JSON but semantically wrong raises :class:`InvalidInput`
        or :class:`RequestTooLarge`.
        """
        payload = self._parse_object(raw)
        return self._historical_var(payload)

    def stress_test(self, raw: bytes | str) -> dict:
        """Validate a batch sensitivity stress-test request and run it.

        Every scenario is applied to the whole position set. For a position,
        the loss on a factor is the negated sensitivity times the factor
        shock; missing referenced factors receive a zero shock and extra
        scenario factors are ignored. Parse failures and non-object payloads
        raise :class:`InvalidRequest`; everything semantically wrong raises
        :class:`InvalidInput` or :class:`RequestTooLarge`. No result is
        partially returned.
        """
        payload = self._parse_object(raw)
        return self._stress_test(payload)

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
        if not _is_nonempty_str(currency):
            raise InvalidInput()

        positions_raw = payload.get("positions")
        if not isinstance(positions_raw, list) or len(positions_raw) == 0:
            raise InvalidInput()
        position_ids: list[str] = []
        # Sensitivities narrowed to finite floats, keyed by position id.
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
            sensitivities: dict[str, float] = {}
            for factor, value in sensitivities_raw.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise InvalidInput()
                sensitivities[factor] = _as_float(value)
            position_ids.append(position_id)
            positions.append((position_id, sensitivities))

        scenarios_raw = payload.get("scenarios")
        if not isinstance(scenarios_raw, list):
            raise InvalidInput()
        scenario_count = len(scenarios_raw)
        if scenario_count > MAX_SCENARIOS:
            raise RequestTooLarge(f"at most {MAX_SCENARIOS} scenarios are allowed")
        if scenario_count * len(positions) > MAX_SCENARIO_POSITION_PAIRS:
            raise RequestTooLarge(
                f"scenario count times position count must not exceed "
                f"{MAX_SCENARIO_POSITION_PAIRS}"
            )
        if scenario_count == 0:
            raise InvalidInput()
        scenario_ids: list[str] = []
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
            shocks: dict[str, float] = {}
            for factor, value in shocks_raw.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise InvalidInput()
                shocks[factor] = _as_float(value)
            scenario_ids.append(scenario_id)
            scenarios.append((scenario_id, shocks))

        seen_position_ids: set[str] = set()
        for position_id in position_ids:
            if position_id in seen_position_ids:
                raise InvalidInput("duplicate_position", "position ids must be unique")
            seen_position_ids.add(position_id)

        seen_scenario_ids: set[str] = set()
        for scenario_id in scenario_ids:
            if scenario_id in seen_scenario_ids:
                raise InvalidInput("duplicate_scenario", "scenario ids must be unique")
            seen_scenario_ids.add(scenario_id)

        # Factor universe is what positions reference, in first-reference
        # order; scenario-only factors are ignored.
        factors: list[str] = []
        seen_factors: set[str] = set()
        for _, sensitivities in positions:
            for factor in sensitivities:
                if factor not in seen_factors:
                    seen_factors.add(factor)
                    factors.append(factor)

        results: list[dict] = []
        worst_index = 0
        worst_loss: float | None = None
        for index, (scenario_id, shocks) in enumerate(scenarios):
            position_contributions: dict[str, float] = {}
            factor_contributions: dict[str, float] = {factor: 0.0 for factor in factors}
            scenario_loss = 0.0
            for pid, sensitivities in positions:
                position_loss = 0.0
                for factor, sensitivity in sensitivities.items():
                    # A missing referenced factor is a zero shock.
                    cell_loss = -sensitivity * shocks.get(factor, 0.0)
                    if not math.isfinite(cell_loss):
                        raise InvalidInput()
                    position_loss += cell_loss
                    factor_contributions[factor] += cell_loss
                if not math.isfinite(position_loss):
                    raise InvalidInput()
                position_contributions[pid] = position_loss
                # loss is the position-major fold of the same per-position
                # subtotals, so the position attribution totals it exactly;
                # the factor attribution folds the identical cells and
                # agrees up to floating-point rounding.
                scenario_loss += position_loss
            if not math.isfinite(scenario_loss):
                raise InvalidInput()
            for value in factor_contributions.values():
                if not math.isfinite(value):
                    raise InvalidInput()
            results.append(
                {
                    "scenario_id": scenario_id,
                    "loss": scenario_loss,
                    "position_loss_contributions": position_contributions,
                    "factor_loss_contributions": factor_contributions,
                }
            )
            # Strictly greater keeps the earliest scenario on a tie.
            if worst_loss is None or scenario_loss > worst_loss:
                worst_loss = scenario_loss
                worst_index = index

        return {
            "currency": currency,
            "results": results,
            "worst_scenario": {
                "id": scenario_ids[worst_index],
                "loss": worst_loss,
            },
        }
