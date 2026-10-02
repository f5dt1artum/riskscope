"""Core service surface for RiskScope.

The frozen baseline reports process health; historical-simulation VaR and
expected shortfall now live behind :meth:`Service.historical_var`. The public
surface stays backward compatible.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, ROUND_CEILING

from . import __version__

MAX_OBSERVATIONS = 10000


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

    def historical_var(self, raw: bytes | str) -> dict:
        """Validate a historical VaR request and compute the result.

        ``raw`` is the unparsed request body. Parse failures and non-object
        payloads raise :class:`InvalidRequest`; everything else that is
        well-formed JSON but semantically wrong raises :class:`InvalidInput`
        or :class:`RequestTooLarge`.
        """
        try:
            payload = json.loads(raw, parse_constant=_reject_constant)
        except _JsonConstant:
            raise InvalidInput(message="NaN and Infinity are not allowed")
        except (TypeError, ValueError):
            raise InvalidRequest("request body must be valid JSON")
        if not isinstance(payload, dict):
            raise InvalidRequest("request body must be a JSON object")
        return self._historical_var(payload)

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
