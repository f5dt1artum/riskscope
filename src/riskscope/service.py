"""Core service surface for RiskScope.

Process health reporting (``Service.health``) is the frozen baseline and
keeps its exact semantics.  The historical-simulation market risk engine
lives behind ``Service.historical_var``; every documented failure mode is
raised as :class:`RiskError` carrying an HTTP status and error code.
"""

from __future__ import annotations

import math
from typing import Any

from . import __version__

MAX_OBSERVATIONS = 10000


class RiskError(Exception):
    """Structured failure: HTTP status, machine-readable code and message."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _is_number(value: Any) -> bool:
    # bool is a subclass of int but is never an accepted risk number.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ensure_finite_tree(value: Any) -> None:
    """Reject NaN/Infinity anywhere in the parsed JSON payload."""
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RiskError(422, "invalid_input", "numeric values must be finite")
    elif isinstance(value, dict):
        for item in value.values():
            _ensure_finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _ensure_finite_tree(item)


def _ensure_number_map(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise RiskError(
            422, "invalid_input", f"{label} must be a non-empty object of finite numbers"
        )
    for name, number in value.items():
        if not isinstance(name, str) or not name:
            raise RiskError(
                422, "invalid_input", f"{label} factor names must be non-empty strings"
            )
        if not _is_number(number) or not math.isfinite(number):
            raise RiskError(
                422, "invalid_input", f"{label}.{name} must be a finite number"
            )
    return value


class Service:
    """Risk engine service. Health reporting plus historical simulation."""

    name = "riskscope"
    version = __version__

    def health(self) -> dict[str, str]:
        return {"status": "ok", "service": self.name, "version": self.version}

    def historical_var(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run portfolio historical-simulation VaR / expected shortfall.

        ``payload`` must already be a parsed JSON object. Structural request
        errors (400) are handled at the HTTP edge.
        """
        _ensure_finite_tree(payload)

        raw_observations = payload.get("observations")
        if isinstance(raw_observations, list) and len(raw_observations) > MAX_OBSERVATIONS:
            raise RiskError(
                413,
                "request_too_large",
                f"at most {MAX_OBSERVATIONS} observations are supported",
            )

        confidence = payload.get("confidence")
        if not _is_number(confidence) or not math.isfinite(confidence):
            raise RiskError(422, "invalid_input", "confidence must be a finite number")
        if not 0 < confidence < 1:
            raise RiskError(
                422, "invalid_input", "confidence must be strictly between 0 and 1"
            )

        currency = payload.get("currency", "USD")
        if not isinstance(currency, str) or not currency:
            raise RiskError(422, "invalid_input", "currency must be a non-empty string")

        raw_positions = payload.get("positions")
        if not isinstance(raw_positions, list) or not raw_positions:
            raise RiskError(422, "invalid_input", "positions must be a non-empty array")

        positions: list[tuple[str, dict[str, float]]] = []
        for position in raw_positions:
            if not isinstance(position, dict):
                raise RiskError(422, "invalid_input", "each position must be an object")
            position_id = position.get("id")
            if not isinstance(position_id, str) or not position_id:
                raise RiskError(
                    422, "invalid_input", "position id must be a non-empty string"
                )
            sensitivities = _ensure_number_map(
                position.get("sensitivities"), "sensitivities"
            )
            positions.append((position_id, dict(sensitivities)))

        position_ids = [position_id for position_id, _ in positions]
        if len(set(position_ids)) != len(position_ids):
            raise RiskError(422, "duplicate_position", "position ids must be unique")

        if not isinstance(raw_observations, list) or len(raw_observations) < 2:
            raise RiskError(
                422, "invalid_input", "observations must contain at least two items"
            )

        observations: list[tuple[str, dict[str, float]]] = []
        for observation in raw_observations:
            if not isinstance(observation, dict):
                raise RiskError(422, "invalid_input", "each observation must be an object")
            date = observation.get("date")
            if not isinstance(date, str) or not date:
                raise RiskError(
                    422, "invalid_input", "observation date must be a non-empty string"
                )
            factor_returns = _ensure_number_map(
                observation.get("factor_returns"), "factor_returns"
            )
            observations.append((date, dict(factor_returns)))

        dates = [date for date, _ in observations]
        if len(set(dates)) != len(dates):
            raise RiskError(422, "duplicate_observation", "observation dates must be unique")

        factors = sorted({factor for _, sensitivities in positions for factor in sensitivities})
        for date, factor_returns in observations:
            missing = [factor for factor in factors if factor not in factor_returns]
            if missing:
                raise RiskError(
                    422,
                    "missing_factor",
                    f"observation {date!r} misses referenced factor(s): "
                    + ", ".join(missing),
                )

        # Aggregate every position's sensitivity to the same factor.
        aggregate = {factor: 0.0 for factor in factors}
        for _, sensitivities in positions:
            for factor, sensitivity in sensitivities.items():
                aggregate[factor] += sensitivity

        # Per-observation portfolio loss and per-factor loss contributions.
        losses: list[float] = []
        factor_losses: list[dict[str, float]] = []
        for _, factor_returns in observations:
            per_factor = {
                factor: -aggregate[factor] * factor_returns[factor] for factor in factors
            }
            loss = math.fsum(per_factor.values()) + 0.0
            losses.append(loss)
            factor_losses.append(per_factor)

        count = len(observations)

        # Ascending loss order; Python's sort is stable, so ties keep input order.
        ascending = sorted(range(count), key=losses.__getitem__)
        var_index = math.ceil(confidence * count) - 1
        var = losses[ascending[var_index]] + 0.0

        # Worst k observations: larger loss first, ties resolved by input order.
        tail_size = max(1, math.ceil((1 - confidence) * count))
        tail = sorted(range(count), key=lambda i: (-losses[i], i))[:tail_size]
        contributions = {
            factor: math.fsum(factor_losses[i][factor] for i in tail) / tail_size + 0.0
            for factor in factors
        }
        # Define ES from the contributions so their sum equals ES exactly.
        expected_shortfall = math.fsum(contributions.values()) + 0.0

        return {
            "currency": currency,
            "var": var,
            "expected_shortfall": expected_shortfall,
            "losses": [
                {"date": date, "loss": loss}
                for (date, _), loss in zip(observations, losses)
            ],
            "factor_expected_shortfall_contributions": contributions,
        }
