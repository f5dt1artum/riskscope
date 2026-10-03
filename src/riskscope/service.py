"""Core service surface for RiskScope.

The frozen baseline reports process health; historical-simulation VaR and
expected shortfall now live behind :meth:`Service.historical_var` and batch
sensitivity stress tests behind :meth:`Service.stress_test`. Counterparty
credit exposure and expected loss live behind
:meth:`Service.counterparty_exposure`. The public surface stays backward
compatible.
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
