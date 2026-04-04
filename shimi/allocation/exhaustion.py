"""Forecast per-lender capital exhaustion under repeated identical loans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shimi.allocation.engine import AllocationParams, allocate_loan
from shimi.data.history import AllocationHistory
from shimi.data.models import LenderProgram, PortfolioPrior


def _history_mean_draws_per_loan(program: LenderProgram) -> dict[str, float] | None:
    """Mean allocated face per loan from in-memory history, per lender."""
    h = program.history
    if h is None or h.shape[0] == 0:
        return None
    ids = sorted(program.lenders.keys())
    cols = [c for c in h.columns if c not in (AllocationHistory.INDEX_COL, AllocationHistory.FICO_COL)]
    if not cols:
        return None
    out: dict[str, float] = {}
    for lid in ids:
        if lid not in h.columns:
            continue
        s = h[lid].astype(float)
        if s.notna().any():
            out[lid] = float(s.mean())
    return out if out else None


@dataclass(frozen=True)
class ExhaustionSimulation:
    """Result of rolling identical loans until stop.

    ``trajectory[k]`` maps lender_id → remaining commitment **after** ``k`` simulated loans
    (``k=0`` is the starting book, before any simulated allocation).
    """

    trajectory: list[dict[str, float]]
    first_draw: dict[str, float] | None
    first_exhaust_loan: dict[str, int | None]
    loans_simulated: int
    stop_reason: str
    max_loans: int


def run_exhaustion_simulation(
    program: LenderProgram,
    loan_amount: float,
    params: AllocationParams,
    *,
    loan_fico: float | None = None,
    portfolio_prior: PortfolioPrior | None = None,
    max_loans: int = 10_000,
    remaining_tol: float = 1e-4,
) -> ExhaustionSimulation:
    """Simulate successive identical loans; return remaining line after each step."""
    if max_loans < 1:
        raise ValueError("max_loans must be at least 1")

    ids = sorted(program.lenders.keys())
    p = program.clone()
    trajectory: list[dict[str, float]] = [
        {lid: float(p.lenders[lid].remaining_commitment) for lid in ids},
    ]
    first_draw: dict[str, float] | None = None
    first_exhaust_loan: dict[str, int | None] = {lid: None for lid in ids}
    loans_simulated = 0
    stop_reason: str = "max_horizon"

    for k in range(1, max_loans + 1):
        try:
            res = allocate_loan(
                p,
                loan_amount,
                params,
                loan_fico=loan_fico,
                portfolio_prior=portfolio_prior,
            )
        except ValueError:
            stop_reason = f"infeasible_at_loan_{k}"
            break

        if first_draw is None:
            first_draw = {lid: float(res.amounts_by_lender[lid]) for lid in ids}

        p.apply_loan_allocation(res.amounts_by_lender, loan_fico=loan_fico)
        loans_simulated = k
        trajectory.append({lid: float(p.lenders[lid].remaining_commitment) for lid in ids})

        for lid in ids:
            if first_exhaust_loan[lid] is None and p.lenders[lid].remaining_commitment <= remaining_tol:
                first_exhaust_loan[lid] = k

        if all(first_exhaust_loan[lid] is not None for lid in ids):
            stop_reason = "all_exhausted"
            break

    return ExhaustionSimulation(
        trajectory=trajectory,
        first_draw=first_draw,
        first_exhaust_loan=first_exhaust_loan,
        loans_simulated=loans_simulated,
        stop_reason=stop_reason,
        max_loans=max_loans,
    )


def per_lender_exhaustion_summary(program: LenderProgram, sim: ExhaustionSimulation) -> pd.DataFrame:
    """Build the same per-lender table as :func:`forecast_per_lender_capital_exhaustion` from a simulation."""
    ids = sorted(program.lenders.keys())
    rem0 = sim.trajectory[0]

    base = program.to_dataframe()[["lender_id", "name"]].copy()
    base["remaining_line"] = base["lender_id"].map(rem0)

    if sim.first_draw is None:
        base["draw_loan_1"] = np.nan
        base["approx_loans_flat_draw"] = np.nan
    else:
        base["draw_loan_1"] = base["lender_id"].map(sim.first_draw)
        base["approx_loans_flat_draw"] = np.where(
            base["draw_loan_1"] > 1e-12,
            base["remaining_line"] / base["draw_loan_1"],
            np.nan,
        )

    exhaust_map = {
        lid: float(sim.first_exhaust_loan[lid]) if sim.first_exhaust_loan[lid] is not None else np.nan
        for lid in ids
    }
    base["loans_until_depleted"] = base["lender_id"].map(exhaust_map)

    hist_means = _history_mean_draws_per_loan(program)
    if hist_means:
        base["avg_hist_draw_per_loan"] = base["lender_id"].map(lambda lid: hist_means.get(lid, np.nan))
        base["approx_loans_at_hist_avg"] = np.where(
            base["avg_hist_draw_per_loan"] > 1e-12,
            base["remaining_line"] / base["avg_hist_draw_per_loan"],
            np.nan,
        )
    else:
        base["avg_hist_draw_per_loan"] = np.nan
        base["approx_loans_at_hist_avg"] = np.nan

    return base


def forecast_per_lender_capital_exhaustion(
    program: LenderProgram,
    loan_amount: float,
    params: AllocationParams,
    *,
    loan_fico: float | None = None,
    portfolio_prior: PortfolioPrior | None = None,
    max_loans: int = 10_000,
    remaining_tol: float = 1e-4,
) -> tuple[pd.DataFrame, dict[str, str | int]]:
    """Simulate successive identical loans; estimate when each lender's line is depleted.

    For each loan, the program is re-optimized with :func:`allocate_loan` on the **current**
    remaining commitments (caps change over time). The optional ``portfolio_prior`` is **held
    fixed** across the simulation (same as the live UI prior block), so γ portfolio terms do not
    incorporate newly booked volume.

    Returns a per-lender table and metadata (``loans_simulated``, ``stop_reason``).
    """
    sim = run_exhaustion_simulation(
        program,
        loan_amount,
        params,
        loan_fico=loan_fico,
        portfolio_prior=portfolio_prior,
        max_loans=max_loans,
        remaining_tol=remaining_tol,
    )
    base = per_lender_exhaustion_summary(program, sim)
    meta: dict[str, str | int] = {
        "loans_simulated": sim.loans_simulated,
        "stop_reason": sim.stop_reason,
        "max_loans": sim.max_loans,
    }
    return base, meta


def simulate_capital_exhaustion_trajectory(
    program: LenderProgram,
    loan_amount: float,
    params: AllocationParams,
    *,
    loan_fico: float | None = None,
    portfolio_prior: PortfolioPrior | None = None,
    max_loans: int = 10_000,
    remaining_tol: float = 1e-4,
) -> tuple[list[dict[str, float]], dict[str, str | int]]:
    """Return ``trajectory`` (remaining after k loans) and metadata for charts.

    ``trajectory[k][lender_id]`` is remaining after ``k`` simulated loans (``k=0``: initial book).
    """
    sim = run_exhaustion_simulation(
        program,
        loan_amount,
        params,
        loan_fico=loan_fico,
        portfolio_prior=portfolio_prior,
        max_loans=max_loans,
        remaining_tol=remaining_tol,
    )
    meta = {
        "loans_simulated": sim.loans_simulated,
        "stop_reason": sim.stop_reason,
        "max_loans": sim.max_loans,
    }
    return sim.trajectory, meta
