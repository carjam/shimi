"""Allocation engine: per-loan QP (CVXPY + OSQP)."""

from shimi.data.models import PortfolioPrior

from shimi.allocation.engine import (
    AllocationParams,
    AllocationResult,
    allocate_loan,
)
from shimi.allocation.exhaustion import (
    ExhaustionSimulation,
    forecast_per_lender_capital_exhaustion,
    per_lender_exhaustion_summary,
    run_exhaustion_simulation,
    simulate_capital_exhaustion_trajectory,
)

__all__ = [
    "AllocationParams",
    "AllocationResult",
    "ExhaustionSimulation",
    "PortfolioPrior",
    "allocate_loan",
    "forecast_per_lender_capital_exhaustion",
    "per_lender_exhaustion_summary",
    "run_exhaustion_simulation",
    "simulate_capital_exhaustion_trajectory",
]
