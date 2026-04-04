"""Allocation engine: per-loan QP (CVXPY + OSQP)."""

from shimi.data.models import PortfolioPrior

from shimi.allocation.engine import (
    AllocationParams,
    AllocationResult,
    allocate_loan,
)

__all__ = [
    "AllocationParams",
    "AllocationResult",
    "PortfolioPrior",
    "allocate_loan",
]
