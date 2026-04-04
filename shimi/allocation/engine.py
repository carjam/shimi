from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from shimi.data.models import LenderProgram, PortfolioPrior


@dataclass(frozen=True)
class AllocationParams:
    """Tuning weights for the per-loan quadratic program."""

    alpha: float = 1.0
    """Weight on deviation from target share (fair risk distribution)."""

    beta: float = 0.25
    """Weight on squared utilization for contractual originators (capital protection)."""

    gamma_fico: float = 0.0
    """Fair-dealing weight on **portfolio** FICO equality over time (see ``PortfolioPrior``).

    With cumulative priors: penalizes imbalance in ``F_i + x_i f - μ (A_i + x_i)`` (same μ for all
    lenders = equal post-deal weighted avg FICO if zeros could be achieved). With **no** prior
    (cold start): falls back to pulling toward **equal shares** on this loan, scaled by loan FICO.
    """

    participation_floor: float = 0.05
    """Minimum share of the loan each lender receives (e.g. 0.05 = 5%)."""

    ridge: float = 1e-8
    """Tiny quadratic regularizer on ``s`` for numerical stability when all weights are zero."""


@dataclass(frozen=True)
class AllocationResult:
    """Optimal (or best-effort) allocation for one loan."""

    amounts_by_lender: dict[str, float]
    shares: dict[str, float]
    loan_fico: float
    """FICO (or representative credit score) for this loan."""
    objective_value: float | None
    solver_status: str
    is_optimal: bool
    fico_fairness_mode: str
    """``"portfolio"`` if γ used cumulative priors; ``"cold_start"`` if γ used equal-share fallback."""


def _portfolio_arrays(ids: list[str], prior: PortfolioPrior) -> tuple[np.ndarray, np.ndarray]:
    a_list: list[float] = []
    f_list: list[float] = []
    for lid in ids:
        if lid not in prior.funded_face_by_lender:
            raise ValueError(f"portfolio prior missing funded_face for lender {lid!r}")
        if lid not in prior.fico_weighted_face_by_lender:
            raise ValueError(f"portfolio prior missing fico_weighted_face for lender {lid!r}")
        a = float(prior.funded_face_by_lender[lid])
        f_w = float(prior.fico_weighted_face_by_lender[lid])
        if a < 0 or f_w < 0:
            raise ValueError(f"prior amounts must be non-negative for {lid!r}")
        if a < 1e-15 and f_w > 1e-9:
            raise ValueError(f"prior fico_weighted must be 0 when funded is 0 for {lid!r}")
        a_list.append(a)
        f_list.append(f_w)
    return np.array(a_list, dtype=float), np.array(f_list, dtype=float)


def allocate_loan(
    program: LenderProgram,
    loan_amount: float,
    params: AllocationParams | None = None,
    *,
    loan_fico: float | None = None,
    portfolio_prior: PortfolioPrior | None = None,
) -> AllocationResult:
    """
    Solve the per-loan QP: choose nonnegative shares ``s`` summing to 1 such that
    ``floor <= s_i <= remaining_i / loan_amount``, minimizing

    - ``alpha * ||s - t||^2`` (target share ``t``),
    - ``beta * sum_{CO} (loan_amount * s_i / r_i)^2`` (contractual utilization),
    - **γ — FICO fair dealing**
        - If ``portfolio_prior`` is set and total prior funded face > 0: minimize a sum of squares
          of ``(F_i - μ A_i) + x_i (f - μ)`` with ``x_i = s_i * loan_amount``, ``μ = sum(F)/sum(A)``
          (group portfolio weighted-average FICO *before* this loan), ``f`` = this loan's FICO.
          That aligns each lender's **incremental** FICO-mass with moving toward a **common**
          portfolio average (equal weighted-average FICO across lenders when the system is balanced).
        - Otherwise (cold start): ``γ * (f/850)^2 * ||s - u||^2`` with ``u_i = 1/n``.
    - plus ``ridge * ||s||^2``.

    Each loan has a **single** representative ``loan_fico``. If omitted, defaults to the mean of
    lenders' ``avg_fico``.

    Solved with OSQP via CVXPY.
    """
    if params is None:
        params = AllocationParams()

    if loan_amount <= 0:
        raise ValueError("loan_amount must be positive")

    ids = sorted(program.lenders.keys())
    n = len(ids)
    if n == 0:
        raise ValueError("program has no lenders")

    rem = np.array([program.lenders[i].remaining_commitment for i in ids], dtype=float)
    t = np.array([program.lenders[i].target_share for i in ids], dtype=float)
    co = np.array([1.0 if program.lenders[i].is_contractual_originator else 0.0 for i in ids])
    lender_fico = np.array([program.lenders[i].avg_fico for i in ids], dtype=float)

    if loan_fico is not None:
        if not np.isfinite(loan_fico) or float(loan_fico) <= 0:
            raise ValueError("loan_fico must be a positive finite number when provided")
        loan_fico_used = float(loan_fico)
    else:
        loan_fico_used = float(np.mean(lender_fico))

    floor = float(params.participation_floor)
    if not (0.0 <= floor < 1.0 / n):
        raise ValueError(f"participation_floor must be in [0, 1/n) with n={n}; got {floor}")

    caps = rem / loan_amount
    if np.any(caps + 1e-12 < floor):
        raise ValueError(
            "Participation floor exceeds remaining/L for at least one lender; problem infeasible."
        )
    if n * floor > 1.0 + 1e-9:
        raise ValueError("Sum of participation floors exceeds 100%; problem infeasible.")
    if float(np.sum(caps)) < 1.0 - 1e-9:
        raise ValueError(
            "Aggregate remaining commitment is less than the loan amount; problem infeasible."
        )

    s = cp.Variable(n)

    constraints = [
        cp.sum(s) == 1.0,
        s >= floor,
        s <= caps,
    ]

    alpha = float(params.alpha)
    beta = float(params.beta)
    gamma = float(params.gamma_fico)
    ridge = float(params.ridge)

    fico_fairness_mode = "none"
    terms: list = []
    if alpha > 0:
        terms.append(alpha * cp.sum_squares(s - t))
    if beta > 0 and np.any(co > 0):
        rem_safe = np.maximum(rem, 1e-9)
        utilization = cp.multiply(s * loan_amount, 1.0 / rem_safe)
        terms.append(beta * cp.sum(cp.multiply(co, cp.square(utilization))))
    if gamma > 0:
        use_portfolio_term = False
        if portfolio_prior is not None:
            A_vec, F_vec = _portfolio_arrays(ids, portfolio_prior)
            sum_a = float(np.sum(A_vec))
            if sum_a > 1e-12:
                mu = float(np.sum(F_vec) / sum_a)
                c_vec = F_vec - mu * A_vec
                d = loan_fico_used - mu
                if abs(d) > 1e-9:
                    # (F_i + x_i f) - mu (A_i + x_i) = c_i + x_i (f - mu); x_i = s_i * L
                    rx = c_vec + s * loan_amount * d
                    scale = max(
                        1.0,
                        float(np.max(np.abs(c_vec))) + 1e-9,
                        loan_amount * abs(d),
                    )
                    terms.append(gamma * cp.sum_squares(rx / scale))
                    use_portfolio_term = True
                    fico_fairness_mode = "portfolio"
        if not use_portfolio_term:
            u = np.ones(n) / n
            fico_scale = 850.0
            fair_scale = (loan_fico_used / fico_scale) ** 2
            terms.append(gamma * fair_scale * cp.sum_squares(s - u))
            fico_fairness_mode = "cold_start"
    if ridge > 0:
        terms.append(ridge * cp.sum_squares(s))
    if not terms:
        terms.append(1e-6 * cp.sum_squares(s))

    objective = terms[0]
    for term in terms[1:]:
        objective += term

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    status = problem.status or "unknown"
    optimal = str(status).lower() in ("optimal", "optimal_inaccurate")
    if s.value is None:
        raise RuntimeError(f"Allocation solver produced no solution (status={status!r})")

    shares = np.asarray(s.value).flatten()
    amounts = shares * loan_amount
    amounts_by_lender = {lid: float(amt) for lid, amt in zip(ids, amounts, strict=True)}
    shares_dict = {lid: float(sh) for lid, sh in zip(ids, shares, strict=True)}

    return AllocationResult(
        amounts_by_lender=amounts_by_lender,
        shares=shares_dict,
        loan_fico=loan_fico_used,
        objective_value=float(problem.value) if problem.value is not None else None,
        solver_status=str(status),
        is_optimal=bool(optimal),
        fico_fairness_mode=fico_fairness_mode,
    )
