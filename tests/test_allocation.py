from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shimi.allocation import AllocationParams, PortfolioPrior, allocate_loan
from shimi.data import load_lender_program_from_csv

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample_lenders.csv"


def test_allocate_sample_sums_to_loan_and_respects_floor() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    L = 10.0
    res = allocate_loan(
        prog,
        L,
        AllocationParams(alpha=1.0, beta=0.0, participation_floor=0.05),
    )
    total = sum(res.amounts_by_lender.values())
    assert total == pytest.approx(L, rel=1e-5)
    for lid, sh in res.shares.items():
        assert sh >= 0.05 - 1e-5
        assert res.amounts_by_lender[lid] <= prog.lenders[lid].remaining_commitment + 1e-5


def test_high_alpha_tracks_target_shares() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    L = 1.0
    ids = sorted(prog.lenders)
    t = np.array([prog.lenders[i].target_share for i in ids])
    res = allocate_loan(
        prog,
        L,
        AllocationParams(alpha=50.0, beta=0.0, participation_floor=0.01),
    )
    got = np.array([res.shares[i] for i in ids])
    assert np.linalg.norm(got - t) < 0.05


def test_beta_reduces_contractual_utilization_when_possible() -> None:
    """Same book; higher beta should lower share on the contractual lender when others can absorb."""
    from shimi.data.models import LenderProgram, LenderState

    lenders = [
        LenderState(
            lender_id="A",
            name="CO",
            total_commitment=100.0,
            remaining_commitment=100.0,
            target_share=1 / 3,
            is_contractual_originator=True,
            avg_fico=720.0,
        ),
        LenderState(
            lender_id="B",
            name="B",
            total_commitment=100.0,
            remaining_commitment=100.0,
            target_share=1 / 3,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
        LenderState(
            lender_id="C",
            name="C",
            total_commitment=100.0,
            remaining_commitment=100.0,
            target_share=1 / 3,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
    ]
    prog = LenderProgram.from_lenders(lenders)
    L = 30.0
    low_b = allocate_loan(prog.clone(), L, AllocationParams(alpha=1.0, beta=0.01, participation_floor=0.05))
    high_b = allocate_loan(prog.clone(), L, AllocationParams(alpha=1.0, beta=50.0, participation_floor=0.05))
    assert high_b.shares["A"] < low_b.shares["A"] - 1e-3


def test_infeasible_when_loan_exceeds_aggregate_remaining() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    total_rem = sum(l.remaining_commitment for l in prog.lenders.values())
    with pytest.raises(ValueError, match="Aggregate remaining"):
        allocate_loan(prog, total_rem * 1.5, AllocationParams())


def test_allocation_result_includes_loan_fico() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    res = allocate_loan(prog, 5.0, AllocationParams(), loan_fico=740.0)
    assert res.loan_fico == 740.0


def test_loan_fico_must_be_positive_when_provided() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    with pytest.raises(ValueError, match="loan_fico"):
        allocate_loan(prog, 5.0, AllocationParams(), loan_fico=0.0)


def test_high_gamma_pulls_toward_equal_shares() -> None:
    """Symmetric book: strong γ and weak α → shares near 1/n (fair FICO-weighted slices)."""
    from shimi.data.models import LenderProgram, LenderState

    lenders = [
        LenderState(
            lender_id="A",
            name="A",
            total_commitment=50.0,
            remaining_commitment=50.0,
            target_share=1 / 3,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
        LenderState(
            lender_id="B",
            name="B",
            total_commitment=50.0,
            remaining_commitment=50.0,
            target_share=1 / 3,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
        LenderState(
            lender_id="C",
            name="C",
            total_commitment=50.0,
            remaining_commitment=50.0,
            target_share=1 / 3,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
    ]
    prog = LenderProgram.from_lenders(lenders)
    res = allocate_loan(
        prog,
        9.0,
        AllocationParams(alpha=0.01, beta=0.0, gamma_fico=50.0, participation_floor=0.05),
        loan_fico=720.0,
    )
    for sh in res.shares.values():
        assert sh == pytest.approx(1 / 3, abs=0.02)
    assert res.fico_fairness_mode == "cold_start"


def test_gamma_with_portfolio_prior_rebalances_weighted_fico() -> None:
    """Imbalanced cumulative FICO×face; high γ steers this loan's split vs γ=0 (weak α)."""
    from shimi.data.models import LenderProgram, LenderState

    lenders = [
        LenderState(
            lender_id="A",
            name="A",
            total_commitment=200.0,
            remaining_commitment=200.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
        LenderState(
            lender_id="B",
            name="B",
            total_commitment=200.0,
            remaining_commitment=200.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
    ]
    prog = LenderProgram.from_lenders(lenders)
    prior = PortfolioPrior(
        funded_face_by_lender={"A": 100.0, "B": 100.0},
        fico_weighted_face_by_lender={"A": 700.0 * 100.0, "B": 740.0 * 100.0},
    )
    L = 10.0
    base = AllocationParams(alpha=0.05, beta=0.0, gamma_fico=0.0, participation_floor=0.05)
    tilt = AllocationParams(alpha=0.05, beta=0.0, gamma_fico=80.0, participation_floor=0.05)
    r0 = allocate_loan(prog.clone(), L, base, loan_fico=760.0, portfolio_prior=prior)
    r1 = allocate_loan(prog.clone(), L, tilt, loan_fico=760.0, portfolio_prior=prior)
    assert r1.fico_fairness_mode == "portfolio"
    # Lender A had lower portfolio avg (700 vs 740); a high FICO new loan should tilt more to A to rebalance.
    assert r1.shares["A"] > r0.shares["A"] + 0.02


def test_portfolio_prior_missing_lender_raises() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    bad = PortfolioPrior(
        funded_face_by_lender={"L001": 1.0},
        fico_weighted_face_by_lender={"L001": 700.0},
    )
    with pytest.raises(ValueError, match="missing"):
        allocate_loan(prog, 5.0, AllocationParams(gamma_fico=1.0), portfolio_prior=bad)


def test_infeasible_participation_floor_too_high() -> None:
    from shimi.data.models import LenderProgram, LenderState

    lenders = [
        LenderState(
            lender_id="A",
            name="A",
            total_commitment=10.0,
            remaining_commitment=10.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
        LenderState(
            lender_id="B",
            name="B",
            total_commitment=10.0,
            remaining_commitment=10.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
    ]
    prog = LenderProgram.from_lenders(lenders)
    with pytest.raises(ValueError, match="participation_floor must be in"):
        allocate_loan(prog, 1.0, AllocationParams(participation_floor=0.51))


def test_loan_amount_must_be_positive() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    with pytest.raises(ValueError, match="loan_amount must be positive"):
        allocate_loan(prog, 0.0, AllocationParams())
    with pytest.raises(ValueError, match="loan_amount must be positive"):
        allocate_loan(prog, -1.0, AllocationParams())


def test_gamma_zero_fairness_mode_none() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    res = allocate_loan(prog, 4.0, AllocationParams(gamma_fico=0.0, participation_floor=0.05))
    assert res.fico_fairness_mode == "none"


def test_participation_floor_exceeds_remaining_per_lender_cap() -> None:
    """One lender's remaining/L is below the floor → infeasible before solve."""
    from shimi.data.models import LenderProgram, LenderState

    lenders = [
        LenderState(
            lender_id="Tight",
            name="Tight",
            total_commitment=10.0,
            remaining_commitment=1.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
        LenderState(
            lender_id="Deep",
            name="Deep",
            total_commitment=100.0,
            remaining_commitment=100.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
    ]
    prog = LenderProgram.from_lenders(lenders)
    with pytest.raises(ValueError, match="Participation floor exceeds remaining"):
        allocate_loan(prog, 10.0, AllocationParams(alpha=1.0, beta=0.0, participation_floor=0.15))
