"""End-to-end checks across data loading, allocation, history, and exhaustion."""

from __future__ import annotations

from pathlib import Path

import pytest

from shimi.allocation import AllocationParams, PortfolioPrior, allocate_loan, run_exhaustion_simulation
from shimi.data import (
    load_lender_program_from_csv,
    load_loan_tape_from_csv,
    load_portfolio_prior_from_csv,
    portfolio_prior_from_loan_tape,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_LENDERS = ROOT / "data" / "sample_lenders.csv"
SAMPLE_LOANS = ROOT / "data" / "sample_loans.csv"
SAMPLE_PRIOR = ROOT / "data" / "sample_portfolio_prior.csv"
IDS4 = ["L001", "L002", "L003", "L004"]


def test_sample_book_allocate_and_exhaustion_share_state() -> None:
    """Load real CSV, allocate on a clone, then simulate exhaustion from the original book."""
    prog = load_lender_program_from_csv(SAMPLE_LENDERS)
    L = 7.5
    params = AllocationParams(alpha=1.0, beta=0.08, gamma_fico=0.0, participation_floor=0.05)
    res = allocate_loan(prog.clone(), L, params)
    assert sum(res.amounts_by_lender.values()) == pytest.approx(L, rel=1e-5)

    sim = run_exhaustion_simulation(prog, L, params, max_loans=800)
    assert sim.loans_simulated >= 1
    assert len(sim.trajectory) == sim.loans_simulated + 1
    assert sim.trajectory[0]["L001"] == pytest.approx(prog.lenders["L001"].remaining_commitment)


def test_prior_from_csv_matches_tape_then_allocate_with_gamma() -> None:
    prior_csv = load_portfolio_prior_from_csv(SAMPLE_PRIOR)
    tape = load_loan_tape_from_csv(SAMPLE_LOANS)
    prior_tape = portfolio_prior_from_loan_tape(tape, IDS4, max_loan_index_inclusive=2)
    assert prior_tape.funded_face_by_lender == pytest.approx(prior_csv.funded_face_by_lender)

    prog = load_lender_program_from_csv(SAMPLE_LENDERS)
    res = allocate_loan(
        prog.clone(),
        5.0,
        AllocationParams(alpha=0.5, beta=0.0, gamma_fico=2.0, participation_floor=0.05),
        loan_fico=710.0,
        portfolio_prior=prior_csv,
    )
    assert res.fico_fairness_mode == "portfolio"
    assert sum(res.amounts_by_lender.values()) == pytest.approx(5.0, rel=1e-5)


def test_apply_loan_then_allocate_on_mutated_book() -> None:
    prog = load_lender_program_from_csv(SAMPLE_LENDERS)
    df = prog.to_dataframe().set_index("lender_id")
    L1 = 12.0
    amounts = {lid: float(df.loc[lid, "target_share"]) * L1 for lid in df.index}
    prog.apply_loan_allocation(amounts, loan_index=0, loan_fico=705.0)

    assert prog.history.shape[0] == 1
    assert float(prog.history["loan_fico"].iloc[0]) == 705.0

    L2 = 4.0
    res = allocate_loan(prog, L2, AllocationParams(alpha=1.0, beta=0.05, participation_floor=0.05))
    assert sum(res.amounts_by_lender.values()) == pytest.approx(L2, rel=1e-5)
    for lid, amt in res.amounts_by_lender.items():
        assert amt <= prog.lenders[lid].remaining_commitment + 1e-4


def test_clone_isolation_after_apply() -> None:
    """Mutating one program must not change a prior clone used for allocation."""
    prog = load_lender_program_from_csv(SAMPLE_LENDERS)
    snap = prog.clone()
    df = prog.to_dataframe().set_index("lender_id")
    amounts = {lid: float(df.loc[lid, "target_share"]) * 5.0 for lid in df.index}
    prog.apply_loan_allocation(amounts, loan_index=0, loan_fico=700.0)

    for lid in prog.lenders:
        assert snap.lenders[lid].remaining_commitment == pytest.approx(
            load_lender_program_from_csv(SAMPLE_LENDERS).lenders[lid].remaining_commitment
        )


def test_custom_prior_round_trip_allocation() -> None:
    prog = load_lender_program_from_csv(SAMPLE_LENDERS)
    prior = PortfolioPrior(
        funded_face_by_lender={lid: 1.0 for lid in IDS4},
        fico_weighted_face_by_lender={lid: 700.0 for lid in IDS4},
    )
    res = allocate_loan(
        prog.clone(),
        3.0,
        AllocationParams(alpha=1.0, gamma_fico=0.1, participation_floor=0.05),
        portfolio_prior=prior,
    )
    assert sum(res.amounts_by_lender.values()) == pytest.approx(3.0, rel=1e-5)
    assert res.fico_fairness_mode == "portfolio"
