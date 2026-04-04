from __future__ import annotations

from pathlib import Path

import pytest

from shimi.data import (
    LenderProgram,
    LenderState,
    load_lender_program_from_csv,
    load_loan_tape_from_csv,
    load_portfolio_prior_from_csv,
    portfolio_prior_from_loan_tape,
)
from shimi.data.history import AllocationHistory

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample_lenders.csv"
SAMPLE_LOANS = ROOT / "data" / "sample_loans.csv"
SAMPLE_PRIOR = ROOT / "data" / "sample_portfolio_prior.csv"
IDS4 = ["L001", "L002", "L003", "L004"]


def test_load_sample_program() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    assert len(prog.lenders) == 4
    df = prog.to_dataframe()
    assert set(df["lender_id"]) == {"L001", "L002", "L003", "L004"}
    shares = df["target_share"].astype(float)
    assert abs(shares.sum() - 1.0) < 1e-6
    assert prog.history.shape == (0, 6)  # loan_index + loan_fico + 4 lenders


def test_legacy_limit_musd_column(tmp_path: Path) -> None:
    p = tmp_path / "legacy.csv"
    p.write_text(
        "lender_id,name,limit_musd\n"
        "A,One,10\n"
        "B,Two,30\n",
        encoding="utf-8",
    )
    prog = load_lender_program_from_csv(p)
    assert prog.lenders["A"].total_commitment == 10.0
    assert prog.lenders["B"].remaining_commitment == 30.0
    s = prog.to_dataframe()["target_share"]
    assert abs(float(s.iloc[0]) - 0.25) < 1e-9
    assert abs(float(s.iloc[1]) - 0.75) < 1e-9


def test_apply_loan_allocation_updates_remaining_and_history() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    # Split a 10M loan proportionally to target_share (not necessarily optimal QP; data-layer only)
    df = prog.to_dataframe().set_index("lender_id")
    loan = 10.0
    amounts = {lid: float(df.loc[lid, "target_share"]) * loan for lid in df.index}
    prog.apply_loan_allocation(amounts, loan_index=0, loan_fico=720.0)
    for lid, amt in amounts.items():
        assert prog.lenders[lid].remaining_commitment == pytest.approx(
            float(df.loc[lid, "total_commitment"]) - amt
        )
    assert prog.history.shape == (1, 6)
    assert list(prog.history.columns) == [
        "loan_index",
        "loan_fico",
        "L001",
        "L002",
        "L003",
        "L004",
    ]
    assert float(prog.history["loan_fico"].iloc[0]) == 720.0


def test_apply_rejects_over_commitment() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    big = {lid: prog.lenders[lid].remaining_commitment * 2 for lid in prog.lenders}
    with pytest.raises(ValueError, match="exceeds remaining"):
        prog.apply_loan_allocation(big)


def test_allocation_history_append_row_column_mismatch() -> None:
    h = AllocationHistory.empty(["A", "B"])
    h2 = AllocationHistory.append_row(
        h, loan_index=0, amounts_by_lender={"A": 1.0, "B": 2.0}, loan_fico=700.0
    )
    with pytest.raises(ValueError, match="columns"):
        AllocationHistory.append_row(h2, loan_index=1, amounts_by_lender={"A": 1.0})


def test_load_portfolio_prior_and_loan_tape_samples() -> None:
    prior = load_portfolio_prior_from_csv(SAMPLE_PRIOR)
    assert prior.funded_face_by_lender["L001"] == pytest.approx(3.0)
    assert prior.fico_weighted_face_by_lender["L002"] == pytest.approx(5172.0)

    tape = load_loan_tape_from_csv(SAMPLE_LOANS)
    assert tape.shape[0] == 5
    derived = portfolio_prior_from_loan_tape(tape, IDS4, max_loan_index_inclusive=2)
    for lid in IDS4:
        assert derived.funded_face_by_lender[lid] == pytest.approx(
            prior.funded_face_by_lender[lid]
        )
        assert derived.fico_weighted_face_by_lender[lid] == pytest.approx(
            prior.fico_weighted_face_by_lender[lid]
        )
