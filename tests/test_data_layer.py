from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shimi.data import (
    LenderProgram,
    LenderState,
    load_lender_program_from_csv,
    load_loan_tape_from_csv,
    load_portfolio_prior_from_csv,
    portfolio_prior_from_loan_tape,
    replay_allocation_history,
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


def test_lender_program_rejects_duplicate_ids() -> None:
    dup = [
        LenderState("A", "One", 10.0, 10.0, 1.0, False, 700.0),
        LenderState("A", "Two", 10.0, 10.0, 1.0, False, 700.0),
    ]
    with pytest.raises(ValueError, match="Duplicate lender_id"):
        LenderProgram.from_lenders(dup)


def test_apply_allocation_rejects_negative_amount() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    amounts = {lid: 1.0 for lid in prog.lenders}
    amounts["L001"] = -1.0
    with pytest.raises(ValueError, match="Negative allocation"):
        prog.apply_loan_allocation(amounts)


def test_apply_allocation_requires_all_lenders() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    amounts = {"L001": 1.0}
    with pytest.raises(ValueError, match="missing"):
        prog.apply_loan_allocation(amounts)


def test_allocation_history_nan_loan_fico_stored() -> None:
    h = AllocationHistory.empty(["X"])
    h2 = AllocationHistory.append_row(h, loan_index=0, amounts_by_lender={"X": 1.0}, loan_fico=None)
    assert np.isnan(float(h2["loan_fico"].iloc[0]))


def test_load_lender_csv_empty_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("lender_id,name,total_commitment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        load_lender_program_from_csv(p)


def test_load_lender_csv_missing_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("lender_id,name\nA,Alice\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_lender_program_from_csv(p)


def test_load_lender_csv_non_positive_total_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("lender_id,name,total_commitment\nA,Alice,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="total_commitment must be positive"):
        load_lender_program_from_csv(p)


def test_load_lender_csv_remaining_exceeds_total_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text(
        "lender_id,name,total_commitment,remaining_commitment\nA,Alice,10,20\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot exceed total"):
        load_lender_program_from_csv(p)


def test_load_lender_csv_target_share_sum_invalid_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text(
        "lender_id,name,total_commitment,target_share\n"
        "A,One,10,0.1\n"
        "B,Two,10,0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sum to"):
        load_lender_program_from_csv(p)


def test_load_portfolio_prior_negative_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text(
        "lender_id,prior_funded,prior_fico_weighted\nL001,-1,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-negative"):
        load_portfolio_prior_from_csv(p)


def test_load_loan_tape_missing_required_columns_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("loan_index,x\n0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="loan_index and loan_fico"):
        load_loan_tape_from_csv(p)


def test_portfolio_prior_from_tape_missing_lender_column_raises() -> None:
    tape = pd.DataFrame({"loan_index": [0], "loan_fico": [700.0]})
    with pytest.raises(ValueError, match="missing columns"):
        portfolio_prior_from_loan_tape(tape, ["L001", "L002"])


def test_replay_allocation_history_updates_remaining_and_rows() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    tape = load_loan_tape_from_csv(SAMPLE_LOANS)
    tape_two = tape.loc[tape["loan_index"] <= 1].copy()
    r0 = {lid: prog.lenders[lid].remaining_commitment for lid in prog.lenders}
    replay_allocation_history(prog, tape_two)
    assert prog.history.shape[0] == 2
    per_loan = 1.0 + 2.4 + 0.7 + 1.6
    assert prog.lenders["L001"].remaining_commitment == pytest.approx(r0["L001"] - 2 * 1.0)
    assert prog.lenders["L002"].remaining_commitment == pytest.approx(r0["L002"] - 2 * 2.4)


def test_replay_allocation_history_rejects_duplicate_loan_index() -> None:
    prog = load_lender_program_from_csv(SAMPLE_CSV)
    bad = pd.DataFrame(
        [
            {"loan_index": 0, "loan_fico": 700.0, "L001": 1.0, "L002": 1.0, "L003": 1.0, "L004": 1.0},
            {"loan_index": 0, "loan_fico": 700.0, "L001": 1.0, "L002": 1.0, "L003": 1.0, "L004": 1.0},
        ]
    )
    with pytest.raises(ValueError, match="Duplicate loan_index"):
        replay_allocation_history(prog, bad)
