from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shimi.allocation import (
    AllocationParams,
    forecast_per_lender_capital_exhaustion,
    per_lender_exhaustion_summary,
    run_exhaustion_simulation,
)
from shimi.data.models import LenderProgram, LenderState


def test_exhaustion_symmetric_two_lender_book() -> None:
    """Equal lines and targets; flat $5 draw per lender each $10 loan → deplete in 4 loans."""
    lenders = [
        LenderState(
            lender_id="A",
            name="A",
            total_commitment=20.0,
            remaining_commitment=20.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
        LenderState(
            lender_id="B",
            name="B",
            total_commitment=20.0,
            remaining_commitment=20.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
    ]
    prog = LenderProgram.from_lenders(lenders)
    df, meta = forecast_per_lender_capital_exhaustion(
        prog,
        10.0,
        AllocationParams(alpha=100.0, beta=0.0, participation_floor=0.01),
        max_loans=50,
        remaining_tol=1e-6,
    )
    assert int(meta["loans_simulated"]) == 4
    assert meta["stop_reason"] == "all_exhausted"
    assert df["loans_until_depleted"].notna().all()
    assert np.allclose(df["loans_until_depleted"].to_numpy(), 4.0)

    sim = run_exhaustion_simulation(
        prog,
        10.0,
        AllocationParams(alpha=100.0, beta=0.0, participation_floor=0.01),
        max_loans=50,
        remaining_tol=1e-6,
    )
    assert len(sim.trajectory) == 5
    assert np.allclose([sim.trajectory[4][lid] for lid in ("A", "B")], 0.0, atol=1e-5)


def test_exhaustion_stops_when_infeasible() -> None:
    """Loan larger than aggregate remaining → zero simulated loans."""
    lenders = [
        LenderState(
            lender_id="A",
            name="A",
            total_commitment=5.0,
            remaining_commitment=5.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
        LenderState(
            lender_id="B",
            name="B",
            total_commitment=5.0,
            remaining_commitment=5.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
    ]
    prog = LenderProgram.from_lenders(lenders)
    df, meta = forecast_per_lender_capital_exhaustion(
        prog,
        100.0,
        AllocationParams(alpha=1.0, beta=0.0, participation_floor=0.01),
        max_loans=20,
    )
    assert int(meta["loans_simulated"]) == 0
    assert str(meta["stop_reason"]).startswith("infeasible_at_loan_")
    assert df["draw_loan_1"].isna().all()

    sim = run_exhaustion_simulation(
        prog,
        100.0,
        AllocationParams(alpha=1.0, beta=0.0, participation_floor=0.01),
        max_loans=20,
    )
    assert len(sim.trajectory) == 1


def test_run_exhaustion_max_loans_must_be_positive() -> None:
    prog = LenderProgram.from_lenders(
        [
            LenderState(
                lender_id="A",
                name="A",
                total_commitment=10.0,
                remaining_commitment=10.0,
                target_share=1.0,
                is_contractual_originator=False,
                avg_fico=720.0,
            ),
        ]
    )
    with pytest.raises(ValueError, match="max_loans"):
        run_exhaustion_simulation(prog, 1.0, AllocationParams(participation_floor=0.0), max_loans=0)


def test_per_lender_summary_matches_forecast_dataframe() -> None:
    lenders = [
        LenderState(
            lender_id="A",
            name="A",
            total_commitment=20.0,
            remaining_commitment=20.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
        LenderState(
            lender_id="B",
            name="B",
            total_commitment=20.0,
            remaining_commitment=20.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
    ]
    prog = LenderProgram.from_lenders(lenders)
    params = AllocationParams(alpha=100.0, beta=0.0, participation_floor=0.01)
    sim = run_exhaustion_simulation(prog, 10.0, params, max_loans=50)
    from_summary = per_lender_exhaustion_summary(prog, sim)
    from_forecast, _ = forecast_per_lender_capital_exhaustion(prog, 10.0, params, max_loans=50)
    pd.testing.assert_frame_equal(
        from_summary.reset_index(drop=True),
        from_forecast.reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_trajectory_remaining_is_non_increasing() -> None:
    """Each simulated step draws nonnegative face; remaining never increases."""
    from pathlib import Path

    from shimi.data import load_lender_program_from_csv

    root = Path(__file__).resolve().parents[1]
    prog = load_lender_program_from_csv(root / "data" / "sample_lenders.csv")
    sim = run_exhaustion_simulation(
        prog,
        6.0,
        AllocationParams(alpha=1.0, beta=0.05, participation_floor=0.05),
        max_loans=400,
    )
    ids = sorted(prog.lenders.keys())
    for lid in ids:
        seq = [float(step[lid]) for step in sim.trajectory]
        for a, b in zip(seq, seq[1:]):
            assert b <= a + 1e-5


def test_max_horizon_stop_reason_when_not_all_depleted() -> None:
    """Tiny max_loans stops before exhaustion → stop_reason max_horizon."""
    lenders = [
        LenderState(
            lender_id="A",
            name="A",
            total_commitment=1000.0,
            remaining_commitment=1000.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
        LenderState(
            lender_id="B",
            name="B",
            total_commitment=1000.0,
            remaining_commitment=1000.0,
            target_share=0.5,
            is_contractual_originator=False,
            avg_fico=720.0,
        ),
    ]
    prog = LenderProgram.from_lenders(lenders)
    sim = run_exhaustion_simulation(
        prog,
        10.0,
        AllocationParams(alpha=10.0, beta=0.0, participation_floor=0.05),
        max_loans=3,
    )
    assert sim.stop_reason == "max_horizon"
    assert sim.loans_simulated == 3
    assert len(sim.trajectory) == 4
