from __future__ import annotations

import numpy as np

from shimi.allocation import AllocationParams, forecast_per_lender_capital_exhaustion, run_exhaustion_simulation
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
