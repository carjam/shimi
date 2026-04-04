from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandas as pd

from shimi.data.history import AllocationHistory


@dataclass
class LenderState:
    """Single lender row used as authoritative input and live simulation state."""

    lender_id: str
    name: str
    total_commitment: float
    remaining_commitment: float
    target_share: float
    is_contractual_originator: bool
    avg_fico: float
    region: str | None = None
    asset_class: str | None = None
    risk_tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lender_id": self.lender_id,
            "name": self.name,
            "total_commitment": self.total_commitment,
            "remaining_commitment": self.remaining_commitment,
            "target_share": self.target_share,
            "is_contractual_originator": self.is_contractual_originator,
            "avg_fico": self.avg_fico,
            "region": self.region,
            "asset_class": self.asset_class,
            "risk_tier": self.risk_tier,
        }


@dataclass
class LenderProgram:
    """Lender book plus optional allocation history for metrics and projections."""

    lenders: dict[str, LenderState]
    history: pd.DataFrame

    @classmethod
    def from_lenders(cls, lenders: list[LenderState]) -> LenderProgram:
        by_id = {l.lender_id: l for l in lenders}
        if len(by_id) != len(lenders):
            raise ValueError("Duplicate lender_id in lender list")
        return cls(
            lenders=by_id,
            history=AllocationHistory.empty(lender_ids=sorted(by_id)),
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows = [s.to_dict() for s in self.lenders.values()]
        return pd.DataFrame(rows).sort_values("lender_id", ignore_index=True)

    def clone(self) -> LenderProgram:
        return LenderProgram(
            lenders={k: deepcopy(v) for k, v in self.lenders.items()},
            history=self.history.copy(),
        )

    def apply_loan_allocation(
        self,
        amounts_by_lender: dict[str, float],
        *,
        loan_index: int | None = None,
    ) -> None:
        """Subtract allocated amounts from remaining_commitment and record history."""
        ids = set(self.lenders)
        if set(amounts_by_lender) != ids:
            missing = ids - set(amounts_by_lender)
            extra = set(amounts_by_lender) - ids
            raise ValueError(f"Allocation keys must match lenders; missing={missing!r}, extra={extra!r}")

        for lid, amt in amounts_by_lender.items():
            if amt < 0:
                raise ValueError(f"Negative allocation for {lid}: {amt}")
            st = self.lenders[lid]
            if amt - st.remaining_commitment > 1e-6:
                raise ValueError(
                    f"Allocation for {lid} ({amt}) exceeds remaining ({st.remaining_commitment})"
                )

        if loan_index is None:
            loan_index = int(self.history.shape[0])

        for lid, amt in amounts_by_lender.items():
            self.lenders[lid].remaining_commitment -= amt

        self.history = AllocationHistory.append_row(
            self.history,
            loan_index=loan_index,
            amounts_by_lender=amounts_by_lender,
        )
