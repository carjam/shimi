from __future__ import annotations

import pandas as pd


class AllocationHistory:
    """Wide-format table: one row per loan, columns ``loan_index`` + one per ``lender_id``."""

    INDEX_COL = "loan_index"

    @staticmethod
    def empty(lender_ids: list[str]) -> pd.DataFrame:
        data: dict[str, pd.Series] = {AllocationHistory.INDEX_COL: pd.Series(dtype="int64")}
        for lid in lender_ids:
            data[lid] = pd.Series(dtype="float64")
        return pd.DataFrame(data)

    @staticmethod
    def append_row(
        history: pd.DataFrame,
        *,
        loan_index: int,
        amounts_by_lender: dict[str, float],
    ) -> pd.DataFrame:
        if history.columns.size == 0:
            history = AllocationHistory.empty(lender_ids=sorted(amounts_by_lender))

        expected = {AllocationHistory.INDEX_COL, *amounts_by_lender.keys()}
        if set(history.columns) != expected:
            raise ValueError(
                "History columns do not match lenders; rebuild with AllocationHistory.empty()"
            )

        row = {AllocationHistory.INDEX_COL: loan_index, **amounts_by_lender}
        added = pd.DataFrame([row], columns=history.columns)
        return pd.concat([history, added], ignore_index=True)
