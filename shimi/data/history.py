from __future__ import annotations

import numpy as np
import pandas as pd


class AllocationHistory:
    """Wide-format table: ``loan_index``, ``loan_fico``, then one column per ``lender_id``."""

    INDEX_COL = "loan_index"
    FICO_COL = "loan_fico"

    @staticmethod
    def empty(lender_ids: list[str]) -> pd.DataFrame:
        data: dict[str, pd.Series] = {
            AllocationHistory.INDEX_COL: pd.Series(dtype="int64"),
            AllocationHistory.FICO_COL: pd.Series(dtype="float64"),
        }
        for lid in lender_ids:
            data[lid] = pd.Series(dtype="float64")
        return pd.DataFrame(data)

    @staticmethod
    def append_row(
        history: pd.DataFrame,
        *,
        loan_index: int,
        amounts_by_lender: dict[str, float],
        loan_fico: float | None = None,
    ) -> pd.DataFrame:
        if history.columns.size == 0:
            history = AllocationHistory.empty(lender_ids=sorted(amounts_by_lender))

        expected = {
            AllocationHistory.INDEX_COL,
            AllocationHistory.FICO_COL,
            *amounts_by_lender.keys(),
        }
        if set(history.columns) != expected:
            raise ValueError(
                "History columns do not match lenders; rebuild with AllocationHistory.empty()"
            )

        fico_val = float(loan_fico) if loan_fico is not None and np.isfinite(loan_fico) else np.nan
        row = {
            AllocationHistory.INDEX_COL: loan_index,
            AllocationHistory.FICO_COL: fico_val,
            **amounts_by_lender,
        }
        added = pd.DataFrame([row], columns=history.columns)
        return pd.concat([history, added], ignore_index=True)
