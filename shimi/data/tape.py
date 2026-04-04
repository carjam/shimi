from __future__ import annotations

import pandas as pd

from shimi.data.history import AllocationHistory
from shimi.data.models import PortfolioPrior


def portfolio_prior_from_loan_tape(
    tape: pd.DataFrame,
    lender_ids: list[str],
    *,
    max_loan_index_inclusive: int | None = None,
) -> PortfolioPrior:
    """
    Build :class:`PortfolioPrior` by aggregating a wide loan tape (as from
    :func:`shimi.data.loaders.load_loan_tape_from_csv`).

    Each row must include ``loan_index``, ``loan_fico``, and amount columns for every id in
    ``lender_ids``. Only rows with ``loan_index <= max_loan_index_inclusive`` are included
    (if ``None``, all rows).
    """
    ids = sorted(lender_ids)
    missing_cols = {AllocationHistory.FICO_COL, AllocationHistory.INDEX_COL, *ids} - set(tape.columns)
    if missing_cols:
        raise ValueError(f"Loan tape missing columns: {sorted(missing_cols)}")

    sub = tape.copy()
    if max_loan_index_inclusive is not None:
        sub = sub.loc[sub[AllocationHistory.INDEX_COL] <= max_loan_index_inclusive]

    funded: dict[str, float] = {lid: 0.0 for lid in ids}
    fico_w: dict[str, float] = {lid: 0.0 for lid in ids}
    for _, row in sub.iterrows():
        fico = float(row[AllocationHistory.FICO_COL])
        for lid in ids:
            amt = float(row[lid])
            funded[lid] += amt
            fico_w[lid] += amt * fico

    return PortfolioPrior(
        funded_face_by_lender=funded,
        fico_weighted_face_by_lender=fico_w,
    )
