"""Pure functions for concentration, FICO-weighted face, and history-based cumulatives."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from shimi.data.history import AllocationHistory


def gini_coefficient(values: Sequence[float] | np.ndarray) -> float:
    """Gini coefficient of nonnegative weights (e.g. dollar amounts or shares on one loan).

    Uses the standard definition for nonnegative ``x``: 0 = perfect equality (including all
    zeros), higher = more concentration. For a single positive element and rest zeros,
    approaches ``(n-1)/n`` as ``n`` grows.

    Raises ``ValueError`` if ``values`` is empty or contains negative entries.
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("gini_coefficient: values must be non-empty")
    if np.any(x < -1e-15):
        raise ValueError("gini_coefficient: negative values are not supported")
    x = np.maximum(x, 0.0)
    total = float(x.sum())
    if total <= 1e-15:
        return 0.0
    xs = np.sort(x)
    n = int(xs.size)
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.dot(idx, xs)) / (n * total) - (n + 1.0) / n)


def gini_of_loan_split(amounts_by_lender: dict[str, float], *, lender_ids: list[str] | None = None) -> float:
    """Gini across lenders for one loan, using amounts in ``lender_ids`` order (default: sorted keys)."""
    ids = sorted(amounts_by_lender.keys()) if lender_ids is None else list(lender_ids)
    vec = np.array([float(amounts_by_lender[i]) for i in ids], dtype=float)
    return gini_coefficient(vec)


def per_lender_fico_weighted_face(amounts_by_lender: dict[str, float], loan_fico: float) -> dict[str, float]:
    """Per lender: allocated face × loan FICO for one loan (same FICO for all slices)."""
    f = float(loan_fico)
    return {lid: float(amt) * f for lid, amt in amounts_by_lender.items()}


def total_fico_weighted_face(amounts_by_lender: dict[str, float], loan_fico: float) -> float:
    """Sum of face×FICO over lenders; equals ``loan_fico * sum(amounts)`` for one loan FICO."""
    return float(sum(per_lender_fico_weighted_face(amounts_by_lender, loan_fico).values()))


def _sorted_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.shape[0] == 0:
        return history.copy()
    return history.sort_values(AllocationHistory.INDEX_COL, ignore_index=True)


def history_lender_columns(history: pd.DataFrame) -> list[str]:
    """Lender id columns present in ``history`` (excludes ``loan_index`` and ``loan_fico``)."""
    skip = {AllocationHistory.INDEX_COL, AllocationHistory.FICO_COL}
    return [c for c in history.columns if c not in skip]


def cumulative_funded_by_lender(history: pd.DataFrame) -> pd.DataFrame:
    """Per-lender cumulative funded face after each row (history sorted by ``loan_index``).

    Columns: ``loan_index``, ``loan_fico``, then one cumulative column per lender (same names
    as in ``history``). Empty history returns an empty frame with expected columns if any.
    """
    if history.shape[0] == 0:
        return history.iloc[0:0].copy()

    h = _sorted_history(history)
    lids = history_lender_columns(h)
    out = h[[AllocationHistory.INDEX_COL, AllocationHistory.FICO_COL]].copy()
    sub = h[lids].astype(float)
    cum = sub.cumsum(axis=0)
    for c in lids:
        out[c] = cum[c].values
    return out


def remaining_after_history(
    initial_remaining_by_lender: dict[str, float],
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Remaining commitment after each historical loan (sorted by ``loan_index``).

    Starts from ``initial_remaining_by_lender``; each row subtracts that row's allocations.
    Lenders missing from ``initial_remaining_by_lender`` are treated as 0 initial remaining.
    """
    lids = history_lender_columns(history)
    init = {lid: float(initial_remaining_by_lender.get(lid, 0.0)) for lid in lids}

    if history.shape[0] == 0:
        if not lids:
            return pd.DataFrame()
        return pd.DataFrame([dict(init)], columns=lids)

    h = _sorted_history(history)
    out_rows: list[dict[str, float]] = []
    rem = dict(init)
    for _, row in h.iterrows():
        for lid in lids:
            rem[lid] -= float(row[lid])
        out_rows.append(dict(rem))
    return pd.DataFrame(out_rows, columns=lids)


def remaining_with_loan_index(
    initial_remaining_by_lender: dict[str, float],
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Like :func:`remaining_after_history` but prefixes ``loan_index`` and ``loan_fico`` from each row."""
    if history.shape[0] == 0:
        lids = history_lender_columns(history)
        return pd.DataFrame(columns=[AllocationHistory.INDEX_COL, AllocationHistory.FICO_COL, *lids])

    h = _sorted_history(history)
    rem_df = remaining_after_history(initial_remaining_by_lender, h)
    meta = h[[AllocationHistory.INDEX_COL, AllocationHistory.FICO_COL]].reset_index(drop=True)
    return pd.concat([meta, rem_df], axis=1)


def slice_history_window(history: pd.DataFrame, window: int | None) -> pd.DataFrame:
    """Last ``window`` loans by ``loan_index`` order; ``None`` means full history."""
    if window is not None and int(window) < 1:
        raise ValueError("window must be at least 1 when not None")
    if history.shape[0] == 0 or window is None:
        return history.copy()
    w = int(window)
    h = _sorted_history(history)
    return h.tail(w).reset_index(drop=True)


def gini_series_by_loan(history: pd.DataFrame) -> pd.Series:
    """Gini of the cross-lender amount vector for each loan row (sorted by ``loan_index``)."""
    if history.shape[0] == 0:
        return pd.Series(dtype=float)
    h = _sorted_history(history)
    lids = history_lender_columns(h)
    ginis: list[float] = []
    for _, row in h.iterrows():
        vec = np.array([float(row[lid]) for lid in lids], dtype=float)
        ginis.append(gini_coefficient(vec))
    return pd.Series(ginis, name="gini_amounts", index=h.index)


def aggregate_metrics_for_window(
    history: pd.DataFrame,
    *,
    window: int | None = None,
) -> dict[str, float]:
    """Summary stats over a tail window: mean Gini per loan, total funded per lender, total face.

    Convenience for dashboards; all monetary sums are in the same units as history amounts.
    """
    h = slice_history_window(history, window)
    if h.shape[0] == 0:
        return {
            "n_loans": 0.0,
            "mean_gini_amounts": float("nan"),
            "total_funded_all_lenders": 0.0,
        }
    lids = history_lender_columns(h)
    ginis = gini_series_by_loan(h)
    total_by_lid = h[lids].astype(float).sum(axis=0)
    return {
        "n_loans": float(h.shape[0]),
        "mean_gini_amounts": float(ginis.mean()),
        "total_funded_all_lenders": float(total_by_lid.sum()),
        **{f"total_funded_{lid}": float(total_by_lid[lid]) for lid in lids},
    }
