from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shimi.data.history import AllocationHistory
from shimi.metrics import (
    aggregate_metrics_for_window,
    cumulative_funded_by_lender,
    gini_coefficient,
    gini_of_loan_split,
    gini_series_by_loan,
    per_lender_fico_weighted_face,
    remaining_after_history,
    remaining_with_loan_index,
    slice_history_window,
    total_fico_weighted_face,
)


def test_gini_equal_splits_is_zero() -> None:
    assert gini_coefficient([2.5, 2.5, 2.5, 2.5]) == pytest.approx(0.0, abs=1e-9)


def test_gini_all_zeros_is_zero() -> None:
    assert gini_coefficient([0.0, 0.0, 0.0]) == 0.0


def test_gini_one_holds_all_is_max_inequality() -> None:
    g = gini_coefficient([1.0, 0.0, 0.0, 0.0])
    assert g == pytest.approx(0.75, abs=1e-9)


def test_gini_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        gini_coefficient([])


def test_gini_negative_raises() -> None:
    with pytest.raises(ValueError, match="negative"):
        gini_coefficient([1.0, -0.1])


def test_gini_of_loan_split_order() -> None:
    d = {"A": 10.0, "B": 10.0}
    assert gini_of_loan_split(d, lender_ids=["A", "B"]) == pytest.approx(0.0, abs=1e-9)


def test_fico_weighted_face() -> None:
    d = {"A": 2.0, "B": 3.0}
    assert per_lender_fico_weighted_face(d, 700.0) == {"A": 1400.0, "B": 2100.0}
    assert total_fico_weighted_face(d, 700.0) == pytest.approx(3500.0)


def test_cumulative_funded_by_lender() -> None:
    h = AllocationHistory.empty(["X", "Y"])
    h = AllocationHistory.append_row(h, loan_index=0, amounts_by_lender={"X": 1.0, "Y": 3.0}, loan_fico=700.0)
    h = AllocationHistory.append_row(h, loan_index=1, amounts_by_lender={"X": 2.0, "Y": 1.0}, loan_fico=710.0)
    cum = cumulative_funded_by_lender(h)
    assert float(cum.loc[0, "X"]) == pytest.approx(1.0)
    assert float(cum.loc[0, "Y"]) == pytest.approx(3.0)
    assert float(cum.loc[1, "X"]) == pytest.approx(3.0)
    assert float(cum.loc[1, "Y"]) == pytest.approx(4.0)


def test_remaining_after_history() -> None:
    h = AllocationHistory.empty(["X", "Y"])
    h = AllocationHistory.append_row(h, loan_index=0, amounts_by_lender={"X": 1.0, "Y": 2.0}, loan_fico=700.0)
    h = AllocationHistory.append_row(h, loan_index=1, amounts_by_lender={"X": 1.0, "Y": 0.0}, loan_fico=700.0)
    rem = remaining_after_history({"X": 10.0, "Y": 10.0}, h)
    assert float(rem.loc[0, "X"]) == pytest.approx(9.0)
    assert float(rem.loc[0, "Y"]) == pytest.approx(8.0)
    assert float(rem.loc[1, "X"]) == pytest.approx(8.0)
    assert float(rem.loc[1, "Y"]) == pytest.approx(8.0)


def test_remaining_does_not_mutate_input_dict() -> None:
    h = AllocationHistory.empty(["X"])
    h = AllocationHistory.append_row(h, loan_index=0, amounts_by_lender={"X": 5.0}, loan_fico=700.0)
    init = {"X": 100.0}
    remaining_after_history(init, h)
    assert init["X"] == 100.0


def test_slice_history_window() -> None:
    h = AllocationHistory.empty(["X"])
    for i, a in enumerate([1.0, 2.0, 3.0]):
        h = AllocationHistory.append_row(h, loan_index=i, amounts_by_lender={"X": a}, loan_fico=700.0)
    tail = slice_history_window(h, 2)
    assert tail.shape[0] == 2
    assert list(tail["loan_index"]) == [1, 2]


def test_slice_history_window_invalid() -> None:
    h = AllocationHistory.empty(["X"])
    with pytest.raises(ValueError, match="window"):
        slice_history_window(h, 0)


def test_gini_series_by_loan() -> None:
    h = AllocationHistory.empty(["X", "Y"])
    h = AllocationHistory.append_row(h, loan_index=0, amounts_by_lender={"X": 5.0, "Y": 5.0}, loan_fico=700.0)
    h = AllocationHistory.append_row(h, loan_index=1, amounts_by_lender={"X": 9.0, "Y": 1.0}, loan_fico=700.0)
    s = gini_series_by_loan(h)
    assert s.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert s.iloc[1] > 0.3


def test_aggregate_metrics_for_window() -> None:
    h = AllocationHistory.empty(["X", "Y"])
    h = AllocationHistory.append_row(h, loan_index=0, amounts_by_lender={"X": 1.0, "Y": 1.0}, loan_fico=700.0)
    h = AllocationHistory.append_row(h, loan_index=1, amounts_by_lender={"X": 3.0, "Y": 1.0}, loan_fico=700.0)
    agg = aggregate_metrics_for_window(h, window=None)
    assert agg["n_loans"] == 2.0
    assert agg["total_funded_all_lenders"] == pytest.approx(6.0)
    assert "mean_gini_amounts" in agg


def test_cumulative_respects_loan_index_order() -> None:
    """Rows out of order in storage should still cumulate by loan_index."""
    rows = [
        {"loan_index": 1, "loan_fico": 700.0, "X": 1.0},
        {"loan_index": 0, "loan_fico": 700.0, "X": 10.0},
    ]
    h = pd.DataFrame(rows)
    cum = cumulative_funded_by_lender(h)
    assert list(cum["loan_index"]) == [0, 1]
    assert float(cum.loc[cum["loan_index"] == 0].iloc[0]["X"]) == pytest.approx(10.0)
    assert float(cum.loc[cum["loan_index"] == 1].iloc[0]["X"]) == pytest.approx(11.0)


def test_remaining_with_loan_index_columns() -> None:
    h = AllocationHistory.empty(["X"])
    h = AllocationHistory.append_row(h, loan_index=0, amounts_by_lender={"X": 1.0}, loan_fico=705.0)
    out = remaining_with_loan_index({"X": 5.0}, h)
    assert list(out.columns[:3]) == ["loan_index", "loan_fico", "X"]
    assert float(out["loan_fico"].iloc[0]) == 705.0
