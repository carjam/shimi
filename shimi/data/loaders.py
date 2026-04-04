from __future__ import annotations

from pathlib import Path

import pandas as pd

from shimi.data.models import LenderProgram, LenderState


def _normalize_column(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    lowered = s.astype(str).str.strip().str.lower()
    return lowered.isin(("1", "true", "t", "yes", "y"))


def load_lender_program_from_csv(path: str | Path) -> LenderProgram:
    """
    Load lenders from CSV into a :class:`LenderProgram`.

    Required: ``lender_id``, ``name``, and either ``total_commitment`` or legacy ``limit_musd``.

    Optional: ``remaining_commitment`` (defaults to total), ``target_share`` (defaults from
    commitment mix), ``is_contractual_originator`` (default False), ``avg_fico`` (default 700),
    ``region``, ``asset_class``, ``risk_tier``.
    """
    path = Path(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV has no rows: {path}")

    df.columns = [_normalize_column(c) for c in df.columns]

    if "limit_musd" in df.columns and "total_commitment" not in df.columns:
        df = df.rename(columns={"limit_musd": "total_commitment"})

    required = {"lender_id", "name", "total_commitment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns {sorted(missing)}; have {sorted(df.columns)}")

    df["lender_id"] = df["lender_id"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    totals = pd.to_numeric(df["total_commitment"], errors="coerce")
    if totals.isna().any() or (totals <= 0).any():
        raise ValueError("total_commitment must be positive for every row")

    if "remaining_commitment" in df.columns:
        rem = pd.to_numeric(df["remaining_commitment"], errors="coerce")
    else:
        rem = totals.copy()

    if rem.isna().any():
        raise ValueError("remaining_commitment invalid (NaN)")
    if (rem < 0).any():
        raise ValueError("remaining_commitment must be >= 0")
    if (rem > totals + 1e-9).any():
        raise ValueError("remaining_commitment cannot exceed total_commitment")

    pool = float(totals.sum())
    if pool <= 0:
        raise ValueError("Sum of total_commitment must be positive")

    if "target_share" in df.columns:
        shares = pd.to_numeric(df["target_share"], errors="coerce")
        if shares.isna().any() or (shares <= 0).any() or (shares > 1).any():
            raise ValueError("target_share must be in (0, 1] when provided")
    else:
        shares = totals / pool

    share_sum = float(shares.sum())
    if abs(share_sum - 1.0) > 0.02:
        raise ValueError(f"target_share values should sum to ~1; got {share_sum:.4f}")

    if "is_contractual_originator" in df.columns:
        co = _bool_series(df["is_contractual_originator"])
    else:
        co = pd.Series([False] * len(df))

    if "avg_fico" in df.columns:
        fico = pd.to_numeric(df["avg_fico"], errors="coerce")
        if fico.isna().any():
            raise ValueError("avg_fico must be numeric when provided")
    else:
        fico = pd.Series([700.0] * len(df))

    lenders: list[LenderState] = []
    for i in range(len(df)):
        lenders.append(
            LenderState(
                lender_id=str(df["lender_id"].iloc[i]),
                name=str(df["name"].iloc[i]),
                total_commitment=float(totals.iloc[i]),
                remaining_commitment=float(rem.iloc[i]),
                target_share=float(shares.iloc[i]),
                is_contractual_originator=bool(co.iloc[i]),
                avg_fico=float(fico.iloc[i]),
                region=_optional_str(df, "region", i),
                asset_class=_optional_str(df, "asset_class", i),
                risk_tier=_optional_str(df, "risk_tier", i),
            )
        )

    return LenderProgram.from_lenders(lenders)


def _optional_str(df: pd.DataFrame, col: str, i: int) -> str | None:
    if col not in df.columns:
        return None
    v = df[col].iloc[i]
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s or None
