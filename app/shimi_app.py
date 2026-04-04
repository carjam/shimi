"""Shimi — main Streamlit entrypoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shimi.allocation import AllocationParams, PortfolioPrior, allocate_loan
from shimi.data import (
    load_lender_program_from_csv,
    load_loan_tape_from_csv,
    load_portfolio_prior_from_csv,
    portfolio_prior_from_loan_tape,
)
from shimi.data.models import LenderProgram

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LENDERS_CSV = DATA_DIR / "sample_lenders.csv"
LOANS_CSV = DATA_DIR / "sample_loans.csv"
PRIOR_CSV = DATA_DIR / "sample_portfolio_prior.csv"


def _lender_ids(program: LenderProgram) -> list[str]:
    return sorted(program.lenders.keys())


def _build_output_table(
    program: LenderProgram,
    result,
    loan_amt: float,
) -> pd.DataFrame:
    df = program.to_dataframe()[["lender_id", "name", "target_share", "is_contractual_originator"]].copy()
    df["suggested_share"] = df["lender_id"].map(result.shares)
    df["amount"] = df["lender_id"].map(result.amounts_by_lender)
    df["share_delta_pp"] = (df["suggested_share"] - df["target_share"]) * 100.0
    rem = df["lender_id"].map(lambda lid: program.lenders[lid].remaining_commitment)
    df["pct_of_remaining"] = np.where(rem > 0, (df["amount"] / rem) * 100.0, np.nan)
    df["fico_weighted_on_loan"] = df["amount"] * float(result.loan_fico)
    return df


def _fig_target_vs_suggested(
    names: list[str],
    target: np.ndarray,
    suggested: np.ndarray,
    *,
    height: int = 380,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Target share", x=names, y=target, marker_color="#94a3b8", marker_line_width=0)
    )
    fig.add_trace(
        go.Bar(
            name="Model share",
            x=names,
            y=suggested,
            marker_color="#2563eb",
            marker_line_width=0,
        )
    )
    fig.update_layout(
        barmode="group",
        title=dict(text="Target vs model share", font=dict(size=13)),
        yaxis=dict(title="Share", tickformat=".1%", range=[0, max(0.15, float(np.max(np.concatenate([target, suggested]))) * 1.15)]),
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        margin=dict(t=48, b=36, l=48, r=16),
        height=height,
    )
    return fig


def _fig_amounts(names: list[str], amounts: np.ndarray, *, height: int = 340) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=names,
            y=amounts,
            marker_color="#059669",
            marker_line_width=0,
            text=[f"{a:.2f}" for a in amounts],
            textposition="outside",
            textfont=dict(size=10),
        )
    )
    fig.update_layout(
        title=dict(text="Dollar allocation (this loan)", font=dict(size=13)),
        yaxis_title="Amount",
        margin=dict(t=48, b=36, l=48, r=16),
        height=height,
    )
    return fig


def _post_loan_portfolio_avg(
    portfolio_prior: PortfolioPrior | None,
    amounts_by_lender: dict[str, float],
    loan_fico: float,
    lender_ids: list[str],
) -> pd.DataFrame | None:
    if portfolio_prior is None:
        return None
    rows = []
    for lid in lender_ids:
        a0 = float(portfolio_prior.funded_face_by_lender.get(lid, 0.0))
        f0 = float(portfolio_prior.fico_weighted_face_by_lender.get(lid, 0.0))
        x = float(amounts_by_lender.get(lid, 0.0))
        a1 = a0 + x
        f1 = f0 + x * loan_fico
        rows.append(
            {
                "lender_id": lid,
                "avg_fico_after": f1 / a1 if a1 > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Shimi", layout="wide")
    st.title("Shimi")
    st.caption("Capital Concentration Decision Engine — 資本密度意思決定エンジン")
    st.write(
        "Interactive loan allocation simulator — lender book, optional **loan tape**, and **portfolio "
        "priors** load from `data/` (see `data/README.md`). Open **View source data** for CSV snapshots. "
        "**Inputs** are on the **left**; **live output** (metrics and charts) stays on the **right** so you can "
        "dial parameters and see impact without scrolling."
    )

    if not LENDERS_CSV.exists():
        st.info(f"No lender file at `{LENDERS_CSV}`. Add `data/sample_lenders.csv` to load the app.")
        return

    try:
        program = load_lender_program_from_csv(LENDERS_CSV)
    except ValueError as e:
        st.error(f"Invalid lender data: {e}")
        return

    ids = _lender_ids(program)

    loan_tape_df: pd.DataFrame | None = None
    with st.expander("View source data (lenders & loan tape)", expanded=False):
        st.subheader("Lender program (`sample_lenders.csv`)")
        st.dataframe(program.to_dataframe(), use_container_width=True)
        st.caption(
            f"In-memory allocation history rows: {program.history.shape[0]} · "
            "Columns include `loan_fico` when you apply allocations from code or future UI actions."
        )

        st.subheader("Sample loan tape (`sample_loans.csv`)")
        if LOANS_CSV.exists():
            try:
                loan_tape_df = load_loan_tape_from_csv(LOANS_CSV)
            except ValueError as e:
                st.warning(f"Could not load loan tape: {e}")
                loan_tape_df = None
            if loan_tape_df is not None:
                st.dataframe(loan_tape_df, use_container_width=True)
                st.caption(
                    "Each row is one loan: **`loan_fico`** (single score for that loan) and allocated **face** "
                    f"per lender ({', '.join(ids)}). Rows with `loan_index` 0–2 roll up to `sample_portfolio_prior.csv`."
                )
        else:
            st.info(f"No `{LOANS_CSV.name}` found next to the lender file.")
            loan_tape_df = None

    st.subheader("Simulation workspace")
    st.caption("Wide layout: adjust sliders in the left column while metrics and charts update on the right.")

    inp_col, out_col = st.columns([0.34, 0.66], gap="large")

    with inp_col:
        st.markdown("##### Inputs")
        total_rem = sum(l.remaining_commitment for l in program.lenders.values())
        default_loan = min(10.0, max(1.0, 0.05 * total_rem))
        loan_amt = st.slider(
            "Loan amount",
            min_value=0.0,
            max_value=float(total_rem),
            value=float(default_loan),
            step=0.5,
            help="Total face to allocate; must not exceed aggregate remaining commitment.",
        )
        floor = st.slider("Participation floor (share)", 0.0, 0.2, 0.05, 0.01)
        alpha = st.slider("α — target share fit", 0.0, 20.0, 1.0, 0.1)
        beta = st.slider("β — contractual utilization", 0.0, 20.0, 0.25, 0.05)
        gamma_fico = st.slider(
            "γ — portfolio FICO fair dealing",
            0.0,
            10.0,
            0.0,
            0.1,
            help="With a portfolio prior: rebalances cross-lender portfolio avg FICO. Without: equal-share proxy.",
        )
        f_mean = float(program.to_dataframe()["avg_fico"].mean())
        loan_fico = st.number_input(
            "Loan FICO",
            min_value=300.0,
            max_value=850.0,
            value=f_mean,
            step=1.0,
        )

        portfolio_prior: PortfolioPrior | None = None
        prior_source = "none"
        with st.expander("Portfolio prior (γ)", expanded=False):
            options = ["None (cold start)"]
            if PRIOR_CSV.exists():
                options.append("Load sample_portfolio_prior.csv")
            if loan_tape_df is not None:
                options.append("Build from sample_loans.csv (loan_index ≤ 2)")
            options.append("Manual table")

            default_ix = 1 if PRIOR_CSV.exists() else 0
            choice = st.radio(
                "Prior source",
                options,
                index=min(default_ix, len(options) - 1),
                horizontal=False,
            )

            if choice == "Load sample_portfolio_prior.csv":
                try:
                    portfolio_prior = load_portfolio_prior_from_csv(PRIOR_CSV)
                    prior_source = "csv"
                except ValueError as e:
                    st.error(str(e))
            elif choice == "Build from sample_loans.csv (loan_index ≤ 2)":
                try:
                    portfolio_prior = portfolio_prior_from_loan_tape(
                        loan_tape_df, ids, max_loan_index_inclusive=2
                    )
                    prior_source = "tape"
                except ValueError as e:
                    st.error(str(e))
            elif choice == "Manual table":
                ed_base = program.to_dataframe()[["lender_id"]].copy()
                ed_base["prior_funded"] = 0.0
                ed_base["prior_fico_weighted"] = 0.0
                edited = st.data_editor(
                    ed_base,
                    disabled=["lender_id"],
                    hide_index=True,
                    use_container_width=True,
                )
                if float(edited["prior_funded"].sum()) > 0:
                    portfolio_prior = PortfolioPrior(
                        funded_face_by_lender=dict(
                            zip(edited["lender_id"], edited["prior_funded"].astype(float), strict=True)
                        ),
                        fico_weighted_face_by_lender=dict(
                            zip(
                                edited["lender_id"],
                                edited["prior_fico_weighted"].astype(float),
                                strict=True,
                            )
                        ),
                    )
                    prior_source = "manual"

            if portfolio_prior is not None:
                prior_view = program.to_dataframe()[["lender_id", "name"]].copy()
                prior_view["prior_funded"] = prior_view["lender_id"].map(portfolio_prior.funded_face_by_lender)
                prior_view["prior_fico_weighted"] = prior_view["lender_id"].map(
                    portfolio_prior.fico_weighted_face_by_lender
                )
                pf = prior_view["prior_funded"].astype(float).to_numpy()
                pw = prior_view["prior_fico_weighted"].astype(float).to_numpy()
                prior_view["portfolio_avg_fico"] = np.where(pf > 0, pw / pf, np.nan)
                st.dataframe(
                    prior_view,
                    column_config={
                        "lender_id": st.column_config.TextColumn("Lender ID"),
                        "name": st.column_config.TextColumn("Name"),
                        "prior_funded": st.column_config.NumberColumn("Prior funded", format="%.2f"),
                        "prior_fico_weighted": st.column_config.NumberColumn("Σ(face×FICO)", format="%.1f"),
                        "portfolio_avg_fico": st.column_config.NumberColumn("Portfolio avg FICO", format="%.1f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(f"Active prior source: **{prior_source}** · Increase γ to emphasize portfolio fair-dealing.")

    _chart_h = 268

    with out_col:
        st.markdown("##### Live output")

        if loan_amt <= 0:
            st.info("Set **Loan amount** above zero to run the optimizer and see suggested shares and charts.")
            return

        try:
            preview = program.clone()
            params = AllocationParams(
                alpha=alpha,
                beta=beta,
                gamma_fico=gamma_fico,
                participation_floor=floor,
            )
            result = allocate_loan(
                preview,
                loan_amt,
                params,
                loan_fico=loan_fico,
                portfolio_prior=portfolio_prior,
            )
        except ValueError as e:
            st.error(f"**Infeasible or invalid:** {e}")
            st.caption("Try a smaller loan, lower participation floor, or check remaining commitments.")
            return

        out = _build_output_table(program, result, loan_amt)
        names = out["name"].tolist()
        target_sh = out["target_share"].astype(float).to_numpy()
        model_sh = out["suggested_share"].astype(float).to_numpy()
        amounts = out["amount"].astype(float).to_numpy()

        total_alloc = float(amounts.sum())
        mae_vs_target = float(np.mean(np.abs(model_sh - target_sh)))
        rmse_vs_target = float(np.sqrt(np.mean((model_sh - target_sh) ** 2)))

        if result.is_optimal:
            st.success("Solver finished with an **optimal** (or optimal-inaccurate) QP solution.")
        else:
            st.warning(f"Solver status: **{result.solver_status}** — review the numbers before relying on them.")

        r1a, r1b, r1c = st.columns(3)
        r1a.metric("Total allocated", f"{total_alloc:.2f}")
        r1a.caption(f"Loan: {loan_amt:.2f}")
        if abs(total_alloc - loan_amt) > 1e-3:
            st.warning("Allocated total differs from the loan amount — check solver / tolerances.")
        r1b.metric(
            "Mean |share − target|",
            f"{mae_vs_target:.3f}",
            help="Lower means closer to commitment mix (α effect).",
        )
        r1c.metric("RMSE (share vs target)", f"{rmse_vs_target:.3f}")
        r2a, r2b = st.columns(2)
        obj = result.objective_value
        r2a.metric("Objective value", f"{obj:.4g}" if obj is not None else "n/a")
        r2b.metric("γ FICO mode", result.fico_fairness_mode)
        r2b.caption(f"Loan FICO **{result.loan_fico:.0f}**")

        st.markdown("**Charts**")
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(
                _fig_target_vs_suggested(names, target_sh, model_sh, height=_chart_h),
                use_container_width=True,
            )
        with ch2:
            st.plotly_chart(_fig_amounts(names, amounts, height=_chart_h), use_container_width=True)

        st.markdown("**Detail table**")
        show = out[
            [
                "lender_id",
                "name",
                "is_contractual_originator",
                "target_share",
                "suggested_share",
                "share_delta_pp",
                "amount",
                "pct_of_remaining",
                "fico_weighted_on_loan",
            ]
        ].copy()
        show.rename(
            columns={
                "is_contractual_originator": "contractual",
                "target_share": "target_share",
                "suggested_share": "model_share",
                "share_delta_pp": "Δ share (pp)",
                "pct_of_remaining": "% of remaining line",
                "fico_weighted_on_loan": "face×FICO (this loan)",
            },
            inplace=True,
        )
        st.dataframe(
            show,
            column_config={
                "target_share": st.column_config.NumberColumn("Target share", format="%.2f"),
                "model_share": st.column_config.NumberColumn("Model share", format="%.2f"),
                "Δ share (pp)": st.column_config.NumberColumn(format="%.2f"),
                "amount": st.column_config.NumberColumn(format="%.2f"),
                "% of remaining line": st.column_config.NumberColumn(format="%.1f"),
                "face×FICO (this loan)": st.column_config.NumberColumn(format="%.0f"),
            },
            hide_index=True,
            use_container_width=True,
        )

        post = _post_loan_portfolio_avg(portfolio_prior, result.amounts_by_lender, result.loan_fico, ids)
        if post is not None and post["avg_fico_after"].notna().any():
            st.markdown("**After this loan** (portfolio avg FICO)")
            merged = show[["name", "lender_id"]].merge(post, on="lender_id")
            merged = merged[["name", "avg_fico_after"]]
            spread = float(merged["avg_fico_after"].max() - merged["avg_fico_after"].min())
            st.dataframe(
                merged,
                column_config={
                    "avg_fico_after": st.column_config.NumberColumn("Portfolio avg FICO", format="%.1f"),
                },
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                f"Cross-lender spread of portfolio avg FICO after this allocation: **{spread:.1f}** points "
                "(γ pushes this down over time when priors are loaded)."
            )

        with st.expander("How to read this output"):
            st.markdown(
                """
- **Target share** comes from the lender book (commitment mix). **Model share** is what the QP chose.
- **Δ share (pp)** is model minus target, in **percentage points** (e.g. +2 means two points more of the loan).
- **% of remaining line** is how much of each lender’s *current* remaining commitment this loan would use (relevant for **β** on contractual originators).
- **face×FICO (this loan)** is amount × this loan’s FICO (same FICO for everyone on this loan; useful for fair-dealing intuition).
- Raise **α** to hug targets; **β** to ease pulls on contractual lenders’ lines; **γ** (with a prior) to tighten portfolio FICO balance across lenders.
"""
            )


if __name__ == "__main__":
    main()
