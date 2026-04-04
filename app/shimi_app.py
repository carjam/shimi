"""Shimi — main Streamlit entrypoint."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st

from shimi.allocation import (
    AllocationParams,
    PortfolioPrior,
    allocate_loan,
    per_lender_exhaustion_summary,
    run_exhaustion_simulation,
)
from shimi.data import (
    load_lender_program_from_csv,
    load_loan_tape_from_csv,
    load_portfolio_prior_from_csv,
    portfolio_prior_from_loan_tape,
    replay_allocation_history,
)
from shimi.data.history import AllocationHistory
from shimi.data.models import LenderProgram
from shimi.metrics import aggregate_metrics_for_window, cumulative_funded_by_lender, gini_series_by_loan

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LENDERS_CSV = DATA_DIR / "sample_lenders.csv"
LOANS_CSV = DATA_DIR / "sample_loans.csv"
HISTORY_CSV = DATA_DIR / "sample_allocation_history.csv"
PRIOR_CSV = DATA_DIR / "sample_portfolio_prior.csv"


def _lender_ids(program: LenderProgram) -> list[str]:
    return sorted(program.lenders.keys())


def _history_for_metrics(
    program: LenderProgram,
    loan_tape_df: pd.DataFrame | None,
    *,
    use_sample_tape: bool,
) -> tuple[pd.DataFrame | None, str]:
    """Return (history-like DataFrame, source tag) for ``shimi.metrics``; source is ``tape``, ``memory``, or reason."""
    ids = _lender_ids(program)
    if use_sample_tape:
        if loan_tape_df is None or loan_tape_df.empty:
            return None, "no_tape"
        req = [AllocationHistory.INDEX_COL, AllocationHistory.FICO_COL, *ids]
        missing = [c for c in req if c not in loan_tape_df.columns]
        if missing:
            return None, f"missing:{','.join(missing)}"
        return loan_tape_df[req].copy(), "tape"
    if program.history.shape[0] > 0:
        return program.history.copy(), "memory"
    return None, "empty"


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


def _lender_qualitative_colors() -> list[str]:
    base = list(pc.qualitative.Plotly)
    extra = getattr(pc.qualitative, "Dark24", ())
    return base + list(extra)


def _fig_remaining_trajectory_loans(
    trajectory: list[dict[str, float]],
    lender_ids: list[str],
    id_to_name: dict[str, str],
    *,
    height: int = 340,
) -> go.Figure | None:
    if not trajectory:
        return None
    colors = _lender_qualitative_colors()
    fig = go.Figure()
    for i, lid in enumerate(lender_ids):
        c = colors[i % len(colors)]
        y0 = float(trajectory[0][lid])
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[y0],
                mode="markers",
                name=f"{id_to_name[lid]} — now",
                legendgroup=lid,
                marker=dict(size=12, color=c, symbol="circle", line=dict(width=1, color="white")),
            )
        )
        if len(trajectory) > 1:
            # Include k=0 so the line has ≥2 points; Plotly draws nothing for mode="lines" with one point.
            xs = list(range(len(trajectory)))
            ys = [float(trajectory[k][lid]) for k in range(len(trajectory))]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=f"{id_to_name[lid]} — projected",
                    legendgroup=lid,
                    line=dict(color=c, dash="dash", width=2.5),
                    marker=dict(color=c, size=[0] + [5] * (len(xs) - 1)),
                    hovertemplate="%{y:.2f}<extra></extra>",
                )
            )
    fig.update_layout(
        title=dict(text="Remaining line: now vs projected (by loan #)", font=dict(size=14)),
        xaxis_title="Loan # after close (0 = current book)",
        yaxis_title="Remaining capital",
        height=height,
        margin=dict(t=56, b=120, l=64, r=24),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )
    return fig


def _fig_remaining_trajectory_dates(
    trajectory: list[dict[str, float]],
    lender_ids: list[str],
    id_to_name: dict[str, str],
    *,
    as_of: date,
    first_loan_date: date,
    interval_days: int,
    height: int = 340,
) -> go.Figure | None:
    if not trajectory:
        return None
    if interval_days < 1:
        interval_days = 1
    colors = _lender_qualitative_colors()
    fig = go.Figure()
    as_of_ts = pd.Timestamp(as_of)
    for i, lid in enumerate(lender_ids):
        c = colors[i % len(colors)]
        y0 = float(trajectory[0][lid])
        fig.add_trace(
            go.Scatter(
                x=[as_of_ts],
                y=[y0],
                mode="markers",
                name=f"{id_to_name[lid]} — now",
                legendgroup=lid,
                marker=dict(size=12, color=c, symbol="circle", line=dict(width=1, color="white")),
            )
        )
        if len(trajectory) > 1:
            xs = [as_of_ts] + [
                pd.Timestamp(first_loan_date + timedelta(days=interval_days * j))
                for j in range(len(trajectory) - 1)
            ]
            ys = [float(trajectory[0][lid])] + [
                float(trajectory[j + 1][lid]) for j in range(len(trajectory) - 1)
            ]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=f"{id_to_name[lid]} — projected",
                    legendgroup=lid,
                    line=dict(color=c, dash="dash", width=2.5),
                    marker=dict(color=c, size=[0] + [5] * (len(xs) - 1)),
                    hovertemplate="%{y:.2f}<extra></extra>",
                )
            )
    fig.update_layout(
        title=dict(text="Remaining line: now vs projected (calendar)", font=dict(size=14)),
        xaxis_title="Date",
        yaxis_title="Remaining capital",
        height=height,
        margin=dict(t=56, b=120, l=64, r=24),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(type="date")
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
        "dial parameters and see impact without scrolling. If `sample_allocation_history.csv` is present, past "
        "loans are **replayed** into the book so **remaining lines**, **History metrics**, and **exhaustion** align."
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
    if HISTORY_CSV.exists():
        try:
            hist_seed = load_loan_tape_from_csv(HISTORY_CSV)
            replay_allocation_history(program, hist_seed)
        except ValueError as e:
            st.warning(f"Could not replay `{HISTORY_CSV.name}`: {e}")

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

        with st.expander("History metrics", expanded=False):
            st.caption(
                "**Gini** measures concentration of each loan’s split across lenders (0 = even, higher = more skewed). "
                "**Cumulative funded** sums face from past rows. If `sample_allocation_history.csv` was replayed at "
                "startup, **in-memory history** already lists those loans and **remaining** lines are reduced accordingly. "
                "You can still use the loan-tape checkbox to analyze the tape as an alternate series."
            )
            use_tape_for_metrics = False
            if loan_tape_df is not None and not loan_tape_df.empty:
                use_tape_for_metrics = st.checkbox(
                    "Use sample loan tape as history (metrics only)",
                    value=False,
                    help="Treats rows in sample_loans.csv like past allocations; does not change the QP or exhaustion book state.",
                )
            hist_m, hist_src = _history_for_metrics(
                program, loan_tape_df, use_sample_tape=use_tape_for_metrics
            )
            if hist_m is None:
                if hist_src == "no_tape":
                    st.warning("No loan tape loaded; cannot use tape-based metrics.")
                elif hist_src.startswith("missing:"):
                    st.warning(f"Loan tape is missing columns for metrics: `{hist_src.split(':', 1)[1]}`.")
                else:
                    st.info(
                        "No rows to analyze. Enable **Use sample loan tape** above, or build "
                        "`program.history` via `apply_loan_allocation` in code."
                    )
            else:
                src_label = "**sample_loans.csv**" if hist_src == "tape" else "**in-memory history**"
                st.caption(f"Source: {src_label} · {hist_m.shape[0]} loan(s)")
                agg = aggregate_metrics_for_window(hist_m, window=None)
                gs = gini_series_by_loan(hist_m)
                last_gini = float(gs.iloc[-1]) if len(gs) > 0 else float("nan")
                mg1, mg2, mg3 = st.columns(3)
                mg1.metric("Loans in series", f"{int(agg['n_loans'])}")
                mg2.metric("Mean Gini (amounts)", f"{agg['mean_gini_amounts']:.3f}")
                mg3.metric("Gini (last loan)", f"{last_gini:.3f}")
                mg3.caption("Concentration on the latest row")
                st.metric("Total funded (series)", f"{agg['total_funded_all_lenders']:.2f}")
                cum = cumulative_funded_by_lender(hist_m)
                tail_n = min(8, len(cum))
                st.markdown("**Cumulative funded face** (tail of series)")
                st.dataframe(
                    cum.tail(tail_n),
                    column_config={
                        "loan_index": st.column_config.NumberColumn("Loan #"),
                        "loan_fico": st.column_config.NumberColumn("Loan FICO", format="%.0f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

        with st.expander("Capital exhaustion forecast", expanded=False):
            st.caption(
                "Projects when each lender’s **remaining line** hits zero if you book **repeated loans** of the "
                "same size with the **same** α, β, γ, floor, and loan FICO. The optimizer is **re-run each loan** "
                "(caps change). The portfolio **prior for γ is fixed** (not rolled forward with new volume)."
            )
            max_fc = st.number_input(
                "Max loans to simulate",
                min_value=50,
                max_value=50_000,
                value=2_000,
                step=50,
                help="Stops early if the next loan is infeasible or every line is depleted.",
            )
            exh_view = st.radio(
                "Exhaustion chart",
                ["Loan index", "Calendar"],
                horizontal=True,
                help="Loan index: x = loan count after each close. Calendar: map loans to dates (schedule below).",
            )
            book_as_of = date.today()
            first_close_date = date.today()
            days_between_closes = 30
            if exh_view == "Calendar":
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    book_as_of = st.date_input(
                        "Book as-of",
                        value=date.today(),
                        help="Date for **now** markers (current remaining line).",
                    )
                with dc2:
                    first_close_date = st.date_input(
                        "First modeled close",
                        value=date.today(),
                        help="Date when the first simulated loan is assumed to fund.",
                    )
                with dc3:
                    days_between_closes = int(
                        st.number_input(
                            "Days between closes",
                            min_value=1,
                            max_value=3650,
                            value=30,
                            step=1,
                            help="Spacing between successive simulated loan closes on the timeline.",
                        )
                    )

            sim_ex = run_exhaustion_simulation(
                program,
                loan_amt,
                params,
                loan_fico=loan_fico,
                portfolio_prior=portfolio_prior,
                max_loans=int(max_fc),
            )
            fcst = per_lender_exhaustion_summary(program, sim_ex)
            fc_meta = {
                "loans_simulated": sim_ex.loans_simulated,
                "stop_reason": sim_ex.stop_reason,
                "max_loans": sim_ex.max_loans,
            }
            disp = fcst.rename(
                columns={
                    "remaining_line": "Remaining",
                    "draw_loan_1": "Modeled $ (loan 1)",
                    "loans_until_depleted": "Loans to deplete (sim)",
                    "approx_loans_flat_draw": "Flat-draw loans (approx)",
                    "avg_hist_draw_per_loan": "Hist avg $/loan",
                    "approx_loans_at_hist_avg": "Loans @ hist avg (approx)",
                }
            )
            st.caption(
                f"Simulated **{fc_meta['loans_simulated']}** loans · stop: **`{fc_meta['stop_reason']}`** "
                f"· horizon cap: **{fc_meta['max_loans']}**."
            )
            show_hist = bool(disp["Hist avg $/loan"].notna().any())
            show_cols = [
                "name",
                "Remaining",
                "Modeled $ (loan 1)",
                "Loans to deplete (sim)",
                "Flat-draw loans (approx)",
            ]
            if show_hist:
                show_cols.extend(["Hist avg $/loan", "Loans @ hist avg (approx)"])
            st.dataframe(
                disp[show_cols],
                column_config={
                    "Remaining": st.column_config.NumberColumn(format="%.2f"),
                    "Modeled $ (loan 1)": st.column_config.NumberColumn(format="%.2f"),
                    "Loans to deplete (sim)": st.column_config.NumberColumn(format="%.0f"),
                    "Flat-draw loans (approx)": st.column_config.NumberColumn(format="%.1f"),
                    "Hist avg $/loan": st.column_config.NumberColumn(format="%.2f"),
                    "Loans @ hist avg (approx)": st.column_config.NumberColumn(format="%.1f"),
                },
                hide_index=True,
                use_container_width=True,
            )
            id_to_name = dict(zip(disp["lender_id"].astype(str), disp["name"].astype(str), strict=True))
            traj_ids = sorted(id_to_name.keys())
            if exh_view == "Loan index":
                tr_fig = _fig_remaining_trajectory_loans(sim_ex.trajectory, traj_ids, id_to_name, height=360)
            else:
                tr_fig = _fig_remaining_trajectory_dates(
                    sim_ex.trajectory,
                    traj_ids,
                    id_to_name,
                    as_of=book_as_of,
                    first_loan_date=first_close_date,
                    interval_days=days_between_closes,
                    height=360,
                )
            if tr_fig is not None:
                st.plotly_chart(tr_fig, use_container_width=True)
            if sim_ex.loans_simulated == 0:
                st.caption("Projection line starts after the first successful simulated loan (none run yet).")

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
