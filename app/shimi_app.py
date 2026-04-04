"""Shimi — main Streamlit entrypoint."""

from pathlib import Path

import numpy as np
import streamlit as st

from shimi.allocation import AllocationParams, PortfolioPrior, allocate_loan
from shimi.data import (
    load_lender_program_from_csv,
    load_loan_tape_from_csv,
    load_portfolio_prior_from_csv,
    portfolio_prior_from_loan_tape,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LENDERS_CSV = DATA_DIR / "sample_lenders.csv"
LOANS_CSV = DATA_DIR / "sample_loans.csv"
PRIOR_CSV = DATA_DIR / "sample_portfolio_prior.csv"


def _lender_ids(program) -> list[str]:
    return sorted(program.lenders.keys())


def main() -> None:
    st.set_page_config(page_title="Shimi", layout="wide")
    st.title("Shimi")
    st.caption("Capital Concentration Decision Engine — 資本密度意思決定エンジン")
    st.write(
        "Interactive loan allocation simulator — lender book, optional **loan tape**, and **portfolio "
        "priors** load from `data/` (see `data/README.md`)."
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

    st.subheader("Lender program (`sample_lenders.csv`)")
    st.dataframe(program.to_dataframe(), use_container_width=True)
    st.caption(
        f"In-memory allocation history rows: {program.history.shape[0]} · "
        "Columns include `loan_fico` when you apply allocations from code or future UI actions."
    )

    # --- Sample loan tape (reflects one FICO per loan) ---
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

    st.subheader("Allocation engine (QP preview)")
    total_rem = sum(l.remaining_commitment for l in program.lenders.values())
    default_loan = min(10.0, max(1.0, 0.05 * total_rem))
    loan_amt = st.slider(
        "Loan amount (next loan to allocate)",
        min_value=0.0,
        max_value=float(total_rem),
        value=float(default_loan),
        step=0.5,
        help="Total to allocate across lenders; capped by aggregate remaining commitment.",
    )
    alpha = st.slider("α — target share fit", 0.0, 20.0, 1.0, 0.1)
    beta = st.slider("β — contractual utilization penalty", 0.0, 20.0, 0.25, 0.05)
    gamma_fico = st.slider(
        "γ — fair dealing (portfolio FICO over time)",
        0.0,
        10.0,
        0.0,
        0.1,
        help="With a portfolio prior below: steers split using cumulative funded & Σ(face×FICO). "
        "Without prior: cold-start equal-share proxy on this loan only.",
    )
    floor = st.slider("Participation floor (share)", 0.0, 0.2, 0.05, 0.01)
    f_mean = float(program.to_dataframe()["avg_fico"].mean())
    loan_fico = st.number_input(
        "Loan FICO (this loan)",
        min_value=300.0,
        max_value=850.0,
        value=f_mean,
        step=1.0,
        help="One representative score for **this** loan only (the next allocation in the preview).",
    )

    # --- Portfolio prior (γ over time) ---
    portfolio_prior: PortfolioPrior | None = None
    prior_source = "none"
    with st.expander("Portfolio prior for γ (`sample_portfolio_prior.csv` or derived from tape)", expanded=True):
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
            horizontal=True,
            help="Cumulative stats **before** the loan above. Sample files live under `data/`.",
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

    if loan_amt > 0:
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
            st.warning(str(e))
        else:
            st.markdown("**Suggested allocation (this loan)**")
            out = program.to_dataframe()[["lender_id", "name"]].copy()
            out["share"] = out["lender_id"].map(result.shares)
            out["amount"] = out["lender_id"].map(result.amounts_by_lender)
            st.dataframe(out, use_container_width=True)
            obj = result.objective_value
            obj_txt = f"{obj:.6g}" if obj is not None else "n/a"
            st.caption(
                f"Solver: {result.solver_status} · objective={obj_txt} · optimal={result.is_optimal} · "
                f"loan FICO={result.loan_fico:.0f} · γ mode={result.fico_fairness_mode}"
            )


if __name__ == "__main__":
    main()
