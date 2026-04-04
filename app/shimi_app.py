"""Shimi — main Streamlit entrypoint."""

from pathlib import Path

import streamlit as st

from shimi.data import load_lender_program_from_csv

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_lenders.csv"


def main() -> None:
    st.set_page_config(page_title="Shimi", layout="wide")
    st.title("Shimi")
    st.caption("Capital Concentration Decision Engine — 資本密度意思決定エンジン")
    st.write(
        "Interactive loan allocation simulator — lenders load from the data layer "
        "(see `shimi.data`)."
    )

    if DATA_PATH.exists():
        try:
            program = load_lender_program_from_csv(DATA_PATH)
        except ValueError as e:
            st.error(f"Invalid lender data: {e}")
            return
        st.subheader("Lender program")
        st.dataframe(program.to_dataframe(), use_container_width=True)
        st.caption(
            f"Allocation history rows: {program.history.shape[0]} · "
            "History widens with one column per lender for projections and drawdowns."
        )
    else:
        st.info(f"No data file at `{DATA_PATH}`. Add `data/sample_lenders.csv` to load the table.")


if __name__ == "__main__":
    main()
