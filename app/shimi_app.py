"""Shimi — main Streamlit entrypoint."""

from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

st.title("Shimi: Interactive Loan Allocation Simulator")
st.write("Welcome to Shimi! Use the sliders and controls to simulate allocations, drawdowns, and risk metrics.")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_lenders.csv"


def main() -> None:
    st.set_page_config(page_title="Shimi", layout="wide")
    st.title("Shimi")
    st.caption("Capital Concentration Decision Engine — 資本密度意思決定エンジン")

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        st.subheader("Sample lenders")
        st.dataframe(df, use_container_width=True)
    else:
        st.info(f"No data file at `{DATA_PATH}`. Add `data/sample_lenders.csv` to load the table.")


if __name__ == "__main__":
    main()
