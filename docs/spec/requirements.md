# Shimi: Interactive Loan Allocation Simulator

## Problem
Financial institutions and program managers need a way to allocate loans across multiple lenders while balancing risk, maintaining fair exposure, and protecting contractual originators.  
Current manual or spreadsheet-based approaches are slow, opaque, and prone to over-allocating certain lenders, risking program continuity.  
Shimi addresses this with a **real-time interactive simulator** that models allocations, metrics, and projections to support decision-making and stakeholder understanding.

---

## Users / Stakeholders
- **Risk Team**: Needs to monitor concentration risk, Gini coefficient, and weighted FICO exposure.
- **Capital Markets / Treasury**: Monitors lender commitments, ensures contractual originators are protected, and uses projected exhaustion metrics for planning.
- **Program Managers / Underwriters**: Explore allocations dynamically and adjust α, β, γ, participation floor, and loan-level FICO to test trade-offs (including fair dealing vs. commitment targets).
- **Architecture / Engineering Team**: Needs clear understanding of data flow, constraints, and how projections interact with the allocation engine.
- **Executive Stakeholders**: Visualize program health and risk distribution.

---

## In Scope
- Interactive web-based tool (Streamlit) for loan allocation simulation.
- Allocation engine solving per-loan Quadratic Program (QP) with:
  - Objective: weighted combination of:
    - **α** — deviation from **target share** (commitment-aligned risk distribution).
    - **β** — penalty on **contractual originators** for high utilization of remaining line.
    - **γ (fair dealing, FICO)** — **Each loan has one representative FICO.** Fairness is judged **in aggregate over time**: we want each lender’s **portfolio weighted-average FICO** (Σ face×FICO / Σ face on loans they took) to stay **roughly equal across lenders** as new loans arrive (scores may vary loan-to-loan, e.g. approximately Gaussian). With **cumulative priors** per lender (prior funded face and prior Σ(face×loan FICO)), γ uses a convex penalty that steers the **current** loan’s split toward **rebalancing** those portfolio averages—not favoritism against any counterparty. **Cold start** (no funded history): γ falls back to equal-share nudging on the current loan.
  - Inputs: loan amount, **loan FICO** for the **current** loan only; lender book; optional **portfolio prior** (cumulative funded and cumulative FICO-weighted face per lender before this loan).
  - Constraints: allocations sum to loan, each ≥ floor, each ≤ remaining commitment.
- Metrics and visualizations:
  - Per-loan allocations (stacked bar chart)
  - Cumulative allocations / drawdown curves
  - Gini coefficient per loan
  - Weighted FICO per loan
- Dynamic controls for α, β, γ, participation floor, **loan FICO**, and (where supported) **optional cumulative portfolio** inputs per lender for γ’s over-time fair-dealing term.
- Capital exhaustion projections:
  - Moving average and linear extrapolation methods
  - Alerts for critical lenders approaching depletion
- Rolling window computations for cumulative metrics.

---

## Out of Scope
- Live allocation enforcement in production systems (Shimi is a **simulation / POC**).
- Machine learning to automatically tune allocations per loan (meta-level α/β tuning only).
- Multi-currency or cross-jurisdiction compliance rules.
- Integration with external treasury systems (optional future enhancement).

---

## Functional Requirements
1. **Loan Allocation Engine**
   - Input: loan amount; **one loan FICO** per allocation (the score for that loan only); lender commitments, remaining capacity, target shares, contractual-originator flags; optional **portfolio prior** (per lender: cumulative funded face before this loan, cumulative Σ(face×loan FICO) on prior loans); optional historical allocation data for metrics/projections.
   - Output: allocation per lender satisfying floor and remaining commitment constraints; solver status; loan FICO used; indicator of whether γ applied **portfolio** fair dealing or **cold-start** equal-share fallback.
   - Optimization: CVXPY + OSQP. **γ** minimizes imbalance in lenders’ **cumulative FICO-mass** relative to a **group** portfolio average when priors exist; otherwise **γ** uses an equal-share proxy on the current loan only.

2. **Metrics Calculation**
   - Compute per-loan and cumulative metrics:
     - Gini coefficient
     - Weighted FICO contribution
     - Cumulative allocation / drawdown
   - Display results dynamically.

3. **Visualization**
   - Stacked bar charts for per-loan allocations.
   - Line charts for cumulative allocations, Gini, weighted FICO.
   - Overlay projected exhaustion points on drawdown curves.

4. **Parameter Tuning**
   - Controls for α, β, γ (portfolio FICO fair dealing over time when priors are supplied), participation floor, loan FICO, and optional per-lender cumulative portfolio fields.
   - Update allocations (and metrics when implemented) in real-time as parameters change.

5. **Capital Exhaustion Projections**
   - Calculate estimated loan number when each lender’s commitment reaches zero.
   - Methods: moving average, linear extrapolation.
   - Highlight contractual originators separately.

6. **Interactivity**
   - Update all visualizations and metrics immediately when parameters change.
   - Include alerts for lenders approaching critical thresholds.

---

## Non-functional Requirements
- **Performance**: Allocation engine must run per-loan in milliseconds for interactive use.
- **Extensibility**: Support addition of lenders, loan products, rolling window adjustments.
- **Auditability**: Each allocation must be deterministic, reproducible, and explainable.
- **Portability**: Runs locally or in cloud environment using Python + Streamlit.
- **Usability**: Visualizations and controls must be intuitive for non-technical stakeholders.

---

## Acceptance Criteria
1. **Allocation Constraints**
   - All allocations sum to loan amount.
   - Each allocation ≥ floor (configurable, e.g., 5%).
   - Each allocation ≤ lender’s remaining commitment.

2. **Metrics Accuracy**
   - Gini coefficient, weighted FICO, and cumulative allocations match expected calculations.

3. **Visualization Functionality**
   - Stacked bars, line charts, and exhaustion projections render correctly.
   - Alerts for critical lenders appear when projected exhaustion is near.

4. **Parameter Tuning**
   - α, β, γ, participation floor, loan FICO, and optional portfolio-prior inputs update the allocation preview in real time without errors (and metrics/charts when those layers are present).

5. **Projection Accuracy**
   - Moving average and linear extrapolation projections are consistent with historical allocation data.

---

## Open Questions
1. Should the tool support multiple loan products simultaneously, or focus on one at a time?  
2. How should extreme edge cases be handled if a contractual originator runs out of capital mid-cycle?  
3. What rolling window size should be default for metrics (e.g., last 90 loans, last 3 months)?  
4. Should projections consider stochastic variation (Monte Carlo) or strictly deterministic moving average/linear trend?  
5. Is there a preferred method for persisting lender state across sessions (CSV, SQLite, or database)?  