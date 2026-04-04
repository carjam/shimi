# Shimi: Interactive Loan Allocation Simulator

## Problem
Financial institutions and program managers need a way to allocate loans across multiple lenders while balancing risk, maintaining fair exposure, and protecting contractual originators.  
Current manual or spreadsheet-based approaches are slow, opaque, and prone to over-allocating certain lenders, risking program continuity.  
Shimi addresses this with a **real-time interactive simulator** that models allocations, metrics, and projections to support decision-making and stakeholder understanding.

---

## Users / Stakeholders
- **Risk Team**: Needs to monitor concentration risk, Gini coefficient, and weighted FICO exposure.
- **Capital Markets / Treasury**: Monitors lender commitments, ensures contractual originators are protected, and uses projected exhaustion metrics for planning.
- **Program Managers / Underwriters**: Explore allocations dynamically and adjust α/β weights to test trade-offs.
- **Architecture / Engineering Team**: Needs clear understanding of data flow, constraints, and how projections interact with the allocation engine.
- **Executive Stakeholders**: Visualize program health and risk distribution.

---

## In Scope
- Interactive web-based tool (Streamlit) for loan allocation simulation.
- Allocation engine solving per-loan Quadratic Program (QP) with:
  - Objective: weighted combination of deviation from target share, contractual originator protection, and weighted FICO.
  - Constraints: allocations sum to loan, each ≥ floor, each ≤ remaining commitment.
- Metrics and visualizations:
  - Per-loan allocations (stacked bar chart)
  - Cumulative allocations / drawdown curves
  - Gini coefficient per loan
  - Weighted FICO per loan
- Dynamic α/β sliders to explore risk vs. contractual originator protection trade-offs.
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
   - Input: loan amount, lender commitments, historical allocation data.
   - Output: allocation per lender satisfying floor and remaining commitment constraints.
   - Optimization: quadratic program solved using CVXPY + OSQP.

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
   - Sliders for α (risk distribution) and β (contractual originator protection).
   - Update allocations and metrics in real-time.

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
   - α/β sliders update metrics and charts in real-time without errors.

5. **Projection Accuracy**
   - Moving average and linear extrapolation projections are consistent with historical allocation data.

---

## Open Questions
1. Should the tool support multiple loan products simultaneously, or focus on one at a time?  
2. How should extreme edge cases be handled if a contractual originator runs out of capital mid-cycle?  
3. What rolling window size should be default for metrics (e.g., last 90 loans, last 3 months)?  
4. Should projections consider stochastic variation (Monte Carlo) or strictly deterministic moving average/linear trend?  
5. Is there a preferred method for persisting lender state across sessions (CSV, SQLite, or database)?  