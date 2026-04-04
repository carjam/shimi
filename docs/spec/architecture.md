# Shimi: Architecture

## Overview
Shimi is structured as a **layered, modular system** to simulate loan allocation, compute metrics, and visualize risk for stakeholders.  
It is designed for **clarity, interactivity, and auditability**, while remaining flexible to extend with additional lenders, loan products, or simulation scenarios.

---

## Components

### 1. Data Layer
- Stores lender information:
  - `name`, `total_commitment`, `remaining_commitment`, `target_share`, `is_contractual_originator`, `avg_fico`.
- Supports optional historical loan series for projections.
- Responsibilities:
  - Serve as the authoritative input for allocations.
  - Track cumulative allocations for rolling window metrics.
  - Provide data for capital exhaustion calculations.

### 2. Allocation Engine
- Per-loan allocation is formulated as a **Quadratic Program (QP)**:
  - **Decision Variables**: allocation percentage per lender (sum = 100% of loan, each ≥ floor, each ≤ remaining commitment).
  - **Objective Function**:
    - Minimize weighted deviation from target share (fair risk distribution).  
    - Penalize depletion of contractual originators (protection weight β).  
    - Include weighted FICO impact (optional cost function).
  - **Constraints**:
    - Participation floor (e.g., ≥ 5%).  
    - Remaining commitment per lender.  
    - Sum of allocations = loan amount.
- Solved using **CVXPY + OSQP**.
- Supports **mixed-integer style behavior** via rounding heuristics for minimal slivers.

### 3. Metrics Layer
- Calculates real-time metrics:
  - **Gini coefficient** → concentration risk per loan or rolling window.
  - **Weighted FICO** → aggregate exposure per loan.
  - **Cumulative allocations / drawdown curves**.
  - **Projected capital exhaustion**:
    - Moving average of historical allocation per lender.
    - Linear regression extrapolation for trend-based projection.
  - Alerts if contractual originators or critical lenders approach zero commitment.

### 4. Presentation Layer
- **Streamlit-based UI**:
  - Sliders for α (risk distribution) and β (contractual originator protection).
  - Dynamic charts:
    - Stacked bar charts (per-loan allocation)
    - Line charts (cumulative allocations, Gini, weighted FICO)
    - Drawdown curves with exhaustion projection markers.
  - Tables showing projected loan number at exhaustion.
  - Alerts and annotations for critical lenders.

---

## Data Flow

Loan Input --> Allocation Engine --> Metrics Layer --> Presentation Layer


- Allocation engine receives loan and lender data, outputs allocation per lender.
- Metrics layer computes Gini, weighted FICO, drawdown, and projections.
- Presentation layer renders charts and tables dynamically.
- Optional: shadow-mode α/β tuning updates suggested allocations without affecting live data.

---

## Technical Stack

- **Language**: Python 3.x  
- **Web Framework**: Streamlit  
- **Optimization**: CVXPY + OSQP  
- **Visualization**: Matplotlib / Streamlit charts  
- **Data Handling**: Pandas / NumPy  
- **Meta-Optimization (Optional)**: scikit-optimize for Bayesian tuning of α/β weights  

---

## Extensibility & Maintainability
- Adding new lenders or loan products requires minimal code change (update data source).  
- Rolling window size for metrics is configurable.  
- The architecture separates **allocation**, **metrics**, and **presentation** for testability.  
- Logging and auditability allow historical allocations to be reviewed and α/β tuning decisions to be tracked.  
- Future enhancements:
  - Stochastic simulation (Monte Carlo) for capital depletion under uncertainty.  
  - Integration with production treasury systems or external dashboards.  