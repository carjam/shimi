# Shimi: Glossary

**Allocation**  
Distribution of a single loan amount across multiple lenders.

**Lender / FI (Financial Institution)**  
A participant in the loan program providing capital for allocations.

**Commitment Total**  
Total capital a lender agrees to contribute to a product or program.

**Commitment Remaining**  
Capital still available to allocate after prior loans.

**Target Share**  
The expected proportional share for a lender based on total commitments.

**Contractual Originator**  
Lender legally holding the loan; their capital is critical to program continuation.

**Floor Participation**  
Minimum allocation percentage to prevent operationally insignificant splits (e.g., 5%).

**α (Alpha)**  
Weight controlling importance of fair risk distribution in the allocation objective.

**β (Beta)**  
Weight controlling importance of contractual originator protection in the allocation objective.

**Weighted FICO**  
For a single loan, how much credit-quality exposure a lender takes via that loan: use the loan’s one representative FICO with that lender’s share and face amount. **Portfolio** weighted-average FICO for a lender is Σ(face×loan FICO)/Σ(face) over loans they funded. **γ (gamma)** targets **equal portfolio averages across lenders over time** (with optional **portfolio priors** in the engine); cold start uses equal-share proxy on the current loan only.

**Portfolio prior (allocation)**  
Per lender, before the current loan: cumulative **funded face** and cumulative **Σ(face × loan FICO)** on all prior loans—inputs that let the γ term rebalance **cross-lender** portfolio weighted-average FICO.

**Gini Coefficient**  
Measure of inequality in allocations; higher values indicate concentration risk.

**Cumulative Allocation / Drawdown Curve**  
Running total of allocations per lender over multiple loans.

**Capital Exhaustion Projection**  
Estimate of when a lender’s remaining commitment will reach zero. Methods:
- **Moving Average**: based on average historical allocation per loan.
- **Linear Extrapolation**: fits a trend line to historical allocations.

**Rolling Window**  
Period (number of loans or days) over which cumulative metrics are computed (e.g., last 90 loans).

**Quadratic Program (QP)**  
Optimization framework minimizing weighted deviation from objectives while satisfying constraints.

**CVXPY / OSQP**  
Python library and solver for quadratic programs.

**Shadow Mode**  
Feature where α/β tuning occurs without affecting live allocations for testing purposes.

**Allocation Sliver**  
Very small allocation (< floor) that is operationally undesirable; handled via rounding heuristics.