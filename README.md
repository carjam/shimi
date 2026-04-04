# Shimi

Capital Concentration Decision Engine — 資本密度意思決定エンジン — Shimi

## Layout

```
Shimi/
├── app/
│   └── shimi_app.py       # Main Streamlit app
├── shimi/                 # Core package (data, allocation, metrics)
│   ├── data/              # Lender program & allocation history
│   ├── allocation/        # Per-loan QP (CVXPY + OSQP)
│   └── metrics/           # Gini, FICO-weighted face, cumulative funded / remaining from history
├── data/
│   ├── sample_lenders.csv        # Lender book snapshot
│   ├── sample_loans.csv          # Loan tape (loan_fico + face per lender)
│   ├── sample_allocation_history.csv  # Optional replay into book (demo history / remaining)
│   ├── sample_portfolio_prior.csv # Cumulative Σface & Σ(face×FICO) per lender
│   └── README.md                 # Describes the sample files
├── docs/
│   ├── spec/              # Requirements, architecture, glossary
│   └── notes/             # Draft / scratch markdown
├── tests/                 # Pytest
├── notebooks/
│   └── prototype.ipynb    # Initial experimentation
├── pyproject.toml         # Package metadata (pip install -e .)
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Documentation

Spec-driven material lives under [docs/spec/](docs/spec/). Start with [requirements](docs/spec/requirements.md), [architecture](docs/spec/architecture.md), and [glossary](docs/spec/glossary.md). Informal notes go in [docs/notes/](docs/notes/).

Sample CSVs under [data/](data/) are described in [data/README.md](data/README.md): lender book, loan tape (one `loan_fico` per row), and optional portfolio priors for γ. Use `load_loan_tape_from_csv`, `load_portfolio_prior_from_csv`, and `portfolio_prior_from_loan_tape` from `shimi.data`.

## How loan allocation works (for stakeholders)

Shimi treats each new loan as a **splitting problem**: how much of this loan should each lender take, right now, given how much capacity they still have and how we want the program to behave?

### 1. Non‑negotiable rules (constraints)

Before we talk about “preferences,” the model enforces the guardrails you would insist on in a committee room:

- **The pieces must add up.** The sum allocated equals the full loan amount—no accidental over- or under-allocation.
- **Nobody is asked to fund more than they have left.** Each lender’s slice is capped by their **remaining commitment** for the program.
- **Everyone can stay meaningfully in the deal (when you want that).** You can set a **participation floor**—for example, each lender must take at least 5% of the loan—so the program does not produce lots of trivial “sliver” participations unless you choose to allow them.

If those rules cannot all be satisfied at once (for example, the loan is larger than the group’s remaining capacity, or the floor is too high), Shimi **says so up front** instead of producing a misleading split.

### 2. Business priorities (what we optimize)

Once the feasible region is clear, we need a principled way to choose *which* feasible split to use. Spreadsheets often hide trade-offs in manual tweaks. Here we make the trade-offs explicit and **tunable**:

- **Stay near fair, agreed targets.** Each lender has a **target share** (for example, reflecting their share of total commitments). We penalize moving away from that split. The weight **α (alpha)** controls how strongly we insist on staying close to those targets—higher α means “stick to the agreed risk distribution.”
- **Protect contractual originators.** For lenders flagged as **contractual originators**, we add a penalty when this loan would **use a large fraction of their remaining line** in one go. The weight **β (beta)** controls how cautious we are about drawing down their capacity relative to others—higher β means “ease off the originators when the math allows.”
- **Fair dealing on credit quality (FICO), aggregate over time.** With **γ (gamma)** we are *not* “punishing weaker lenders.” Everyone uses the **same risk assumptions**; **each loan has one representative FICO**, and over a sequence of loans those scores can look **roughly bell-shaped (Gaussian)** in the aggregate. What we care about for fairness is **each lender’s portfolio**: the **weighted average FICO of everything they have funded so far** (using that loan-level score each time). The goal is for those **portfolio averages to stay roughly the same across lenders** as loans roll on. In the engine, when you supply **cumulative prior funded face** and **cumulative Σ(face × loan FICO)** per lender, γ penalizes imbalances in how this new loan moves everyone toward a **common** portfolio average. If you have **no** history yet (cold start), γ falls back to nudging toward **equal shares** on the current loan as a simple proxy. You still trade all of this off against commitment-based targets (α) and originator protection (β).

You do not need to pick a single “magic” allocation by hand. You **turn the dials** (α, β, γ, floor, loan size, loan FICO, and optional cumulative portfolio inputs) and see how the recommended split responds—similar in spirit to a stress-testing dashboard, but grounded in a single transparent optimization.

### 3. Why use an optimizer at all?

This is a **small constrained decision problem** solved many times as you explore scenarios. A **quadratic program** (QP) is a standard, well-understood formulation: “choose the split that **minimizes weighted squared gaps** from what we want, subject to linear rules.” In practice that means:

- **Stable, intuitive behavior** when you move sliders—no wild jumps from tiny input changes.
- **Fast answers** suitable for an interactive tool (milliseconds per loan on a laptop).
- **Reproducibility**—the same inputs yield the same allocation, which supports audit and review.
- **Separation of concerns**—policy lives in the weights and floors; the solver’s job is only to find the best feasible split.

Under the hood we use **CVXPY** with the **OSQP** solver—mature, open-source building blocks—so the approach is not a one-off script but something you could extend, test, and eventually wire to richer data as the program grows.

### 4. What this demo is (and is not)

Shimi here is a **simulation and transparency layer**: it shows how a disciplined, constraint-first allocation behaves under different priorities. It is **not** claiming to replace legal documentation, credit committees, or production treasury systems—but it **does** demonstrate that a stakeholder-friendly, auditable allocation workflow can sit on top of clear rules and transparent tuning.

## Technical primer: the per-loan model (1–3)

A compact mathematical picture of what `shimi.allocation` implements—useful if you are comfortable with vectors, basic optimization, and want to connect the code to equations.

### 1. What problem are we solving?

For one new loan of face $L$ and $n$ lenders, choose **shares** $s_1,\ldots,s_n$ so lender $i$ takes amount $x_i = L\,s_i$.

**Hard constraints (feasible set):**

- **Full allocation:** $\sum_i s_i = 1$.
- **Capacity:** $s_i \le r_i / L$ where $r_i$ is remaining commitment.
- **Participation floor:** $s_i \ge f$ for all $i$ (e.g. $f = 0.05$).

These are linear equalities and inequalities; the feasible set is a **polytope**. If it is empty (floors too high, aggregate remaining $< L$, etc.), the problem is declared infeasible.

### 2. What are we optimizing?

We minimize a **sum of convex quadratic penalties** in $s$. Weights $\alpha$, $\beta$, $\gamma$ (and a tiny ridge) set how strongly each preference matters.

**Term A — $\alpha$ (target mix):**

$$\alpha \,\lVert s - t \rVert^2 = \alpha \sum_i (s_i - t_i)^2$$

where $t_i$ are **target shares** (e.g. from commitment mix). Large $\alpha$ keeps this loan’s split close to the agreed risk distribution.

**Term B — $\beta$ (contractual originator utilization):**

For lenders flagged as contractual originators (CO), penalize squared **utilization** of their remaining line:

$$\beta \sum_{i \in \mathrm{CO}} \left(\frac{L s_i}{r_i}\right)^2$$

Large $\beta$ discourages taking a big fraction of a CO’s remaining capacity when others can absorb more.

**Term C — $\gamma$ (FICO / fair dealing):**

One scalar $f$ is the loan’s representative FICO. Two regimes:

- **Cold start (no useful portfolio prior):** nudge toward equal shares $u_i = 1/n$, scaled by loan FICO, e.g. $\gamma (f/850)^2 \lVert s - u \rVert^2$.
- **With a portfolio prior:** cumulative funded $A_i$ and $\Sigma(\text{face}\times\text{FICO})$ per lender define a group average $\mu$ before the loan; the code penalizes squared imbalances built from how each lender’s **post-deal** FICO-mass relates to moving toward a **common** portfolio average (still quadratic in $s$ because $x_i = L s_i$ is linear).

**Ridge:** a small $\mathrm{ridge}\,\lVert s\rVert^2$ keeps the problem well-posed if other weights are zero and helps numerics.

### 3. Why quadratic programming (QP) and OSQP?

The objective is a **convex quadratic** and the constraints are **linear**—that is a **convex QP**. Convexity means any local minimum is **global**; the structure is the same family as least squares with linear constraints (KKT conditions: linear algebra plus complementarity for active inequalities).

**CVXPY** models the problem; **OSQP** solves it efficiently. This is transparent, reproducible optimization—not a learned black box: policy lives in $(\alpha,\beta,\gamma,f,\ldots)$; the solver finds the best feasible split.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install -e .         # makes the `shimi` package importable for Streamlit & tests
```

## Run the app

```bash
streamlit run app/shimi_app.py
```

## Tests

```bash
pytest
```

## License

See [LICENSE](LICENSE).
