# Shimi

Capital Concentration Decision Engine — 資本密度意思決定エンジン — Shimi

## Layout

```
Shimi/
├── app/
│   └── shimi_app.py       # Main Streamlit app
├── shimi/                 # Core package (data, allocation, metrics)
│   ├── data/              # Lender program & allocation history
│   └── allocation/        # Per-loan QP (CVXPY + OSQP)
├── data/
│   ├── sample_lenders.csv        # Lender book snapshot
│   ├── sample_loans.csv          # Loan tape (loan_fico + face per lender)
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
