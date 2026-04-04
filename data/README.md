# Data files

| File | Purpose |
|------|---------|
| `sample_lenders.csv` | Point-in-time lender book (commitments, remaining, contractual flag, `avg_fico`, …). |
| `sample_loans.csv` | Wide **loan tape**: each row is one loan with a single `loan_fico` and allocated **face** per lender (same column ids as `lender_id` in the lender file). |
| `sample_portfolio_prior.csv` | Per-lender **cumulative** stats *before* the next allocation: `prior_funded` (Σ face on prior loans) and `prior_fico_weighted` (Σ face×loan_FICO). The sample matches aggregating `sample_loans.csv` for `loan_index` 0–2. |

Use `load_lender_program_from_csv`, `load_loan_tape_from_csv`, `load_portfolio_prior_from_csv`, and `portfolio_prior_from_loan_tape` in `shimi.data` to load these in code.
