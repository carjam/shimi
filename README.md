# Shimi

Capital Concentration Decision Engine — 資本密度意思決定エンジン — Shimi

## Layout

```
Shimi/
├── app/
│   └── shimi_app.py       # Main Streamlit app
├── shimi/                 # Core package (data, allocation, metrics)
│   └── data/              # Lender program & allocation history
├── data/
│   └── sample_lenders.csv # Starter lender dataset
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
