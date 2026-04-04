# Shimi

Capital Concentration Decision Engine — 資本密度意思決定エンジン — Shimi

## Layout

```
Shimi/
├── app/
│   └── shimi_app.py       # Main Streamlit app
├── data/
│   └── sample_lenders.csv # Starter lender dataset
├── docs/
│   ├── spec/              # Requirements, architecture, glossary
│   └── notes/             # Draft / scratch markdown
├── notebooks/
│   └── prototype.ipynb    # Initial experimentation
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
```

## Run the app

```bash
streamlit run app/shimi_app.py
```

## License

See [LICENSE](LICENSE).
