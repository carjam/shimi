# Shimi

Capital Concentration Decision Engine — 資本密度意思決定エンジン — Shimi

## Layout

```
Shimi/
├── app/
│   └── shimi_app.py       # Main Streamlit app
├── data/
│   └── sample_lenders.csv # Starter lender dataset
├── notebooks/
│   └── prototype.ipynb    # Initial experimentation
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

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
