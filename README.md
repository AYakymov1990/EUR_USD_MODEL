# EUR/USD ML Thesis Project

This project builds a simple PyTorch linear regression model to predict EUR/USD M15 returns (3-bar horizon) and uses the predictions in a basic trend-following backtest.

## Pipeline
1. Download EUR/USD candles (M15 + H1) from OANDA v20 using `/v3/instruments/{instrument}/candles`.
2. Build features and target.
3. Train a linear regression model in PyTorch.
4. Evaluate predictive quality and run a simple backtest.

## Setup
### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure OANDA
Copy the example config and fill in your credentials:
```bash
cp config/oanda_config.example.json config/oanda_config.json
```
Edit `config/oanda_config.json` and set your `api_key`, `account_id`, and `environment`.

### 4) Run notebooks
Run the notebooks in order:
1. `notebooks/01_oanda_download.ipynb`
2. `notebooks/02_features_and_target.ipynb`
3. `notebooks/03_model_and_backtest.ipynb`
