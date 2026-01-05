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

## CLI train (скрипт)
Запуск обучения и сохранение артефактов в `data/artifacts/`:
```bash
python scripts/train_model.py --retrain
```
Артефакты: `model.pt`, `scaler.pkl`, `selected_config.json`, `metadata.json`.

## Streamlit CRM
- Демо (офлайн, реплей parquet):  
  ```bash
  DEMO_MODE=true streamlit run app.py -- --demo
  ```
- Live practice (реальные свечи/ордера, требуется `.env` с OANDA_API_KEY/OANDA_ACCOUNT_ID):  
  ```bash
  DEMO_MODE=false OANDA_ENV=practice streamlit run app.py
  ```
В live режиме данные подтягиваются через OANDA, модель даёт сигнал, после подтверждения кнопкой LONG/SHORT отправляется market-ордер в practice.

## Backend API (FastAPI)
- Запуск API (реюзит те же артефакты/демо-данные, CORS для localhost:3000):
  ```bash
  source .venv/bin/activate
  uvicorn src.api.main:app --reload --port 8000
  ```
- Основные эндпоинты:  
  - `GET /health`, `GET /config` — статус/режимы  
  - `GET /account` — данные счёта (demo -> `{demo: true}`)  
  - `POST /signals/generate` — сгенерировать сигнал (демо: следующий ряд из parquet)  
  - `GET /signals/recent?limit=50` — журнал сигналов из sqlite  
  - `POST /orders/market` — отправить рыночный ордер (в demo не отправляет, только логирует)  
  - `GET /news?limit=5` — Новости EURUSD (NewsAPI при наличии NEWS_API_KEY)

## Frontend (Next.js UI)
- New landing UI lives in `frontend/` (Next.js 13 + Tailwind), mirroring the twenty.com hero for the Trader CRM.
- Install deps and run locally: `cd frontend && npm install && npm run dev` (default port 3000).
- Follow `docs/05_frontend_doc.md` for pixel-parity workflow (capture reference screenshot, compare via Playwright, iterate).
