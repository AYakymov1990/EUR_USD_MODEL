# Projekt pracy dyplomowej EUR/USD ML

Projekt buduje prosty model regresji liniowej w PyTorch do prognozowania zwrotów EUR/USD M15 (horyzont 3 świec) i wykorzystuje predykcje w podstawowym backteście trend-following.

## Pipeline
1. Pobierz świece EUR/USD (M15 + H1) z OANDA v20 przez `/v3/instruments/{instrument}/candles`.
2. Zbuduj cechy i target.
3. Wytrenuj model regresji liniowej w PyTorch.
4. Oceń jakość predykcji i uruchom prosty backtest.

## Konfiguracja
### 1) Utwórz i aktywuj wirtualne środowisko
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Zainstaluj zależności
```bash
pip install -r requirements.txt
```

### 3) Skonfiguruj OANDA
Skopiuj przykładowy config i uzupełnij dane uwierzytelniające:
```bash
cp config/oanda_config.example.json config/oanda_config.json
```
Edytuj `config/oanda_config.json` i ustaw `api_key`, `account_id` oraz `environment`.

### 4) Uruchom notebooki
Uruchamiaj notebooki w kolejności:
1. `notebooks/01_oanda_download.ipynb`
2. `notebooks/02_features_and_target.ipynb`
3. `notebooks/03_model_and_backtest.ipynb`

## Trenowanie z CLI (skrypt)
Uruchom trening i zapisz artefakty w `data/artifacts/`:
```bash
python scripts/train_model.py --retrain
```
Artefakty: `model.pt`, `scaler.pkl`, `selected_config.json`, `metadata.json`.

## Streamlit CRM
- Demo (offline, replay parquet):
  ```bash
  DEMO_MODE=true streamlit run app.py -- --demo
  ```
- Live practice (realne świece/zlecenia, wymagane `.env` z OANDA_API_KEY/OANDA_ACCOUNT_ID):
  ```bash
  DEMO_MODE=false OANDA_ENV=practice streamlit run app.py
  ```
W trybie live dane są pobierane z OANDA, model generuje sygnał, a po potwierdzeniu przyciskiem LONG/SHORT wysyłany jest market order na practice.

## Backend API (FastAPI)
- Uruchomienie API (wykorzystuje te same artefakty/demo dane, CORS dla localhost:3000):
  ```bash
  source .venv/bin/activate
  uvicorn src.api.main:app --reload --port 8000
  ```
- Kluczowe endpointy:
  - `GET /health`, `GET /config` — status/tryby
  - `GET /account` — dane konta (demo -> `{demo: true}`)
  - `POST /signals/generate` — wygeneruj sygnał (demo: kolejny wiersz z parquet)
  - `GET /signals/recent?limit=50` — dziennik sygnałów z SQLite
  - `POST /orders/market` — wyślij zlecenie rynkowe (w demo nie wysyła, tylko loguje)
  - `GET /news?limit=5` — wiadomości EURUSD (NewsAPI przy obecności NEWS_API_KEY)

## Frontend (Next.js UI)
- Nowy landing znajduje się w `frontend/` (Next.js 13 + Tailwind), odwzorowuje hero twenty.com dla Trader CRM.
- Instalacja i lokalne uruchomienie: `cd frontend && npm install && npm run dev` (domyślnie port 3000).
- Postępuj wg `docs/05_frontend_doc.md` dla workflow porównania pikseli (zrzut referencyjny, porównanie Playwright, iteracje).
