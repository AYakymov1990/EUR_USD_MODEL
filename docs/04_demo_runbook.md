# Runbook demonstracyjny

## Przygotowanie
1. Zainstaluj zależności `pip install -r requirements.txt` (Streamlit w razie potrzeby `pip install streamlit`).
2. Upewnij się, że masz `data/eurusd_features.parquet` i `data/artifacts/selected_config.json` (+ zapisany model/scaler).
3. Utwórz `.env` (opcjonalnie dla live):
   ```
   OANDA_API_KEY=...
   OANDA_ACCOUNT_ID=...
   OANDA_ENV=practice
   DEMO_MODE=true
   ```

## Uruchomienie demo (offline)
1. `python -c "import src.crm"` — sprawdzenie importów.
2. `streamlit run app.py -- --demo` (lub zmienna `DEMO_MODE=true`):
   - Źródło danych: replay testowego wycinka.
   - Sygnały zapisywane do `data/artifacts/trader_crm.sqlite`.
   - Przyciski LONG/SHORT aktywne, ale wykonanie nie trafia do OANDA.

## Uruchomienie live (opcjonalnie)
1. Wpisz klucze do `.env`, ustaw `DEMO_MODE=false`.
2. `streamlit run app.py` — dane z OANDA, potwierdzone zlecenia trafiają do practice/live wg flagi.

## Co powinno działać
- Widać ostatni sygnał, wyjaśnienie, confidence.
- Przyciski potwierdzenia → zapis akcji w SQLite.
- W trybie demo: pojawia się syntetyczna equity/metryki w panelu.
- W trybie live (przy kluczach): widać balans/NAV, otwarte transakcje; wysyłka zlecenia po potwierdzeniu.

## Rozwiązywanie problemów
- Brak danych: upewnij się, że `data/eurusd_features.parquet` jest na miejscu.
- Brak modelu: zapisz wytrenowany model/scaler w `data/artifacts/` (zob. notebook 03).
- Błędy OANDA: sprawdź klucze i `OANDA_ENV`; zajrzyj do tabel orders/errors w SQLite.
