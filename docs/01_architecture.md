# Architektura (opis)

## Warstwy
- **Warstwa modelu (zamrożona)**: `src/features.py`, `src/model.py`, `src/backtest.py`, notebooki 02/03. Odpowiada za dane/cechy, trening, backtest, wybór konfiguracji.
- **Warstwa CRM**: `src/crm/` — praca z configiem, pobieranie świec/cech, inferencja, wyjaśnienia, generowanie sygnałów/progów, magazyn zdarzeń, integracja OANDA, liczenie bieżących metryk, scheduler.
- **Warstwa UI**: `app.py` (Streamlit) — wizualizacja sygnałów/metryk, przyciski potwierdzeń.

## Przepływ danych (tryb demo)
1. `data_feed` czyta świece/cechy z testowego wycinka (replay).
2. `features_adapter` przygotowuje wejścia dla modelu.
3. `inference` ładuje scaler/model, zwraca `y_hat` + metadane.
4. `signals` stosuje thresholds/regime z `selected_config.json`, liczy sygnał i confidence.
5. `explain` buduje krótkie objaśnienie PL po kluczowych cechach/predykcji.
6. `storage` zapisuje sygnał, `metrics_live` odświeża podsumowania, UI wyświetla.
7. Po kliknięciu LONG/SHORT zapisujemy akcję, w trybie live `oanda_executor` wysyła zlecenie.

## Przepływ danych (tryb live)
1. `data_feed` pobiera świece z OANDA (bezpiecznie, domyślnie practice).
2. Dalej kroki jak w trybie demo.
3. `oanda_executor` wysyła/zamyka zlecenia, zapisuje odpowiedzi w `storage`.

## Artefakty
- `data/artifacts/selected_config.json` — próg/reżim/hold_bars/polarity.
- Zapisane model/scaler (oczekiwane obok w `data/artifacts/`).
- Baza audytu SQLite (`data/artifacts/trader_crm.sqlite`).

## Bezpieczeństwo
- Klucze/konta tylko przez `.env`/środowisko.
- Logi bez sekretów, błędy sieciowe — z czytelnym tekstem.
- Wykonanie w trybie demo domyślnie.
