# Integracja OANDA v20 (skrót)

## Środowisko
- ENV: `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `OANDA_ENV` (practice|live, domyślnie practice), `OANDA_TIMEOUT` (sek).
- Plik konfiguracyjny bez sekretów: `config/oanda_config.json` (instrument, domyślna granularność).

## Endpointy (REST)
- GET `/v3/accounts/{accountId}/summary` — balans/NAV.
- GET `/v3/accounts/{accountId}/openTrades` — otwarte pozycje.
- GET `/v3/instruments/{instrument}/candles` — świece dla danych live.
- POST `/v3/accounts/{accountId}/orders` — zlecenie rynkowe (market buy/sell).
- PUT `/v3/accounts/{accountId}/orders/{orderID}/cancel` lub zamknięcie pozycji zleceniem market w przeciwnym kierunku.

## Przepływ wykonania
1) Sprawdź tryb: jeśli `demo_mode=True` lub brak kluczy — nie wysyłaj zleceń, tylko loguj.
2) Zbuduj payload: market order, units ze znakiem (long >0, short <0), instrument EUR_USD.
3) Wyślij z time-outem, obsłuż błędy HTTP (429/5xx z retry i backoff), zapisuj w SQLite (orders/fills).
4) Do zamknięcia — wyślij zlecenie przeciwnego znaku na wolumen otwartej pozycji.

## Błędy i logowanie
- Dowolny błąd sieci → zapisz status/text w storage, zwróć operatorowi jasny komunikat.
- Nie loguj kluczy. Pokazuj w UI ostatni kod błędu/wiadomość.

## Bezpieczne wartości domyślne
- `practice` środowisko.
- `demo_mode=True` dopóki operator nie włączy live.
- Time-outy 10–15 sekund, bez nieskończonych retry.
