# Plan modułów (kamienie milowe)

## Etap 1. Szkielet CRM
- [x] Config/środowisko: `src/crm/config.py` (domyślnie practice, ładowanie .env).
- [x] Adapter danych: `data_feed` (replay z testowego wycinka) + szkic trybu live.
- [x] Inferencja/wyjaśnienie: ładowanie modelu/scalera, krótki tekst wyjaśnienia.
- [x] Sygnały: progi/reżimy z `selected_config.json`, tryb demo/live.
- [x] Magazyn SQLite: tabele signals/actions/orders/fills/metrics.
- [x] Egzekutor OANDA (zastępstwo + wywołania na practice).
- [x] Planer: prosty loop/apscheduler.

## Etap 2. UI i demo
- [ ] Streamlit `app.py`: panele sygnałów/wyjaśnień, przyciski LONG/SHORT, konto/metryki, przełącznik demo/live.
- [ ] Uruchomienie demo: replay świec → sygnał → zapis w SQLite → wyświetlenie.

## Etap 3. Live (opcjonalnie)
- [ ] Podłączyć realne klucze z .env, przetestować wywołania balansu/zleceń.
- [ ] Obsługa błędów/retry OANDA, time-outy.

## Etap 4. Ulepszenia (po demo)
- [ ] Powiadomienia (email/telegram) z sygnałów.
- [ ] Rozszerzone wyjaśnienie (SHAP lub top-N cech).
- [ ] Więcej trybów zarządzania ryzykiem (pozycjonowanie, stop/take).
