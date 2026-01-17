# Przegląd projektu (Trader CRM)

## Co już jest
- Zbiór danych M15/H1 EURUSD z wyliczonymi cechami (`data/eurusd_features.parquet`).
- Pipeline cech/targetu w `src/features.py` oraz weryfikacja wyrównania.
- Model (MLP) oraz trening/walidacja w notebookach `02_features_and_target.ipynb`, `03_model_and_backtest.ipynb`.
- Backtest i metryki w `src/backtest.py`.
- Wybór najlepszej konfiguracji (threshold/regime/sizing/hold_bars) i zapis `data/artifacts/selected_config.json`.

## Co trzeba dodać (demo do 15 stycznia)
- Warstwa CRM nad modelem: generowanie sygnałów, wyjaśnienia, logowanie, potwierdzanie przez człowieka.
- Integracja z OANDA v20 (domyślnie practice), opcjonalnie live.
- Tryb demo bez sieci: replay świec/cech z testowego wycinka, sygnał → log → UI.
- Streamlit UI: ostatni sygnał i wyjaśnienie, przyciski LONG/SHORT, stan konta, mini-metryki.

## Ograniczenia
- Architektury modelu i treningu nie zmieniamy (zamrożone).
- Configi/klucze tylko przez `.env` lub zmienne środowiskowe, bez sekretów w repo.
- Bezpieczeństwo: wykonanie w trybie demo domyślnie; live — tylko z jawnym flagiem.
