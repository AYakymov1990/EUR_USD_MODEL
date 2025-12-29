# Демонстрационный ранбук

## Подготовка
1. Установить зависимости `pip install -r requirements.txt` (Streamlit при необходимости `pip install streamlit`).
2. Проверить, что есть `data/eurusd_features.parquet` и `data/artifacts/selected_config.json` (+ сохранённые модель/скейлер).
3. Создать `.env` (опционально для live):
   ```
   OANDA_API_KEY=...
   OANDA_ACCOUNT_ID=...
   OANDA_ENV=practice
   DEMO_MODE=true
   ```

## Запуск демо (offline)
1. `python -c "import src.crm"` — проверка импортов.
2. `streamlit run app.py -- --demo` (или переменная `DEMO_MODE=true`):
   - Источник данных: реплей тестового среза.
   - Сигналы пишутся в `data/artifacts/trader_crm.sqlite`.
   - Кнопки LONG/SHORT активны, но исполнение не уходит в OANDA.

## Запуск live (опционально)
1. Прописать ключи в `.env`, установить `DEMO_MODE=false`.
2. `streamlit run app.py` — данные с OANDA, подтверждённые заявки уходят в practice/live по флагу.

## Что должно работать
- Видно последний сигнал, объяснение, confidence.
- Кнопки подтверждения → запись действия в SQLite.
- В demo mode: появляется синтетическая equity/метрики в панели.
- В live (при ключах): отображается баланс/NAV, открытые сделки; отправка ордера после подтверждения.

## Устранение неполадок
- Нет данных: убедитесь, что `data/eurusd_features.parquet` на месте.
- Нет модели: сохраните обученную модель/скейлер в `data/artifacts/` (см. ноутбук 03).
- Ошибки OANDA: проверьте ключи и `OANDA_ENV`; смотрите таблицу orders/errors в SQLite.
