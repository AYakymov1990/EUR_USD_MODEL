# Интеграция OANDA v20 (кратко)

## Окружение
- ENV: `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, `OANDA_ENV` (practice|live, default=practice), `OANDA_TIMEOUT` (сек).
- Файл-конфиг без секретов: `config/oanda_config.json` (инструмент, гранулярность по умолчанию).

## Эндпоинты (REST)
- GET `/v3/accounts/{accountId}/summary` — баланс/NAV.
- GET `/v3/accounts/{accountId}/openTrades` — открытые позиции.
- GET `/v3/instruments/{instrument}/candles` — свечи для live-данных.
- POST `/v3/accounts/{accountId}/orders` — рыночный ордер (маркет buy/sell).
- PUT `/v3/accounts/{accountId}/orders/{orderID}/cancel` или закрытие позиции через market order встречного направления.

## Поток исполнения
1) Проверить режим: если `demo_mode=True` или нет ключей — не отправлять ордера, только логировать.
2) Сформировать payload: market order, units со знаком (long >0, short <0), инструмент EUR_USD.
3) Отправить с таймаутом, обработать HTTP ошибки (429/5xx с ретраем по бэкофу), писать в SQLite (orders/fills).
4) Для закрытия — отправить ордер противоположного знака на объем открытой позиции.

## Ошибки и логирование
- Любая ошибка сети → записать status/text в storage, вернуть оператору понятное сообщение.
- Не логировать ключи. Показывать в UI последний код ошибки/сообщение.

## Безопасные значения по умолчанию
- `practice` окружение.
- `demo_mode=True` пока оператор не включит live.
- Таймауты 10–15 секунд, без бесконечных ретраев.
