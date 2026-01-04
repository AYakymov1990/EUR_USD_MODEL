## UI и данные

- Новости EUR/USD: использовать публичный API (например, NewsAPI.org / GNews). Запрос по ключу “EURUSD”, выводить заголовки (без сохранения). Пример:
  ```python
  import requests
  news_api_url = "https://newsapi.org/v2/top-headlines"
  params = {"q": "EURUSD", "apiKey": YOUR_NEWSAPI_KEY, "language": "en", "pageSize": 5}
  res = requests.get(news_api_url, params=params).json()
  for article in res.get("articles", []):
      st.write(f"{article['publishedAt'][:10]} – [{article['title']}]({article['url']})")
  ```

- UI через вкладки `st.tabs`: ["Dashboard", "Журнал сигналов", "Статистика", "Настройки"].
  - Dashboard: `fetch_account(cfg)` (st.json), последний сигнал (ts, action, y_hat, confidence, explanation), кнопки LONG/SHORT (только если не demo и сигнал соответствует).
  - Журнал сигналов: перенести вывод из SQLite (`fetch_recent_signals(conn, limit=100)`).
  - Статистика: базовые метрики (Sharpe, max drawdown, total return). Можно QuantStats (`qs.stats.sharpe`, `qs.stats.max_drawdown`, графики drawdown/returns).
  - Настройки: переключатели demo/live, авто-режим (аналог sidebar), отображать активный режим.

- Данные по аккаунту и сигналам брать из уже реализованных функций (fetch_account, fetch_recent_signals). Ничего лишнего не сохранять — только отображение.
