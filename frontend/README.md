Aplikacja Next.js (13.5, App Router) + Tailwind hostuje nową stronę lądowania Trader CRM inspirowaną sekcją hero twenty.com.

## Uruchom lokalnie
```bash
cd frontend
npm install
npm run dev
```
Otwórz http://localhost:3000, aby zobaczyć stronę. Edytuj `src/app/page.tsx` dla zmian layoutu/treści oraz `src/app/globals.css` dla stylów globalnych.

### Dashboard (integracja z API)
- Backend API: `uvicorn src.api.main:app --reload --port 8000` (z katalogu głównego repo, w venv).
- Frontend: odwiedź http://localhost:3000/dashboard, aby zobaczyć JSON konta, wygenerować sygnały i obejrzeć log korzystając z endpointów FastAPI.
- Skonfiguruj bazę API przez zmienną `NEXT_PUBLIC_API_BASE` (domyślnie `http://localhost:8000`).

## Workflow porównania wizualnego
- Zrób zrzut referencyjnej sekcji hero (twenty.com), np. `original.png` via MCP Playwright.
- Zrób zrzut lokalnego buildu (`npm run dev`) do `page1.png` z tym samym viewportem.
- Porównuj i iteruj (pełna checklista w `docs/05_frontend_doc.md`).
