This Next.js (13.5, App Router) + Tailwind app hosts the new Trader CRM landing page that mirrors the twenty.com hero.

## Run locally
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:3000 to view the page. Edit `src/app/page.tsx` for layout/content updates and `src/app/globals.css` for global styling.

### Dashboard (API integration)
- Backend API: `uvicorn src.api.main:app --reload --port 8000` (from repo root, venv).
- Frontend: visit http://localhost:3000/dashboard to view account JSON, generate signals, and see the log using the FastAPI endpoints.
- Configure API base via `NEXT_PUBLIC_API_BASE` env var (default `http://localhost:8000`).

## Visual parity workflow
- Capture the reference (twenty.com) hero screenshot (e.g., `original.png` via MCP Playwright).
- Capture the local build (`npm run dev`) to `page1.png` with the same viewport.
- Compare and iterate (see `docs/05_frontend_doc.md` for the full checklist).
