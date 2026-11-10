# SilentVoice frontend

React 19 + TypeScript + Vite + Tailwind CSS 4 + Recharts + lucide-react.

```bash
npm install
npm run dev      # http://localhost:5173; proxies /api to http://127.0.0.1:8000 (override: SILENTVOICE_API)
npm run build    # type-check + production build into dist/
```

Structure:
- `src/services/api.ts` is the typed API client.
- `src/hooks/useApi.ts` holds the fetch and job-polling hooks.
- `src/components/` holds UI primitives and feature components.
- `src/pages/` has one component per route.

Design tokens live in `src/index.css` (`@theme`). API types in `src/types/api.ts` mirror
`backend/app/schemas/api.py`.
