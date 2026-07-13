# NeuroGraphRAG — Frontend (Vite + React)

A modern React UI for NeuroGraphRAG. There is also a zero-build static UI in `../web/` that the API
serves directly; this Vite app is the richer, hot-reloading version.

## Run

```bash
# 1. start the backend (from the repo root)
python -m neurographrag.cli serve         # http://127.0.0.1:8000

# 2. start the frontend dev server
cd frontend
npm install
npm run dev                               # http://127.0.0.1:5173
```

`vite.config.js` proxies `/api/*` to the backend, so the UI and API can run on separate ports in dev.

## Build

```bash
npm run build      # emits static assets to frontend/dist/
```

## What it shows

- Multilingual search box with retriever toggles (BM25 / Dense / Graph) and a fusion selector (C²RF / RRF / Single).
- Synthesized answer + community context, and ranked passages annotated with language, community, per-retriever ranks, and matched concepts.
- A live, force-directed knowledge-graph canvas coloured by community (nodes light up for the current results).
- A results panel that reads the latest evaluation run from the API.
