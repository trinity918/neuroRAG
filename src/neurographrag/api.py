"""FastAPI backend exposing search / answer / graph / eval, plus the static UI.

Launch with:  python -m neurographrag.cli serve
The pipeline is built once at import time from the config named in $NGR_CONFIG.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Config
from .pipeline import NeuroGraphRAG
from .utils import read_json

_CONFIG_PATH = os.environ.get("NGR_CONFIG", "configs/default.yaml")
_cfg = Config.load(_CONFIG_PATH)
_ngr = NeuroGraphRAG.build(_cfg)

app = FastAPI(title="NeuroGraphRAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    retrievers: list[str] | None = None
    fusion: str | None = None
    top_k: int | None = None


def _result_json(r) -> dict:
    return {
        "passage_id": r.passage.id,
        "doc_id": r.passage.doc_id,
        "lang": r.passage.lang,
        "title": r.passage.title,
        "source": r.passage.source,
        "text": r.passage.text,
        "score": round(r.score, 5),
        "community": r.community,
        "ranks": r.ranks,
        "concepts": sorted(r.passage.concepts),
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/api/stats")
def stats() -> dict:
    s = _ngr.stats()
    return s.__dict__


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    results = _ngr.search(req.query, req.retrievers, req.fusion, req.top_k)
    return {"query": req.query, "results": [_result_json(r) for r in results]}


@app.post("/api/answer")
def answer(req: SearchRequest) -> dict:
    ans = _ngr.answer(req.query, req.retrievers, req.fusion, req.top_k)
    return {
        "query": ans.query,
        "answer": ans.text,
        "provider": ans.provider,
        "community_summary": ans.community_summary,
        "citations": ans.citations,
        "results": [_result_json(r) for r in ans.contexts],
    }


@app.get("/api/graph")
def graph(max_nodes: int = 200) -> dict:
    return _ngr.graph_view(max_nodes=max_nodes)


@app.get("/api/eval")
def eval_results() -> dict:
    latest = Path(_cfg.root) / _cfg.paths.runs / "latest.json"
    if latest.exists():
        return read_json(latest)
    return {"error": "no evaluation run found; run `neurographrag eval` first."}


# --- static single-file UI (works with zero front-end build) ---
_WEB_DIR = Path(_cfg.root) / "web"
if _WEB_DIR.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(_WEB_DIR / "index.html"))

    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
