"""Command-line interface: index | query | eval | report | demo | serve."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import Config
from .pipeline import NeuroGraphRAG
from .report import render_main_table, write_reports
from .utils import read_json, write_json

DEFAULT_CONFIG = "configs/default.yaml"


def _build(config_path: str) -> NeuroGraphRAG:
    cfg = Config.load(config_path)
    t0 = time.time()
    ngr = NeuroGraphRAG.build(cfg)
    s = ngr.stats()
    print(
        f"[build] {s.passages} passages · {s.concepts_in_graph} concept nodes · "
        f"{s.edges} edges · {s.communities} communities · langs={','.join(s.languages)} · "
        f"emb={s.embedding_backend}({s.embedding_dim}d) · {time.time()-t0:.2f}s",
        file=sys.stderr,
    )
    return ngr


def cmd_index(args) -> None:
    ngr = _build(args.config)
    art = Path(ngr.cfg.root) / ngr.cfg.paths.artifacts
    art.mkdir(parents=True, exist_ok=True)
    write_json(art / "graph.json", ngr.graph_view())
    write_json(art / "stats.json", ngr.stats().__dict__)
    print(f"[index] wrote knowledge graph + stats to {art}")


def cmd_query(args) -> None:
    ngr = _build(args.config)
    retrievers = args.retrievers.split(",") if args.retrievers else None
    ans = ngr.answer(args.text, retrievers=retrievers, fusion=args.fusion, top_k=args.top_k)
    print(f"\nQ: {args.text}\n")
    print("Top passages:")
    for i, r in enumerate(ans.contexts, 1):
        contrib = ", ".join(f"{k}#{v}" for k, v in sorted(r.ranks.items()))
        print(f"  {i}. [{r.passage.doc_id}] ({r.passage.lang}) score={r.score:.4f} "
              f"comm={r.community} via[{contrib}]")
        print(f"     {r.passage.title}")
    print(f"\nAnswer ({ans.provider}):\n  {ans.text}")
    if ans.community_summary:
        print(f"\nCommunity context:\n  {ans.community_summary}")


def cmd_eval(args) -> None:
    ngr = _build(args.config)
    run = ngr.evaluator().run()
    runs_dir = Path(ngr.cfg.root) / ngr.cfg.paths.runs
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    write_json(runs_dir / f"eval_{stamp}.json", run)
    write_json(runs_dir / "latest.json", run)
    print("\n" + render_main_table(run) + "\n")
    primary = run["meta"]["primary_metric"]
    best = max(run["results"], key=lambda r: r["metrics"].get(primary, 0))
    print(f"[eval] primary metric = {primary}; best = '{best['name']}' "
          f"({best['metrics'].get(primary, 0):.3f}) · saved to {runs_dir/'latest.json'}")


def cmd_report(args) -> None:
    cfg = Config.load(args.config)
    latest = Path(cfg.root) / cfg.paths.runs / "latest.json"
    if not latest.exists():
        print("[report] no run found — run `eval` first.", file=sys.stderr)
        sys.exit(1)
    run = read_json(latest)
    paths = write_reports(cfg, run)
    print("\n" + render_main_table(run) + "\n")
    for name, p in paths.items():
        print(f"[report] {name}: {p}")


def cmd_demo(args) -> None:
    ngr = _build(args.config)
    run = ngr.evaluator().run()
    runs_dir = Path(ngr.cfg.root) / ngr.cfg.paths.runs
    runs_dir.mkdir(parents=True, exist_ok=True)
    write_json(runs_dir / "latest.json", run)
    paths = write_reports(ngr.cfg, run)
    art = Path(ngr.cfg.root) / ngr.cfg.paths.artifacts
    art.mkdir(parents=True, exist_ok=True)
    write_json(art / "graph.json", ngr.graph_view())
    print("\n" + render_main_table(run) + "\n")
    print(f"[demo] results -> {paths['results_md']} · figure -> {paths['figure']}")


def cmd_conformal(args) -> None:
    from .conformal import run_conformal
    from .conformal_report import write_conformal_reports

    ngr = _build(args.config)
    run = run_conformal(ngr, ngr.cfg)
    runs_dir = Path(ngr.cfg.root) / ngr.cfg.paths.runs
    runs_dir.mkdir(parents=True, exist_ok=True)
    write_json(runs_dir / "conformal_latest.json", run)
    paths = write_conformal_reports(ngr.cfg, run)

    mc = run["mondrian_community"]
    cl = run["cross_lingual"]
    print("\n[conformal] marginal coverage vs target:")
    for r in run["marginal"]:
        print(f"    1-alpha={r['target']:.2f}  empirical={r['coverage']:.3f}  |set|={r['avg_set_size']:.1f}")
    print(f"[conformal] worst-community coverage: marginal={mc['worst_group_coverage_marginal']:.3f} "
          f"-> Mondrian={mc['worst_group_coverage_mondrian']:.3f} (target {mc['target']:.2f})")
    print("[conformal] cross-lingual coverage (English-calibrated):")
    for lg in cl["languages"]:
        if lg in cl["english_calibrated"]:
            print(f"    {lg}: {cl['english_calibrated'][lg]:.3f}  (pooled {cl['pooled_calibrated'][lg]:.3f})")
    print(f"[conformal] report -> {paths['report']}")


def cmd_serve(args) -> None:
    # config path is passed to the app via env so the factory can load it
    import os

    import uvicorn

    os.environ["NGR_CONFIG"] = args.config
    uvicorn.run("neurographrag.api:app", host=args.host, port=args.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neurographrag", description="Community-Aware Cross-Lingual GraphRAG for neuroscience.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_config(sp):
        sp.add_argument("--config", default=DEFAULT_CONFIG, help="path to YAML config")

    sp = sub.add_parser("index", help="build the KG + indexes and dump artifacts")
    add_config(sp)
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("query", help="run a single query end-to-end")
    add_config(sp)
    sp.add_argument("text", help="the query text (any supported language)")
    sp.add_argument("--retrievers", default=None, help="comma list: bm25,dense,graph")
    sp.add_argument("--fusion", default=None, help="single|rrf|c2rf|weighted")
    sp.add_argument("--top-k", type=int, default=None)
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("eval", help="run the reproducible evaluation grid")
    add_config(sp)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("report", help="render tables + figure from the last run")
    add_config(sp)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("demo", help="index + eval + report in one shot")
    add_config(sp)
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("conformal", help="run the risk-controlled (conformal) retrieval study")
    add_config(sp)
    sp.set_defaults(func=cmd_conformal)

    sp = sub.add_parser("serve", help="launch the FastAPI backend + static UI")
    add_config(sp)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
