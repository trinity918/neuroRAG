"""Turn an evaluation run into human-readable tables + an SVG figure.

Outputs (under paper/):
    paper/results.md          -- markdown tables for the paper / README
    paper/figures/main.svg    -- horizontal bar chart of the primary metric
"""
from __future__ import annotations

from pathlib import Path

from .config import Config

_KEY_METRICS = ["mrr", "map", "recall@5", "ndcg@10"]


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def render_main_table(run: dict) -> str:
    header = "| Configuration | MRR | MAP | Recall@5 | nDCG@10 | Faithfulness | Ans. Rel. |"
    sep = "|" + "|".join(["---"] * 7) + "|"
    lines = [header, sep]
    best = _best_by(run, "ndcg@10")
    for r in run["results"]:
        m = r["metrics"]
        rag = r.get("ragas", {})
        name = r["name"]
        cells = [
            f"**{name}**" if name == best else name,
            _fmt(m.get("mrr", 0)),
            _fmt(m.get("map", 0)),
            _fmt(m.get("recall@5", 0)),
            _fmt(m.get("ndcg@10", 0)),
            _fmt(rag.get("faithfulness", 0)),
            _fmt(rag.get("answer_relevancy", 0)),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_by_type_table(run: dict, metric: str = "ndcg@10") -> str:
    types: list[str] = []
    for r in run["results"]:
        for t in r["by_type"]:
            if t not in types:
                types.append(t)
    types.sort()
    header = "| Configuration | " + " | ".join(types) + " |"
    sep = "|" + "|".join(["---"] * (len(types) + 1)) + "|"
    lines = [header, sep]
    for r in run["results"]:
        cells = [r["name"]]
        for t in types:
            cells.append(_fmt(r["by_type"].get(t, {}).get(metric, 0.0)))
        lines.append("| " + " | ".join(cells) + " |")
    return f"_Per-query-type {metric} (rows = configuration)._\n\n" + "\n".join(lines)


def _best_by(run: dict, metric: str) -> str:
    best_name, best_val = "", -1.0
    for r in run["results"]:
        v = r["metrics"].get(metric, 0.0)
        if v > best_val:
            best_val, best_name = v, r["name"]
    return best_name


def render_bar_svg(run: dict, metric: str = "ndcg@10") -> str:
    rows = [(r["name"], r["metrics"].get(metric, 0.0)) for r in run["results"]]
    best = max((v for _, v in rows), default=1.0) or 1.0
    W, rowh, pad, label_w = 640, 34, 16, 210
    bar_max = W - label_w - pad - 70
    H = pad * 2 + rowh * len(rows) + 30
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
        f'<text x="{pad}" y="{pad+4}" font-size="15" font-weight="700" fill="#1f2937">'
        f'NeuroGraphRAG — {metric} by retrieval configuration</text>',
    ]
    for i, (name, val) in enumerate(rows):
        y = pad + 24 + i * rowh
        w = max(2.0, bar_max * (val / best))
        is_best = val == max(v for _, v in rows)
        color = "#4f46e5" if is_best else "#94a3b8"
        out.append(f'<text x="{pad}" y="{y+15}" font-size="12.5" fill="#334155">{_esc(name)}</text>')
        out.append(f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="20" rx="4" fill="{color}"/>')
        out.append(f'<text x="{label_w+w+8:.1f}" y="{y+15}" font-size="12.5" fill="#1f2937">{val:.3f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_reports(cfg: Config, run: dict) -> dict[str, Path]:
    root = Path(cfg.root)
    paper = root / "paper"
    figures = paper / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    main = render_main_table(run)
    by_type = render_by_type_table(run)
    svg = render_bar_svg(run)

    meta = run["meta"]
    md = (
        "# NeuroGraphRAG — Evaluation Results\n\n"
        f"- Embedding backend: `{meta['embedding_backend']}`\n"
        f"- Queries: {meta['num_queries']} across languages {', '.join(meta['languages'])}\n"
        f"- Seed: {meta['seed']} · wall-clock: {meta['elapsed_sec']}s\n\n"
        "## Main results\n\n" + main + "\n\n"
        "![main results](figures/main.svg)\n\n"
        "## Breakdown by query type\n\n" + by_type + "\n"
    )
    paths = {
        "results_md": paper / "results.md",
        "figure": figures / "main.svg",
    }
    paths["results_md"].write_text(md, encoding="utf-8")
    paths["figure"].write_text(svg, encoding="utf-8")
    return paths
