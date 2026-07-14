"""Render a conformal run into paper-ready SVG figures + a markdown report.

Figures (under paper/figures/):
    conformal_reliability.svg   coverage vs target (the guarantee)
    conformal_setsize.svg       efficiency: set size vs 1-alpha
    conformal_community.svg     conditional coverage: marginal vs Mondrian
    conformal_crosslingual.svg  cross-lingual coverage shift
    conformal_riskcoverage.svg  selective-generation risk vs answer rate
"""
from __future__ import annotations

from pathlib import Path

from .config import Config

BLUE, GREY, GREEN, AMBER, RED = "#4f46e5", "#94a3b8", "#10b981", "#f59e0b", "#ef4444"


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_open(w, h, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif">',
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>',
        f'<text x="16" y="22" font-size="15" font-weight="700" fill="#1f2937">{_esc(title)}</text>',
    ]


# --------------------------------------------------------------------------- #
def reliability_svg(run: dict) -> str:
    rows = run["marginal"]
    W, H, ox, oy, pw, ph = 480, 360, 60, 300, 380, 250
    out = _svg_open(W, H, "Marginal coverage vs target (guarantee)")
    # axes
    out.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#cbd5e1"/>')
    out.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#cbd5e1"/>')
    # diagonal y=x over [0.6,1]
    def X(v):  # target in [0.6,1]
        return ox + pw * (v - 0.6) / 0.4
    def Y(v):
        return oy - ph * (v - 0.6) / 0.4
    out.append(f'<line x1="{X(0.6)}" y1="{Y(0.6)}" x2="{X(1.0)}" y2="{Y(1.0)}" stroke="{GREY}" stroke-dasharray="4 4"/>')
    out.append(f'<text x="{X(0.62)}" y="{Y(0.72)}" font-size="10" fill="{GREY}">y = x (ideal)</text>')
    # points + polyline
    pts = [(X(r["target"]), Y(min(1.0, r["coverage"]))) for r in rows]
    out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    for (x, y), r in zip(pts, rows):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{BLUE}"/>')
        out.append(f'<text x="{x:.1f}" y="{y-8:.1f}" font-size="9" fill="#334155" text-anchor="middle">{r["coverage"]:.2f}</text>')
    for v in (0.6, 0.7, 0.8, 0.9, 1.0):
        out.append(f'<text x="{X(v):.1f}" y="{oy+16}" font-size="10" fill="#64748b" text-anchor="middle">{v:.1f}</text>')
        out.append(f'<text x="{ox-10}" y="{Y(v)+3:.1f}" font-size="10" fill="#64748b" text-anchor="end">{v:.1f}</text>')
    out.append(f'<text x="{ox+pw/2}" y="{H-6}" font-size="11" fill="#475569" text-anchor="middle">target coverage (1 - α)</text>')
    out.append(f'<text x="16" y="{oy-ph-6}" font-size="11" fill="#475569">empirical coverage</text>')
    out.append("</svg>")
    return "\n".join(out)


def setsize_svg(run: dict) -> str:
    rows = run["marginal"]
    n_docs = run["meta"]["n_docs"]
    W, H, ox, oy, pw, ph = 480, 340, 60, 280, 380, 230
    out = _svg_open(W, H, "Efficiency: prediction-set size vs 1-α")
    mx = max(r["avg_set_size"] for r in rows) or 1
    bw = pw / (len(rows) * 1.6)
    for i, r in enumerate(rows):
        x = ox + 20 + i * (pw / len(rows))
        h = ph * (r["avg_set_size"] / mx)
        out.append(f'<rect x="{x:.1f}" y="{oy-h:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" fill="{BLUE}"/>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{oy-h-6:.1f}" font-size="10" fill="#1f2937" text-anchor="middle">{r["avg_set_size"]:.1f}</text>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{oy+16}" font-size="10" fill="#64748b" text-anchor="middle">{r["target"]:.2f}</text>')
    out.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#cbd5e1"/>')
    out.append(f'<text x="{ox+pw/2}" y="{H-6}" font-size="11" fill="#475569" text-anchor="middle">target coverage (1 - α) — universe = {n_docs} docs</text>')
    out.append("</svg>")
    return "\n".join(out)


def _grouped_bar_svg(title, groups, series, target, ylabel, colors):
    """groups: list[str]; series: list[(name, [values])]; target: float or None."""
    W, H = 520, 360
    ox, oy, pw, ph = 60, 290, 430, 235
    out = _svg_open(W, H, title)
    gw = pw / max(1, len(groups))
    nb = len(series)
    bw = gw / (nb + 1)
    for gi, g in enumerate(groups):
        gx = ox + gi * gw
        for si, (_, vals) in enumerate(series):
            v = vals[gi] if vals[gi] is not None else 0
            h = ph * min(1.0, v)
            x = gx + bw * (si + 0.5)
            out.append(f'<rect x="{x:.1f}" y="{oy-h:.1f}" width="{bw*0.9:.1f}" height="{h:.1f}" rx="2" fill="{colors[si]}"/>')
            out.append(f'<text x="{x+bw*0.45:.1f}" y="{oy-h-4:.1f}" font-size="8.5" fill="#334155" text-anchor="middle">{v:.2f}</text>')
        out.append(f'<text x="{gx+gw/2:.1f}" y="{oy+15}" font-size="9.5" fill="#475569" text-anchor="middle">{_esc(g)}</text>')
    if target is not None:
        ty = oy - ph * target
        out.append(f'<line x1="{ox}" y1="{ty:.1f}" x2="{ox+pw}" y2="{ty:.1f}" stroke="{RED}" stroke-dasharray="5 4"/>')
        out.append(f'<text x="{ox+pw-4}" y="{ty-4:.1f}" font-size="9" fill="{RED}" text-anchor="end">target 1-α = {target:.2f}</text>')
    out.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#cbd5e1"/>')
    # legend
    lx = ox
    for si, (name, _) in enumerate(series):
        out.append(f'<rect x="{lx}" y="{H-20}" width="11" height="11" fill="{colors[si]}"/>')
        out.append(f'<text x="{lx+15}" y="{H-11}" font-size="10" fill="#475569">{_esc(name)}</text>')
        lx += 40 + 7 * len(name)
    out.append(f'<text x="14" y="{oy-ph-6}" font-size="10" fill="#475569">{_esc(ylabel)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def community_svg(run: dict) -> str:
    mc = run["mondrian_community"]
    pg = {k: v for k, v in mc["per_group"].items() if v["marginal"] is not None}
    groups = sorted(pg, key=lambda k: pg[k]["marginal"])
    labels = [f"c{g} (n={pg[g]['n']})" for g in groups]
    marg = [pg[g]["marginal"] for g in groups]
    mond = [pg[g]["mondrian"] for g in groups]
    return _grouped_bar_svg(
        "Conditional coverage by KG community", labels,
        [("marginal", marg), ("Mondrian (ours)", mond)], mc["target"],
        "coverage", [GREY, GREEN],
    )


def crosslingual_svg(run: dict) -> str:
    cl = run["cross_lingual"]
    langs = [lg for lg in cl["languages"] if lg in cl["english_calibrated"]]
    en = [cl["english_calibrated"][lg] for lg in langs]
    pool = [cl["pooled_calibrated"][lg] for lg in langs]
    return _grouped_bar_svg(
        "Cross-lingual coverage transfer", [lg.upper() for lg in langs],
        [("English-calibrated", en), ("pooled-calibrated", pool)], cl["target"],
        "coverage", [AMBER, BLUE],
    )


def riskcoverage_svg(run: dict) -> str:
    rc = run["selective"]["risk_coverage"]
    ltt = run["selective"]["ltt"]
    W, H, ox, oy, pw, ph = 480, 340, 60, 280, 380, 230
    out = _svg_open(W, H, "Selective generation: risk vs answer rate")
    xs, ys = rc["answer_rate"], rc["risk"]
    my = max(max(ys), *[r["beta"] for r in ltt]) or 1
    def X(a):
        return ox + pw * a
    def Y(r):
        return oy - ph * (r / my)
    pts = sorted(zip(xs, ys))
    out.append('<polyline points="' + " ".join(f"{X(a):.1f},{Y(r):.1f}" for a, r in pts) + f'" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    for r in ltt:
        out.append(f'<circle cx="{X(r["answer_rate"]):.1f}" cy="{Y(r["test_risk"]):.1f}" r="4" fill="{GREEN}"/>')
        out.append(f'<text x="{X(r["answer_rate"]):.1f}" y="{Y(r["test_risk"])-7:.1f}" font-size="8.5" fill="#166534" text-anchor="middle">β={r["beta"]:.1f}</text>')
    out.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#cbd5e1"/>')
    out.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#cbd5e1"/>')
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<text x="{X(a):.1f}" y="{oy+16}" font-size="10" fill="#64748b" text-anchor="middle">{a:.2f}</text>')
    out.append(f'<text x="{ox+pw/2}" y="{H-6}" font-size="11" fill="#475569" text-anchor="middle">answer rate (coverage)</text>')
    out.append(f'<text x="14" y="{oy-ph-6}" font-size="11" fill="#475569">selective risk (answered error)</text>')
    out.append(f'<text x="{ox+pw-4}" y="{oy-ph+8}" font-size="9" fill="{GREEN}" text-anchor="end">● LTT-controlled thresholds</text>')
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def _md(run: dict) -> str:
    m = run["meta"]
    mc, ml, cl = run["mondrian_community"], run["mondrian_language"], run["cross_lingual"]
    sel = run["selective"]
    lines = [
        "# Conformal GraphRAG — Risk-Controlled Retrieval Results\n",
        f"- {m['n_queries']} queries · {m['n_docs']} docs · languages {', '.join(m['languages'])}",
        f"- System under test: `{m['fusion']}` fusion of {', '.join(m['retrievers'])}",
        f"- {m['trials']} random calibration/test resamples · seed {m['seed']} · α* = {m['primary_alpha']}\n",
        "## 1. Marginal coverage guarantee\n",
        "| target 1−α | empirical coverage | avg set size |",
        "|---|---|---|",
    ]
    for r in run["marginal"]:
        lines.append(f"| {r['target']:.2f} | {r['coverage']:.3f} ± {r['coverage_std']:.3f} | {r['avg_set_size']:.1f} / {m['n_docs']} |")
    lines += [
        "\n![reliability](figures/conformal_reliability.svg) ![set size](figures/conformal_setsize.svg)\n",
        "## 2. Conditional coverage — marginal conformal hides per-group failures\n",
        f"At target {mc['target']:.2f}, the **worst knowledge-graph community** gets only "
        f"**{mc['worst_group_coverage_marginal']:.3f}** coverage under marginal calibration; "
        f"community-conditional (Mondrian) calibration raises the worst group to "
        f"**{mc['worst_group_coverage_mondrian']:.3f}**.\n",
        "![community](figures/conformal_community.svg)\n",
        "## 3. Cross-lingual coverage transfer\n",
        "Calibrating on English and testing per language reveals that the guarantee does **not** transfer "
        "uniformly; pooled (multilingual) calibration narrows the gap.\n",
        "| language | English-calibrated | pooled-calibrated |",
        "|---|---|---|",
    ]
    for lg in cl["languages"]:
        if lg in cl["english_calibrated"]:
            lines.append(f"| {lg} | {cl['english_calibrated'][lg]:.3f} | {cl['pooled_calibrated'][lg]:.3f} |")
    lines += [
        "\n![crosslingual](figures/conformal_crosslingual.svg)\n",
        "## 4. Risk-controlled selective generation\n",
        f"Base top-1 error is {sel['base_error_rate']:.3f}. Learn-then-Test picks a confidence threshold so the "
        "answered-query error stays under a target β:\n",
        "| target risk β | achieved test risk | answer rate |",
        "|---|---|---|",
    ]
    for r in sel["ltt"]:
        lines.append(f"| {r['beta']:.2f} | {r['test_risk']:.3f} | {r['answer_rate']:.3f} |")
    lines.append("\n![riskcoverage](figures/conformal_riskcoverage.svg)\n")
    return "\n".join(lines)


def write_conformal_reports(cfg: Config, run: dict) -> dict[str, Path]:
    figures = Path(cfg.root) / "paper" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figs = {
        "conformal_reliability.svg": reliability_svg(run),
        "conformal_setsize.svg": setsize_svg(run),
        "conformal_community.svg": community_svg(run),
        "conformal_crosslingual.svg": crosslingual_svg(run),
        "conformal_riskcoverage.svg": riskcoverage_svg(run),
    }
    for name, svg in figs.items():
        (figures / name).write_text(svg, encoding="utf-8")
    md = Path(cfg.root) / "paper" / "conformal_results.md"
    md.write_text(_md(run), encoding="utf-8")
    return {"report": md, "figures_dir": figures}
