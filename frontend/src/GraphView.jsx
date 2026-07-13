import React, { useEffect, useRef } from "react";

const PALETTE = ["#6366f1", "#22d3ee", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#f87171", "#38bdf8"];

// Hand-rolled force-directed layout on a canvas (no external graph library).
export default function GraphView({ graph, highlight }) {
  const canvasRef = useRef(null);
  const stateRef = useRef({ nodes: [], edges: [], idx: {} });
  const rafRef = useRef(0);

  useEffect(() => {
    if (!graph || !graph.nodes) return;
    const canvas = canvasRef.current;
    const W = (canvas.width = canvas.clientWidth);
    const H = (canvas.height = 340);
    const nodes = graph.nodes.map((n) => ({ ...n, x: Math.random() * W, y: Math.random() * H, vx: 0, vy: 0 }));
    const idx = Object.fromEntries(nodes.map((n, i) => [n.id, i]));
    stateRef.current = { nodes, edges: graph.edges, idx };
    const ctx = canvas.getContext("2d");

    const step = () => {
      const { nodes, edges, idx } = stateRef.current;
      const hot = highlight.current;
      for (let a = 0; a < nodes.length; a++) {
        const na = nodes[a];
        for (let b = a + 1; b < nodes.length; b++) {
          const nb = nodes[b];
          let dx = na.x - nb.x, dy = na.y - nb.y, d2 = dx * dx + dy * dy + 0.01, d = Math.sqrt(d2);
          const f = 1400 / d2;
          na.vx += (f * dx) / d; na.vy += (f * dy) / d;
          nb.vx -= (f * dx) / d; nb.vy -= (f * dy) / d;
        }
      }
      edges.forEach((e) => {
        const a = nodes[idx[e.source]], b = nodes[idx[e.target]];
        if (!a || !b) return;
        let dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 0.01;
        const f = (d - 70) * 0.012 * (e.weight > 2 ? 1.6 : 1);
        a.vx += (f * dx) / d; a.vy += (f * dy) / d;
        b.vx -= (f * dx) / d; b.vy -= (f * dy) / d;
      });
      nodes.forEach((n) => {
        n.vx += (W / 2 - n.x) * 0.0016; n.vy += (H / 2 - n.y) * 0.0016;
        n.x += n.vx *= 0.86; n.y += n.vy *= 0.86;
        n.x = Math.max(12, Math.min(W - 12, n.x)); n.y = Math.max(12, Math.min(H - 12, n.y));
      });
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1;
      edges.forEach((e) => {
        const a = nodes[idx[e.source]], b = nodes[idx[e.target]];
        if (!a || !b) return;
        ctx.strokeStyle = hot.has(a.id) && hot.has(b.id) ? "rgba(99,102,241,.6)" : "rgba(120,135,180,.13)";
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      });
      nodes.forEach((n) => {
        const r = 4 + Math.min(7, n.degree), c = PALETTE[n.community % PALETTE.length];
        ctx.globalAlpha = hot.size && !hot.has(n.id) ? 0.35 : 1;
        ctx.fillStyle = c; ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 7); ctx.fill();
        if (n.degree >= 5 || hot.has(n.id)) {
          ctx.globalAlpha = 1; ctx.fillStyle = "#dfe6f7"; ctx.font = "10px Segoe UI";
          ctx.fillText(n.label, n.x + r + 2, n.y + 3);
        }
        ctx.globalAlpha = 1;
      });
      rafRef.current = requestAnimationFrame(step);
    };
    step();
    return () => cancelAnimationFrame(rafRef.current);
  }, [graph]);

  return <canvas ref={canvasRef} className="graph-canvas" />;
}
