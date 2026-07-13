import React, { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import GraphView from "./GraphView.jsx";

const EXAMPLES = [
  "What does the N400 component measure?",
  "अल्फा लय की आवृत्ति क्या है?",
  "Which ERP components are altered in schizophrenia?",
  "মৃগীরোগ নির্ণয়ে EEG কীভাবে সাহায্য করে?",
  "signals and methods for a motor-imagery BCI",
];
const PALETTE = ["#6366f1", "#22d3ee", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#f87171", "#38bdf8"];
const LANG = { en: "EN", hi: "HI", bn: "BN", ta: "TA" };

export default function App() {
  const [stats, setStats] = useState(null);
  const [graph, setGraph] = useState(null);
  const [evalRun, setEvalRun] = useState(null);
  const [query, setQuery] = useState("");
  const [retrievers, setRetrievers] = useState({ bm25: true, dense: true, graph: true });
  const [fusion, setFusion] = useState("c2rf");
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const highlight = useRef(new Set());

  useEffect(() => {
    api.stats().then(setStats).catch(() => {});
    api.graph(90).then(setGraph).catch(() => {});
    api.evalResults().then(setEvalRun).catch(() => {});
  }, []);

  const toggle = (r) =>
    setRetrievers((prev) => {
      const on = Object.values(prev).filter(Boolean).length;
      if (prev[r] && on <= 1) return prev; // keep at least one
      return { ...prev, [r]: !prev[r] };
    });

  const runSearch = async (q) => {
    const text = (q ?? query).trim();
    if (!text) return;
    setQuery(text);
    setLoading(true);
    try {
      const active = Object.entries(retrievers).filter(([, v]) => v).map(([k]) => k);
      const data = await api.answer({ query: text, retrievers: active, fusion, top_k: 8 });
      setAnswer(data);
      const hot = new Set();
      const blob = data.results.map((r) => (r.title + " " + r.text).toLowerCase()).join(" ");
      (graph?.nodes || []).forEach((n) => {
        if (blob.includes(n.label.toLowerCase())) hot.add(n.id);
      });
      highlight.current = hot;
    } finally {
      setLoading(false);
    }
  };

  const best =
    evalRun?.results?.reduce((a, b) => (b.metrics["ndcg@10"] > a.metrics["ndcg@10"] ? b : a));

  return (
    <div>
      <header>
        <div className="logo">Neuro<span>GraphRAG</span></div>
        <div className="tag">Community-Aware Cross-Lingual GraphRAG · EEG / ERP & biomedical literature</div>
        {stats && (
          <div className="stats">
            <span><b>{stats.passages}</b> passages</span>
            <span><b>{stats.concepts_in_graph}</b> concepts</span>
            <span><b>{stats.edges}</b> edges</span>
            <span><b>{stats.communities}</b> communities</span>
            <span><b>{stats.languages?.join(", ")}</b></span>
            <span><b>{stats.embedding_backend}</b> {stats.embedding_dim}d</span>
          </div>
        )}
      </header>

      <main>
        <section className="card">
          <h2>Ask across languages</h2>
          <div className="searchbar">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runSearch()}
              placeholder="e.g. What does the N400 measure?  ·  अल्फा लय की आवृत्ति क्या है?"
            />
            <button onClick={() => runSearch()}>{loading ? "…" : "Search"}</button>
          </div>
          <div className="controls">
            <span>Retrievers:</span>
            {["bm25", "dense", "graph"].map((r) => (
              <span key={r} className={"chip" + (retrievers[r] ? " on" : "")} onClick={() => toggle(r)}>
                {r.toUpperCase()}
              </span>
            ))}
            <span style={{ marginLeft: 8 }}>Fusion:</span>
            <select value={fusion} onChange={(e) => setFusion(e.target.value)}>
              <option value="c2rf">C²RF (ours)</option>
              <option value="rrf">RRF</option>
              <option value="single">Single</option>
            </select>
          </div>
          <div className="examples">
            {EXAMPLES.map((e) => (
              <a key={e} onClick={() => runSearch(e)}>{e}</a>
            ))}
          </div>

          {answer && (
            <>
              <div className="answer">
                <div className="lbl">Synthesized answer</div>
                <div>{answer.answer}</div>
                {answer.community_summary && <div className="commsum">🕸 {answer.community_summary}</div>}
              </div>
              {answer.results.map((r, i) => (
                <div className="result" key={i}>
                  <div className="top">
                    <span className="badge lang">{LANG[r.lang] || r.lang}</span>
                    <span className="badge comm">community {r.community}</span>
                    <span className="score">{r.score.toFixed(4)}</span>
                  </div>
                  <h3>{r.title}</h3>
                  <p>{r.text}</p>
                  <div className="src">
                    <span><b>source:</b> {r.source}</span>
                    <span><b>via:</b> {Object.entries(r.ranks || {}).map(([k, v]) => `${k}#${v}`).join(" · ") || "—"}</span>
                    <span><b>concepts:</b> {r.concepts.slice(0, 6).join(", ")}</span>
                  </div>
                </div>
              ))}
            </>
          )}
        </section>

        <section>
          <div className="card">
            <h2>Knowledge graph</h2>
            <GraphView graph={graph} highlight={highlight} />
            {graph && (
              <div className="legend">
                {[...new Set(graph.nodes.map((n) => n.community))].sort((a, b) => a - b).map((c) => {
                  const theme = (graph.communities.find((x) => x.id === c) || {}).theme || `community ${c}`;
                  return (
                    <span key={c}>
                      <span className="dot" style={{ background: PALETTE[c % PALETTE.length] }} />
                      {theme}
                    </span>
                  );
                })}
              </div>
            )}
          </div>

          <div className="card" style={{ marginTop: 18 }}>
            <h2>Evaluation (nDCG@10 · MRR · Recall@5)</h2>
            {!evalRun || evalRun.error ? (
              <div className="muted">{evalRun?.error || "loading…"}</div>
            ) : (
              <>
                <table>
                  <thead>
                    <tr><th>Config</th><th>nDCG@10</th><th>MRR</th><th>Recall@5</th><th>Faith.</th></tr>
                  </thead>
                  <tbody>
                    {evalRun.results.map((r) => (
                      <tr key={r.name} className={r === best ? "best" : ""}>
                        <td>{r.name}</td>
                        <td>{r.metrics["ndcg@10"].toFixed(3)}</td>
                        <td>{r.metrics["mrr"].toFixed(3)}</td>
                        <td>{r.metrics["recall@5"].toFixed(3)}</td>
                        <td>{(r.ragas.faithfulness || 0).toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="muted" style={{ marginTop: 8 }}>
                  {evalRun.meta.num_queries} queries · langs {evalRun.meta.languages.join(", ")} · backend {evalRun.meta.embedding_backend}
                </div>
              </>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
