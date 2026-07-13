// Thin API client. In dev, Vite proxies /api to the FastAPI backend.
const base = "";

async function jsonFetch(path, options) {
  const res = await fetch(base + path, options);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  stats: () => jsonFetch("/api/stats"),
  graph: (maxNodes = 90) => jsonFetch(`/api/graph?max_nodes=${maxNodes}`),
  evalResults: () => jsonFetch("/api/eval"),
  answer: (body) =>
    jsonFetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};
