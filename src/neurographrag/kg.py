"""Knowledge-graph construction, community detection, and community summaries.

The graph is over *concept* nodes. Edges come from two sources:
  1. typed seed relations asserted in the ontology, and
  2. corpus co-occurrence (two concepts sharing a passage).
Communities partition the concept graph (GraphRAG-style); each community gets a
templated summary that provides global context for community-aware retrieval and
for answer synthesis.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import networkx as nx

from .config import Config
from .ingestion import Passage
from .ontology import Ontology


@dataclass
class Community:
    id: int
    concepts: list[str]              # concept ids
    theme: str                       # human-readable label
    passages: list[str]              # representative passage ids
    summary: str


@dataclass
class KnowledgeGraph:
    graph: nx.Graph
    ontology: Ontology
    communities: list[Community]
    concept_community: dict[str, int]           # concept id -> community id
    passage_community: dict[str, int]           # passage id -> dominant community id
    concept_passages: dict[str, list[str]]      # concept id -> passage ids
    passage_concepts: dict[str, set[str]]       # passage id -> concept ids

    # ---- graph queries used by the graph retriever ----
    def expand(self, seeds: set[str], hops: int, max_nodes: int) -> dict[str, int]:
        """BFS from seed concepts; return {concept_id: hop_distance}."""
        dist: dict[str, int] = {c: 0 for c in seeds if c in self.graph}
        frontier = set(dist)
        for h in range(1, hops + 1):
            nxt: set[str] = set()
            for node in frontier:
                for nb in self.graph.neighbors(node):
                    if nb not in dist:
                        dist[nb] = h
                        nxt.add(nb)
                        if len(dist) >= max_nodes:
                            return dist
            frontier = nxt
            if not frontier:
                break
        return dist

    def to_dict(self, max_nodes: int = 200) -> dict:
        """JSON-serializable view for the /graph API endpoint and figures."""
        nodes = []
        for n, _data in self.graph.nodes(data=True):
            c = self.ontology.concepts.get(n)
            nodes.append({
                "id": n,
                "label": c.canonical if c else n,
                "type": c.type if c else "Concept",
                "community": self.concept_community.get(n, -1),
                "degree": self.graph.degree(n),
            })
            if len(nodes) >= max_nodes:
                break
        keep = {n["id"] for n in nodes}
        edges = [
            {"source": u, "target": v, "weight": d.get("weight", 1), "type": d.get("type", "cooccur")}
            for u, v, d in self.graph.edges(data=True)
            if u in keep and v in keep
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "communities": [
                {"id": c.id, "theme": c.theme, "size": len(c.concepts), "summary": c.summary}
                for c in self.communities
            ],
        }


def build_graph(cfg: Config, ontology: Ontology, passages: list[Passage]) -> KnowledgeGraph:
    # --- inverted indexes over concepts ---
    concept_passages: dict[str, list[str]] = defaultdict(list)
    passage_concepts: dict[str, set[str]] = {}
    concept_freq: Counter[str] = Counter()
    for p in passages:
        passage_concepts[p.id] = set(p.concepts)
        for c in p.concepts:
            concept_passages[c].append(p.id)
            concept_freq[c] += 1

    active = {c for c, f in concept_freq.items() if f >= cfg.kg.min_concept_freq}

    G = nx.Graph()
    for c in active:
        concept = ontology.concepts[c]
        G.add_node(c, canonical=concept.canonical, type=concept.type)

    # --- seed (typed) edges from the ontology ---
    for r in ontology.relations:
        if r.source in active and r.target in active:
            G.add_edge(r.source, r.target, weight=2.0, type=r.type)

    # --- co-occurrence edges from the corpus ---
    pair_weight: Counter[tuple[str, str]] = Counter()
    for p in passages:
        cs = sorted(c for c in p.concepts if c in active)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                pair_weight[(cs[i], cs[j])] += 1
    for (u, v), w in pair_weight.items():
        if w < cfg.kg.edge_min_weight:
            continue
        if G.has_edge(u, v):
            G[u][v]["weight"] += float(w)
        else:
            G.add_edge(u, v, weight=float(w), type="cooccur")

    communities, concept_community = _detect_communities(cfg, G)
    community_summaries = _summarize_communities(
        cfg, ontology, G, communities, concept_community, concept_passages, passages
    )

    # dominant community per passage = majority community of its concepts
    passage_community: dict[str, int] = {}
    for p in passages:
        votes = Counter(concept_community[c] for c in p.concepts if c in concept_community)
        passage_community[p.id] = votes.most_common(1)[0][0] if votes else -1

    return KnowledgeGraph(
        graph=G,
        ontology=ontology,
        communities=community_summaries,
        concept_community=concept_community,
        passage_community=passage_community,
        concept_passages=dict(concept_passages),
        passage_concepts=passage_concepts,
    )


def _detect_communities(cfg: Config, G: nx.Graph) -> tuple[list[list[str]], dict[str, int]]:
    if G.number_of_nodes() == 0:
        return [], {}
    algo = cfg.kg.community.algorithm
    if algo == "label_propagation":
        raw = list(nx.community.label_propagation_communities(G))
    else:
        raw = list(
            nx.community.greedy_modularity_communities(
                G, weight="weight", resolution=cfg.kg.community.resolution
            )
        )
    raw.sort(key=len, reverse=True)
    concept_community: dict[str, int] = {}
    groups: list[list[str]] = []
    for cid, members in enumerate(raw):
        members = sorted(members)
        groups.append(members)
        for m in members:
            concept_community[m] = cid
    return groups, concept_community


def _summarize_communities(
    cfg: Config,
    ontology: Ontology,
    G: nx.Graph,
    groups: list[list[str]],
    concept_community: dict[str, int],
    concept_passages: dict[str, list[str]],
    passages: list[Passage],
) -> list[Community]:
    by_id = {p.id: p for p in passages}
    out: list[Community] = []
    for cid, members in enumerate(groups):
        # theme = highest-degree concept in the community
        anchor = max(members, key=lambda m: G.degree(m, weight="weight"))
        anchor_name = ontology.concepts[anchor].canonical
        types = Counter(ontology.concepts[m].type for m in members)
        theme = f"{anchor_name} & related {types.most_common(1)[0][0].lower()}s"

        # representative passages: those covering the most community concepts
        member_set = set(members)
        scored: Counter[str] = Counter()
        for m in members:
            for pid in concept_passages.get(m, []):
                overlap = len(by_id[pid].concepts & member_set)
                scored[pid] = max(scored[pid], overlap)
        rep = [pid for pid, _ in scored.most_common(cfg.kg.community.max_summary_passages)]

        names = [ontology.concepts[m].canonical for m in members[:8]]
        finding = " ".join(
            f"{by_id[pid].title.rstrip('.')}." for pid in rep[:2]
        )
        summary = (
            f"This community centers on {anchor_name} and links "
            f"{len(members)} concepts including {', '.join(names)}. "
            f"Representative evidence: {finding}"
        ).strip()
        out.append(Community(id=cid, concepts=members, theme=theme, passages=rep, summary=summary))
    return out
