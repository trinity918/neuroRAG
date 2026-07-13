"""End-to-end pipeline, retrieval behaviour, and fusion."""


def test_build_stats_are_sane(ngr):
    s = ngr.stats()
    assert s.passages >= 40
    assert s.concepts_in_graph > 10
    assert s.edges > s.concepts_in_graph  # a connected-ish graph
    assert s.communities >= 2
    assert set(["en", "hi"]).issubset(set(s.languages))


def test_each_retriever_returns_results(ngr):
    for r in ["bm25", "dense", "graph"]:
        res = ngr.search("What does the N400 measure?", retrievers=[r], fusion="single", top_k=5)
        assert res, f"{r} returned nothing"
        assert all(x.score >= 0 for x in res)


def test_top_result_is_on_topic(ngr):
    res = ngr.search("What does the N400 component measure?", fusion="c2rf", top_k=3)
    assert res[0].passage.doc_id.endswith("n400")


def test_cross_lingual_retrieval(ngr):
    # a Hindi query should retrieve the English (or Hindi) N400 passage
    res = ngr.search("N400 घटक क्या मापता है?", fusion="c2rf", top_k=5)
    doc_ids = {r.passage.doc_id for r in res}
    assert doc_ids & {"en-n400", "hi-n400", "bn-n400"}


def test_alpha_query_precision(ngr):
    # regression: the community prior must not demote the on-topic alpha passage
    res = ngr.search("अल्फा लय की आवृत्ति क्या है?", fusion="c2rf", top_k=3)
    assert res[0].passage.doc_id == "hi-alpha"


def test_c2rf_beats_single_retrievers_on_ndcg(ngr):
    ev = ngr.evaluator()
    run = ev.run()
    by_name = {r["name"]: r["metrics"]["ndcg@10"] for r in run["results"]}
    ours = by_name["C²RF (ours)"]
    assert ours >= by_name["BM25"]
    assert ours >= by_name["Dense"]
    assert ours >= by_name["RRF (BM25+Dense)"]


def test_answer_has_citations(ngr):
    ans = ngr.answer("How is the P300 used in a brain-computer interface?")
    assert ans.citations
    assert ans.text
    assert ans.contexts[0].passage.doc_id.endswith(("p300",))
