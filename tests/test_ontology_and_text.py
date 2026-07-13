"""Ontology matching, tokenization, and the cross-lingual concept bridge."""
from pathlib import Path

import numpy as np

from neurographrag.embeddings import ConceptHashEmbedder
from neurographrag.ontology import Ontology
from neurographrag.utils import char_ngrams, tokenize

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "data" / "ontology" / "neuro_ontology.yaml"


def test_tokenize_is_script_agnostic():
    assert tokenize("The N400 effect") == ["the", "n400", "effect"]
    # Devanagari stays intact as word tokens
    assert "अल्फा" in tokenize("अल्फा लय की आवृत्ति")


def test_char_ngrams_have_boundaries():
    grams = list(char_ngrams("eeg", 3, 4))
    assert "#ee" in grams and "eg#" in grams


def test_ontology_matches_multilingual_aliases():
    onto = Ontology.load(ONTOLOGY)
    assert "n400" in onto.match("The N400 is a negative ERP")
    assert "n400" in onto.match("एन400 एक ऋणात्मक तरंग है")  # Hindi alias -> same id
    assert "alpha" in onto.match("अल्फा लय")                  # Hindi phrase alias
    assert "bci" in onto.match("மூளை-கணினி இடைமுகம் BCI")     # Tamil + Latin acronym


def test_concept_embedding_aligns_across_languages(config):
    onto = Ontology.load(ONTOLOGY)
    emb = ConceptHashEmbedder(config, onto)
    en = emb.encode(["The N400 indexes semantic processing"])
    hi = emb.encode(["एन400 अर्थपूर्ण प्रसंस्करण का सूचकांक है"])
    unrelated = emb.encode(["Delta waves dominate deep sleep"])
    sim_cross = float(en[0] @ hi[0])
    sim_unrelated = float(en[0] @ unrelated[0])
    # shared concept dimensions make the cross-lingual pair more similar than an
    # unrelated same-language passage
    assert sim_cross > sim_unrelated
    assert sim_cross > 0.1
