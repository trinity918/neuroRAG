"""NeuroGraphRAG: Community-Aware Cross-Lingual GraphRAG for neuroscience retrieval.

Public entry points:
    from neurographrag import Config, NeuroGraphRAG
    ngr = NeuroGraphRAG.build(Config.load("configs/default.yaml"))
    result = ngr.answer("What does the N400 measure?")
"""
from __future__ import annotations

from .config import Config
from .pipeline import NeuroGraphRAG

__all__ = ["Config", "NeuroGraphRAG", "__version__"]
__version__ = "0.1.0"
