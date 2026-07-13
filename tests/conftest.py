import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurographrag import Config, NeuroGraphRAG  # noqa: E402


@pytest.fixture(scope="session")
def config() -> Config:
    return Config.load(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def ngr(config) -> NeuroGraphRAG:
    return NeuroGraphRAG.build(config)
