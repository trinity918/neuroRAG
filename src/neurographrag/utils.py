"""Shared, dependency-light helpers: tokenization, hashing, IO, seeding."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

# Unicode-aware word tokenizer. Plain \w drops Indic combining marks (viramas,
# dependent vowel signs) because they are Unicode marks rather than letters,
# which would fragment words like "अल्फा". We therefore also admit the Indic
# block range U+0900–U+0FFF (Devanagari/Bengali/Tamil/… incl. their marks),
# excluding the danda punctuation U+0964–U+0965.
_TOKEN_RE = re.compile(r"[\wऀ-ॣ०-࿿]+", re.UNICODE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Split text into unicode word tokens (script-agnostic)."""
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace; used for phrase (alias) matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def char_ngrams(token: str, n_min: int, n_max: int) -> Iterator[str]:
    """Character n-grams of a single token, padded to expose word boundaries."""
    padded = f"#{token}#"
    for n in range(n_min, n_max + 1):
        if len(padded) < n:
            continue
        for i in range(len(padded) - n + 1):
            yield padded[i : i + n]


def stable_hash(text: str, dim: int) -> int:
    """Deterministic bucket index in [0, dim) for a string (md5-based)."""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h, 16) % dim


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def l2_normalize(mat: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=axis, keepdims=True)
    return mat / np.maximum(norm, eps)


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
