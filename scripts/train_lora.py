"""LoRA/PEFT domain adaptation of a multilingual sentence encoder for neuroscience.

This is the OPTIONAL "full-stack" path from the paper. It requires the extras:

    pip install -r requirements-extras.txt

It mines (anchor, positive) pairs from the ontology + corpus (a concept's
definition and the passages that mention it are treated as positives), attaches
a low-rank adapter to the encoder's transformer, and fine-tunes with a
multiple-negatives contrastive objective. The resulting adapter is saved to
`models/adapters/<name>` and can be merged at inference by setting
`embedding.lora_adapter` in the config.

Example:
    python scripts/train_lora.py --base sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --epochs 2 --out models/adapters/neuro-mpnet
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurographrag.config import Config          # noqa: E402
from neurographrag.ingestion import load_passages  # noqa: E402
from neurographrag.ontology import Ontology       # noqa: E402


def build_pairs(cfg: Config) -> list[tuple[str, str]]:
    """Mine (anchor, positive) training pairs from ontology + corpus."""
    onto = Ontology.load(cfg.resolve(cfg.paths.ontology))
    passages = load_passages(cfg, onto)
    by_concept: dict[str, list[str]] = {}
    for p in passages:
        for c in p.concepts:
            by_concept.setdefault(c, []).append(p.searchable_text())

    pairs: list[tuple[str, str]] = []
    for cid, concept in onto.concepts.items():
        texts = by_concept.get(cid, [])
        # definition <-> each mentioning passage
        if concept.definition:
            for t in texts:
                pairs.append((f"{concept.canonical}: {concept.definition}", t))
        # passage <-> passage sharing the concept (cross-lingual positives included)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                pairs.append((texts[i], texts[j]))
    random.shuffle(pairs)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--out", default="models/adapters/neuro-adapter")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260713)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        from peft import LoraConfig, get_peft_model
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise SystemExit(
            "train_lora.py needs the extras: pip install -r requirements-extras.txt\n"
            f"(missing: {e.name})"
        )

    random.seed(args.seed)
    cfg = Config.load(ROOT / args.config)
    pairs = build_pairs(cfg)
    print(f"[data] mined {len(pairs)} contrastive pairs")

    model = SentenceTransformer(args.base)
    transformer = model[0].auto_model
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.05,
        bias="none",
    )
    model[0].auto_model = get_peft_model(transformer, lora)
    model[0].auto_model.print_trainable_parameters()

    examples = [InputExample(texts=[a, b]) for a, b in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        warmup_steps=max(1, len(loader) // 10),
        show_progress_bar=True,
    )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    model[0].auto_model.save_pretrained(str(out))
    print(f"[done] saved LoRA adapter -> {out}")
    print(f"       set embedding.lora_adapter: {args.out} and backend: sentence-transformers")


if __name__ == "__main__":
    main()
