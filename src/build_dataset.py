# src/build_dataset.py
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from features import extract_features

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw_generations.jsonl"
OUTPUT_PATH = REPO_ROOT / "data" / "features.csv"

TOP_K_TRIGRAMS = 100  # cap the trigram vocab size


def load_raw_records() -> list:
    records = []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_global_trigram_vocab(records: list, top_k: int) -> list:
    """Compute the most common character trigrams across the ENTIRE corpus.
    This must be done globally, not per-sample, so every row of the final
    feature matrix has the same columns."""
    counter = Counter()
    for rec in records:
        text = rec["generated_text"].lower()
        counter.update(text[i:i + 3] for i in range(len(text) - 2))
    return [tg for tg, _ in counter.most_common(top_k)]


def main():
    records = load_raw_records()
    print(f"Loaded {len(records)} raw generations")

    print("Building global trigram vocabulary...")
    top_trigrams = build_global_trigram_vocab(records, TOP_K_TRIGRAMS)
    print(f"Top 10 trigrams: {top_trigrams[:10]}")

    rows = []
    for rec in tqdm(records, desc="Extracting features"):
        feats = extract_features(rec["generated_text"], top_k_trigrams=top_trigrams)
        row = {
            "model": rec["model"],
            "prompt_id": rec["prompt_id"],
            "category": rec["category"],
            "gen_idx": rec["gen_idx"],
            **feats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nFeature matrix shape: {df.shape}")
    print(f"Columns: {len(df.columns)} total "
          f"({len(df.columns) - 4} features + 4 metadata cols)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    # quick per-model sanity check
    print("\nSamples per model:")
    print(df["model"].value_counts())


if __name__ == "__main__":
    main()