# src/build_dataset.py
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from features import extract_features

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw_generations.jsonl"
PERPLEXITY_PATH = REPO_ROOT / "data" / "perplexity.jsonl"
OUTPUT_PATH = REPO_ROOT / "data" / "features.csv"

TOP_K_TRIGRAMS = 100


def load_raw_records() -> list:
    records = []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_perplexity() -> pd.DataFrame:
    records = []
    with open(PERPLEXITY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def build_global_trigram_vocab(records: list, top_k: int) -> list:
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

    # --- merge in perplexity ---
    ppl_df = load_perplexity()
    before_rows = len(df)
    df = df.merge(
        ppl_df[["model", "prompt_id", "gen_idx", "perplexity"]],
        on=["model", "prompt_id", "gen_idx"],
        how="left",
    )
    assert len(df) == before_rows, "Merge changed row count - duplicate keys somewhere"
    missing_ppl = df["perplexity"].isna().sum()
    print(f"\nRows missing perplexity after merge: {missing_ppl}")
    if missing_ppl > 0:
        df["perplexity"] = df["perplexity"].fillna(df["perplexity"].median())
        print("Filled missing perplexity with median (should be 0 normally - investigate if not)")

    print(f"\nFeature matrix shape: {df.shape}")
    print(f"Columns: {len(df.columns)} total "
          f"({len(df.columns) - 4} features + 4 metadata cols)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    print("\nSamples per model:")
    print(df["model"].value_counts())


if __name__ == "__main__":
    main()