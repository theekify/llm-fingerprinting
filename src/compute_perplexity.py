# src/compute_perplexity.py
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw_generations.jsonl"
OUTPUT_PATH = REPO_ROOT / "data" / "perplexity.jsonl"

REFERENCE_MODEL = "distilgpt2"  # small, CPU-friendly, NOT one of our 4 target models


def load_reference_model():
    tokenizer = GPT2TokenizerFast.from_pretrained(REFERENCE_MODEL)
    model = GPT2LMHeadModel.from_pretrained(REFERENCE_MODEL)
    model.eval()
    return tokenizer, model


def compute_perplexity(text: str, tokenizer, model, max_length: int = 512) -> float:
    """Perplexity of `text` under the reference model. Lower = more
    'predictable'/generic text under a standard LM; higher = more
    unusual phrasing. This is a topic-independent style signal because
    it's about HOW predictable the token sequence is, not WHAT it's about."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids

    if input_ids.size(1) < 2:
        return float("nan")  # too short to score meaningfully

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss  # mean NLL per token

    return torch.exp(neg_log_likelihood).item()


def main():
    tokenizer, model = load_reference_model()

    records = []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Scoring {len(records)} samples with {REFERENCE_MODEL}...")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for rec in tqdm(records, desc="Perplexity"):
            ppl = compute_perplexity(rec["generated_text"], tokenizer, model)
            row = {
                "model": rec["model"],
                "prompt_id": rec["prompt_id"],
                "gen_idx": rec["gen_idx"],
                "perplexity": ppl,
            }
            out.write(json.dumps(row) + "\n")

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()