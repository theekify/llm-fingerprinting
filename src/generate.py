# src/generate.py
import json
import time
from pathlib import Path

import ollama
from prompts import PROMPTS

MODELS = ["llama3.2:3b", "phi3:mini", "gemma2:2b", "mistral:7b-instruct-q4_0"]
GENERATIONS_PER_PROMPT = 4
TEMPERATURE = 0.7
MAX_TOKENS = 300

# resolve path relative to repo root regardless of where script is run from
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_generations.jsonl"


def already_done() -> set:
    """Returns set of (model, prompt_id, gen_idx) tuples already generated,
    so reruns after a crash don't duplicate work."""
    done = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                done.add((rec["model"], rec["prompt_id"], rec["gen_idx"]))
    return done


def generate_one(model: str, prompt_text: str) -> str:
    response = ollama.generate(
        model=model,
        prompt=prompt_text,
        options={
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    )
    return response["response"]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = already_done()
    total = len(MODELS) * len(PROMPTS) * GENERATIONS_PER_PROMPT
    count = len(done)

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for model in MODELS:
            for prompt in PROMPTS:
                for gen_idx in range(GENERATIONS_PER_PROMPT):
                    key = (model, prompt["id"], gen_idx)
                    if key in done:
                        continue

                    start = time.time()
                    try:
                        text = generate_one(model, prompt["text"])
                    except Exception as e:
                        print(f"FAILED: {model} | {prompt['id']} | gen {gen_idx} | {e}")
                        continue
                    elapsed = time.time() - start

                    record = {
                        "model": model,
                        "prompt_id": prompt["id"],
                        "category": prompt["category"],
                        "gen_idx": gen_idx,
                        "prompt_text": prompt["text"],
                        "generated_text": text,
                        "gen_time_sec": round(elapsed, 2),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

                    count += 1
                    print(f"[{count}/{total}] {model} | {prompt['id']} | gen {gen_idx} | {elapsed:.1f}s")

    print("Done.")


if __name__ == "__main__":
    main()