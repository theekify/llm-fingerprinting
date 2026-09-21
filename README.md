# LLM Fingerprinting: Authorship Attribution via Stylometric Analysis

A classifier that identifies which open-source LLM generated a piece of text,
using purely stylometric and statistical features — no semantic/content
features, no paid APIs. Built end-to-end: local generation, feature
engineering, leak-free evaluation, and interpretability analysis.

## Problem

Large language models are increasingly used to generate text at scale, and
distinguishing AI-generated text from human text (or one model's output from
another's) is a growing practical concern — content moderation, academic
integrity, provenance tracking. Most public "AI text detectors" are opaque
black boxes. This project asks a narrower, more tractable question: **can a
small, fully interpretable classifier distinguish between four specific
open-source LLMs based on writing style alone**, and if so, *what* about
their style is discriminative?

The constraint that made this a real modeling problem rather than a lookup
table: the classifier should learn *style*, not *topic*. If Model A always
talks about cats and Model B always talks about dogs, a classifier could
"succeed" by learning content words without learning anything about writing
style at all. The entire pipeline is designed around eliminating that
shortcut.

## Models used

Four open-source models, all run locally via Ollama (no paid APIs), chosen
for architectural/training diversity rather than just size differences:

| Model | Params | Source |
|---|---|---|
| Llama 3.2 | 3B | Meta |
| Phi-3 mini | 3.8B | Microsoft (synthetic-data-heavy training) |
| Gemma2 | 2B | Google |
| Mistral 7B Instruct | 7B (Q4 quantized) | Mistral AI |

## Approach

### 1. Controlled data generation
33 prompts spanning 6 task categories (factual explanation, creative
writing, persuasive argument, step-by-step instruction, dialogue, and
summarization of a fixed source paragraph), each run through all 4 models
with 4 sampled generations per prompt (temperature=0.7), giving 528 samples
total, perfectly balanced across models (132 each). Using the *same* prompt
set across all models, spread across diverse task types, is what keeps the
learned signal style-based rather than topic-based.

### 2. Feature extraction — three independent signal families
195 features per sample, deliberately drawn from different signal types so
no single kind of feature can dominate by accident:

- **Lexical/surface stylometry**: sentence length stats, word length,
  punctuation frequency (normalized per 100 words), function-word frequency
  (closed-class words — articles, prepositions, conjunctions — chosen
  because they carry style, not topic), vocabulary richness (type-token
  ratio, hapax legomena ratio), character trigram frequencies (top 100,
  built from a global vocabulary so all samples share the same feature
  space), and standard readability scores (Flesch Reading Ease,
  Flesch-Kincaid Grade).
- **Syntactic structure**: POS-tag distribution (noun/verb/adjective/adverb/
  pronoun/determiner/conjunction/preposition ratios).
- **Predictability under an external reference model**: perplexity of each
  sample scored by DistilGPT-2 — deliberately *not* one of the four target
  models, to avoid any target-model leakage into the feature itself. This
  captures how "expected" a model's token sequences are under a generic
  language model, independent of surface word choice.

### 3. Leak-free evaluation
Samples were split by **prompt ID**, not by row: all generations tied to a
given prompt end up entirely in train or entirely in test, never split
across both. This matters because a naive random row-level split lets the
classifier partially memorize prompt-specific phrasing patterns it saw
during training, inflating apparent accuracy. I measured this directly:

| Split method | Accuracy |
|---|---|
| Naive row-level split (leaky) | 91.5% |
| Prompt-level split (leak-free) | 78.1% (baseline) / **79.7%** (with improvement pass) |

That's a **~12-point inflation from leakage alone** — quantifying this gap,
rather than just avoiding the mistake silently, was one of the more
deliberate design choices in this project.

### 4. Baseline → improvement pass
- **Baseline** (surface stylometry only, 186 features): Random Forest,
  78.1% prompt-level test accuracy, 84.5% ± 4.1% grouped 5-fold CV.
- **Improvement pass** (+ POS-tag distribution + reference-model perplexity,
  195 features): **79.7%** test accuracy, **85.7% ± 3.1%** CV — a modest
  accuracy gain, but a meaningfully *lower-variance* one, and it pushed
  Phi-3 to perfect separation (32/32) while lifting Llama's recall from
  0.56 to 0.62.

## Results

Final model: Random Forest (300 trees), 195 features, prompt-level held-out
test set (8 prompts, 128 samples never seen in training).

**Per-class performance (final model):**

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Phi-3 mini | 0.86 | **1.00** | 0.93 |
| Mistral 7B | 0.82 | 0.84 | 0.83 |
| Gemma2 | 0.74 | 0.72 | 0.73 |
| Llama 3.2 | 0.74 | 0.62 | 0.68 |

**What the classifier actually relies on (SHAP):** perplexity under the
reference model is the single most important feature by a wide margin
(mean |SHAP| ~1.6x the next-highest feature), followed by vocabulary
richness (type-token ratio), sentence-final punctuation frequency, and
hapax legomena ratio. Function-word frequencies and POS ratios contribute
secondary, smaller signal.

**Per-model fingerprints, as read from SHAP direction:**
- **Phi-3**: shortest sentences, distinctly highest perplexity and its
  widest spread — the most "unusual"/least-predictable phrasing of the
  four under a generic reference LM. Cleanly separable from every other
  class.
- **Mistral**: longer sentences, higher period frequency, more
  preposition-heavy — reads as more structurally formal.
- **Gemma2**: highest noun ratio, lowest perplexity, heavier article usage.
- **Llama 3.2**: no strong distinguishing signature. On nearly every
  top-ranked feature, its value sits *between* Gemma2 and Mistral rather
  than at either extreme — this is the direct, feature-level explanation
  for why it's the hardest class to separate (lowest recall, most spread
  confusion matrix row). This isn't a pipeline artifact; it's a genuine
  finding that at this parameter scale, Llama 3.2's default generation
  style overlaps substantially with its peers.

## Limitations & what I'd improve

- **Test set size**: only 8 held-out prompts (128 samples) after the
  prompt-level split. The gap between CV accuracy (85.7%) and single
  held-out test accuracy (79.7%) suggests real variance from this small
  holdout — a stronger version of this project would repeat the
  prompt-level split across multiple random seeds and report mean ± std
  test accuracy, not a single number. (Not yet done — the CV std of ±3.1
  points gives a rough sense of this variance in the meantime.)
- **Formatting artifacts in trigram features — checked, ruled out**: one
  top-ranked trigram (`.` followed by a paragraph break) looked like it
  might reflect Ollama's default chat template formatting rather than
  genuine authorial style. I ran an ablation removing it and retraining:
  test accuracy moved by +0.8 points and CV accuracy by −1.0 points — both
  within the CV fold-to-fold noise (±3.1 points) — confirming the model's
  performance isn't propped up by this artifact.
- **Single decoding setting**: all generations used temperature=0.7 with no
  system prompt. I don't know whether the classifier is robust to different
  decoding parameters or prompting styles — that's an untested
  generalization question, not a validated result.
- **Four models, 2-7B scale only**: the fingerprints found here are
  specific to this size class. Larger models within the same families
  might have more/less distinct styles; this doesn't generalize to that
  claim without more data.
- **Llama's ambiguity is unresolved, not just diagnosed**: I identified
  *why* Llama is hard to classify (its features sit between other models')
  but didn't find a feature that fixes it. A next step would be exploring
  features that specifically target inter-model syntax overlap rather than
  more of the same lexical/stylometric family.

## Setup
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

To regenerate from scratch, you'll need [Ollama](https://ollama.com) with
`llama3.2:3b`, `phi3:mini`, `gemma2:2b`, `mistral:7b-instruct-q4_0` pulled.
Otherwise, `data/raw_generations.jsonl` is committed, so `build_dataset.py`
and `train.py` run standalone.

## Reproducing results
```bash
cd src
python build_dataset.py          # rebuild features.csv from raw generations
python train.py                  # train + evaluate, prints leaky vs leak-free comparison
python analyze.py                # SHAP feature importance + per-class analysis
python ablation_trigrams.py      # whitespace-trigram artifact check
```