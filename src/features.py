# src/features.py
import re
import string
from collections import Counter

import nltk
import textstat
from nltk import pos_tag

# one-time downloads (safe to call every run, cached after first time)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize

# Standard English function words (closed-class, topic-independent)
FUNCTION_WORDS = [
    "the", "a", "an", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "in", "on", "off",
    "over", "under", "again", "further", "then", "once", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "i", "you", "he", "she", "it", "we", "they",
    "this", "that", "these", "those", "not", "no", "so", "as", "very",
]

# Coarse POS categories we care about (Penn Treebank tags grouped)
POS_GROUPS = {
    "noun": {"NN", "NNS", "NNP", "NNPS"},
    "verb": {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"},
    "adj": {"JJ", "JJR", "JJS"},
    "adv": {"RB", "RBR", "RBS"},
    "pronoun": {"PRP", "PRP$", "WP", "WP$"},
    "det": {"DT", "PDT", "WDT"},
    "conj": {"CC"},
    "prep": {"IN"},
}


def basic_tokenize(text: str):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words_alpha = [w.lower() for w in words if any(c.isalpha() for c in w)]
    return sentences, words, words_alpha


def length_features(sentences, words_alpha) -> dict:
    if not sentences or not words_alpha:
        return {
            "avg_sentence_len": 0, "std_sentence_len": 0,
            "avg_word_len": 0, "total_words": 0,
        }
    sent_lengths = [len(word_tokenize(s)) for s in sentences]
    avg_sent_len = sum(sent_lengths) / len(sent_lengths)
    std_sent_len = (
        sum((x - avg_sent_len) ** 2 for x in sent_lengths) / len(sent_lengths)
    ) ** 0.5
    avg_word_len = sum(len(w) for w in words_alpha) / len(words_alpha)
    return {
        "avg_sentence_len": avg_sent_len,
        "std_sentence_len": std_sent_len,
        "avg_word_len": avg_word_len,
        "total_words": len(words_alpha),
    }


def punctuation_features(text: str, n_words: int) -> dict:
    n_words = max(n_words, 1)  # avoid divide-by-zero
    marks = {
        "comma": ",", "period": ".", "semicolon": ";", "exclaim": "!",
        "question": "?", "colon": ":", "dash": "-",
    }
    return {
        f"punct_{name}_per100w": (text.count(char) / n_words) * 100
        for name, char in marks.items()
    }


def function_word_features(words_alpha) -> dict:
    n_words = max(len(words_alpha), 1)
    counts = Counter(words_alpha)
    return {
        f"fw_{fw}": (counts.get(fw, 0) / n_words) * 100
        for fw in FUNCTION_WORDS
    }


def vocab_richness_features(words_alpha) -> dict:
    if not words_alpha:
        return {"type_token_ratio": 0, "hapax_ratio": 0}
    counts = Counter(words_alpha)
    n_words = len(words_alpha)
    n_unique = len(counts)
    n_hapax = sum(1 for c in counts.values() if c == 1)
    return {
        "type_token_ratio": n_unique / n_words,
        "hapax_ratio": n_hapax / n_words,
    }


def readability_features(text: str) -> dict:
    try:
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        }
    except Exception:
        return {"flesch_reading_ease": 0, "flesch_kincaid_grade": 0}


def pos_features(words) -> dict:
    if not words:
        return {f"pos_{k}": 0 for k in POS_GROUPS}
    tagged = pos_tag(words)
    tag_counts = Counter(tag for _, tag in tagged)
    n_words = len(words)
    result = {}
    for group_name, tag_set in POS_GROUPS.items():
        group_count = sum(tag_counts.get(t, 0) for t in tag_set)
        result[f"pos_{group_name}"] = (group_count / n_words) * 100
    return result


def char_trigram_features(text: str, top_k_trigrams: list) -> dict:
    text_clean = text.lower()
    n_total = max(len(text_clean) - 2, 1)
    trigram_counts = Counter(
        text_clean[i:i + 3] for i in range(len(text_clean) - 2)
    )
    return {
        f"trigram_{tg}": (trigram_counts.get(tg, 0) / n_total) * 1000
        for tg in top_k_trigrams
    }


def extract_features(text: str, top_k_trigrams: list = None) -> dict:
    """Extract the full feature vector for one generated text sample."""
    sentences, words, words_alpha = basic_tokenize(text)

    features = {}
    features.update(length_features(sentences, words_alpha))
    features.update(punctuation_features(text, len(words_alpha)))
    features.update(function_word_features(words_alpha))
    features.update(vocab_richness_features(words_alpha))
    features.update(readability_features(text))
    features.update(pos_features(words))
    if top_k_trigrams:
        features.update(char_trigram_features(text, top_k_trigrams))

    return features