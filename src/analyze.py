# src/analyze.py
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = REPO_ROOT / "data" / "features.csv"
MODEL_PATH = REPO_ROOT / "results" / "rf_model.joblib"
FIG_DIR = REPO_ROOT / "results" / "figures"

METADATA_COLS = ["model", "prompt_id", "category", "gen_idx"]


def load_everything():
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    df = pd.read_csv(FEATURES_PATH)
    X = df[feature_names]
    y = df["model"]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)

    return clf, X_scaled, X, y, feature_names


def global_importance(clf, X_scaled, feature_names, class_names, top_n=20):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_scaled)
    # shap_values shape: (n_samples, n_features, n_classes) for multiclass RF in newer shap versions
    # handle both old (list of arrays) and new (3D array) API shapes
    if isinstance(shap_values, list):
        mean_abs_per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
        mean_abs_overall = np.mean(mean_abs_per_class, axis=0)
    else:
        mean_abs_overall = np.abs(shap_values).mean(axis=(0, 2))

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_overall,
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\n=== Top {top_n} globally important features ===")
    print(importance_df.head(top_n).to_string(index=False))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_names,
                       class_names=list(class_names), show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_global_summary.png", dpi=150)
    plt.close()
    print(f"Saved {FIG_DIR / 'shap_global_summary.png'}")

    return importance_df, shap_values, explainer


def per_class_importance(shap_values, feature_names, class_names, top_n=10):
    """What features push the model TOWARD predicting each specific class."""
    for i, cls in enumerate(class_names):
        if isinstance(shap_values, list):
            sv = shap_values[i]
        else:
            sv = shap_values[:, :, i]
        mean_shap = sv.mean(axis=0)  # signed, not abs - shows direction
        top_idx = np.argsort(np.abs(mean_shap))[::-1][:top_n]
        print(f"\n--- Top features pushing toward '{cls}' ---")
        for idx in top_idx:
            direction = "+" if mean_shap[idx] > 0 else "-"
            print(f"  {direction} {feature_names[idx]:30s}  mean_shap={mean_shap[idx]:.4f}")


def investigate_llama_confusion(df: pd.DataFrame, feature_names):
    """Llama was the weakest class - compare its feature distributions against
    the classes it gets confused with (gemma2, mistral, phi3) on the specific
    features that matter most, to see WHY it's ambiguous."""
    key_features = [
        "perplexity", "avg_sentence_len", "type_token_ratio",
        "pos_noun", "pos_verb", "hapax_ratio", "flesch_kincaid_grade",
    ]
    key_features = [f for f in key_features if f in feature_names]

    print("\n=== Llama vs other classes: key feature means ===")
    summary = df.groupby("model")[key_features].mean().T
    print(summary.round(3).to_string())

    print("\nInterpretation aid: look for rows where llama3.2:3b's value sits "
          "BETWEEN two other models rather than being distinct - that's a feature "
          "where Llama has no clear signature of its own.")


def main():
    clf, X_scaled, X_raw, y, feature_names = load_everything()
    class_names = sorted(y.unique())

    importance_df, shap_values, explainer = global_importance(
        clf, X_scaled, feature_names, class_names
    )

    per_class_importance(shap_values, feature_names, class_names)

    full_df = pd.read_csv(FEATURES_PATH)
    investigate_llama_confusion(full_df, feature_names)

    importance_df.to_csv(REPO_ROOT / "results" / "feature_importance.csv", index=False)
    print(f"\nSaved feature importance table to results/feature_importance.csv")


if __name__ == "__main__":
    main()