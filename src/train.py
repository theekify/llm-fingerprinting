# src/train.py
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = REPO_ROOT / "data" / "features.csv"
MODEL_OUT = REPO_ROOT / "results" / "rf_model.joblib"
RANDOM_STATE = 42

METADATA_COLS = ["model", "prompt_id", "category", "gen_idx"]


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=METADATA_COLS)
    y = df["model"]
    groups = df["prompt_id"]          # the grouping key for leak-free split
    return df, X, y, groups


def prompt_level_split(df, X, y, groups, test_size=0.22):
    """Split so that ALL generations from a given prompt_id stay together
    in either train or test - never split across both."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    train_prompts = set(df.iloc[train_idx]["prompt_id"])
    test_prompts = set(df.iloc[test_idx]["prompt_id"])
    assert train_prompts.isdisjoint(test_prompts), "LEAKAGE: prompt overlap between train/test!"

    print(f"Train prompts: {len(train_prompts)} | Test prompts: {len(test_prompts)}")
    print(f"Train rows: {len(train_idx)} | Test rows: {len(test_idx)}")

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], \
           groups.iloc[train_idx], df.iloc[test_idx]


def naive_split_baseline(X, y):
    """For comparison ONLY - shows how inflated accuracy looks with row-level
    (leaky) splitting, so we can quantify the leakage effect explicitly."""
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.22, random_state=RANDOM_STATE, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    return accuracy_score(y_te, preds)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds))
        results[name] = {"model": clf, "accuracy": acc, "preds": preds}

    return results, scaler


def main():
    df, X, y, groups = load_data()
    print(f"Full dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.nunique()} classes")
    print(f"Class balance:\n{y.value_counts()}\n")

    # --- leaky baseline, for comparison ---
    leaky_acc = naive_split_baseline(X, y)
    print(f"\n[COMPARISON] Naive row-level split accuracy (LEAKY, inflated): {leaky_acc:.3f}")

    # --- proper prompt-level split ---
    print("\n--- Prompt-level (leak-free) split ---")
    X_train, X_test, y_train, y_test, train_groups, test_df = prompt_level_split(df, X, y, groups)

    results, scaler = train_and_evaluate(X_train, X_test, y_train, y_test)

    # --- cross-validation on the training set, still grouped by prompt ---
    print("\n--- 5-fold grouped cross-validation (train set only, random forest) ---")
    cv = StratifiedGroupKFold(n_splits=5)
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        rf, X_train, y_train, cv=cv, groups=train_groups, scoring="accuracy"
    )
    print(f"CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # --- confusion matrix for the best model ---
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest model: {best_name} (acc={results[best_name]['accuracy']:.3f})")
    cm = confusion_matrix(y_test, results[best_name]["preds"], labels=sorted(y.unique()))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique())))

    # save best model + scaler for Phase 4 (SHAP analysis)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": results[best_name]["model"], "scaler": scaler,
                 "feature_names": list(X.columns)}, MODEL_OUT)
    print(f"\nSaved best model to {MODEL_OUT}")


if __name__ == "__main__":
    main()