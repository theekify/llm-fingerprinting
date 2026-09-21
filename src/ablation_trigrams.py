# src/ablation_trigrams.py
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = REPO_ROOT / "data" / "features.csv"
RANDOM_STATE = 42
METADATA_COLS = ["model", "prompt_id", "category", "gen_idx"]


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    all_feature_cols = [c for c in df.columns if c not in METADATA_COLS]

    # identify whitespace-adjacent trigram columns
    whitespace_trigram_cols = [
        c for c in all_feature_cols
        if c.startswith("trigram_") and ("\n" in c or "\\n" in c)
    ]
    print(f"Found {len(whitespace_trigram_cols)} whitespace-adjacent trigram columns:")
    print(whitespace_trigram_cols)

    clean_feature_cols = [c for c in all_feature_cols if c not in whitespace_trigram_cols]
    return df, all_feature_cols, clean_feature_cols, whitespace_trigram_cols


def evaluate(df, feature_cols, label: str):
    X = df[feature_cols]
    y = df["model"]
    groups = df["prompt_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_test = scaler.transform(X.iloc[test_idx])
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    cv = StratifiedGroupKFold(n_splits=5)
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        X.iloc[train_idx], y_train, cv=cv, groups=train_groups, scoring="accuracy"
    )

    print(f"\n=== {label} ({len(feature_cols)} features) ===")
    print(f"Test accuracy: {acc:.3f}")
    print(f"CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(classification_report(y_test, preds))

    return acc, cv_scores.mean(), cv_scores.std()


def main():
    df, all_cols, clean_cols, dropped_cols = load_data()

    print("\n" + "=" * 60)
    full_acc, full_cv_mean, full_cv_std = evaluate(df, all_cols, "FULL feature set (with whitespace trigrams)")

    print("\n" + "=" * 60)
    clean_acc, clean_cv_mean, clean_cv_std = evaluate(df, clean_cols, "ABLATED (whitespace trigrams removed)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"Full:     test={full_acc:.3f}  cv={full_cv_mean:.3f}+/-{full_cv_std:.3f}")
    print(f"Ablated:  test={clean_acc:.3f}  cv={clean_cv_mean:.3f}+/-{clean_cv_std:.3f}")
    print(f"Delta:    test={clean_acc - full_acc:+.3f}  cv={clean_cv_mean - full_cv_mean:+.3f}")


if __name__ == "__main__":
    main()