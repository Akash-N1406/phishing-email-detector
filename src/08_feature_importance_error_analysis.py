"""
08_feature_importance_error_analysis.py

Two analyses that turn "the models work" into "here's what they learned
and where they still fail":

1. FEATURE IMPORTANCE
   - Logistic Regression: coefficients directly tell us, per feature, how
     strongly it pushes toward "phishing" (positive) or "safe" (negative).
   - Random Forest: feature_importances_ tells us which features the trees
     found most useful for splitting, but NOT direction (just "important").
   Both models share the same feature indexing from Day 3: columns 0-4999
   are TF-IDF words, columns 5000-5006 are the 7 metadata features.

2. ERROR ANALYSIS
   - Re-creates the exact same train/test split from Day 3 (same random_state,
     same stratify) but this time keeping the raw cleaned_text alongside, so
     we can actually read the emails the best model (Neural Network) got
     wrong -- not just see a number.
   - Separates false positives (safe emails wrongly flagged as phishing) from
     false negatives (phishing emails that slipped through) since these are
     different failure modes with different real-world costs.

Run from project root or from src/ (paths auto-resolve):
    python src/08_feature_importance_error_analysis.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_emails.csv"
METADATA_COLS = [
    "char_count", "word_count", "url_count",
    "has_url", "exclamation_count", "digit_count", "caps_ratio",
]
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N = 15


def plot_lr_feature_importance(feature_names):
    model = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    coefs = model.coef_[0]

    order = np.argsort(coefs)
    top_phishing_idx = order[-TOP_N:][::-1]   # most positive = most phishing-indicative
    top_safe_idx = order[:TOP_N]              # most negative = most safe-indicative

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].barh(
        [feature_names[i] for i in top_phishing_idx][::-1],
        [coefs[i] for i in top_phishing_idx][::-1],
        color="crimson",
    )
    axes[0].set_title("Top features pushing toward PHISHING")
    axes[0].set_xlabel("Logistic Regression coefficient")

    axes[1].barh(
        [feature_names[i] for i in top_safe_idx][::-1],
        [coefs[i] for i in top_safe_idx][::-1],
        color="seagreen",
    )
    axes[1].set_title("Top features pushing toward SAFE")
    axes[1].set_xlabel("Logistic Regression coefficient")

    plt.tight_layout()
    out_path = FIGURES_DIR / "feature_importance_logistic_regression.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")

    print("\nTop phishing-indicative features (Logistic Regression):")
    for i in top_phishing_idx[:10]:
        print(f"  {feature_names[i]:<20} coef={coefs[i]:.3f}")
    print("\nTop safe-indicative features (Logistic Regression):")
    for i in top_safe_idx[:10]:
        print(f"  {feature_names[i]:<20} coef={coefs[i]:.3f}")


def plot_rf_feature_importance(feature_names):
    model = joblib.load(MODELS_DIR / "random_forest.joblib")
    importances = model.feature_importances_

    order = np.argsort(importances)[::-1][:TOP_N]

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="steelblue",
    )
    plt.title(f"Top {TOP_N} feature importances - Random Forest")
    plt.xlabel("Importance (Gini)")
    plt.tight_layout()
    out_path = FIGURES_DIR / "feature_importance_random_forest.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved -> {out_path}")

    print(f"\nTop {TOP_N} features by Random Forest importance:")
    for i in order:
        print(f"  {feature_names[i]:<20} importance={importances[i]:.4f}")


def error_analysis(n_examples: int = 3):
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS (Neural Network - best overall model)")
    print("=" * 60)

    # Recreate the exact same split as Day 3, but keep raw text this time
    df = pd.read_csv(CLEANED_DATA_PATH)
    df["cleaned_text"] = df["cleaned_text"].fillna("")

    X_text = df["cleaned_text"]
    X_meta = df[METADATA_COLS]
    y = df["Email Type"]

    _, X_text_test, _, _, _, y_test = train_test_split(
        X_text, X_meta, y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    from scipy import sparse
    X_test = sparse.load_npz(DATA_PROCESSED_DIR / "X_test.npz")
    X_test_dense = X_test.toarray().astype("float32")

    model = keras.models.load_model(MODELS_DIR / "neural_network.keras")
    y_pred_proba = model.predict(X_test_dense, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    y_test_arr = y_test.to_numpy()
    text_test_arr = X_text_test.to_numpy()

    false_positives = np.where((y_test_arr == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_test_arr == 1) & (y_pred == 0))[0]

    print(f"\nTotal false positives (safe flagged as phishing): {len(false_positives)}")
    print(f"Total false negatives (phishing missed as safe):  {len(false_negatives)}")

    print(f"\n--- {n_examples} FALSE POSITIVE examples (safe email wrongly flagged) ---")
    for i in false_positives[:n_examples]:
        print(f"\n[confidence phishing={y_pred_proba[i][0]:.3f}] {text_test_arr[i][:300]}")

    print(f"\n--- {n_examples} FALSE NEGATIVE examples (phishing that slipped through) ---")
    for i in false_negatives[:n_examples]:
        print(f"\n[confidence phishing={y_pred_proba[i][0]:.3f}] {text_test_arr[i][:300]}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")

    print("Analyzing Logistic Regression feature importance...")
    plot_lr_feature_importance(feature_names)

    print("\nAnalyzing Random Forest feature importance...")
    plot_rf_feature_importance(feature_names)

    error_analysis()


if __name__ == "__main__":
    main()