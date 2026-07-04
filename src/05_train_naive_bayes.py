"""
05_train_naive_bayes.py

Trains a Multinomial Naive Bayes classifier on the TF-IDF portion of the
Day 3 features (metadata columns excluded).

Why TF-IDF only: MultinomialNB requires non-negative feature values (it
models features as counts/frequencies). Our combined matrix includes
StandardScaler-scaled metadata, which can be negative -- incompatible with
MultinomialNB's assumptions. TF-IDF values are always >= 0, so we slice out
just those columns. This also plays to Naive Bayes' actual strength: pure
word-frequency reasoning, without structural features muddying the model.

Run from project root or from src/ (paths auto-resolve):
    python src/05_train_naive_bayes.py
"""

import joblib
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB

from eval_utils import evaluate_model, load_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Must match TFIDF_MAX_FEATURES from 03_feature_engineering.py --
# the first N columns of the combined matrix are TF-IDF, the rest are metadata.
TFIDF_FEATURE_COUNT = 5000


def main():
    print("Loading features from Day 3...")
    X_train, X_test, y_train, y_test = load_features()

    # Slice to TF-IDF columns only (drop the trailing metadata columns)
    X_train_tfidf = X_train[:, :TFIDF_FEATURE_COUNT]
    X_test_tfidf = X_test[:, :TFIDF_FEATURE_COUNT]
    print(f"Using TF-IDF-only features: {X_train_tfidf.shape}")

    print("\nTraining Multinomial Naive Bayes...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    evaluate_model("Naive Bayes", y_test, y_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "naive_bayes.joblib")
    print(f"\nSaved model -> {MODELS_DIR / 'naive_bayes.joblib'}")


if __name__ == "__main__":
    main()