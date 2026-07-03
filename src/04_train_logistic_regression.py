"""
04_train_logistic_regression.py

Trains a Logistic Regression baseline on the TF-IDF + metadata features
from Day 3, evaluates it, and saves the model + metrics.

Why Logistic Regression first: it's fast, interpretable (coefficients map
directly to words/features), and gives you an honest baseline number that
Random Forest, Naive Bayes, and the Neural Network all need to beat to
justify their added complexity.

Run from project root:
    python src/04_train_logistic_regression.py
"""

import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from eval_utils import evaluate_model, load_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    print("Loading features from Day 3...")
    X_train, X_test, y_train, y_test = load_features()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # accounts for the 60/40 class imbalance
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    evaluate_model("Logistic Regression", y_test, y_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "logistic_regression.joblib")
    print(f"\nSaved model -> {MODELS_DIR / 'logistic_regression.joblib'}")


if __name__ == "__main__":
    main()