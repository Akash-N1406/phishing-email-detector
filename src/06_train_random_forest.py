"""
06_train_random_forest.py

Trains a Random Forest classifier on the full combined feature set
(TF-IDF + metadata) from Day 3.

Unlike Naive Bayes, Random Forest has no non-negativity requirement, so it
can use the complete 5007-column matrix (5000 TF-IDF + 7 metadata features).
This means RF gets the same "unfair advantage" Logistic Regression had over
Naive Bayes -- a genuinely fair three-way comparison so far is:
    - Logistic Regression: TF-IDF + metadata
    - Random Forest:       TF-IDF + metadata
    - Naive Bayes:         TF-IDF only (by necessity)

Includes an optional (commented-out) RandomizedSearchCV block for
hyperparameter tuning -- left off by default because it takes noticeably
longer to run; the manual hyperparameters below are already sensible
defaults for this dataset size.

Run from project root or from src/ (paths auto-resolve):
    python src/06_train_random_forest.py
"""

import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV  # uncomment for tuning

from eval_utils import evaluate_model, load_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    print("Loading features from Day 3...")
    X_train, X_test, y_train, y_test = load_features()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300,          # more trees = more stable predictions, diminishing returns past ~300 here
        max_depth=50,              # caps tree depth to reduce overfitting on the 5000-dim TF-IDF space
        min_samples_leaf=2,        # requires at least 2 samples per leaf, smooths out noisy single-sample splits
        class_weight="balanced",   # same imbalance handling as Logistic Regression
        n_jobs=-1,                 # use all available CPU cores
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ---- Optional: hyperparameter tuning (slower, run separately if you want it) ----
    # param_dist = {
    #     "n_estimators": [200, 300, 500],
    #     "max_depth": [30, 50, None],
    #     "min_samples_leaf": [1, 2, 4],
    # }
    # search = RandomizedSearchCV(
    #     RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    #     param_distributions=param_dist,
    #     n_iter=8, cv=3, scoring="f1", random_state=42, n_jobs=-1, verbose=2,
    # )
    # search.fit(X_train, y_train)
    # model = search.best_estimator_
    # print("Best params:", search.best_params_)

    y_pred = model.predict(X_test)
    evaluate_model("Random Forest", y_test, y_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "random_forest.joblib")
    print(f"\nSaved model -> {MODELS_DIR / 'random_forest.joblib'}")


if __name__ == "__main__":
    main()