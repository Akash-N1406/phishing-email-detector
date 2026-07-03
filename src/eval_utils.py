"""
eval_utils.py

Shared evaluation utilities used by every model training script (04-07).
Keeps the metrics computation, confusion matrix plotting, and results
tracking consistent across Logistic Regression, Random Forest, Naive Bayes,
and the Neural Network -- so the Week 2 comparison table is apples-to-apples.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# Resolve paths relative to the project root (parent of src/), not the
# current working directory -- this way it doesn't matter whether you run
# scripts from the project root or from inside src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_PATH = REPORTS_DIR / "model_comparison.json"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def evaluate_model(model_name: str, y_true, y_pred, save: bool = True) -> dict:
    """Compute standard classification metrics and optionally persist them."""
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {model_name}")
    print(f"{'=' * 50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k.capitalize():<10}: {v}")

    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=["Safe", "Phishing"]))

    if save:
        save_results(metrics)
        plot_confusion_matrix(model_name, y_true, y_pred)

    return metrics


def plot_confusion_matrix(model_name: str, y_true, y_pred):
    """Save a confusion matrix heatmap to reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Safe", "Phishing"], yticklabels=["Safe", "Phishing"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    out_path = FIGURES_DIR / f"confusion_matrix_{safe_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {out_path}")


def save_results(metrics: dict):
    """Append/update this model's results in the shared comparison file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[metrics["model"]] = metrics

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Updated {RESULTS_PATH}")


def load_features():
    """Convenience loader for the Day 3 feature matrices, used by every model script."""
    from scipy import sparse

    X_train = sparse.load_npz(DATA_PROCESSED_DIR / "X_train.npz")
    X_test = sparse.load_npz(DATA_PROCESSED_DIR / "X_test.npz")
    y_train = np.load(DATA_PROCESSED_DIR / "y_train.npy")
    y_test = np.load(DATA_PROCESSED_DIR / "y_test.npy")

    return X_train, X_test, y_train, y_test