"""
07_train_neural_network.py

Trains a simple feedforward Neural Network (Keras) on the full combined
feature set (TF-IDF + metadata) from Day 3.

Key differences from the previous three models:
1. Keras needs DENSE arrays, not the sparse matrices we've been using.
   Converting (14903, 5007) sparse -> dense costs ~285MB in memory, which is
   fine for a one-off conversion but is exactly why we kept everything sparse
   until now (RF/LR/NB all accept sparse input directly and never pay this
   cost).
2. Class imbalance is handled differently -- Keras doesn't have a
   `class_weight="balanced"` string shortcut like scikit-learn. We compute
   the weights explicitly via sklearn's `compute_class_weight` and pass them
   into `model.fit(..., class_weight=...)`.
3. EarlyStopping monitors validation loss so we don't overfit past the point
   where the model stops actually improving -- important since a 5007-input
   network has plenty of capacity to memorize a ~15K-row training set.

Run from project root or from src/ (paths auto-resolve):
    python src/07_train_neural_network.py
"""

import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from eval_utils import evaluate_model, load_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

EPOCHS = 30
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),  # binary output: phishing probability
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model


def main():
    print("Loading features from Day 3...")
    X_train, X_test, y_train, y_test = load_features()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    print("\nConverting sparse matrices to dense (required by Keras)...")
    X_train_dense = X_train.toarray().astype("float32")
    X_test_dense = X_test.toarray().astype("float32")
    print(f"Dense train matrix memory: {X_train_dense.nbytes / 1e6:.1f} MB")

    # Compute class weights the same way "balanced" does in scikit-learn,
    # since Keras has no built-in string shortcut for this.
    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight = {0: class_weights_arr[0], 1: class_weights_arr[1]}
    print(f"Class weights: {class_weight}")

    print("\nBuilding model...")
    model = build_model(input_dim=X_train_dense.shape[1])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    print("\nTraining...")
    history = model.fit(
        X_train_dense, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    print(f"\nStopped after {len(history.history['loss'])} epochs (early stopping).")

    # Predict probabilities, threshold at 0.5 for hard labels
    y_pred_proba = model.predict(X_test_dense, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    evaluate_model("Neural Network", y_test, y_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "neural_network.keras")
    print(f"\nSaved model -> {MODELS_DIR / 'neural_network.keras'}")


if __name__ == "__main__":
    main()