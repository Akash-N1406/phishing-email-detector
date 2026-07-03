"""
03_feature_engineering.py

Builds the feature matrix that Week 2's models will train on:
1. Load cleaned data (text + metadata features) from Day 2.
2. Train/test split FIRST (stratified) -- critical to avoid data leakage.
   The TF-IDF vectorizer and metadata scaler must only ever see training data
   during .fit(); if you fit on the full dataset before splitting, test-set
   information leaks into the vocabulary/scaling and your evaluation numbers
   in Week 2 will be optimistically biased.
3. Fit TF-IDF on training text only, transform both train and test.
4. Fit StandardScaler on training metadata only, transform both train and test.
5. Combine TF-IDF (sparse) + scaled metadata (dense) into one sparse matrix.
6. Save everything needed for Week 2: feature matrices, labels, and the
   fitted vectorizer/scaler (needed later for the optional live-demo deployment).

Run from project root:
    python src/03_feature_engineering.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- Config ----
PROCESSED_DATA_PATH = Path("../data/processed/cleaned_emails.csv")
OUTPUT_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")

TEXT_COL = "cleaned_text"
LABEL_COL = "Email Type"
METADATA_COLS = [
    "char_count", "word_count", "url_count",
    "has_url", "exclamation_count", "digit_count", "caps_ratio",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)   # unigrams + bigrams -- catches phrases like "act now"
TFIDF_MIN_DF = 5             # ignore terms that appear in fewer than 5 emails


def main():
    print(f"Loading {PROCESSED_DATA_PATH} ...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("")

    X_text = df[TEXT_COL]
    X_meta = df[METADATA_COLS]
    y = df[LABEL_COL]

    # ---- Split FIRST, before any fitting, to avoid leakage ----
    X_text_train, X_text_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
        X_text, X_meta, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # preserves the 60/40 class ratio in both splits
    )

    print(f"Train size: {len(y_train)}  |  Test size: {len(y_test)}")
    print("Train class balance:")
    print(y_train.value_counts(normalize=True).round(3))
    print("Test class balance:")
    print(y_test.value_counts(normalize=True).round(3))

    # ---- TF-IDF: fit on train only ----
    print("\nFitting TF-IDF vectorizer on training text...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True,  # dampens the effect of very high term frequencies
    )
    X_tfidf_train = vectorizer.fit_transform(X_text_train)
    X_tfidf_test = vectorizer.transform(X_text_test)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"TF-IDF train matrix shape: {X_tfidf_train.shape}")

    # ---- Metadata scaling: fit on train only ----
    print("\nScaling metadata features...")
    scaler = StandardScaler()
    X_meta_train_scaled = scaler.fit_transform(X_meta_train)
    X_meta_test_scaled = scaler.transform(X_meta_test)

    # ---- Combine TF-IDF (sparse) + metadata (dense -> sparse) ----
    X_train_combined = sparse.hstack([
        X_tfidf_train, sparse.csr_matrix(X_meta_train_scaled)
    ]).tocsr()
    X_test_combined = sparse.hstack([
        X_tfidf_test, sparse.csr_matrix(X_meta_test_scaled)
    ]).tocsr()

    print(f"\nFinal combined train matrix shape: {X_train_combined.shape}")
    print(f"Final combined test matrix shape:  {X_test_combined.shape}")
    print(f"(TF-IDF features: {X_tfidf_train.shape[1]}  +  metadata features: {len(METADATA_COLS)})")

    # ---- Save everything ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(OUTPUT_DIR / "X_train.npz", X_train_combined)
    sparse.save_npz(OUTPUT_DIR / "X_test.npz", X_test_combined)
    np.save(OUTPUT_DIR / "y_train.npy", y_train.to_numpy())
    np.save(OUTPUT_DIR / "y_test.npy", y_test.to_numpy())

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(scaler, MODELS_DIR / "metadata_scaler.joblib")

    # Save feature names for later interpretation (e.g. feature importance in Week 3)
    feature_names = list(vectorizer.get_feature_names_out()) + METADATA_COLS
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'X_train.npz'}")
    print(f"  {OUTPUT_DIR / 'X_test.npz'}")
    print(f"  {OUTPUT_DIR / 'y_train.npy'}")
    print(f"  {OUTPUT_DIR / 'y_test.npy'}")
    print(f"  {MODELS_DIR / 'tfidf_vectorizer.joblib'}")
    print(f"  {MODELS_DIR / 'metadata_scaler.joblib'}")
    print(f"  {MODELS_DIR / 'feature_names.joblib'}")
    print("\nReady for Week 2 model training.")


if __name__ == "__main__":
    main()