"""
detector/ml.py

Loads the trained model artifacts (produced by src/03_feature_engineering.py
and src/04_train_logistic_regression.py) once at server startup, and exposes
a single predict() function the Django view calls.

Reuses clean_text() and extract_metadata_features() directly from
src/02_clean_data.py via importlib -- same reasoning as the Week 3 notebook:
the preprocessing logic lives in ONE place (src/), not duplicated here. If
you ever change how cleaning works, this demo automatically stays in sync
as long as you rerun 02-03 and retrain.

Uses Logistic Regression rather than the Neural Network for serving:
it's the second-best model overall (96.78% vs 97.18% accuracy), but loads
almost instantly and has no TensorFlow startup cost -- better fit for a
responsive web demo. Swap MODEL_FILE below if you'd rather serve the NN.
"""

import importlib.util
import warnings
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse

warnings.filterwarnings("ignore")

# webapp/detector/ml.py -> webapp/detector -> webapp -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

MODEL_FILE = "logistic_regression.joblib"
METADATA_COLS = [
    "char_count", "word_count", "url_count",
    "has_url", "exclamation_count", "digit_count", "caps_ratio",
]
MAX_EMAIL_LENGTH = 20_000  # must match src/02_clean_data.py

_artifacts = {}


def _load_clean_module():
    """Load src/02_clean_data.py as a module (filename starts with a digit,
    so a normal `import` statement can't reach it)."""
    spec = importlib.util.spec_from_file_location("clean_data_mod", SRC_DIR / "02_clean_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_artifacts():
    """Called once from DetectorConfig.ready() at server startup."""
    import nltk
    from nltk.corpus import stopwords

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    _artifacts["clean_mod"] = _load_clean_module()
    _artifacts["stop_words"] = stop_words
    _artifacts["vectorizer"] = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    _artifacts["scaler"] = joblib.load(MODELS_DIR / "metadata_scaler.joblib")
    _artifacts["model"] = joblib.load(MODELS_DIR / MODEL_FILE)
    _artifacts["feature_names"] = joblib.load(MODELS_DIR / "feature_names.joblib")
    print(f"[detector.ml] Loaded model artifacts from {MODELS_DIR}")


def _extract_single_metadata(raw_text: str) -> pd.DataFrame:
    """Reuse the exact same metadata extraction logic as training, on a
    single-row dataframe so extract_metadata_features() (built for the full
    dataset) works unmodified here."""
    clean_mod = _artifacts["clean_mod"]
    df = pd.DataFrame({clean_mod.TEXT_COL: [raw_text]})
    df = clean_mod.extract_metadata_features(df)
    return df[METADATA_COLS]


def predict(raw_text: str) -> dict:
    """Run the full inference pipeline on one raw email string:
    metadata extraction -> cleaning -> TF-IDF -> scaling -> combine -> predict.
    Mirrors src/03_feature_engineering.py's transform steps exactly, using
    .transform() (never .fit()) since the vectorizer/scaler were already
    fitted on training data.
    """
    if "model" not in _artifacts:
        load_artifacts()

    clean_mod = _artifacts["clean_mod"]
    stop_words = _artifacts["stop_words"]
    vectorizer = _artifacts["vectorizer"]
    scaler = _artifacts["scaler"]
    model = _artifacts["model"]
    feature_names = _artifacts["feature_names"]

    # Metadata must be extracted from the RAW text, same as training
    meta_df = _extract_single_metadata(raw_text)

    truncated = raw_text[:MAX_EMAIL_LENGTH]
    cleaned = clean_mod.clean_text(truncated, stop_words)

    X_tfidf = vectorizer.transform([cleaned])
    X_meta = scaler.transform(meta_df)
    X_combined = sparse.hstack([X_tfidf, sparse.csr_matrix(X_meta)]).tocsr()

    proba = model.predict_proba(X_combined)[0]
    pred = model.predict(X_combined)[0]

    # Show which words in THIS email pushed the decision, using the
    # non-zero TF-IDF columns and their Logistic Regression coefficients.
    coefs = model.coef_[0]
    nonzero_idx = X_tfidf.nonzero()[1]
    contributions = [(feature_names[i], round(float(coefs[i]), 3)) for i in nonzero_idx]
    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)

    return {
        "label": "Phishing" if pred == 1 else "Safe",
        "phishing_probability": round(float(proba[1]) * 100, 1),
        "safe_probability": round(float(proba[0]) * 100, 1),
        "top_contributions": contributions[:8],
        "metadata": meta_df.iloc[0].to_dict(),
        "cleaned_preview": cleaned[:200],
    }
