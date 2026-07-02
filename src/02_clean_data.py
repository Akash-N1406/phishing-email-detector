"""
02_clean_data.py

Cleans the raw phishing email dataset and extracts metadata features.

Steps:
1. Load raw CSV, drop missing text rows.
2. Inspect and cap extreme outlier-length emails (data quality issue).
3. Extract metadata features BEFORE cleaning (URLs, punctuation, caps ratio,
   digit count) -- these signals often get destroyed by cleaning, so we grab
   them from the raw text first.
4. Clean text: strip HTML, lowercase, remove punctuation/numbers, remove
   stopwords.
5. Save cleaned dataset + metadata features to data/processed/.

Run from project root:
    python src/02_clean_data.py
"""

import re
import string
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

import warnings
import nltk
from nltk.corpus import stopwords
from bs4 import MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ---- Config ----
RAW_DATA_PATH = Path("../data/raw/Phishing_Email.csv")
PROCESSED_DATA_PATH = Path("../data/processed/cleaned_emails.csv")

TEXT_COL = "Email Text"
LABEL_COL = "Email Type"

# Cap on email length in characters. Anything beyond this is almost certainly
# junk (encoded attachments, corrupted rows) rather than real email body text.
# 20,000 chars ~ a genuinely very long email; adjust after inspecting outliers.
MAX_EMAIL_LENGTH = 20_000

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")


def ensure_nltk_data():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def load_and_filter(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=[TEXT_COL]).copy()
    print(f"Dropped {before - len(df)} rows with missing '{TEXT_COL}'")

    df["Email Type"] = df[LABEL_COL].map(
        {"Safe Email": 0, "Phishing Email": 1}
    )
    if df["Email Type"].isnull().any():
        unmapped = df[df["Email Type"].isnull()][LABEL_COL].unique()
        raise ValueError(f"Unmapped label values found: {unmapped}")

    return df


def inspect_outliers(df: pd.DataFrame, n: int = 5):
    lengths = df[TEXT_COL].astype(str).str.len()
    print("\nTop", n, "longest emails (char count):")
    top = df.loc[lengths.nlargest(n).index]
    for idx, row in top.iterrows():
        preview = str(row[TEXT_COL])[:150].replace("\n", " ")
        print(f"  len={len(str(row[TEXT_COL])):>10}  label={row[LABEL_COL]:<15}  preview: {preview}...")


def extract_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structural/metadata features from RAW text, before cleaning."""
    text = df[TEXT_COL].astype(str)

    df["char_count"] = text.str.len()
    df["word_count"] = text.str.split().str.len()
    df["url_count"] = text.apply(lambda t: len(URL_PATTERN.findall(t)))
    df["has_url"] = (df["url_count"] > 0).astype(int)
    df["exclamation_count"] = text.str.count("!")
    df["digit_count"] = text.apply(lambda t: sum(c.isdigit() for c in t))

    def caps_ratio(t: str) -> float:
        letters = [c for c in t if c.isalpha()]
        if not letters:
            return 0.0
        return sum(c.isupper() for c in letters) / len(letters)

    df["caps_ratio"] = text.apply(caps_ratio)

    return df


def clean_text(text: str, stop_words: set) -> str:
    # Strip HTML
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    # Lowercase
    text = text.lower()
    # Remove URLs (already captured as a metadata feature above)
    text = URL_PATTERN.sub(" ", text)
    # Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)


def main():
    ensure_nltk_data()
    stop_words = set(stopwords.words("english"))

    print(f"Loading {RAW_DATA_PATH} ...")
    df = load_and_filter(RAW_DATA_PATH)

    inspect_outliers(df)

    n_outliers = (df[TEXT_COL].astype(str).str.len() > MAX_EMAIL_LENGTH).sum()
    print(f"\n{n_outliers} rows exceed {MAX_EMAIL_LENGTH} chars -- capping (truncating) these.")
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.slice(0, MAX_EMAIL_LENGTH)

    print("\nExtracting metadata features from raw text...")
    df = extract_metadata_features(df)

    print("Cleaning text (HTML strip, lowercase, punctuation/stopword removal)...")
    df["cleaned_text"] = df[TEXT_COL].apply(lambda t: clean_text(t, stop_words))

    # Drop rows that became empty after cleaning (rare, but possible)
    before = len(df)
    df = df[df["cleaned_text"].str.strip().str.len() > 0]
    print(f"Dropped {before - len(df)} rows that were empty after cleaning")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_cols = [
        "cleaned_text", "Email Type", "char_count", "word_count",
        "url_count", "has_url", "exclamation_count", "digit_count", "caps_ratio",
    ]
    df[out_cols].to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\nSaved cleaned dataset -> {PROCESSED_DATA_PATH}")
    print("Final shape:", df[out_cols].shape)
    print("\nClass balance after cleaning:")
    print(df["Email Type"].value_counts())


if __name__ == "__main__":
    main()