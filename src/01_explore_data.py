"""
01_explore_data.py

First look at the raw phishing email dataset:
- Load the CSV
- Check shape, columns, dtypes
- Check class balance (phishing vs legitimate)
- Check for missing values / duplicates
- Peek at a few sample rows from each class

Run from project root:
    python src/01_explore_data.py
"""

import pandas as pd
from pathlib import Path

# ---- Config ----
# Adjust this once you know the actual filename after unzipping the Kaggle dataset.
RAW_DATA_PATH = Path("../data/raw/Phishing_Email.csv")

# Common column name variants across phishing datasets - adjust after first run
# once you've seen the real column names printed below.
TEXT_COL_CANDIDATES = ["Email Text", "text", "body", "Text"]
LABEL_COL_CANDIDATES = ["Email Type", "label", "Category", "class"]


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first candidate column name that actually exists in df."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"None of {candidates} found in columns: {list(df.columns)}. "
        "Update TEXT_COL_CANDIDATES / LABEL_COL_CANDIDATES in this script."
    )


def main():
    if not RAW_DATA_PATH.exists():
        print(f"[!] File not found: {RAW_DATA_PATH}")
        print("    Download it first, e.g.:")
        print("    kaggle datasets download -d cyber-cop/phishing-email-detection -p data/raw")
        print("    unzip data/raw/phishing-email-detection.zip -d data/raw")
        return

    df = pd.read_csv(RAW_DATA_PATH)

    print("=" * 60)
    print("SHAPE:", df.shape)
    print("=" * 60)
    print("\nCOLUMNS:")
    print(df.columns.tolist())

    print("\nDTYPES:")
    print(df.dtypes)

    print("\nFIRST 3 ROWS:")
    print(df.head(3))

    print("\nMISSING VALUES PER COLUMN:")
    print(df.isnull().sum())

    print("\nDUPLICATE ROWS:", df.duplicated().sum())

    # Try to auto-detect the text/label columns; fall back gracefully if not found
    try:
        text_col = find_column(df, TEXT_COL_CANDIDATES)
        label_col = find_column(df, LABEL_COL_CANDIDATES)

        print(f"\nDetected text column:  '{text_col}'")
        print(f"Detected label column: '{label_col}'")

        print("\nCLASS BALANCE:")
        print(df[label_col].value_counts())
        print(df[label_col].value_counts(normalize=True).round(3))

        print("\nSAMPLE PHISHING EMAIL:")
        phishing_sample = df[df[label_col] != df[label_col].mode()[0]]
        if not phishing_sample.empty:
            print(phishing_sample[text_col].iloc[0][:500])

        print("\nEMAIL LENGTH STATS (characters):")
        print(df[text_col].astype(str).str.len().describe())

    except ValueError as e:
        print(f"\n[!] {e}")
        print("    Inspect the COLUMNS list above and update the candidate lists.")


if __name__ == "__main__":
    main()