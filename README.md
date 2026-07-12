# AI-Driven Phishing Email Detection Using NLP

Machine-learning system that detects phishing emails using text analysis and metadata
features, built from scratch: data preprocessing, feature extraction, model training,
evaluation across four classifier families, and a live web demo.

**Internship project — Indian Institute of Computing and Technology (IICT)**

## Project Status

✅ Complete — all deliverables from the project brief have been built.

## Results Summary

Four classifiers trained and evaluated on the same held-out test set (3,726 emails):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Neural Network** | **97.18%** | 94.84% | 98.15% | **96.47%** |
| Logistic Regression | 96.78% | 94.26% | 97.74% | 95.97% |
| Naive Bayes | 95.87% | 94.49% | 95.01% | 94.75% |
| Random Forest | 95.81% | 91.33% | **98.70%** | 94.87% |

- **Neural Network** — best overall performance.
- **Logistic Regression** — best interpretability (fully explainable coefficients), within 0.4 points of the NN.
- **Random Forest** — highest recall, catches the most phishing emails at the cost of more false alarms.
- **Naive Bayes** — trained on text alone (no metadata) and still lands within ~1 point of Logistic Regression.

Full analysis, feature importance, and error analysis: [`reports/limitations.md`](reports/limitations.md) and [`reports/Phishing_Detection_Comparative_Analysis_Report.docx`](reports/).

## Stack

- **Language:** Python 3.10+
- **ML:** scikit-learn (Logistic Regression, Random Forest, Naive Bayes), TensorFlow/Keras (Neural Network)
- **NLP:** NLTK, TF-IDF
- **Data:** pandas, numpy, scipy (sparse matrices)
- **Viz:** matplotlib, seaborn
- **Web demo:** Django

## Project Structure

```
phishing-email-detector/
├── data/
│   ├── raw/                 # original Kaggle dataset
│   └── processed/           # cleaned data + train/test feature matrices
├── notebooks/
│   └── phishing_detection_walkthrough.ipynb   # end-to-end presentation notebook
├── src/                      # the actual pipeline, run in order:
│   ├── 01_explore_data.py
│   ├── 02_clean_data.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_logistic_regression.py
│   ├── 05_train_naive_bayes.py
│   ├── 06_train_random_forest.py
│   ├── 07_train_neural_network.py
│   ├── 08_feature_importance_error_analysis.py
│   └── eval_utils.py         # shared evaluation/plotting utilities
├── webapp/                    # Django live-demo (see below)
├── models/                    # trained model artifacts (committed for clone-and-run)
├── reports/
│   ├── figures/               # confusion matrices, feature importance plots
│   ├── model_comparison.json  # machine-readable results, updated by each training script
│   ├── limitations.md         # documented findings & known limitations
│   └── Phishing_Detection_Comparative_Analysis_Report.docx
├── requirements.txt
└── README.md
```

## Methodology

| Phase | Description |
|---|---|
| Data Collection | Kaggle "Phishing Email Detection" dataset (18,650 emails, Enron corpus + known phishing samples). |
| Data Cleaning | HTML stripping, lowercasing, punctuation/stopword removal; outlier-length rows capped. |
| Metadata Extraction | 7 structural features (URL count, caps ratio, exclamation count, etc.) extracted from raw text before cleaning destroys those signals. |
| Feature Engineering | TF-IDF (5,000 features, unigrams+bigrams) combined with scaled metadata into a 5,007-dim matrix. Train/test split performed *before* fitting, to avoid data leakage. |
| Model Development | 4 classifiers trained: Logistic Regression, Random Forest, Naive Bayes (TF-IDF only — non-negativity requirement), Neural Network (Keras). |
| Evaluation | Accuracy, precision, recall, F1, confusion matrices for every model. |
| Analysis | Feature importance (LR coefficients, RF importances) + error analysis on misclassifications. |
| Deployment | Django web demo for live email classification. |

## Setup: ML Pipeline

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords

kaggle datasets download -d cyber-cop/phishing-email-detection -p data/raw
cd data/raw && unzip *.zip && cd ../..

# Run the pipeline in order:
python src/01_explore_data.py
python src/02_clean_data.py
python src/03_feature_engineering.py
python src/04_train_logistic_regression.py
python src/05_train_naive_bayes.py
python src/06_train_random_forest.py
python src/07_train_neural_network.py
python src/08_feature_importance_error_analysis.py
```

Each training script prints metrics, saves a confusion matrix to `reports/figures/`,
saves the trained model to `models/`, and appends its results to
`reports/model_comparison.json`.

## Setup: Notebook Walkthrough

```bash
jupyter notebook notebooks/phishing_detection_walkthrough.ipynb
```

Presents the full pipeline end-to-end (data → cleaning → features → all 4 models →
comparison → feature importance → error analysis → conclusion) by importing the
real `src/` modules rather than duplicating logic — this notebook is a presentation
layer over the actual pipeline, not a separate copy of it.

## Setup: Live Web Demo (Django)

Requires the ML pipeline to have been run at least once (needs `models/*.joblib`).

```bash
cd webapp
python manage.py runserver
```

Open `http://127.0.0.1:8000/`, paste in email text, and get a classification with
confidence scores, the specific words that influenced the prediction, and detected
structural signals (URL count, exclamation count, etc.).

Serves the Logistic Regression model (fast startup, fully interpretable) rather than
the higher-accuracy Neural Network, since responsiveness matters more than the ~0.4
point accuracy gap for a live demo. See `webapp/detector/ml.py` to swap models.

**Scope note:** intentionally a single view, no database, no auth — this is a
focused prediction demo, not a full application. See `reports/limitations.md` for
the reasoning behind this and other design choices.

## Key Findings

- Phishing-indicative language matches the project brief's hypothesis directly:
  `click`, `remove`, `free`, `money`, `site` — classic urgency/sales/call-to-action terms.
- Engineered metadata features (`exclamation_count`, `caps_ratio`) rank among the most
  predictive features overall, on par with top TF-IDF words.
- The Neural Network's false negatives cluster around **content obfuscation** —
  phishing emails padded with unrelated text to dodge keyword-based detection — a
  structural blind spot of TF-IDF-based models that context-aware embeddings
  (transformers) could address.
- The "safe" class is shaped by this dataset's specific source (Enron corporate +
  academic email); generalization to other email sources is a documented caveat,
  not an assumption.

Full discussion: [`reports/limitations.md`](reports/limitations.md).

## Learning Outcomes

- How NLP supports cybersecurity threat detection
- Hands-on text classification and model evaluation across multiple algorithm families
- Data leakage prevention in feature engineering pipelines
- Reusable, DRY pipeline design (shared modules across scripts, notebook, and web demo)
- Ethical AI practices: documenting limitations and generalization boundaries rather than overstating results