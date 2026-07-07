# AI-Driven Phishing Email Detection Using NLP

Machine-learning system that detects phishing emails using text analysis and metadata
features. Built from scratch: data preprocessing, feature extraction, model training,
and evaluation across four classifier families.

## Objective

To design and implement a machine-learning system that automatically detects phishing
emails using text analysis and metadata features, applying data preprocessing, feature
extraction, model training, and evaluation.

## Project Status

🚧 In progress — Week 2 of 3 (Model Training & Comparison)

Completed:
- Day 1: Repo scaffold + GitHub setup
- Day 2: Data cleaning + metadata feature extraction
- Day 3: TF-IDF + metadata feature engineering, train/test split
- Day 4: Logistic Regression baseline (96.78% acc, 95.97% F1)
- Day 5: Naive Bayes on TF-IDF (95.87% acc, 94.75% F1)

Pending for Week 3:
- Consolidated `notebooks/phishing_detection_walkthrough.ipynb` — walks through
  the full pipeline (data → cleaning → features → each model → comparison →
  conclusion) by importing the existing `src/` modules rather than duplicating
  logic. This is the "Python notebook with complete code and documentation"
  deliverable from the original brief.
- Comparative analysis report
- Presentation slides

## Stack

- **Language:** Python 3.10+
- **ML:** scikit-learn (Logistic Regression, Random Forest, Naive Bayes), TensorFlow/Keras (Neural Network)
- **NLP:** NLTK, spaCy, TF-IDF
- **Data:** pandas, numpy
- **Viz:** matplotlib, seaborn
- **Deployment (optional):** Flask / Streamlit

## Project Structure

```
phishing-email-detector/
├── data/
│   ├── raw/            # original, untouched datasets
│   └── processed/      # cleaned/feature-engineered data
├── notebooks/           # exploratory analysis
├── src/                  # reusable scripts (cleaning, features, training, eval)
├── models/               # saved trained models
├── reports/              # comparative analysis, figures, slides
├── requirements.txt
└── README.md
```

## Methodology

| Phase | Description |
|---|---|
| Data Collection | Gather phishing and legitimate email samples (Kaggle). |
| Data Cleaning | Remove HTML tags, punctuation, and stopwords. |
| Feature Engineering | Apply TF-IDF, word embeddings, and metadata extraction. |
| Model Development | Train multiple classifiers and tune hyperparameters. |
| Evaluation | Compare models and interpret results. |
| Deployment (Optional) | Simple web interface for live email classification. |

## Dataset

- Primary: [Phishing Email Detection (Cyber Cop, Kaggle)](https://www.kaggle.com/datasets/cyber-cop/phishing-email-detection)
- Robustness check (Week 3): synthetic LLM-generated phishing samples for generalization testing

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Results

_To be filled in as models are trained (Week 2)._

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Random Forest | | | | |
| Naive Bayes | | | | |
| Neural Network | | | | |

## Learning Outcomes

- How NLP supports cybersecurity threat detection
- Hands-on text classification and model evaluation
- Ethical AI practices for digital threat detection