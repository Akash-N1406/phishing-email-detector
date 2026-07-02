# AI-Driven Phishing Email Detection Using NLP

Machine-learning system that detects phishing emails using text analysis and metadata
features. Built from scratch: data preprocessing, feature extraction, model training,
and evaluation across four classifier families.

## Objective

To design and implement a machine-learning system that automatically detects phishing
emails using text analysis and metadata features, applying data preprocessing, feature
extraction, model training, and evaluation.

## Project Status

🚧 In progress — Week 1 of 3 (Data & Feature Engineering)

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