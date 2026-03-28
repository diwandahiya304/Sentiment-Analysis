# 🎭 Sentiment Analysis — NLP Pipeline

> **Binary sentiment classification of 20,000+ customer reviews using spaCy, TF-IDF, and Naive Bayes.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.x-09A3D5?logo=spacy)](https://spacy.io/)

---

## 📌 Project Overview

End-to-end NLP sentiment analysis pipeline on the **IMDB Movie Reviews** dataset (50,000 reviews) from Kaggle. Reviews are classified as **Positive** or **Negative** using classical NLP techniques.

**Dataset:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
_(Only 20,000 reviews are used for efficiency — easily configurable.)_

---

## 📂 Repository Structure

```
Sentiment-Analysis/
│
├── data/
│   └── IMDB Dataset.csv          # ⚠️ Download from Kaggle (not tracked by Git)
│
├── src/
│   ├── preprocess.py             # spaCy text cleaning pipeline
│   ├── features.py               # TF-IDF feature engineering
│   ├── train.py                  # Naive Bayes model training + CLI inference
│   └── evaluate.py               # Metrics, confusion matrix & top features plot
│
├── outputs/
│   ├── confusion_matrix.png      # Confusion matrix heatmap
│   ├── top_features.png          # Most informative TF-IDF words per class
│   ├── inference_demo.png        # Live inference screenshot
│   └── results.csv               # Accuracy, F1, Precision, Recall
│
├── models/
│   ├── nb_model.joblib           # Saved Naive Bayes model
│   └── tfidf_vectorizer.joblib   # Saved TF-IDF vectorizer
│
├── notebooks/
│   └── sentiment_analysis.ipynb  # Full end-to-end notebook
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Column      | Description                     |
|-------------|---------------------------------|
| `review`    | Raw movie review text           |
| `sentiment` | Label: `positive` or `negative` |

- **Total rows:** 50,000 (20,000 used)
- **Class balance:** 50% positive / 50% negative

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/diwandahiya304/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Download the dataset
Download `IMDB Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it at:
```
data/IMDB Dataset.csv
```

---

## 🚀 Usage

### Option A — Jupyter Notebook *(recommended)*
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

### Option B — Full training pipeline
```bash
python src/train.py
```

### Option C — Predict on new text
```bash
python src/train.py --predict "This movie was absolutely fantastic!"
```

---

## 🧠 Pipeline

```
Raw CSV (IMDB Dataset.csv)
  │
  ▼
src/preprocess.py  ──  spaCy pipeline
  ├─ Lowercase
  ├─ Remove HTML tags & URLs
  ├─ Tokenise with spaCy (en_core_web_sm)
  ├─ Remove stopwords & punctuation
  └─ Lemmatise tokens
  │
  ▼
src/features.py  ──  TF-IDF Vectorization
  ├─ Unigrams + Bigrams
  ├─ max_features = 20,000
  └─ sublinear_tf = True
  │
  ▼
src/train.py  ──  Naive Bayes Classifier
  ├─ MultinomialNB (scikit-learn)
  ├─ 80/20 stratified train–test split
  └─ Save model → models/nb_model.joblib
  │
  ▼
src/evaluate.py  ──  Metrics & Visualizations
  ├─ Accuracy, F1, Precision, Recall
  ├─ Confusion matrix → outputs/confusion_matrix.png
  └─ Top TF-IDF features → outputs/top_features.png
```

---

## 📈 Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.86  |
| F1 Score  | 0.86  |
| Precision | 0.86  |
| Recall    | 0.86  |

---

## 🔍 Live Inference Demo

Run predictions on any custom review text directly from the terminal:

```bash
python src/train.py --predict "This movie was absolutely brilliant!"
python src/train.py --predict "Worst film I have ever seen. Complete waste of time."
```

![Inference Demo](outputs/inference_demo.png)

> **Positive review** detected with **89.7% confidence**  
> **Negative review** detected with **99.3% confidence**

---

## 🔑 Key Features

- **spaCy NLP pipeline** — efficient tokenization, lemmatization, and stopword removal on 20,000+ reviews
- **Advanced text cleaning** — strips HTML tags, URLs, and special characters to reduce noise
- **TF-IDF vectorization** — extracts contextual unigram + bigram features (20k vocab, `sublinear_tf`)
- **80/20 stratified split** — ensures balanced class distribution across train and test sets
- **Naive Bayes classifier** — fast, interpretable, and scalable sentiment classification with scikit-learn

---

## 👤 Author

**Diwan Dahiya**  
[GitHub](https://github.com/diwandahiya304)
