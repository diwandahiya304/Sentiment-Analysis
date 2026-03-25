# 🏨 Hotel Review Sentiment Analysis

> **Binary sentiment classification of 515K+ European hotel reviews using Classical ML, Deep Learning (LSTM), and Transformers (BERT).**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

This project performs **sentiment analysis** on the [Hotel Reviews dataset](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) — 515,000+ reviews scraped from Booking.com. Each review contains a separate positive and negative comment along with a numeric reviewer score (0–10).

**Goal:** Predict whether a hotel review is **Positive** (score ≥ 7) or **Negative** (score < 7).

---

## 📂 Repository Structure

```
Sentiment-Analysis/
│
├── notebooks/
│   └── hotel_sentiment_analysis.ipynb   # Main end-to-end notebook
│
├── src/
│   ├── preprocess.py    # Text cleaning & label engineering
│   ├── features.py      # TF-IDF, BoW, meta-feature extraction
│   ├── train.py         # CLI training script (all models)
│   ├── evaluate.py      # Metrics, confusion matrix, ROC curve
│   └── predict.py       # Inference on new reviews
│
├── data/                # Place your CSV here (not tracked by Git)
│   └── .gitkeep
│
├── models/              # Saved model artefacts (not tracked by Git)
│   └── .gitkeep
│
├── outputs/             # Plots, charts, result JSONs (not tracked by Git)
│   └── .gitkeep
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset Columns

| Column | Description |
|--------|-------------|
| `Hotel_Name` | Hotel name |
| `Hotel_Address` | Full address |
| `Average_Score` | Hotel's overall Booking.com score |
| `Reviewer_Nationality` | Reviewer's country |
| `Negative_Review` | Free-text negative comment |
| `Positive_Review` | Free-text positive comment |
| `Review_Total_Negative_Word_Counts` | Word count of negative review |
| `Review_Total_Positive_Word_Counts` | Word count of positive review |
| `Reviewer_Score` | ⭐ Numeric score 0–10 (target variable basis) |
| `Total_Number_of_Reviews` | Total reviews for the hotel |
| `Total_Number_of_Reviews_Reviewer_Has_Given` | Reviewer experience |
| `Tags` | Traveller type, room type, trip duration |
| `days_since_review` | Days since the review was posted |
| `lat`, `lng` | Hotel geographic coordinates |

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
```

### 4. Add the dataset
Download the CSV from [Kaggle](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) and place it at:
```
data/Hotel_Reviews.csv
```

---

## 🚀 Usage

### Option A — Jupyter Notebook (recommended)
```bash
jupyter notebook notebooks/hotel_sentiment_analysis.ipynb
```
Run all cells top-to-bottom. The notebook covers EDA → preprocessing → modelling → evaluation → inference.

### Option B — CLI Scripts

**Train a model:**
```bash
python src/train.py --data data/hotel_reviews.csv --model logreg
python src/train.py --data data/hotel_reviews.csv --model all   # train all models
```

**Predict on new text:**
```bash
python src/predict.py --text "The room was spotless and the staff were amazing!"
```

**Batch prediction on a CSV:**
```bash
python src/predict.py --csv data/new_reviews.csv --out outputs/predictions.csv
```

---

## 🧠 Models

| Model | Type | Library |
|-------|------|---------|
| Logistic Regression | Classical ML | scikit-learn |
| Linear SVM | Classical ML | scikit-learn |
| Random Forest | Ensemble | scikit-learn |
| XGBoost | Gradient Boosting | xgboost |
| Bi-LSTM | Deep Learning | TensorFlow/Keras |
| BERT | Transformer | HuggingFace Transformers |

---

## 📈 Pipeline

```
Raw CSV
  │
  ▼
Preprocessing
  ├─ Combine Negative_Review + Positive_Review
  ├─ Remove placeholders ("No Negative", etc.)
  ├─ Lowercase → strip HTML/URLs → remove punctuation
  ├─ Tokenise → remove stopwords (keep negations) → lemmatise
  └─ Label: score < 7 → Negative (0), score ≥ 7 → Positive (1)
  │
  ▼
Feature Engineering
  ├─ TF-IDF (unigrams + bigrams, 50k vocab)
  └─ Meta-features: word counts, neg_ratio, avg_score, reviewer exp, geo coords
  │
  ▼
Model Training & Evaluation
  ├─ Train/test split (80/20, stratified)
  ├─ 5-fold cross-validation
  └─ Metrics: Accuracy, F1, Precision, Recall, AUC-ROC
  │
  ▼
Outputs
  ├─ outputs/model_comparison.png
  ├─ outputs/confusion_matrix_*.png
  ├─ outputs/roc_curves.png
  ├─ outputs/wordclouds.png
  └─ models/best_model.joblib
```

---

## 📉 Sample Results

| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| Logistic Regression | ~0.93 | ~0.93 | ~0.97 |
| Linear SVM | ~0.93 | ~0.93 | — |
| Random Forest *(speed-opt)* | ~0.91 | ~0.91 | ~0.96 |
| Bi-LSTM | ~0.94 | ~0.94 | ~0.98 |

> *Random Forest uses 100 trees / 30% row sub-sample / max depth 20 for <1 min training. Exact results depend on dataset version and hyperparameters.*

---

## 📁 Outputs

After running the notebook, the `outputs/` directory contains:

- `score_distributions.png` — reviewer & hotel score histograms
- `label_distribution.png` — class balance chart
- `top_nationalities.png` — reviewer nationality bar chart
- `positive_review_wordcloud.png` — word cloud for positive reviews
- `negative_review_wordcloud.png` — word cloud for negative reviews
- `top_words_by_class.png` — top 20 words per class
- `confusion_matrix_*.png` — per-model confusion matrices
- `roc_curves.png` — all model ROC curves
- `model_comparison.png` — metric bar chart
- `model_results.csv` — tabular results
- `hotel_heatmap.html` — interactive geo map (requires folium)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Diwan Dahiya**  
[GitHub](https://github.com/diwandahiya304)
