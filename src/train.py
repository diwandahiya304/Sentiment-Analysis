"""
train.py
────────
Train a Naive Bayes classifier for sentiment analysis.

Usage:
    python src/train.py                              # full pipeline
    python src/train.py --predict "Great movie!"    # predict a single review
"""

import os
import sys
import argparse

import joblib
from sklearn.naive_bayes import MultinomialNB

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess, raw_clean, load_spacy, spacy_clean
from features   import build_features
from evaluate   import evaluate_model

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join('data', 'IMDB Dataset.csv')
CLEANED_PATH = os.path.join('data', 'cleaned_reviews.csv')
MODEL_PATH   = os.path.join('models', 'nb_model.joblib')
TFIDF_PATH   = os.path.join('models', 'tfidf_vectorizer.joblib')


def train():
    """Full training pipeline: preprocess → features → train → evaluate."""

    # Step 1: Preprocess (skip if already done)
    if not os.path.exists(CLEANED_PATH):
        print('=' * 55)
        print(' STEP 1 — Preprocessing')
        print('=' * 55)
        preprocess(data_path=DATA_PATH, output_path=CLEANED_PATH)
    else:
        print(f'✔ Cleaned data already exists: {CLEANED_PATH}')

    # Step 2: Feature engineering
    print('\n' + '=' * 55)
    print(' STEP 2 — Feature Engineering (TF-IDF)')
    print('=' * 55)
    X_train, X_test, y_train, y_test, vectorizer = build_features(
        cleaned_path=CLEANED_PATH, tfidf_path=TFIDF_PATH
    )

    # Step 3: Train Naive Bayes
    print('\n' + '=' * 55)
    print(' STEP 3 — Training Naive Bayes Classifier')
    print('=' * 55)
    model = MultinomialNB(alpha=0.1)   # alpha: Laplace smoothing
    model.fit(X_train, y_train)
    print('  MultinomialNB trained ✅')

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f'  Model saved → {MODEL_PATH}')

    # Step 4: Evaluate
    print('\n' + '=' * 55)
    print(' STEP 4 — Evaluation')
    print('=' * 55)
    evaluate_model(model, vectorizer, X_test, y_test)

    return model, vectorizer


def predict_single(text: str,
                   model_path: str = MODEL_PATH,
                   tfidf_path: str = TFIDF_PATH) -> str:
    """
    Predict sentiment for a single raw review string.
    Loads saved model and vectorizer from disk.
    """
    model      = joblib.load(model_path)
    vectorizer = joblib.load(tfidf_path)
    nlp        = load_spacy()

    # Replicate the same cleaning pipeline
    raw   = raw_clean(text)
    clean = spacy_clean([raw], nlp)[0]

    X    = vectorizer.transform([clean])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()

    label = 'Positive ✅' if pred == 1 else 'Negative ❌'
    print(f'\nReview    : {text}')
    print(f'Sentiment : {label}  (confidence: {prob:.1%})')
    return label


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment Analysis — Naive Bayes')
    parser.add_argument('--predict', type=str, default=None,
                        help='Predict sentiment for a single review string')
    args = parser.parse_args()

    if args.predict:
        predict_single(args.predict)
    else:
        train()
