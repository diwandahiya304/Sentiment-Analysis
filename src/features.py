"""
features.py
───────────
TF-IDF feature engineering for sentiment analysis.

Usage (standalone):
    python src/features.py
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# ── Constants ──────────────────────────────────────────────────────────────────
CLEANED_PATH  = os.path.join('data', 'cleaned_reviews.csv')
TFIDF_PATH    = os.path.join('models', 'tfidf_vectorizer.joblib')

TFIDF_CONFIG = dict(
    max_features  = 20_000,      # vocabulary size
    ngram_range   = (1, 2),      # unigrams + bigrams
    sublinear_tf  = True,        # apply log(1+tf) scaling
    min_df        = 2,           # ignore very rare terms
    max_df        = 0.95,        # ignore very common terms
    strip_accents = 'unicode',
)

TEST_SIZE    = 0.20
RANDOM_STATE = 42


def build_features(cleaned_path: str = CLEANED_PATH,
                   tfidf_path:   str = TFIDF_PATH,
                   test_size:    float = TEST_SIZE):
    """
    Load cleaned text, vectorise with TF-IDF, and return train/test splits.

    Returns
    -------
    X_train, X_test : sparse matrices
    y_train, y_test : Series
    vectorizer       : fitted TfidfVectorizer
    """
    print(f'Loading cleaned data from: {cleaned_path}')
    df = pd.read_csv(cleaned_path)
    print(f'  → {len(df):,} rows loaded')

    X = df['clean_text']
    y = df['label']

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        stratify     = y,
        random_state = RANDOM_STATE,
    )
    print(f'\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}')
    print(f'Class balance (train): {y_train.mean():.1%} positive')

    # TF-IDF vectorisation
    print('\nFitting TF-IDF vectorizer…')
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    print(f'  → Train matrix: {X_train_tfidf.shape}')
    print(f'  → Test  matrix: {X_test_tfidf.shape}')

    # Persist vectorizer
    os.makedirs(os.path.dirname(tfidf_path), exist_ok=True)
    joblib.dump(vectorizer, tfidf_path)
    print(f'\n✅ TF-IDF vectorizer saved → {tfidf_path}')

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


if __name__ == '__main__':
    build_features()
