"""
src/features.py
---------------
Feature extraction: TF-IDF, Bag-of-Words, and engineered meta-features
for the hotel-review sentiment analysis pipeline.
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import os


# ─────────────────────────────────────────────────────────────────────
# TF-IDF / BoW vectorisers
# ─────────────────────────────────────────────────────────────────────

def build_tfidf(
    texts: pd.Series,
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2),
    sublinear_tf: bool = True,
) -> tuple:
    """
    Fit a TF-IDF vectoriser on *texts* and return (matrix, vectoriser).

    Parameters
    ----------
    texts : pd.Series
        Clean text column.
    max_features : int
        Vocabulary cap.
    ngram_range : tuple
        (min_n, max_n) for n-gram extraction.
    sublinear_tf : bool
        Apply log normalisation to term frequencies.

    Returns
    -------
    (sparse_matrix, TfidfVectorizer)
    """
    vectoriser = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=2,
        max_df=0.95,
    )
    matrix = vectoriser.fit_transform(texts)
    return matrix, vectoriser


def transform_tfidf(texts: pd.Series, vectoriser: TfidfVectorizer):
    """Transform *texts* using a pre-fitted TF-IDF vectoriser."""
    return vectoriser.transform(texts)


def build_bow(
    texts: pd.Series,
    max_features: int = 30_000,
) -> tuple:
    """Fit a Bag-of-Words (CountVectorizer) and return (matrix, vectoriser)."""
    vectoriser = CountVectorizer(max_features=max_features, min_df=2, max_df=0.95)
    matrix = vectoriser.fit_transform(texts)
    return matrix, vectoriser


# ─────────────────────────────────────────────────────────────────────
# Engineered meta-features from structured columns
# ─────────────────────────────────────────────────────────────────────

def engineer_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numeric meta-features from the structured columns.

    Returns a DataFrame with new engineered columns.
    """
    feat = pd.DataFrame(index=df.index)

    # Review length features
    feat["neg_word_count"] = df["Review_Total_Negative_Word_Counts"].fillna(0)
    feat["pos_word_count"] = df["Review_Total_Positive_Word_Counts"].fillna(0)
    feat["total_word_count"] = feat["neg_word_count"] + feat["pos_word_count"]

    # Ratio of negative words (proxy for negativity)
    feat["neg_ratio"] = feat["neg_word_count"] / (feat["total_word_count"] + 1e-6)

    # Hotel average score
    feat["average_score"] = df["Average_Score"].fillna(df["Average_Score"].median())

    # Total reviews for the hotel (popularity signal)
    feat["log_total_reviews"] = np.log1p(
        df["Total_Number_of_Reviews"].fillna(0)
    )

    # Reviewer experience (how many reviews they've written)
    feat["reviewer_experience"] = np.log1p(
        df["Total_Number_of_Reviews_Reviewer_Has_Given"].fillna(0)
    )

    # Days since review (recency)
    feat["days_since_review"] = df["days_since_review"].fillna(
        df["days_since_review"].median() if "days_since_review" in df.columns else 0
    )

    # Geographic features
    feat["lat"] = df["lat"].fillna(df["lat"].median())
    feat["lng"] = df["lng"].fillna(df["lng"].median())

    return feat


def scale_meta_features(
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame = None,
) -> tuple:
    """
    StandardScale meta-features.

    Returns
    -------
    (scaled_train_array, scaled_test_array_or_None, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(meta_train)
    X_test_scaled = scaler.transform(meta_test) if meta_test is not None else None
    return X_train_scaled, X_test_scaled, scaler


def combine_features(tfidf_matrix, meta_array: np.ndarray):
    """
    Horizontally stack TF-IDF sparse matrix with dense meta-feature array.

    Returns a scipy sparse matrix.
    """
    meta_sparse = csr_matrix(meta_array)
    return hstack([tfidf_matrix, meta_sparse])


# ─────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────

def save_vectoriser(vectoriser, path: str = "models/tfidf_vectoriser.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectoriser, path)
    print(f"Vectoriser saved → {path}")


def load_vectoriser(path: str = "models/tfidf_vectoriser.joblib"):
    return joblib.load(path)


def save_scaler(scaler, path: str = "models/meta_scaler.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved → {path}")


def load_scaler(path: str = "models/meta_scaler.joblib"):
    return joblib.load(path)
