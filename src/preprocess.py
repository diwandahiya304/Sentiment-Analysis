"""
src/preprocess.py
-----------------
Text cleaning & preprocessing utilities for hotel-review sentiment analysis.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── Download required NLTK data ───────────────────────────────────────
for _pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(_pkg, quiet=True)

# Keep negation words – they flip sentiment
_STOPWORDS = set(stopwords.words("english"))
_KEEP = {"not", "no", "nor", "never", "neither", "n't", "cannot", "hardly"}
STOPWORDS = _STOPWORDS - _KEEP

LEMMATIZER = WordNetLemmatizer()


# ─────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────

def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\S+", "", text)


def remove_html(text: str) -> str:
    return re.sub(r"<.*?>", "", text)


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def lemmatize(tokens: list[str]) -> list[str]:
    return [LEMMATIZER.lemmatize(t) for t in tokens]


# ─────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────

def clean_text(text: str, keep_stopwords: bool = False) -> str:
    """
    Full text-cleaning pipeline.

    Parameters
    ----------
    text : str
        Raw review text.
    keep_stopwords : bool
        If True, skip stopword removal (useful for transformer models).

    Returns
    -------
    str
        Cleaned text, ready for vectorisation or tokenisation.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_whitespace(text)

    tokens = tokenize(text)
    if not keep_stopwords:
        tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────
# Dataset-specific helpers
# ─────────────────────────────────────────────────────────────────────

PLACEHOLDER_NEGATIVES = {
    "no negative", "nothing", "none", "n/a", "na", "nil", "no negatives",
    "nothing negative", "no complaints", "nothing to complain",
}

PLACEHOLDER_POSITIVES = {
    "no positive", "nothing", "none", "n/a", "na", "nil", "no positives",
    "nothing positive",
}


def is_placeholder(text: str, placeholders: set) -> bool:
    """Return True if the text is a known placeholder (not a real review)."""
    if not isinstance(text, str):
        return True
    return text.strip().lower() in placeholders


def combine_reviews(row: pd.Series) -> str:
    """
    Combine Negative_Review and Positive_Review into one text field,
    filtering out placeholder values.

    Parameters
    ----------
    row : pd.Series
        A DataFrame row containing 'Negative_Review' and 'Positive_Review'.

    Returns
    -------
    str
        Combined review text.
    """
    neg = "" if is_placeholder(row["Negative_Review"], PLACEHOLDER_NEGATIVES) \
        else str(row["Negative_Review"])
    pos = "" if is_placeholder(row["Positive_Review"], PLACEHOLDER_POSITIVES) \
        else str(row["Positive_Review"])
    return (neg + " " + pos).strip()


def label_from_score(score: float) -> int:
    """
    Convert a Reviewer_Score (0-10) to a binary sentiment label.

    0-6  → Negative (0)
    7-10 → Positive (1)
    """
    if score < 7:
        return 0  # Negative
    return 1      # Positive


def label_from_score_multiclass(score: float) -> str:
    """
    Convert a Reviewer_Score (0-10) to a 3-class label.

    0-4   → Negative
    5-7   → Neutral
    8-10  → Positive
    """
    if score <= 4:
        return "Negative"
    if score <= 7:
        return "Neutral"
    return "Positive"


def load_and_preprocess(
    filepath: str,
    mode: str = "binary",
    clean: bool = True,
) -> pd.DataFrame:
    """
    Load the hotel-reviews CSV and return a preprocessed DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    mode : str
        'binary' or 'multiclass'.
    clean : bool
        Whether to apply text cleaning.

    Returns
    -------
    pd.DataFrame
        Columns: review_text, label, and original metadata columns.
    """
    df = pd.read_csv(filepath)

    # Drop rows with missing scores
    df.dropna(subset=["Reviewer_Score"], inplace=True)

    # Combine positive + negative review text
    df["review_text"] = df.apply(combine_reviews, axis=1)

    # Remove rows where combined text is empty
    df = df[df["review_text"].str.strip() != ""].copy()

    # Clean text
    if clean:
        print("Cleaning text … (this may take a minute)")
        df["clean_text"] = df["review_text"].apply(clean_text)
    else:
        df["clean_text"] = df["review_text"]

    # Create label
    if mode == "binary":
        df["label"] = df["Reviewer_Score"].apply(label_from_score)
        df["sentiment"] = df["label"].map({0: "Negative", 1: "Positive"})
    else:
        df["sentiment"] = df["Reviewer_Score"].apply(label_from_score_multiclass)
        df["label"] = df["sentiment"].map(
            {"Negative": 0, "Neutral": 1, "Positive": 2}
        )

    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/hotel_reviews.csv"
    df = load_and_preprocess(path)
    print(df[["clean_text", "label", "sentiment"]].head())
    print(f"\nLabel distribution:\n{df['sentiment'].value_counts()}")
