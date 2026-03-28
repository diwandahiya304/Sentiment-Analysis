"""
preprocess.py
─────────────
spaCy-based NLP preprocessing pipeline for IMDB sentiment analysis.

Usage (standalone):
    python src/preprocess.py
"""

import re
import os
import pandas as pd
import spacy

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join('data', 'IMDB Dataset.csv')
OUTPUT_PATH  = os.path.join('data', 'cleaned_reviews.csv')
SAMPLE_SIZE  = 20_000   # Set to None to use all 50k rows
RANDOM_STATE = 42


# ── Load spaCy model ───────────────────────────────────────────────────────────
def load_spacy():
    try:
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    except OSError:
        raise OSError(
            "spaCy model not found.\n"
            "Run: python -m spacy download en_core_web_sm"
        )
    return nlp


# ── Text cleaning helpers ──────────────────────────────────────────────────────
def remove_html(text: str) -> str:
    """Strip HTML tags (common in IMDB reviews)."""
    return re.sub(r'<.*?>', ' ', text)


def remove_urls(text: str) -> str:
    return re.sub(r'http\S+|www\S+', ' ', text)


def remove_special_chars(text: str) -> str:
    """Keep only letters and spaces."""
    return re.sub(r'[^a-zA-Z\s]', ' ', text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def raw_clean(text: str) -> str:
    """Apply all non-spaCy cleaning steps."""
    text = text.lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)
    return text


# ── spaCy pipeline ─────────────────────────────────────────────────────────────
def spacy_clean(texts: list[str], nlp, batch_size: int = 500) -> list[str]:
    """
    Tokenise, remove stopwords & punctuation, and lemmatise
    using spaCy's pipe for efficiency.
    """
    cleaned = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.lemma_) > 1
        ]
        cleaned.append(' '.join(tokens))
    return cleaned


# ── Main ───────────────────────────────────────────────────────────────────────
def preprocess(data_path: str = DATA_PATH,
               output_path: str = OUTPUT_PATH,
               sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:

    print(f'Loading data from: {data_path}')
    df = pd.read_csv(data_path)
    print(f'  → Loaded {len(df):,} rows')

    # Optional sub-sample
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f'  → Sampled {sample_size:,} rows')

    # Encode label
    df['label'] = (df['sentiment'].str.lower() == 'positive').astype(int)
    print(f'\nLabel distribution:\n{df["sentiment"].value_counts().to_string()}')

    # Step 1: Basic cleaning
    print('\nStep 1/2 — Basic text cleaning…')
    df['raw_clean'] = df['review'].apply(raw_clean)

    # Step 2: spaCy tokenisation + lemmatisation
    print('Step 2/2 — spaCy lemmatisation (may take ~30s)…')
    nlp = load_spacy()
    df['clean_text'] = spacy_clean(df['raw_clean'].tolist(), nlp)

    # Drop rows where cleaning left empty text
    before = len(df)
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    print(f'  → Dropped {before - len(df)} empty rows after cleaning')

    # Save
    df[['review', 'clean_text', 'sentiment', 'label']].to_csv(output_path, index=False)
    print(f'\n✅ Cleaned data saved → {output_path}  ({len(df):,} rows)')

    return df


if __name__ == '__main__':
    preprocess()
