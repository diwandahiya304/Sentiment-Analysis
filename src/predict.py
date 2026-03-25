"""
src/predict.py
--------------
Load a trained model + vectoriser and predict sentiment for new reviews.

Usage
-----
    # Single text
    python src/predict.py --text "The room was spotless and the staff were amazing!"

    # Batch CSV
    python src/predict.py --csv path/to/reviews.csv
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

from preprocess import clean_text
from features import load_vectoriser, load_scaler, engineer_meta_features

LABEL_MAP = {0: "Negative", 1: "Positive"}


def load_artefacts(
    model_path: str = "models/logreg_model.joblib",
    vec_path:   str = "models/tfidf_vectoriser.joblib",
    scaler_path: str = "models/meta_scaler.joblib",
):
    """Load and return model, vectoriser, and optional scaler."""
    model     = joblib.load(model_path)
    vectoriser = load_vectoriser(vec_path)
    try:
        scaler = load_scaler(scaler_path)
    except FileNotFoundError:
        scaler = None
    return model, vectoriser, scaler


def predict_text(
    text: str,
    model,
    vectoriser,
    scaler=None,
) -> dict:
    """
    Predict sentiment for a single raw review string.

    Returns
    -------
    dict with keys: clean_text, prediction, label, confidence
    """
    cleaned = clean_text(text)
    X = vectoriser.transform([cleaned])

    if scaler is not None:
        # Build a dummy meta row with zeros (no structured data for ad-hoc text)
        dummy = np.zeros((1, scaler.n_features_in_))
        X = hstack([X, csr_matrix(dummy)])

    pred = model.predict(X)[0]
    label = LABEL_MAP.get(int(pred), str(pred))

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))

    return {
        "clean_text":  cleaned,
        "prediction":  int(pred),
        "label":       label,
        "confidence":  confidence,
    }


def predict_batch(
    df: pd.DataFrame,
    model,
    vectoriser,
    scaler=None,
    text_col: str = "review_text",
) -> pd.DataFrame:
    """
    Predict sentiment for a DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *text_col*.
    text_col : str
        Column with raw review text.

    Returns
    -------
    pd.DataFrame with added columns: clean_text, prediction, label, confidence.
    """
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)

    X = vectoriser.transform(df["clean_text"])

    if scaler is not None and all(c in df.columns for c in [
        "Review_Total_Negative_Word_Counts",
        "Review_Total_Positive_Word_Counts",
    ]):
        meta = engineer_meta_features(df)
        meta_s = scaler.transform(meta)
        X = hstack([X, csr_matrix(meta_s)])

    preds = model.predict(X)
    df["prediction"] = preds
    df["label"] = df["prediction"].map(LABEL_MAP)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        df["confidence"] = probas.max(axis=1)
    else:
        df["confidence"] = None

    return df


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Predict sentiment from hotel reviews.")
    parser.add_argument("--text",   type=str, default=None,
                        help="Single review text to classify.")
    parser.add_argument("--csv",    type=str, default=None,
                        help="Path to CSV with a 'review_text' column.")
    parser.add_argument("--model",  default="models/logreg_model.joblib")
    parser.add_argument("--vec",    default="models/tfidf_vectoriser.joblib")
    parser.add_argument("--scaler", default="models/meta_scaler.joblib")
    parser.add_argument("--out",    default="outputs/predictions.csv",
                        help="Output path for batch predictions.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, vectoriser, scaler = load_artefacts(args.model, args.vec, args.scaler)

    if args.text:
        result = predict_text(args.text, model, vectoriser, scaler)
        print(f"\nText      : {args.text}")
        print(f"Sentiment : {result['label']}")
        if result["confidence"]:
            print(f"Confidence: {result['confidence']:.2%}")

    elif args.csv:
        df = pd.read_csv(args.csv)
        out = predict_batch(df, model, vectoriser, scaler)
        out.to_csv(args.out, index=False)
        print(f"\nPredictions saved → {args.out}")
        print(out[["label", "confidence"]].value_counts())

    else:
        print("Provide --text or --csv. See --help for usage.")
