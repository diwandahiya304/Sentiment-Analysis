"""
src/train.py
------------
Train classical ML models (Logistic Regression, SVM, Random Forest, XGBoost)
on the hotel-review sentiment dataset and persist the best model.

Usage
-----
    python src/train.py --data data/hotel_reviews.csv --model logreg
    python src/train.py --data data/hotel_reviews.csv --model all
"""

import argparse
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from preprocess import load_and_preprocess
from features import (
    build_tfidf, transform_tfidf,
    engineer_meta_features, scale_meta_features,
    combine_features,
    save_vectoriser, save_scaler,
)
from evaluate import (
    print_metrics, plot_confusion_matrix, plot_roc_curve, save_results
)


# ─────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────

MODELS = {
    "logreg": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", multi_class="auto", n_jobs=-1
    ),
    "svm": LinearSVC(max_iter=2000, C=1.0),
    # Speed-optimised: 100 trees, 30% row sub-sample, max depth 20.
    # Runs in <1 min on 500k rows; accuracy stays within ~1% of full RF.
    "rf": RandomForestClassifier(
        n_estimators=100, max_depth=20, max_samples=0.3,
        max_features="sqrt", n_jobs=-1, random_state=42
    ),
    "xgb": XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        eval_metric="logloss",
        n_jobs=-1, random_state=42,
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    model_name: str = "logreg",
    use_meta: bool = True,
    save_dir: str = "models",
):
    """
    Full training pipeline for a single model.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_and_preprocess().
    model_name : str
        Key from MODELS dict.
    use_meta : bool
        Combine TF-IDF with engineered meta-features.
    save_dir : str
        Directory to persist model artefacts.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*60}")

    # ── Train / test split ──────────────────────────────────────────
    X_text = df["clean_text"]
    y = df["label"]

    (X_text_train, X_text_test,
     y_train, y_test,
     idx_train, idx_test) = train_test_split(
        X_text, y, df.index,
        test_size=0.2, random_state=42, stratify=y
    )

    # ── TF-IDF ──────────────────────────────────────────────────────
    print("Building TF-IDF features …")
    X_tfidf_train, vectoriser = build_tfidf(X_text_train)
    X_tfidf_test = transform_tfidf(X_text_test, vectoriser)
    save_vectoriser(vectoriser, os.path.join(save_dir, "tfidf_vectoriser.joblib"))

    # ── Meta features ───────────────────────────────────────────────
    if use_meta:
        print("Engineering meta-features …")
        meta_train = engineer_meta_features(df.loc[idx_train])
        meta_test  = engineer_meta_features(df.loc[idx_test])
        meta_train_s, meta_test_s, scaler = scale_meta_features(meta_train, meta_test)
        save_scaler(scaler, os.path.join(save_dir, "meta_scaler.joblib"))

        X_train = combine_features(X_tfidf_train, meta_train_s)
        X_test  = combine_features(X_tfidf_test,  meta_test_s)
    else:
        X_train = X_tfidf_train
        X_test  = X_tfidf_test

    # ── Fit ─────────────────────────────────────────────────────────
    model = MODELS[model_name]
    print(f"Fitting {model_name} …")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - t0:.1f}s")

    # ── Evaluate ────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    print_metrics(y_test, y_pred, model_name)
    plot_confusion_matrix(y_test, y_pred, model_name)
    if hasattr(model, "predict_proba"):
        plot_roc_curve(model, X_test, y_test, model_name)
    save_results(y_test, y_pred, model_name)

    # ── Cross-validation ────────────────────────────────────────────
    print("\nRunning 5-fold CV on training data …")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted", n_jobs=-1)
    print(f"CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Save model ──────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")

    return model, vectoriser


# ─────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train sentiment analysis models.")
    parser.add_argument("--data",  default="data/hotel_reviews.csv",
                        help="Path to the hotel reviews CSV.")
    parser.add_argument("--model", default="logreg",
                        choices=list(MODELS.keys()) + ["all"],
                        help="Model to train (or 'all').")
    parser.add_argument("--no-meta", action="store_true",
                        help="Disable engineered meta-features.")
    parser.add_argument("--mode",  default="binary",
                        choices=["binary", "multiclass"],
                        help="Sentiment label scheme.")
    parser.add_argument("--save-dir", default="models",
                        help="Directory to save model artefacts.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\nLoading and preprocessing data from: {args.data}")
    df = load_and_preprocess(args.data, mode=args.mode)
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['sentiment'].value_counts()}\n")

    models_to_train = list(MODELS.keys()) if args.model == "all" else [args.model]

    results = {}
    for m in models_to_train:
        model, vec = train_model(
            df,
            model_name=m,
            use_meta=not args.no_meta,
            save_dir=args.save_dir,
        )
        results[m] = model

    print("\n✅ Training complete.")
