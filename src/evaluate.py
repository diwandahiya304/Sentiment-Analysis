"""
evaluate.py
───────────
Metrics, confusion matrix, and top TF-IDF feature plots.

Usage (standalone — requires trained model):
    python src/evaluate.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size']  = 12

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Main evaluation function ───────────────────────────────────────────────────
def evaluate_model(model, vectorizer, X_test, y_test):
    """
    Print classification report, save confusion matrix and top-feature plot.
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f'\n  Accuracy  : {acc:.4f}')
    print(f'  F1 Score  : {f1:.4f}')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print('\nDetailed Classification Report:')
    print(classification_report(y_test, y_pred,
                                 target_names=['Negative', 'Positive'],
                                 zero_division=0))

    # Save metrics to CSV
    results = pd.DataFrame([{
        'Model': 'Naive Bayes',
        'Accuracy': round(acc, 4),
        'F1': round(f1, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
    }])
    results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
    print(f'Results saved → {OUTPUT_DIR}/results.csv')

    # Plot confusion matrix
    _plot_confusion_matrix(y_test, y_pred)

    # Plot top TF-IDF features
    _plot_top_features(model, vectorizer)


def _plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        ax=ax,
    )
    ax.set_title('Confusion Matrix — Naive Bayes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(path)
    plt.show()
    print(f'Confusion matrix saved → {path}')


def _plot_top_features(model, vectorizer, n: int = 20):
    """
    Plot the top N words most associated with each class
    using the Naive Bayes log-probability difference.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    # log P(feature | class=1) - log P(feature | class=0)
    log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]

    top_pos_idx = np.argsort(log_prob_diff)[-n:]
    top_neg_idx = np.argsort(log_prob_diff)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, idx, color, title in [
        (axes[0], top_pos_idx, 'steelblue',  f'Top {n} Positive Words'),
        (axes[1], top_neg_idx, 'tomato',     f'Top {n} Negative Words'),
    ]:
        words  = feature_names[idx]
        scores = log_prob_diff[idx]
        ax.barh(words, np.abs(scores), color=color, edgecolor='white')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Log-Prob Difference (abs)')
        ax.invert_yaxis()

    plt.suptitle('Most Informative TF-IDF Features', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'top_features.png')
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    print(f'Top features plot saved → {path}')


# ── Standalone run ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from features import build_features

    MODEL_PATH = os.path.join('models', 'nb_model.joblib')
    TFIDF_PATH = os.path.join('models', 'tfidf_vectorizer.joblib')
    CLEANED    = os.path.join('data',   'cleaned_reviews.csv')

    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(TFIDF_PATH)
    _, X_test, _, y_test, _ = build_features(CLEANED, TFIDF_PATH)

    evaluate_model(model, vectorizer, X_test, y_test)
