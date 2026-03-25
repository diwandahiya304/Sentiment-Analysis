"""
src/evaluate.py
---------------
Evaluation helpers: metrics, confusion matrix, ROC curve, result logging.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay,
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAP_BINARY = {0: "Negative", 1: "Positive"}
LABEL_MAP_MULTI  = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────

def print_metrics(y_true, y_pred, model_name: str = "") -> dict:
    """Print and return a dict of common classification metrics."""
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n── {model_name} Metrics ──────────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print("\nDetailed report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


# ─────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true, y_pred, model_name: str = "", normalize: bool = False
):
    """Save a confusion matrix heatmap to outputs/."""
    labels = sorted(set(y_true) | set(y_pred))
    n_classes = len(labels)
    display_labels = (
        [LABEL_MAP_BINARY.get(l, l) for l in labels]
        if n_classes == 2
        else [LABEL_MAP_MULTI.get(l, l) for l in labels]
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels,
                          normalize="true" if normalize else None)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {path}")


# ─────────────────────────────────────────────────────────────────────
# ROC curve (binary only)
# ─────────────────────────────────────────────────────────────────────

def plot_roc_curve(model, X_test, y_test, model_name: str = ""):
    """Save an ROC curve to outputs/ (binary classification only)."""
    if len(set(y_test)) != 2:
        print("ROC curve skipped (not binary).")
        return

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        print(f"{model_name} does not support predict_proba — ROC skipped.")
        return

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"roc_curve_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved → {path}")
    return auc


# ─────────────────────────────────────────────────────────────────────
# Results persistence
# ─────────────────────────────────────────────────────────────────────

def save_results(y_true, y_pred, model_name: str = ""):
    """Append model metrics to outputs/results.json."""
    metrics = {
        "model": model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred, average="weighted"), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted",
                                           zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="weighted",
                                        zero_division=0), 4),
    }

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    all_results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)

    # Replace or append
    all_results = [r for r in all_results if r.get("model") != model_name]
    all_results.append(metrics)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results appended → {results_path}")


def compare_models(results_path: str = "outputs/results.json"):
    """Load and display a comparison table of all saved model results."""
    if not os.path.exists(results_path):
        print("No results file found.")
        return

    with open(results_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data).set_index("model")
    print("\n── Model Comparison ─────────────────────────────────────")
    print(df.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    df[["accuracy", "f1", "precision", "recall"]].plot(
        kind="bar", ax=ax, rot=0, colormap="Set2"
    )
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Comparison chart saved → {path}")
    return df
