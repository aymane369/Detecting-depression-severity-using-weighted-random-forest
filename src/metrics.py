"""Evaluation metrics for binary and multiclass problems (article-compatible)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def auc_roc(y_true, proba, task: str) -> float:
    """ROC-AUC; macro one-vs-rest for the multiclass task (as in the article)."""
    if task == "binary":
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")


def auc_pr(y_true, proba, task: str) -> float:
    """Area under the precision-recall curve (average precision); macro-OvR if multiclass."""
    if task == "binary":
        return average_precision_score(y_true, proba[:, 1])
    classes = np.arange(proba.shape[1])
    y_bin = label_binarize(y_true, classes=classes)
    return float(np.mean([average_precision_score(y_bin[:, k], proba[:, k]) for k in classes]))


def compute_metrics(y_true, proba, task: str) -> dict[str, float]:
    """All metrics reported in the project.

    Precision / recall / F1 use the support-weighted average (this is what makes the
    article's recall equal to its accuracy); macro-F1 and balanced accuracy are added
    because they are not dominated by the majority class.
    """
    y_pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "auc": auc_roc(y_true, proba, task),
        "auc_pr": auc_pr(y_true, proba, task),
    }


def bootstrap_auc_ci(y_true, proba, task: str, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap confidence interval of the test AUC."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n, k = len(y_true), proba.shape[1]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < k:  # a class is missing in this resample
            continue
        vals.append(auc_roc(y_true[idx], proba[idx], task))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
