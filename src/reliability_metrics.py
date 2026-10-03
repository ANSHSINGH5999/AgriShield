"""Metrics for the reliability task. Positive class = INCORRECT disease prediction."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score)


def best_threshold(y: np.ndarray, score: np.ndarray, beta: float = 2.0) -> float:
    """Threshold maximising F-beta (beta=2 favours recall of incorrect predictions). Use on VALIDATION data only."""
    cands = np.unique(np.quantile(score, np.linspace(0, 1, 501)))
    best_t, best_f = cands[0], -1
    for t in cands:
        f = fbeta_score(y, score >= t, beta=beta, zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    return float(best_t)


def binary_report(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    flag = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "n": int(len(y)), "incorrect_rate": float(y.mean()),
        "precision": float(precision_score(y, flag, zero_division=0)), "recall": float(recall_score(y, flag, zero_division=0)),
        "f1": float(f1_score(y, flag, zero_division=0)), "f2": float(fbeta_score(y, flag, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if 0 < y.mean() < 1 else None,
        "pr_auc": float(average_precision_score(y, score)) if y.sum() else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "false_negative_rate": float(fn / max(tp + fn, 1)), "false_alarm_rate": float(fp / max(fp + tn, 1)),
        "flagged_share": float(flag.mean()),
    }


def risk_coverage(y: np.ndarray, score: np.ndarray, points: int = 50) -> tuple[np.ndarray, np.ndarray, float]:
    """Accept the least risky predictions first; error rate among accepted vs coverage. Returns AURC too."""
    order = np.argsort(score, kind="stable")
    err = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    cov = np.arange(1, len(y) + 1) / len(y)
    idx = np.unique(np.linspace(0, len(y) - 1, points).astype(int))
    return cov[idx], err[idx], float(err.mean())


def calibration_table(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> tuple[list, float]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "mean_predicted": float(prob[m].mean()), "observed_incorrect": float(y[m].mean()), "n": int(m.sum())})
    return rows, float(brier_score_loss(y, prob))
