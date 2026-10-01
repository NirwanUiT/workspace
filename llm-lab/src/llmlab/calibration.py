"""Calibration metrics for K-way classifiers.

All functions take `probs` (N, K) rows summing to 1 and integer `labels` (N,).
Conventions are stated explicitly because calibration numbers are only
comparable when the binning and Brier variant are known:

- ECE is *top-label* ECE: confidence = max_k p_k, correct = argmax == label.
  Bins are equal-width on [0, 1] ("width") or equal-mass by confidence
  quantiles ("mass"). Default 15 bins, as in Guo et al. 2017.
  ECE = sum_b (n_b / N) * |acc_b - conf_b|.
- Brier is the multiclass Brier score: mean over items of sum_k (p_k - y_k)^2,
  range [0, 2]. `brier_top` is the binary Brier of the top-label confidence
  against correctness, range [0, 1].
- NLL is the mean negative log-probability of the true label (natural log).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def _check(probs: np.ndarray, labels: np.ndarray) -> None:
    if probs.ndim != 2 or labels.shape != (probs.shape[0],):
        raise ValueError(f"bad shapes: probs {probs.shape}, labels {labels.shape}")
    if not np.allclose(probs.sum(1), 1.0, atol=1e-4):
        raise ValueError("probability rows must sum to 1")


def confidence_and_correct(probs: np.ndarray, labels: np.ndarray):
    pred = probs.argmax(1)
    return probs.max(1), (pred == labels).astype(float)


def accuracy(probs, labels) -> float:
    _check(probs, labels)
    return float((probs.argmax(1) == labels).mean())


def nll(probs, labels) -> float:
    _check(probs, labels)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], EPS, 1)).mean())


def brier(probs, labels) -> float:
    _check(probs, labels)
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def brier_top(probs, labels) -> float:
    conf, correct = confidence_and_correct(probs, labels)
    return float(((conf - correct) ** 2).mean())


def bin_edges(conf: np.ndarray, n_bins: int = 15, scheme: str = "width") -> np.ndarray:
    if scheme == "width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if scheme == "mass":
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return edges
    raise ValueError(f"unknown binning scheme {scheme!r}")


def reliability_bins(probs, labels, n_bins: int = 15, scheme: str = "width") -> dict:
    """Per-bin count, mean confidence and accuracy for the top label.

    Bins are right-closed except the first, which also includes 0:
    (e_{b-1}, e_b], so confidence 1.0 lands in the last bin.
    """
    _check(probs, labels)
    conf, correct = confidence_and_correct(probs, labels)
    edges = bin_edges(conf, n_bins, scheme)
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, len(edges) - 2)
    count = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.bincount(idx, weights=conf, minlength=len(edges) - 1) / count
        acc = np.bincount(idx, weights=correct, minlength=len(edges) - 1) / count
    return {"edges": edges, "count": count, "conf": mean_conf, "acc": acc}


def ece(probs, labels, n_bins: int = 15, scheme: str = "width") -> float:
    b = reliability_bins(probs, labels, n_bins, scheme)
    m = b["count"] > 0
    return float((b["count"][m] / b["count"].sum() * np.abs(b["acc"][m] - b["conf"][m])).sum())


def mce(probs, labels, n_bins: int = 15, scheme: str = "width", min_count: int = 20) -> float:
    """Max calibration error over bins holding at least `min_count` items."""
    b = reliability_bins(probs, labels, n_bins, scheme)
    m = b["count"] >= min_count
    return float(np.abs(b["acc"][m] - b["conf"][m]).max()) if m.any() else float("nan")


def auroc(probs, labels) -> float:
    """How well top-label confidence separates right from wrong answers."""
    conf, correct = confidence_and_correct(probs, labels)
    if correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, conf))


def all_metrics(probs, labels, n_bins: int = 15) -> dict:
    return {
        "n": int(len(labels)),
        "accuracy": accuracy(probs, labels),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
        "brier_top": brier_top(probs, labels),
        "ece_width": ece(probs, labels, n_bins, "width"),
        "ece_mass": ece(probs, labels, n_bins, "mass"),
        "mce_width": mce(probs, labels, n_bins, "width"),
        "auroc": auroc(probs, labels),
        "mean_confidence": float(probs.max(1).mean()),
    }


def bootstrap_ci(probs, labels, metric, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI over items for any metric(probs, labels) -> float."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    stats = np.array([metric(probs[i], labels[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def softmax_t(logits: np.ndarray, temperature: float) -> np.ndarray:
    return np.exp(log_softmax(logits / temperature, axis=1))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, bounds=(0.05, 20.0)) -> float:
    """Single temperature minimising NLL on a held-out split (Guo et al. 2017)."""
    def loss(log_t):
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return -lp[np.arange(len(labels)), labels].mean()

    res = minimize_scalar(loss, bounds=np.log(bounds), method="bounded")
    return float(np.exp(res.x))
