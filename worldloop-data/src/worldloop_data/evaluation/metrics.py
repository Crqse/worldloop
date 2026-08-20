"""Metrics for M6 evaluation.

Pre-registered primary metrics (§16.6):
    - mae: mean absolute error for regression targets (energy_delta, edge_change_count).
    - accuracy: classification accuracy for categorical targets (position_change).
    - ndcg_at_k: normalized discounted cumulative gain for ranking targets (executed_candidate_rank).
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error. Returns NaN if inputs are empty."""
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy. Returns NaN if inputs are empty."""
    if y_true.size == 0:
        return float("nan")
    y_true_int = np.round(y_true).astype(int)
    y_pred_int = np.round(y_pred).astype(int)
    return float(np.mean(y_true_int == y_pred_int))


def ndcg_at_k(y_true_ranks: np.ndarray, y_pred_scores: np.ndarray, k: int = 3) -> float:
    """Normalized Discounted Cumulative Gain @ k for ranking evaluation.

    Args:
        y_true_ranks: (N,) array of true ranks (1-indexed; 0 means single candidate).
        y_pred_scores: (N,) array of predicted scores (higher = more likely executed).

    Returns:
        Mean NDCG@k over all samples with rank > 0 (i.e., multi-candidate samples).
        Returns NaN if no multi-candidate samples.
    """
    mask = y_true_ranks > 0
    if not np.any(mask):
        return float("nan")

    y_true_r = y_true_ranks[mask]
    y_pred_s = y_pred_scores[mask]

    ndcgs = []
    for true_rank, pred_score in zip(y_true_r, y_pred_s):
        # For a single prediction (no list), NDCG = 1.0 if pred matches true, else 0.
        # Here we treat each sample independently: the "list" is the candidate set,
        # but we only have the executed candidate's predicted score.
        # Simplification: NDCG = 1 / log2(true_rank + 1) if pred_score > 0, else 0.
        # This is a per-sample proxy; full NDCG requires per-candidate scores.
        if pred_score > 0:
            ndcgs.append(1.0 / np.log2(true_rank + 1))
        else:
            ndcgs.append(0.0)

    return float(np.mean(ndcgs))
