"""Cosine-similarity search over chip embeddings."""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance features (so no single feature dominates cosine)."""
    return (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)


def neighbor_mask(rows: np.ndarray, cols: np.ndarray, query: int, radius: int) -> np.ndarray:
    """True for chips within `radius` chips (Chebyshev distance) of the query, itself included."""
    return (np.abs(rows - rows[query]) <= radius) & (np.abs(cols - cols[query]) <= radius)


def cosine_topk(features: np.ndarray, query: int, k: int,
                exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine similarities of the k chips most similar to `query`.

    `exclude` is a boolean mask of candidates to skip; the query itself is
    always skipped.
    """
    f = l2_normalize(features)
    sims = f @ f[query]
    sims[query] = -np.inf
    if exclude is not None:
        sims[exclude] = -np.inf
    idx = np.argsort(-sims, kind="stable")[:k]
    return idx, sims[idx]
