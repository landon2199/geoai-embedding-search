"""Retrieval precision@k, spectral baseline features and a spatially blocked linear probe."""

from __future__ import annotations

import numpy as np

from .search import l2_normalize, standardize


def spectral_features(chips: np.ndarray) -> np.ndarray:
    """Baseline: per-band mean and standard deviation of each chip, plus NDVI mean.

    `chips` is (N, C, H, W) in L2A band order (see data.L2A_BANDS).
    """
    x = chips.astype(np.float32)
    mean = x.mean(axis=(2, 3))
    std = x.std(axis=(2, 3))
    red, nir = x[:, 3], x[:, 7]
    ndvi = ((nir - red) / np.maximum(nir + red, 1.0)).mean(axis=(1, 2))[:, None]
    return np.concatenate([mean, std, ndvi], axis=1)


def precision_at_k(features: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                   cols: np.ndarray, k: int, exclude_radius: int = 1,
                   block: int = 1024) -> dict:
    """Mean share of each chip's top-k cosine neighbours that share its label.

    Every chip is used once as a query. Chips within `exclude_radius`
    chips of the query (its immediate neighbours) are excluded so that
    spatial autocorrelation does not inflate the score. Returns overall
    (micro) precision, per-class precision and the class-balanced mean.
    """
    f = l2_normalize(features.astype(np.float32))
    n = len(f)
    hits = np.zeros(n)
    for s in range(0, n, block):
        q = np.arange(s, min(s + block, n))
        sims = f[q] @ f.T
        near = ((np.abs(rows[q, None] - rows[None]) <= exclude_radius)
                & (np.abs(cols[q, None] - cols[None]) <= exclude_radius))
        sims[near] = -np.inf  # also removes the query itself
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        hits[q] = (labels[top] == labels[q, None]).mean(1)
    per_class = {int(c): float(hits[labels == c].mean()) for c in np.unique(labels)}
    return {"micro": float(hits.mean()),
            "macro": float(np.mean(list(per_class.values()))),
            "per_class": per_class}


def chance_precision(labels: np.ndarray) -> dict:
    """Expected precision@k of random retrieval (class frequency)."""
    classes, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return {"micro": float((p ** 2).sum()), "macro": float(p.mean())}


def spatial_blocks(rows: np.ndarray, cols: np.ndarray, block: int) -> np.ndarray:
    """Group id of the block x block-chip square each chip falls in."""
    n_bc = cols.max() // block + 1
    return (rows // block) * n_bc + (cols // block)


def linear_probe(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                 n_splits: int = 5, seed: int = 0) -> dict:
    """Logistic regression with spatially grouped K-fold cross-validation.

    Whole spatial blocks are held out together, so test chips are never
    adjacent to training chips except along block edges.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    accs, f1s = [], []
    cv = GroupKFold(n_splits=n_splits)
    for tr, te in cv.split(features, labels, groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=1.0,
                                               class_weight="balanced", random_state=seed))
        clf.fit(features[tr], labels[tr])
        pred = clf.predict(features[te])
        accs.append(accuracy_score(labels[te], pred))
        f1s.append(f1_score(labels[te], pred, average="macro"))
    return {"accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "n_splits": n_splits}


__all__ = ["spectral_features", "precision_at_k", "chance_precision",
           "spatial_blocks", "linear_probe", "standardize"]
