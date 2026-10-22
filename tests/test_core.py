"""Fast CPU-only tests: tiling, band handling, normalisation and cosine top-k."""

import numpy as np
import pytest

from embedsearch import data, embed, evaluate, search


def test_tile_shapes_and_order():
    arr = np.arange(3 * 130 * 200).reshape(3, 130, 200)
    chips, rows, cols = data.tile(arr, 64)
    assert chips.shape == (2 * 3, 3, 64, 64)          # edge remainder dropped
    assert list(zip(rows, cols))[:4] == [(0, 0), (0, 1), (0, 2), (1, 0)]
    # chip (row 1, col 2) is the window starting at pixel (64, 128)
    np.testing.assert_array_equal(chips[5], arr[:, 64:128, 128:192])


def test_tile_2d_labels():
    lab = np.zeros((128, 128), dtype=np.uint8)
    lab[:64, 64:] = 80
    chips, _, _ = data.tile(lab, 64)
    assert chips.shape == (4, 64, 64)
    assert (chips[1] == 80).all() and (chips[0] == 0).all()


def test_dominant_class_and_purity():
    chips = np.full((2, 4, 4), 30, dtype=np.uint8)
    chips[0, :3] = 40                   # 12 of 16 pixels cropland
    chips[1, 0, 0] = 0                  # nodata pixel is never the dominant class
    dom, pur = data.dominant_class(chips)
    assert dom.tolist() == [40, 30]
    assert pur.tolist() == pytest.approx([0.75, 15 / 16])


def test_band_reorder_inserts_zero_b10():
    chips = np.stack([np.full((1, 2, 2), i + 1.0) for i in range(12)], axis=1)[0][None]
    out = embed.to_model_bands(chips, data.L2A_BANDS)
    assert out.shape == (1, 13, 2, 2)
    assert embed.MODEL_BANDS.index("B10") == 10
    assert (out[0, 10] == 0).all()
    assert out[0, 11, 0, 0] == 11 and out[0, 12, 0, 0] == 12   # B11, B12 shifted by one
    assert out[0, 9, 0, 0] == 10                                # B09 unchanged


def test_normalize_matches_torchgeo_scale():
    x = np.array([0.0, 1000.0, 10000.0])
    np.testing.assert_allclose(embed.normalize(x), [0.0, 0.1, 1.0])


def test_cosine_topk_and_exclusion():
    f = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.99, 0.01], [-1, 0]], dtype=float)
    idx, sims = search.cosine_topk(f, 0, 2)
    assert idx.tolist() == [3, 1]
    assert sims[0] > sims[1]
    excl = np.zeros(5, bool)
    excl[3] = True
    idx, _ = search.cosine_topk(f, 0, 2, exclude=excl)
    assert idx.tolist() == [1, 2]
    assert 0 not in idx


def test_cosine_is_scale_invariant():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(50, 8))
    a, _ = search.cosine_topk(f, 7, 5)
    b, _ = search.cosine_topk(f * rng.uniform(0.1, 10, size=(50, 1)), 7, 5)
    assert a.tolist() == b.tolist()


def test_precision_at_k_perfect_and_neighbor_exclusion():
    # two well-separated clusters laid out on a 1 x 20 strip
    labels = np.array([0] * 10 + [1] * 10)
    feats = np.where(labels[:, None] == 0, [[1.0, 0.0]], [[0.0, 1.0]])
    rows, cols = np.zeros(20, int), np.arange(20)
    p = evaluate.precision_at_k(feats, labels, rows, cols, k=3, exclude_radius=1)
    assert p["micro"] == pytest.approx(1.0) and p["macro"] == pytest.approx(1.0)


def test_spatial_blocks():
    rows, cols = data.chip_grid(4 * 64, 4 * 64, 64)
    g = evaluate.spatial_blocks(rows, cols, 2)
    assert len(np.unique(g)) == 4
    assert g[0] == g[1] == g[4] == g[5]
