"""Find places that look alike from space.

End-to-end run: Sentinel-2 composite -> chips -> SSL4EO-S12 embeddings ->
query-by-example search, retrieval precision@k, linear probe and a 2-D
projection, each compared with a simple spectral-statistics baseline.

    python run.py                 # full run with defaults (Brazos Valley, TX)
    python run.py --device cpu    # slower, no GPU needed
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# On Windows, pip PyTorch ships Intel OpenMP while conda-forge libraries
# load LLVM OpenMP; allow both so the process does not abort on import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

from embedsearch import data, embed, evaluate, search

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--center-lon", type=float, default=-96.33, help="area centre longitude")
    p.add_argument("--center-lat", type=float, default=30.62, help="area centre latitude")
    p.add_argument("--size-km", type=float, default=57.6, help="side of the square area (km)")
    p.add_argument("--epsg", type=int, default=32614, help="UTM grid (14N covers Bryan-College Station)")
    p.add_argument("--start", default="2024-05-01")
    p.add_argument("--end", default="2024-09-30")
    p.add_argument("--max-cloud", type=float, default=10.0, help="max scene cloud cover (%%)")
    p.add_argument("--n-days", type=int, default=4, help="clearest acquisition days in the median")
    p.add_argument("--chip", type=int, default=64, help="chip size in 10 m pixels")
    p.add_argument("--purity", type=float, default=0.5,
                   help="min dominant-class share for a chip to enter the evaluation")
    p.add_argument("--min-class-chips", type=int, default=30,
                   help="drop classes with fewer evaluation chips than this")
    p.add_argument("--block", type=int, default=10, help="spatial CV block size (chips)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return p.parse_args()


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------

def rgb(chip: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """True-colour (B04, B03, B02) image with a fixed scene-wide stretch."""
    x = chip[[3, 2, 1]].astype(np.float32)
    x = (x - lo[:, None, None]) / (hi - lo)[:, None, None]
    return np.clip(x, 0, 1).transpose(1, 2, 0) ** 0.8


def class_name(c: int) -> str:
    return data.WORLDCOVER_CLASSES.get(int(c), str(c))


def plot_query_rows(path, rows_spec, chips, labels, lo, hi, title):
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(rows_spec), 9
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.9, n_rows * 2.25))
    axes = np.atleast_2d(axes)
    for r, (qname, q, idx, sims) in enumerate(rows_spec):
        for c in range(n_cols):
            ax = axes[r, c]
            i = q if c == 0 else idx[c - 1]
            ax.imshow(rgb(chips[i], lo, hi))
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_title(f"QUERY\n{qname}", fontsize=8.5, fontweight="bold")
                for s in ax.spines.values():
                    s.set_edgecolor("#d62728"), s.set_linewidth(3)
            else:
                ax.set_title(f"#{c}  cos {sims[c - 1]:.2f}\n{class_name(labels[i])}", fontsize=7.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_hero(path, q, idx, sims, chips, labels, lo, hi, qname):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="#0f1720")
    gs = GridSpec(2, 6, figure=fig, left=0.03, right=0.98, top=0.80, bottom=0.05,
                  wspace=0.08, hspace=0.18, width_ratios=[1.0, 1.0, 0.18, 1, 1, 1])
    ax = fig.add_subplot(gs[:, :2])
    ax.imshow(rgb(chips[q], lo, hi))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#f5b942"), s.set_linewidth(5)
    ax.set_title(f"Query: {qname}", color="white", fontsize=17, pad=10)
    slots = [(r, c) for r in range(2) for c in (3, 4, 5)]
    for j, (r, c) in enumerate(slots):
        a = fig.add_subplot(gs[r, c])
        a.imshow(rgb(chips[idx[j]], lo, hi))
        a.set_xticks([]), a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor("#5b6b7a")
        a.set_title(f"match {j + 1}   cosine {sims[j]:.2f}", color="#d0d7de", fontsize=12)
    fig.text(0.03, 0.93, "Find places that look alike from space", color="white",
             fontsize=30, fontweight="bold")
    fig.text(0.03, 0.865, "Query-by-example search over 640 m Sentinel-2 chips of the Brazos Valley, Texas, "
             "using self-supervised SSL4EO-S12 ResNet-50 embeddings", color="#aab4be", fontsize=15)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_projection(path, emb, spec, labels):
    import matplotlib.pyplot as plt

    try:
        import umap
        method = "UMAP"

        def project(x):
            return umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine",
                             random_state=0).fit_transform(x)
    except ImportError:
        from sklearn.decomposition import PCA
        method = "PCA"

        def project(x):
            return PCA(2, random_state=0).fit_transform(x)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, x, name in [(axes[0], emb, "SSL4EO-S12 embedding (2048-D)"),
                        (axes[1], spec, "Spectral baseline (25-D)")]:
        xy = project(search.standardize(x))
        for c in np.unique(labels):
            m = labels == c
            ax.scatter(xy[m, 0], xy[m, 1], s=5, alpha=0.7, c=data.WORLDCOVER_COLORS[int(c)],
                       label=f"{class_name(c)} (n={m.sum()})", edgecolors="none")
        ax.set_title(f"{method} of {name}")
        ax.set_xticks([]), ax.set_yticks([])
    axes[1].legend(markerscale=3, fontsize=8, loc="best", frameon=True)
    fig.suptitle("Chips coloured by dominant ESA WorldCover 2021 class", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return method


def plot_metrics(path, m):
    import matplotlib.pyplot as plt

    names = ["precision@10\n(micro)", "precision@10\n(class-balanced)",
             "probe accuracy", "probe macro-F1"]
    keys = [("retrieval", "k10", "micro"), ("retrieval", "k10", "macro"),
            ("linear_probe", "accuracy_mean"), ("linear_probe", "macro_f1_mean")]

    def get(feat, key):
        d = m[key[0]][feat]
        for k in key[1:]:
            d = d[k]
        return d

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for off, feat, label, col in [(-w / 2, "embedding", "SSL4EO-S12 embedding", "#2f6db5"),
                                  (w / 2, "spectral", "Spectral baseline", "#a0a7af")]:
        vals = [get(feat, k) for k in keys]
        bars = ax.bar(x + off, vals, w, label=label, color=col)
        ax.bar_label(bars, fmt="%.2f", fontsize=8.5, padding=2)
    chance = [m["retrieval"]["chance"]["micro"], m["retrieval"]["chance"]["macro"]]
    for i, cval in enumerate(chance):
        ax.hlines(cval, i - 0.42, i + 0.42, colors="k", linestyles=":", lw=1)
    ax.text(0.45, chance[0] + 0.02, "chance", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Same-class retrieval and linear probe (spatially blocked CV)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    a = parse_args()
    t0 = time.time()
    timings = {}
    for d in (a.data_dir, a.fig_dir, a.results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Data --------------------------------------------------------------
    geobox = data.make_geobox(a.center_lon, a.center_lat, a.size_km, a.epsg)
    s2_path, wc_path = a.data_dir / "s2_composite.tif", a.data_dir / "worldcover_2021.tif"
    meta_path = a.data_dir / "data_meta.json"
    if not (s2_path.exists() and wc_path.exists() and meta_path.exists()):
        print("Building Sentinel-2 composite (Planetary Computer) ...", flush=True)
        s2_meta = data.build_s2_composite(geobox, a.start, a.end, a.max_cloud, a.n_days, s2_path)
        print("  days:", s2_meta["days"], flush=True)
        wc_meta = data.build_worldcover(geobox, wc_path)
        meta_path.write_text(json.dumps({"sentinel2": s2_meta, "worldcover": wc_meta}, indent=2))
    meta = json.loads(meta_path.read_text())
    timings["data_s"] = time.time() - t0

    img = data.read_raster(s2_path)            # (12, H, W) uint16
    lab = data.read_raster(wc_path)[0]         # (H, W) uint8
    chips, rows, cols = data.tile(img, a.chip)
    lab_chips, _, _ = data.tile(lab, a.chip)
    dom, purity = data.dominant_class(lab_chips)
    nodata_frac = (chips == 0).all(1).mean((1, 2))
    valid = nodata_frac < 0.01
    chips, rows, cols, dom, purity = chips[valid], rows[valid], cols[valid], dom[valid], purity[valid]
    print(f"{len(chips)} valid chips of {a.chip}x{a.chip} px", flush=True)

    # 2. Features ------------------------------------------------------------
    t1 = time.time()
    emb_path = a.data_dir / "embeddings.npy"
    if emb_path.exists() and np.load(emb_path, mmap_mode="r").shape[0] == len(chips):
        emb = np.load(emb_path)
    else:
        model, weights = embed.load_model(device=a.device)
        emb = embed.embed_chips(embed.to_model_bands(chips, data.L2A_BANDS), model, a.device)
        np.save(emb_path, emb)
    timings["embed_s"] = time.time() - t1
    spec = evaluate.spectral_features(chips)
    feats = {"embedding": emb, "spectral": search.standardize(spec)}

    # 3. Query-by-example ---------------------------------------------------
    lo = np.percentile(chips[:, [3, 2, 1]], 1, axis=(0, 2, 3))
    hi = np.percentile(chips[:, [3, 2, 1]], 99.8, axis=(0, 2, 3))
    query_specs = [("Open water (reservoir)", 80), ("Dense built-up", 50), ("Cropland", 40)]
    queries, rows_emb, rows_spec = [], [], []
    with __import__("rasterio").open(s2_path) as src:
        transform, crs = src.transform, src.crs
    from pyproj import Transformer
    to_ll = Transformer.from_crs(crs, 4326, always_xy=True)
    for qname, cls in query_specs:
        cand = np.where(dom == cls)[0]
        if len(cand) == 0:
            continue
        q = int(cand[np.argmax(purity[cand])])
        excl = search.neighbor_mask(rows, cols, q, radius=2)
        idx_e, sim_e = search.cosine_topk(feats["embedding"], q, 8, excl)
        idx_s, sim_s = search.cosine_topk(feats["spectral"], q, 8, excl)
        rows_emb.append((qname, q, idx_e, sim_e))
        rows_spec.append((qname, q, idx_s, sim_s))
        x, y = transform * ((cols[q] + 0.5) * a.chip, (rows[q] + 0.5) * a.chip)
        lon, lat = to_ll.transform(x, y)
        queries.append({
            "name": qname, "chip_index": q, "lon": round(lon, 5), "lat": round(lat, 5),
            "worldcover_class": class_name(cls), "purity": round(float(purity[q]), 3),
            "top8_same_class_embedding": float(np.mean(dom[idx_e] == cls)),
            "top8_same_class_spectral": float(np.mean(dom[idx_s] == cls)),
        })
    plot_query_rows(a.fig_dir / "query_results_embedding.png", rows_emb, chips, dom, lo, hi,
                    "Top-8 most similar chips: SSL4EO-S12 embedding (cosine)")
    plot_query_rows(a.fig_dir / "query_results_spectral_baseline.png", rows_spec, chips, dom, lo, hi,
                    "Top-8 most similar chips: spectral baseline (band mean/std + NDVI, cosine)")
    qname, q, idx_e, sim_e = next(r for r in rows_emb if r[0] == "Dense built-up")
    plot_hero(a.fig_dir / "hero.png", q, idx_e, sim_e, chips, dom, lo, hi, qname)

    # 4. Quantitative evaluation (chips with a clear dominant class) -------
    t2 = time.time()
    keep = purity >= a.purity
    cls_ids, counts = np.unique(dom[keep], return_counts=True)
    good = cls_ids[counts >= a.min_class_chips]
    keep &= np.isin(dom, good)
    y, r_, c_ = dom[keep], rows[keep], cols[keep]
    groups = evaluate.spatial_blocks(r_, c_, a.block)
    metrics = {
        "retrieval": {"chance": evaluate.chance_precision(y)},
        "linear_probe": {},
    }
    # Retrieval variant with z-scored embedding dimensions, scaled the same
    # way as the spectral features. (The probe standardises internally, so
    # it would give the same result for both embedding variants.)
    retrieval_feats = {**feats, "embedding_standardized": search.standardize(emb)}
    metrics["exclusion_sensitivity_p10"] = {}
    for name, f in retrieval_feats.items():
        fk = f[keep]
        metrics["retrieval"][name] = {
            f"k{k}": evaluate.precision_at_k(fk, y, r_, c_, k, exclude_radius=1) for k in (5, 10, 20)}
        metrics["exclusion_sensitivity_p10"][name] = {
            f"radius{r}": evaluate.precision_at_k(fk, y, r_, c_, 10, exclude_radius=r)
            for r in (1, 3, 5)}
        print(name, "P@10 micro/macro:",
              round(metrics["retrieval"][name]["k10"]["micro"], 3),
              round(metrics["retrieval"][name]["k10"]["macro"], 3), flush=True)
    for name, f in feats.items():
        metrics["linear_probe"][name] = evaluate.linear_probe(f[keep], y, groups)
        print(name, "probe acc/F1:", round(metrics["linear_probe"][name]["accuracy_mean"], 3),
              round(metrics["linear_probe"][name]["macro_f1_mean"], 3), flush=True)
    timings["evaluate_s"] = time.time() - t2

    method = plot_projection(a.fig_dir / "projection.png", feats["embedding"][keep],
                             feats["spectral"][keep], y)
    plot_metrics(a.fig_dir / "metrics.png", metrics)
    timings["total_s"] = time.time() - t0

    out = {
        "area": {"center_lon": a.center_lon, "center_lat": a.center_lat, "size_km": a.size_km,
                 "epsg": a.epsg, "resolution_m": 10},
        "imagery": {"date_range": [a.start, a.end], "max_scene_cloud": a.max_cloud, **meta["sentinel2"]},
        "labels": {"product": "ESA WorldCover 2021 v200", **meta["worldcover"]},
        "model": {"torchgeo_weights": f"ResNet50_Weights.{embed.WEIGHTS_NAME}",
                  "embedding_dim": int(emb.shape[1]), "input_size": embed.MODEL_INPUT,
                  "normalization": "x / 10000 (TorchGeo SSL4EO-S12 S2 transform)",
                  "band_order": embed.MODEL_BANDS, "zero_filled_bands": ["B10"]},
        "chips": {"size_px": a.chip, "size_m": a.chip * 10, "n_total": int(len(chips)),
                  "n_eval": int(keep.sum()), "purity_threshold": a.purity,
                  "eval_classes": {class_name(c): int((y == c).sum()) for c in np.unique(y)}},
        "retrieval_protocol": "every eval chip queries all other eval chips; the 8 adjacent chips are excluded",
        "exclusion_sensitivity_protocol": "P@10 with chips within radius 1, 3 or 5 chips "
                                          "(Chebyshev; 640 m, 1.9 km, 3.2 km) of the query excluded",
        "embedding_standardized_note": "embedding z-scored per dimension before cosine retrieval, "
                                       "as the spectral features are; linear probe standardises all features",
        "probe_protocol": f"logistic regression, 5-fold GroupKFold over {a.block}x{a.block}-chip "
                          f"({a.block * a.chip * 10 / 1000:.1f} km) spatial blocks, balanced class weights",
        "projection": method,
        **metrics,
        "queries": queries,
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
    }
    # Readable class names in per-class precision.
    for block in ("retrieval", "exclusion_sensitivity_p10"):
        for name in retrieval_feats:
            for kk in out[block][name].values():
                kk["per_class"] = {class_name(c): round(v, 4) for c, v in kk["per_class"].items()}
    (a.results_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"Done in {timings['total_s'] / 60:.1f} min -> {a.results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
