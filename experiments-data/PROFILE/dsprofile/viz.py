"""Figures: distribution-shift heatmaps, mean-vs-cov decomposition, difficulty scatter.

matplotlib only (no seaborn); saves PNGs to results/figures/. Import errors are non-fatal.
"""
from __future__ import annotations

import numpy as np

from . import config


def _ax():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def shift_heatmap(dataset):
    plt = _ax()
    f = config.find("module3", f"{dataset}__shift_matrices.npz")
    if not f.exists():
        return None
    d = np.load(f)
    subs = d["subjects"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(d["mmd"], cmap="magma")
    ax.set_title(f"{dataset}: inter-subject MMD")
    ax.set_xticks(range(len(subs))); ax.set_xticklabels(subs, fontsize=6, rotation=90)
    ax.set_yticks(range(len(subs))); ax.set_yticklabels(subs, fontsize=6)
    fig.colorbar(im)
    out = config.RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{dataset}__mmd_heatmap.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return str(p)


def difficulty_scatter(dataset):
    plt = _ax()
    import pandas as pd
    f = config.find("module5", f"{dataset}__difficulty.parquet")
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df["mmd_to_pool"], df["loso_acc"])
    ax.set_xlabel("MMD to training pool (cheap statistic)")
    ax.set_ylabel("LOSO accuracy (target)")
    ax.set_title(f"{dataset}: difficulty predictor")
    out = config.RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{dataset}__difficulty_scatter.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return str(p)
