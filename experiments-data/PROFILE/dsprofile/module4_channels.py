"""Module 4 — channel & sensor analysis.

Channel redundancy via Pearson correlation + normalized mutual information (NMI) between
channels (summaries 08/12), a greedy min-redundancy/max-relevance minimal channel subset,
and per-channel Fisher discriminability. Sampling-rate sufficiency is a Phase-C extension
(needs re-windowing at decimated fs); stubbed here with a note.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, windows


def _channel_signal_matrix(frame, base="RMS"):
    """One representative value per (window, channel) -> (n_windows, n_channels)."""
    cols = sorted([c for c in frame.columns if c.startswith(base + "_c")],
                  key=lambda c: int(c.split("_c")[1]))
    return np.nan_to_num(frame[cols].to_numpy(np.float64)), cols


def nmi_matrix(M, bins=16):
    from sklearn.metrics import normalized_mutual_info_score
    C = M.shape[1]
    disc = np.zeros_like(M, dtype=int)
    for c in range(C):
        disc[:, c] = np.digitize(M[:, c], np.histogram_bin_edges(M[:, c], bins=bins)[1:-1])
    out = np.eye(C)
    for i in range(C):
        for j in range(i + 1, C):
            out[i, j] = out[j, i] = normalized_mutual_info_score(disc[:, i], disc[:, j])
    return out


def per_channel_fisher(frame):
    """Fisher ratio using each channel's RMS across classes -> channel relevance."""
    M, cols = _channel_signal_matrix(frame, "RMS")
    y = frame.label.to_numpy()
    classes = np.unique(y)
    rel = []
    for c in range(M.shape[1]):
        x = M[:, c]; overall = x.mean()
        sb = sum(len(x[y == k]) * (x[y == k].mean() - overall) ** 2 for k in classes)
        sw = sum(((x[y == k] - x[y == k].mean()) ** 2).sum() for k in classes)
        rel.append(sb / (sw + 1e-12))
    return np.array(rel)


def greedy_mrmr(relevance, redundancy, n_select=None):
    """min-Redundancy Max-Relevance greedy channel ranking (summary 08)."""
    C = len(relevance)
    n_select = n_select or C
    selected = [int(np.argmax(relevance))]
    remaining = set(range(C)) - set(selected)
    while remaining and len(selected) < n_select:
        best, best_score = None, -np.inf
        for c in remaining:
            red = np.mean([redundancy[c, s] for s in selected])
            score = relevance[c] - red
            if score > best_score:
                best, best_score = c, score
        selected.append(best); remaining.discard(best)
    return selected


def run(dataset, seed=42):
    config.ensure_dirs()
    frame = windows.build_fast_frame(dataset, seed=seed)
    M, cols = _channel_signal_matrix(frame, "RMS")
    C = M.shape[1]
    corr = np.corrcoef(M, rowvar=False)
    nmi = nmi_matrix(M)
    relevance = per_channel_fisher(frame)
    ranking = greedy_mrmr(relevance, nmi)

    # minimal subset reaching 90% of full-set summed relevance
    order_rel = relevance[ranking]
    cum = np.cumsum(order_rel) / (order_rel.sum() + 1e-12)
    k90 = int(np.searchsorted(cum, 0.90) + 1)

    result = dict(
        dataset=dataset, n_channels=C,
        mean_abs_corr=float(np.abs(corr[~np.eye(C, dtype=bool)]).mean()),
        mean_nmi=float(nmi[~np.eye(C, dtype=bool)].mean()),
        channel_relevance=[float(x) for x in relevance],
        mrmr_ranking=[int(x) for x in ranking],
        min_channels_for_90pct_relevance=k90,
        sampling_rate_sufficiency="Phase-C: re-run separability after decimation (not yet implemented)",
    )
    outdir = config.RESULTS_DIR / "module4"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / f"{dataset}__channels.npz", corr=corr, nmi=nmi, relevance=relevance)
    (outdir / f"{dataset}__channels.json").write_text(json.dumps(result, indent=2))
    return result
