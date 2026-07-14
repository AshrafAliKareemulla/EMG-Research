"""Block F — cross-dataset transfer-compatibility matrix.

Datasets have different channel counts, so we project every window onto a CHANNEL-AGNOSTIC shared
space: the representative-basis features averaged across channels (fixed length regardless of #ch).
Then we compute pairwise MMD between datasets on this shared space -> a dataset x dataset
compatibility matrix (low MMD = distributions close = good transfer/pretraining candidates).
Answers "which dataset is the best pretraining source for which" (scientific question A6).

Aggregate experiment: iterates all datasets once. Scalable + dataset-agnostic.
"""
from __future__ import annotations

import json

import numpy as np

from . import config, windows, progress
from .module3_shift import mmd_rbf


def _shared_vector(frame):
    """Channel-average each representative feature -> one row per window, len = |REPR_BASIS|.

    NOTE (2026-07-10 audit): this used to z-score EACH dataset independently before the MMD.
    That maps every dataset to mean 0 / unit variance and therefore deletes exactly the
    between-dataset location and scale difference that "compatibility" is supposed to measure.
    The surviving MMDs (0.002-0.108) were shape-only. Standardisation is now done ONCE with a
    scaler fitted on the POOLED windows of all datasets (see `run`), so a dataset that sits far
    from the others in feature space stays far.
    """
    cols = {}
    for fb in config.REPR_BASIS:
        cc = [c for c in frame.columns if c.startswith(fb + "_c")]
        if cc:
            cols[fb] = frame[cc].mean(axis=1).to_numpy(np.float64)
    if not cols:
        return None
    return np.nan_to_num(np.column_stack([cols[k] for k in sorted(cols)]))


def _matrix(datasets, seed, sample, signal_norm):
    """Pairwise MMD in a shared, POOLED-standardised, channel-agnostic space."""
    rng = np.random.default_rng(seed)
    reps = {}
    for ds in datasets:
        try:
            frame = windows.build_fast_frame(ds, seed=seed, normalize=signal_norm)
        except Exception as e:
            progress.log(f"transfer[{signal_norm}]: skip {ds} ({type(e).__name__})"); continue
        V = _shared_vector(frame)
        if V is None:
            continue
        reps[ds] = V if len(V) <= sample else V[rng.choice(len(V), sample, replace=False)]
    keys = sorted(reps)
    if len(keys) < 2:
        return keys, np.zeros((len(keys), len(keys)))

    # ONE scaler for the union of all datasets -> between-dataset location/scale is preserved.
    P = np.vstack([reps[k] for k in keys])
    mu, sd = P.mean(0), P.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    reps = {k: (v - mu) / sd for k, v in reps.items()}

    n = len(keys)
    M = np.zeros((n, n))
    with progress.timer(f"transfer[{signal_norm}]: {n*(n-1)//2} dataset pairs"):
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = M[j, i] = mmd_rbf(reps[keys[i]], reps[keys[j]], rng=rng, n=sample)
    return keys, M


def run(datasets=None, seed=42, sample=2000):
    config.ensure_dirs()
    outdir = config.RESULTS_DIR / "transfer"
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = datasets or config.ALL14

    # `none`   : signals kept at native scale -> the true cross-dataset distance (device, gain,
    #            electrode, population all still present). This is the compatibility matrix.
    # `global` : signals per-dataset z-scored -> location/scale already removed upstream, so the
    #            MMD reflects distribution SHAPE only. Kept as the "after normalisation" view.
    keys_raw, M_raw = _matrix(datasets, seed, sample, "none")
    keys_g, M_g = _matrix(datasets, seed, sample, "global")

    def best(keys, M):
        out = {}
        for i, ds in enumerate(keys):
            others = [(keys[j], M[i, j]) for j in range(len(keys)) if j != i]
            out[ds] = min(others, key=lambda kv: kv[1])[0] if others else None
        return out

    result = dict(
        datasets=keys_raw,
        signal_normalization_note=(
            "compatibility_mmd is computed on NATIVE-scale signals (normalize='none'). The "
            "previous version standardised each dataset independently before the MMD, which "
            "removes exactly the between-dataset location/scale difference it claims to "
            "measure; those distances (0.002-0.108) were shape-only. That view is retained "
            "as `shape_only_mmd`."),
        compatibility_mmd={keys_raw[i]: {keys_raw[j]: float(M_raw[i, j])
                                         for j in range(len(keys_raw))}
                           for i in range(len(keys_raw))},
        shape_only_mmd={keys_g[i]: {keys_g[j]: float(M_g[i, j]) for j in range(len(keys_g))}
                        for i in range(len(keys_g))},
        best_pretraining_source=best(keys_raw, M_raw),
        best_pretraining_source_shape_only=best(keys_g, M_g),
        validated=False,
        validation_caveat=(
            "EXPLORATORY. No actual transfer/pretraining accuracy was measured, so "
            "`best_pretraining_source` is an unvalidated ranking of a distance, not a "
            "demonstrated transfer result. It answers A6 only as groundwork. Datasets also "
            "differ in device, fs and channel count, all of which the channel-averaged shared "
            "space confounds with population difference."),
    )
    np.savez(outdir / "transfer_matrix.npz", datasets=np.array(keys_raw),
             mmd=M_raw, mmd_shape_only=M_g)
    (outdir / "transfer.json").write_text(json.dumps(result, indent=2))
    return result
