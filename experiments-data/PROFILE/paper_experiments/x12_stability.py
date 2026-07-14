"""X12 — Subsample-stability / convergence. Buries the "are the caps adequate?" limitation (§8).

Sweeps the per-subject window cap and re-estimates the inter-subject MMD over repeats; a converged
estimate has a standard deviation that shrinks ~1/sqrt(n). Establishes the 40/600 caps are adequate
(or exposes where they are not).

GROUND TRUTH: on a synthetic stationary source the estimate's std must shrink monotonically as the
subsample grows.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import common


def stability_curve(X, subjects, caps=(50, 100, 200, 400, 600), reps=8, seed=42):
    X = np.asarray(X, float)
    subjects = np.asarray(subjects)
    subs = [s for s in np.unique(subjects) if (subjects == s).sum() >= 40]
    curve = {}
    for cap in caps:
        vals = []
        for rp in range(reps):
            rng = np.random.default_rng(seed + rp)
            groups = {}
            for s in subs:
                idx = np.where(subjects == s)[0]
                take = min(cap, len(idx))
                groups[s] = X[rng.choice(idx, take, replace=False)]
            keys = list(groups)
            if len(keys) < 2:
                continue
            pv = [common.mmd_rbf(groups[a], groups[b], rng=rng)
                  for a, b in itertools.combinations(keys, 2)]
            vals.append(float(np.mean(pv)))
        if vals:
            curve[int(cap)] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), reps=len(vals))
    return curve


def run_one(dataset, seed=42, n_jobs=1):
    with common.timer(f"X12 :: {dataset}"):
        frame = common.build_frame(dataset, seed=seed)
        X, _ = common.basis(frame)
        curve = stability_curve(X, frame.subject.to_numpy(), seed=seed)
    caps = sorted(curve)
    stds = [curve[c]["std"] for c in caps]
    converged = bool(len(stds) >= 2 and stds[-1] <= stds[0])
    return dict(dataset=dataset, curve=curve, converged=converged,
                std_first=stds[0] if stds else None, std_last=stds[-1] if stds else None)


# ------------------------------------------------ ground truth
def selftest(check):
    fr = common.synth_frame("separable", n_subjects=10, n_classes=6, per_class=120, seed=5)
    X, _ = common.basis(fr)
    curve = stability_curve(X, fr.subject.to_numpy(), caps=(50, 100, 200, 400), reps=8, seed=5)
    caps = sorted(curve)
    stds = [curve[c]["std"] for c in caps]
    check("X12 estimate std shrinks as subsample grows", stds[-1] < stds[0],
          f"std {caps[0]}={stds[0]:.4f} -> {caps[-1]}={stds[-1]:.4f}")
