"""E8 — calibration curve (Block E): cross-subject accuracy vs #calibration repetitions.

For each held-out target subject, an LDA is trained on ALL other subjects (the source pool) PLUS the
first k repetitions from the target subject, and tested on the target's remaining repetitions.
k=0 is zero-shot (source only). Averaged over subjects -> the calibration curve (accuracy vs k).
Reframes subject difficulty as onboarding cost (scientific question A8). Self-contained (self-LDA).

Scalable + dataset-agnostic; uses the cached fast frame.
"""
from __future__ import annotations

import json

import numpy as np

from . import config, windows
from .module3_shift import _basis


def calibration_curve(frame, kmax=5, seed=42):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = _basis(frame); y = frame["label"].to_numpy()
    subj = frame["subject"].to_numpy(); rep = frame["repetition"].to_numpy()
    subs = sorted(np.unique(subj))
    per_k = {k: [] for k in range(0, kmax + 1)}
    for s in subs:
        tmask = subj == s; omask = ~tmask
        if len(np.unique(y[omask])) < 2 or tmask.sum() < 10:
            continue
        treps = sorted(np.unique(rep[tmask]))
        for k in range(0, kmax + 1):
            if k >= len(treps):
                continue
            cal = tmask & np.isin(rep, treps[:k]) if k > 0 else np.zeros_like(tmask)
            test = tmask & np.isin(rep, treps[k:])
            if test.sum() < 5:
                continue
            Xtr = np.vstack([X[omask], X[cal]]) if k > 0 else X[omask]
            ytr = np.concatenate([y[omask], y[cal]]) if k > 0 else y[omask]
            if len(np.unique(ytr)) < 2:
                continue
            clf = LinearDiscriminantAnalysis().fit(Xtr, ytr)
            per_k[k].append(float((clf.predict(X[test]) == y[test]).mean()))
    curve = {k: (float(np.mean(v)) if v else None) for k, v in per_k.items()}
    valid = {k: v for k, v in curve.items() if v is not None}
    gain = None
    if 0 in valid and 1 in valid:
        gain = float(valid[1] - valid[0])
    return dict(accuracy_vs_k=curve, n_subjects_used=len(per_k[0]),
                zero_shot=valid.get(0), one_shot=valid.get(1),
                one_shot_gain=gain)


def run(dataset, seed=42):
    config.ensure_dirs()
    outdir = config.RESULTS_DIR / "calibration"
    outdir.mkdir(parents=True, exist_ok=True)
    frame = windows.build_fast_frame(dataset, seed=seed)
    result = dict(dataset=dataset, E8_calibration=calibration_curve(frame, seed=seed))
    (outdir / f"{dataset}__calibration.json").write_text(json.dumps(result, indent=2))
    return result
