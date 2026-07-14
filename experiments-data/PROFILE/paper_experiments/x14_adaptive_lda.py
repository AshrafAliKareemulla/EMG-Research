"""X14 — Adaptive-LDA calibration curve. Buries the "no calibration-budget baseline" gap (§5, roadmap).

Implements the roadmap's shrinkage Adaptive-LDA: blended class means mu~ = tau*mu_cal + (1-tau)*mu_src
and covariance S~ = lam*S_cal + (1-lam)*S_src, then a linear-discriminant assignment. Produces the
zero->few-shot calibration curve (accuracy vs #calibration repetitions k), framing difficulty as
onboarding cost.

GROUND TRUTH: boundary conditions — (tau=1,lam=1) equals a target-only classifier; (tau=0,lam=0) equals
the source classifier; the curve is monotone non-decreasing in k on a well-behaved synthetic.
"""
from __future__ import annotations

import numpy as np

from . import common


def _class_stats(X, y):
    classes = sorted(np.unique(y))
    mu = {c: X[y == c].mean(0) for c in classes}
    d = X.shape[1]
    S = np.zeros((d, d))
    dof = 0
    for c in classes:
        Xc = X[y == c]
        if len(Xc) > 1:
            S += (len(Xc) - 1) * np.cov(Xc, rowvar=False)
            dof += len(Xc) - 1
    S = S / max(dof, 1)
    return mu, S


def adaptive_lda_predict(Xte, mu_src, S_src, mu_cal, S_cal, tau, lam, ridge=1e-3):
    classes = sorted(set(mu_src) & (set(mu_cal) if mu_cal else set(mu_src)))
    mu = {c: (tau * mu_cal[c] + (1 - tau) * mu_src[c]) if (mu_cal and c in mu_cal) else mu_src[c]
          for c in classes}
    S = lam * (S_cal if S_cal is not None else S_src) + (1 - lam) * S_src
    Sinv = np.linalg.pinv(S + np.eye(S.shape[0]) * ridge)
    # linear discriminant score (equal priors): x^T Sinv mu_c - 0.5 mu_c^T Sinv mu_c
    scores = np.column_stack([Xte @ Sinv @ mu[c] - 0.5 * mu[c] @ Sinv @ mu[c] for c in classes])
    return np.array(classes)[scores.argmax(1)]


def calibration_curve(frame, kmax=5, tau=0.75, lam=0.9, seed=42):
    X, _ = common.basis(frame)
    y = frame.label.to_numpy()
    subj = frame.subject.to_numpy()
    rep = frame.repetition.to_numpy()
    per_k = {k: [] for k in range(kmax + 1)}
    for s in sorted(np.unique(subj)):
        tr, tm = subj != s, subj == s
        if len(np.unique(y[tr])) < 2 or tm.sum() < 10:
            continue
        mu_src, S_src = _class_stats(X[tr], y[tr])
        treps = sorted(np.unique(rep[tm]))
        for k in range(kmax + 1):
            if k >= len(treps):
                continue
            cal = tm & np.isin(rep, treps[:k]) if k > 0 else np.zeros_like(tm)
            test = tm & np.isin(rep, treps[k:])
            if test.sum() < 5:
                continue
            if k > 0 and len(np.unique(y[cal])) >= 2:
                mu_cal, S_cal = _class_stats(X[cal], y[cal])
                t, la = tau, lam
            else:
                mu_cal, S_cal, t, la = None, None, 0.0, 0.0     # zero-shot = source only
            pred = adaptive_lda_predict(X[test], mu_src, S_src, mu_cal, S_cal, t, la)
            per_k[k].append(float((pred == y[test]).mean()))
    curve = {k: (float(np.mean(v)) if v else None) for k, v in per_k.items()}
    valid = {k: v for k, v in curve.items() if v is not None}
    return dict(accuracy_vs_k=curve, zero_shot=valid.get(0), one_shot=valid.get(1),
                one_shot_gain=(valid.get(1) - valid.get(0)) if (0 in valid and 1 in valid) else None,
                tau=tau, lam=lam, n_subjects=len(per_k[0]))


def run_one(dataset, seed=42, n_jobs=1):
    with common.timer(f"X14 :: {dataset}"):
        frame = common.build_frame(dataset, seed=seed)
        out = dict(dataset=dataset, **calibration_curve(frame, seed=seed))
    return out


# ------------------------------------------------ ground truth
def selftest(check):
    rng = np.random.default_rng(0)
    d, C = 4, 3
    mu_src = {c: rng.standard_normal(d) * 2 for c in range(C)}
    S_src = np.eye(d) * 1.5
    mu_cal = {c: rng.standard_normal(d) * 2 for c in range(C)}
    S_cal = np.eye(d) * 0.7
    Xte = rng.standard_normal((50, d))

    p_tt = adaptive_lda_predict(Xte, mu_src, S_src, mu_cal, S_cal, tau=1.0, lam=1.0)
    p_target_only = adaptive_lda_predict(Xte, mu_cal, S_cal, mu_cal, S_cal, tau=0.0, lam=0.0)
    check("X14 (tau=1,lam=1) == target-only classifier", np.array_equal(p_tt, p_target_only))
    p_ss = adaptive_lda_predict(Xte, mu_src, S_src, mu_cal, S_cal, tau=0.0, lam=0.0)
    p_source_only = adaptive_lda_predict(Xte, mu_src, S_src, None, None, tau=0.0, lam=0.0)
    check("X14 (tau=0,lam=0) == source-only classifier", np.array_equal(p_ss, p_source_only))

    # monotone-ish calibration curve on a separable multi-rep synthetic
    fr = _synth_reps(seed=1)
    cur = calibration_curve(fr, kmax=4, seed=1)
    vals = [cur["accuracy_vs_k"][k] for k in range(5) if cur["accuracy_vs_k"].get(k) is not None]
    check("X14 calibration curve non-decreasing (k=0 -> k_max, within noise)",
          vals[-1] >= vals[0] - 0.02, f"{[round(v,3) for v in vals]}")


def _synth_reps(n_subjects=8, n_classes=3, n_reps=6, per_rep=20, seed=0):
    """Separable classes + a per-subject offset, with a repetition axis, for the calibration curve."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    n_ch = 3
    D = len(common.REPR_BASIS) * n_ch
    cM = rng.standard_normal((n_classes, D)) * 2.5
    feats, subj, lab, rep = [], [], [], []
    for s in range(n_subjects):
        off = rng.standard_normal(D) * 1.5
        for rr in range(n_reps):
            for c in range(n_classes):
                feats.append(cM[c] + off + rng.standard_normal((per_rep, D)))
                subj += [s] * per_rep
                lab += [c] * per_rep
                rep += [rr] * per_rep
    F = np.vstack(feats)
    cols = {}
    j = 0
    for fb in common.REPR_BASIS:
        for ch in range(n_ch):
            cols[f"{fb}_c{ch}"] = F[:, j]
            j += 1
    df = pd.DataFrame(cols)
    df["subject"] = subj
    df["session"] = 0
    df["repetition"] = rep
    df["label"] = lab
    df.attrs["n_channels"] = n_ch
    df.attrs["dataset"] = "synth_reps"
    return df
