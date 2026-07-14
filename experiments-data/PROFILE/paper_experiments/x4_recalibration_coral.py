"""X4 — CORAL / covariance-alignment vs mean-centring recalibration. Strengthens N5, reconciles E3.

exp_B showed per-subject mean-centring recovers +3.6pp on 13/14. E3 said the between-subject KL is
covariance-dominated. So: does aligning per-subject COVARIANCE (CORAL) beat aligning only the MEAN?
Four label-free LOSO-LDA arms per subject: baseline / subject_center / subject_coral / center+coral,
paired Wilcoxon vs baseline.

GROUND TRUTH:
  pure covariance rotation -> CORAL recovers accuracy, centring does NOT.
  pure mean offset         -> centring recovers, CORAL ~ baseline.
"""
from __future__ import annotations

import numpy as np

from . import common
from .common import coral, per_subject_center


def _fit_predict(Xtr, ytr, Xte, yte):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    if len(np.unique(ytr)) < 2:
        return float("nan")
    try:
        return float((LinearDiscriminantAnalysis().fit(Xtr, ytr).predict(Xte) == yte).mean())
    except Exception:
        return float("nan")


def loso_variants(X, y, subjects, seed=42):
    """Per-subject LOSO accuracy under 4 label-free normalisations. All standardisation train-only."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    acc = {k: {} for k in ("baseline", "subject_center", "subject_coral", "center_plus_coral")}
    for s in sorted(np.unique(subjects)):
        tr, te = subjects != s, subjects == s
        if len(np.unique(y[tr])) < 2 or te.sum() < 5:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        acc["baseline"][int(s)] = _fit_predict(Ztr, y[tr], Zte, y[te])
        # centring (deployment: new user removes its own unlabeled mean; train subjects each centred)
        Ctr = per_subject_center(Ztr, subjects[tr])
        acc["subject_center"][int(s)] = _fit_predict(Ctr, y[tr], Zte - Zte.mean(0), y[te])
        # CORAL: align the new user's covariance to the training pool (model trained on Ztr)
        acc["subject_coral"][int(s)] = _fit_predict(Ztr, y[tr], coral(Zte, Ztr), y[te])
        # both
        acc["center_plus_coral"][int(s)] = _fit_predict(Ctr, y[tr], coral(Zte - Zte.mean(0), Ctr), y[te])
    return acc


def _paired(a, b):
    common_s = sorted(set(a) & set(b))
    xa = np.array([a[s] for s in common_s], float)
    xb = np.array([b[s] for s in common_s], float)
    m = np.isfinite(xa) & np.isfinite(xb)
    xa, xb = xa[m], xb[m]
    if len(xa) < 5:
        return dict(n=int(len(xa)), note="too few subjects")
    from scipy.stats import wilcoxon
    d = xb - xa
    if np.allclose(d, 0):
        p = 1.0
    else:
        try:
            _, p = wilcoxon(xb, xa, alternative="greater")
        except ValueError:
            p = 1.0
    return dict(n=int(len(xa)), mean_baseline=float(xa.mean()), mean_variant=float(xb.mean()),
                mean_delta=float(d.mean()), n_improved=int((d > 0).sum()),
                wilcoxon_p=float(p), helps=bool(d.mean() > 0 and p < 0.05))


def summarise(acc):
    base = acc["baseline"]
    return {k: _paired(base, acc[k]) for k in ("subject_center", "subject_coral", "center_plus_coral")}


def run_one(dataset, seed=42, n_jobs=1):
    with common.timer(f"X4 :: {dataset}"):
        frame = common.build_frame(dataset, seed=seed)
        X, _ = common.basis(frame)
        acc = loso_variants(X, frame.label.to_numpy(), frame.subject.to_numpy(), seed)
    out = summarise(acc)
    out["dataset"] = dataset
    out["mean_acc"] = {k: float(np.nanmean(list(v.values()))) for k, v in acc.items() if v}
    return out


# ------------------------------------------------ ground truth
def selftest(check):
    # The transforms' DEFINING properties are provable (the end-to-end accuracy gain is a
    # data-dependent RESULT to report, not a ground truth to assert).
    rng = np.random.default_rng(0)
    d = 5
    # CORAL: cov(coral(Xs -> Xt)) must match cov(Xt)
    Xs = rng.standard_normal((800, d)) @ rng.standard_normal((d, d))
    Xt = rng.standard_normal((800, d)) @ rng.standard_normal((d, d))
    Cc = np.cov(coral(Xs, Xt), rowvar=False)
    Ct = np.cov(Xt, rowvar=False)
    rel = float(np.linalg.norm(Cc - Ct) / (np.linalg.norm(Ct) + 1e-12))
    check("X4 CORAL aligns covariance: cov(coral(Xs->Xt)) ~ cov(Xt)", rel < 0.15, f"relerr={rel:.3f}")
    check("X4 CORAL is ~identity when source and target already match",
          float(np.linalg.norm(np.cov(coral(Xs, Xs), rowvar=False) - np.cov(Xs, rowvar=False))
                / (np.linalg.norm(np.cov(Xs, rowvar=False)) + 1e-12)) < 0.05)
    # per-subject centering zeroes each subject's mean
    subj = np.repeat([0, 1, 2], 100)
    means = np.array([[0.0, 0.0, 0.0, 0.0], [3.0, -2.0, 1.0, 0.5], [-1.0, 4.0, -2.0, 1.0]])
    X = rng.standard_normal((300, 4)) + means[subj]           # distinct per-subject offset
    Xc = per_subject_center(X, subj)
    max_mean = max(abs(Xc[subj == s].mean(0)).max() for s in np.unique(subj))
    check("X4 per-subject centering zeroes every subject mean", max_mean < 1e-9, f"max|mean|={max_mean:.1e}")
    # the LOSO variant loop runs and returns finite accuracies on a real-shaped synthetic frame
    fr = common.synth_frame("real_difficulty", n_subjects=10, n_classes=5, per_class=40, seed=3)
    Xfr, _ = common.basis(fr)
    ac = loso_variants(Xfr, fr.label.to_numpy(), fr.subject.to_numpy(), 3)
    check("X4 all four LOSO arms produce finite accuracies",
          all(len(v) >= 5 and all(np.isfinite(list(v.values()))) for v in ac.values()),
          f"n={ {k: len(v) for k, v in ac.items()} }")
