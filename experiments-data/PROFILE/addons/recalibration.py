"""Experiment B — does per-subject mean-recalibration improve cross-subject accuracy?

Closes the E3 loop. Block C/E3 found that between-subject divergence has a MEAN component
(`kl_removed_by_subject_center` 0.03-0.39): removing each subject's own feature mean eliminates a
measurable share of the divergence. That is descriptive. This experiment asks the ACTIONABLE
question: does removing the per-subject mean actually IMPROVE leave-one-subject-out accuracy?

Three normalisations, same LOSO-LDA loop:
  baseline        : global z-score (train-fit)  -- the standard the paper already uses
  subject_center  : + remove each subject's OWN feature mean (label-free -> available at test time
                    from a few unlabeled windows of the new user)
  subject_zscore  : + remove each subject's own mean AND scale

If subject_center > baseline, the E3 decomposition becomes a concrete recommendation: a cheap,
unsupervised, per-user mean alignment recovers accuracy — the mechanism behind why z-score helps,
made operational. This is the classical mean-only Correlation-Alignment / feature-centering DA
baseline, grounded in Yoneda & Furui 2025.

Caveat reported: per-subject centering uses the subject's OVERALL mean (across their classes), so a
strongly class-imbalanced subject has a class-biased mean. We report per-subject class balance.

Run on the box:  python exp_B_recalibration.py --datasets all --jobs 8
Output: results/experiments/exp_B_recalibration.json
"""
from __future__ import annotations

import numpy as np

from dsprofile import config, windows, progress
from dsprofile.module5_difficulty import _basis


def _fit_acc(Xtr, ytr, Xte, yte):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    if len(np.unique(ytr)) < 2:
        return float("nan")
    clf = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean())


def _per_subject_center(X, subjects):
    Y = X.copy()
    for s in np.unique(subjects):
        m = subjects == s
        Y[m] = Y[m] - Y[m].mean(0)
    return Y


def _per_subject_zscore(X, subjects):
    Y = X.copy()
    for s in np.unique(subjects):
        m = subjects == s
        mu, sd = Y[m].mean(0), Y[m].std(0)
        Y[m] = (Y[m] - mu) / np.where(sd < 1e-12, 1.0, sd)
    return Y


def loso_variants(X, y, subjects, seed=42):
    """Per-subject LOSO accuracy under the three normalisations. Testable on a synthetic frame."""
    subs = sorted(np.unique(subjects))
    acc = {"baseline": {}, "subject_center": {}, "subject_zscore": {}}
    for s in subs:
        tr = subjects != s; te = subjects == s
        if len(np.unique(y[tr])) < 2 or te.sum() < 5:
            continue
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        # baseline: global z-score fit on TRAIN only
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        acc["baseline"][int(s)] = _fit_acc(Ztr, ytr, Zte, yte)

        # subject_center: additionally remove each subject's own mean (train subjects individually;
        # the held-out subject uses ITS OWN mean, estimated unsupervised from its windows)
        Ctr = _per_subject_center(Ztr, subjects[tr])
        Cte = Zte - Zte.mean(0)
        acc["subject_center"][int(s)] = _fit_acc(Ctr, ytr, Cte, yte)

        # subject_zscore: remove per-subject mean and scale
        Ztr2 = _per_subject_zscore(Ztr, subjects[tr])
        Zte2 = (Zte - Zte.mean(0)) / np.where(Zte.std(0) < 1e-12, 1.0, Zte.std(0))
        acc["subject_zscore"][int(s)] = _fit_acc(Ztr2, ytr, Zte2, yte)
    return acc


def _paired_delta(a, b):
    """Paired (per-subject) improvement of b over a, with a Wilcoxon signed-rank p."""
    common = sorted(set(a) & set(b))
    xa = np.array([a[s] for s in common], float)
    xb = np.array([b[s] for s in common], float)
    ok = np.isfinite(xa) & np.isfinite(xb)
    xa, xb = xa[ok], xb[ok]
    if len(xa) < 5:
        return dict(n=int(len(xa)), note="too few subjects")
    from scipy.stats import wilcoxon
    d = xb - xa
    if np.allclose(d, 0.0):                          # identical -> Wilcoxon undefined; p := 1
        w, p = float("nan"), 1.0
    else:
        try:
            w, p = wilcoxon(xb, xa, alternative="greater")
        except ValueError:
            w, p = float("nan"), 1.0
    return dict(n=int(len(xa)), mean_baseline=float(xa.mean()), mean_recal=float(xb.mean()),
                mean_delta=float(d.mean()), median_delta=float(np.median(d)),
                n_improved=int((d > 0).sum()), frac_improved=float((d > 0).mean()),
                wilcoxon_p=float(p), helps=bool(d.mean() > 0 and p < 0.05))


def run_dataset(dataset, seed=42, n_jobs=None):
    """n_jobs is accepted for a uniform driver signature; B has no internal joblib (it is a plain
    LOSO-LDA loop). Parallelise B by SHARDING datasets across terminals."""
    with progress.timer(f"B :: {dataset}"):
        frame = windows.build_fast_frame(dataset, seed=seed)
        X = _basis(frame); y = frame.label.to_numpy(); subj = frame.subject.to_numpy()
        acc = loso_variants(X, y, subj, seed)
    bal = []
    for s in np.unique(subj):                        # per-subject class balance (imbalance caveat)
        _, c = np.unique(y[subj == s], return_counts=True)
        bal.append(float(c.max() / c.min()) if c.min() > 0 else float("inf"))
    return dict(
        dataset=dataset, n_subjects=len(acc["baseline"]),
        mean_acc={k: float(np.nanmean(list(v.values()))) for k, v in acc.items() if v},
        center_vs_baseline=_paired_delta(acc["baseline"], acc["subject_center"]),
        zscore_vs_baseline=_paired_delta(acc["baseline"], acc["subject_zscore"]),
        median_class_imbalance_ratio=float(np.median(bal)) if bal else float("nan"),
        per_subject_acc=acc,
    )


def build_summary(results):
    valid = [r for r in results.values()
             if "error" not in r and "mean_delta" in (r.get("center_vs_baseline") or {})]
    n_help = sum(1 for r in valid if r["center_vs_baseline"].get("helps"))
    mean_delta = (float(np.mean([r["center_vs_baseline"]["mean_delta"] for r in valid]))
                  if valid else float("nan"))
    return dict(
        n_datasets=len(valid), n_datasets_center_helps_significantly=n_help,
        mean_delta_across_datasets=mean_delta,
        per_dataset={ds: dict(mean_acc=r.get("mean_acc"),
                              center_vs_baseline=r.get("center_vs_baseline"),
                              zscore_vs_baseline=r.get("zscore_vs_baseline"))
                     for ds, r in results.items()},
        verdict=("per-subject mean-centering improves LOSO accuracy on "
                 f"{n_help}/{len(valid)} datasets -> E3's mean-shift finding is ACTIONABLE"
                 if valid and n_help > len(valid) / 2 else
                 "per-subject mean-centering does not reliably help -> E3's mean component is real "
                 "but not the dominant obstacle to cross-subject accuracy"),
        _console=[f"centering helps significantly on {n_help}/{len(valid)} datasets",
                  f"mean accuracy delta across datasets: {mean_delta:+.4f}"],
    )


if __name__ == "__main__":
    from addons import common as exp_common
    exp_common.main("B", run_one=run_dataset, build_summary=build_summary,
                    all_datasets=config.ALL14)
