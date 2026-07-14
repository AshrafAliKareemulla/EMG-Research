"""X13 — Class-imbalance stratification. Buries exp_B's own caveat about biased subject means (§2).

Does per-subject mean-centring help LESS (or hurt) on class-imbalanced subjects, whose global mean is
class-biased? For each subject we compute (a) its class-imbalance ratio and (b) its centring benefit
(subject_center LOSO acc - baseline LOSO acc), and correlate them.

GROUND TRUTH: with a PLANTED negative relationship between imbalance and benefit, the analysis recovers
a negative Spearman; with imbalance independent of benefit, it recovers ~0.
"""
from __future__ import annotations

import numpy as np

from . import common
from .x4_recalibration_coral import loso_variants


def imbalance_ratio(y_subject):
    _, c = np.unique(y_subject, return_counts=True)
    return float(c.max() / c.min()) if c.min() > 0 else float("inf")


def stratify(frame, seed=42):
    X, _ = common.basis(frame)
    y, subj = frame.label.to_numpy(), frame.subject.to_numpy()
    acc = loso_variants(X, y, subj, seed)
    base, cen = acc["baseline"], acc["subject_center"]
    subs = sorted(set(base) & set(cen))
    imb = np.array([imbalance_ratio(y[subj == s]) for s in subs], float)
    benefit = np.array([cen[s] - base[s] for s in subs], float)
    r, p, n = common.spearman(imb, benefit)
    return dict(n_subjects=len(subs), spearman_imbalance_vs_center_benefit=r, p=p, n=n,
                median_imbalance=float(np.median(imb)) if len(imb) else float("nan"),
                mean_center_benefit=float(np.nanmean(benefit)) if len(benefit) else float("nan"),
                interpretation="negative Spearman => centring helps imbalanced subjects less (biased mean).")


def run_one(dataset, seed=42, n_jobs=1):
    with common.timer(f"X13 :: {dataset}"):
        frame = common.build_frame(dataset, seed=seed)
        out = dict(dataset=dataset, **stratify(frame, seed))
    return out


# ------------------------------------------------ ground truth
def selftest(check):
    rng = np.random.default_rng(0)
    imb = rng.uniform(1, 8, 40)
    # planted NEGATIVE relationship
    ben_neg = -0.01 * imb + 0.01 * rng.standard_normal(40)
    r_neg, _, _ = common.spearman(imb, ben_neg)
    check("X13 recovers a planted negative imbalance<->benefit correlation", r_neg < -0.4, f"{r_neg:.3f}")
    # independent
    ben_ind = 0.01 * rng.standard_normal(40)
    r_ind, _, _ = common.spearman(imb, ben_ind)
    check("X13 finds ~0 when imbalance is independent of benefit", abs(r_ind) < 0.4, f"{r_ind:.3f}")
