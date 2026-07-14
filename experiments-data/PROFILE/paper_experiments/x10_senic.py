"""X10 — senic as an electrode-shift testbed + within-condition null. Buries the senic outlier (§6).

senic's "sessions" are electrode-shift / rotation / fatigue CONDITIONS. This (a) measures whether
MMD-across-conditions predicts per-condition accuracy drop within a subject, (b) subtracts a
within-condition NULL (MMD between trial-disjoint halves of the SAME condition, which must be ~0 if the
shift signal is not trial memorisation), and (c) decomposes the shift into amplitude vs shape (X5
bases) to probe the sign reversal. Converts a quarantined outlier into an electrode-shift contribution.

GROUND TRUTH: on stationary synthetic data the within-condition null is ~0 (two halves of one
condition are indistinguishable); a genuine between-condition shift is detected above it.
"""
from __future__ import annotations

import numpy as np

from . import common

AMP = ["MAV", "WL", "RMS", "MFL"]
SHAPE = ["HJ_MOB", "HJ_COM", "WAMP"]


def _halves(reps, rng):
    u = rng.permutation(np.unique(reps))
    h = len(u) // 2
    return set(u[:h].tolist()), set(u[h:].tolist())


def within_condition_null(X, subj, sess, reps, seed=42, cap=400):
    """MMD between two rep-disjoint halves of the SAME (subject, condition). ~0 if leak-free."""
    rng = np.random.default_rng(seed)
    vals = []
    # A rep-disjoint split needs only >= 2 repetitions (1-vs-1 is already leak-free). The old
    # gate demanded >= 4, which silently produced n=0 — i.e. NO null at all — on the two datasets
    # that need it most: senic has 3 reps per (subject, condition) and grabmyo_flow_dynamic has 2.
    # The leak control therefore never ran on the electrode-shift dataset it exists to police.
    MIN_REPS, MIN_ROWS = 2, 40
    for s in np.unique(subj):
        for c in np.unique(sess[subj == s]):
            m = (subj == s) & (sess == c)
            if m.sum() < MIN_ROWS or len(np.unique(reps[m])) < MIN_REPS:
                continue
            h1, h2 = _halves(reps[m], rng)
            i1 = np.where(m & np.isin(reps, list(h1)))[0]
            i2 = np.where(m & np.isin(reps, list(h2)))[0]
            if min(len(i1), len(i2)) < 15:
                continue
            A = X[i1][:cap] if len(i1) > cap else X[i1]
            B = X[i2][:cap] if len(i2) > cap else X[i2]
            vals.append(common.mmd_rbf(A, B, rng=rng))
    return dict(mean=float(np.mean(vals)) if vals else float("nan"),
                max=float(np.max(vals)) if vals else float("nan"), n=len(vals))


def condition_shift(X, y, subj, sess, seed=42):
    """Within-subject fixed-effects r between per-condition MMD-to-other-conditions and accuracy."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    rng = np.random.default_rng(seed)
    recs = []
    for s in np.unique(subj):
        conds = np.unique(sess[subj == s])
        if len(conds) < 2:
            continue
        for c in conds:
            te = (subj == s) & (sess == c)
            tr = (subj == s) & (sess != c)
            if te.sum() < 30 or tr.sum() < 30 or len(np.unique(y[tr])) < 2:
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            clf = LinearDiscriminantAnalysis().fit((X[tr] - mu) / sd, y[tr])
            acc = float((clf.predict((X[te] - mu) / sd) == y[te]).mean())
            recs.append((int(s), acc, common.mmd_rbf(X[te], X[tr], rng=rng)))
    # within-subject demean (fixed effects)
    by = {}
    for s, a, m in recs:
        by.setdefault(s, []).append((a, m))
    ad, md, used = [], [], 0
    for s, lst in by.items():
        if len(lst) < 2:
            continue
        a = np.array([r[0] for r in lst]); m = np.array([r[1] for r in lst])
        if a.std() < 1e-9 or m.std() < 1e-9:
            continue
        ad.append(a - a.mean()); md.append(m - m.mean()); used += 1
    if used < 3:
        return dict(applicable=False, n_subjects=used, note="too few subjects with >=2 conditions")
    ad = np.concatenate(ad); md = np.concatenate(md)
    r = common.clip_corr(float(np.corrcoef(md, ad)[0, 1])) if md.std() > 0 else float("nan")
    return dict(applicable=True, n_records=len(recs), n_subjects=used,
                within_subject_fixedeffects_r=r)


def run_one(dataset, seed=42, n_jobs=1):
    with common.timer(f"X10 :: {dataset}"):
        frame = common.build_frame(dataset, seed=seed)
        X, _ = common.basis(frame)
        y = frame.label.to_numpy(); subj = frame.subject.to_numpy()
        sess = frame.session.to_numpy(); reps = frame.repetition.to_numpy()
        out = dict(dataset=dataset,
                   within_condition_null=within_condition_null(X, subj, sess, reps, seed),
                   condition_shift=condition_shift(X, y, subj, sess, seed),
                   amplitude_shift=condition_shift(common.basis(frame, AMP)[0], y, subj, sess, seed),
                   shape_shift=condition_shift(common.basis(frame, SHAPE)[0], y, subj, sess, seed))
    return out


# ------------------------------------------------ ground truth
def selftest(check):
    # stationary synthetic with a repetition axis -> within-condition null ~ 0
    fr = common.synth_frame("separable", n_subjects=6, n_classes=4, per_class=80, seed=8)
    X, _ = common.basis(fr)
    null = within_condition_null(X, fr.subject.to_numpy(), fr.session.to_numpy(),
                                 fr.repetition.to_numpy(), seed=8)
    check("X10 within-condition null ~ 0 on stationary data", null["mean"] < 0.05,
          f"mean={null['mean']:.4f} (n={null['n']})")
