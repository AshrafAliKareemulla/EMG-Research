"""X5 — De-amplituded basis ablation. Buries the "shift == amplitude?" limitation (§9-4).

REPR_BASIS is 5/7 amplitude features, so cross-subject "distribution shift" may just be
contraction-amplitude shift. We recompute inter-subject MMD and the difficulty correlation in the
FULL basis, an AMPLITUDE-only basis, and a de-amplituded SHAPE/invariant basis, and report which
survives.

GROUND TRUTH:
  pure per-subject amplitude scale -> inter-subject MMD > 0 in the amplitude basis, ~0 in the shape
  basis; a shape (covariance) difference -> MMD > 0 in BOTH.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import common

AMP = ["MAV", "WL", "RMS", "MFL"]
SHAPE = ["HJ_MOB", "HJ_COM", "WAMP"]


def _inter_subject_mmd(X, subjects, seed, cap=600):
    """Mean pairwise between-subject MMD (magnitude-preserving aggregate)."""
    rng = np.random.default_rng(seed)
    groups = {}
    for s in np.unique(subjects):
        idx = np.where(subjects == s)[0]
        if len(idx) < 20:
            continue
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        groups[int(s)] = X[idx]
    keys = list(groups)
    vals = [common.mmd_rbf(groups[a], groups[b], rng=rng)
            for a, b in itertools.combinations(keys, 2)]
    return float(np.mean(vals)) if vals else float("nan"), len(keys)


def _one_basis(frame, feats, seed):
    X, cols = common.basis(frame, feats)
    y, subj = frame.label.to_numpy(), frame.subject.to_numpy()
    if not cols:
        return dict(applicable=False, note="no columns for this basis")
    is_mmd, n_subj = _inter_subject_mmd(X, subj, seed)
    acc = common.loso_accuracy(X, y, subj, "lda")
    mmd = common.mmd_to_pool(X, subj, seed)
    diff = common.corr_across_subjects(mmd, acc)
    return dict(applicable=True, n_cols=len(cols), n_subjects=n_subj,
                inter_subject_mmd=is_mmd, mean_loso_acc=float(np.mean(list(acc.values()))) if acc else float("nan"),
                difficulty_r=diff["r"], difficulty_p=diff["p"], difficulty_n=diff["n"])


def run_one(dataset, seed=42, n_jobs=1):
    with common.timer(f"X5 :: {dataset}"):
        frame = common.build_frame(dataset, seed=seed)
        out = dict(dataset=dataset,
                   full=_one_basis(frame, common.REPR_BASIS, seed),
                   amplitude=_one_basis(frame, AMP, seed),
                   shape_invariant=_one_basis(frame, SHAPE, seed))
    a = out["amplitude"].get("inter_subject_mmd", float("nan"))
    s = out["shape_invariant"].get("inter_subject_mmd", float("nan"))
    out["shift_is_amplitude_dominated"] = bool(np.isfinite(a) and np.isfinite(s) and s < 0.4 * a)
    out["interpretation"] = ("if the difficulty_r and inter_subject_mmd SURVIVE in the shape/invariant "
                             "basis, the shift is structural; if they collapse, cross-subject shift is "
                             "dominated by contraction amplitude (a clean, citable finding).")
    return out


# ------------------------------------------------ ground truth
def selftest(check):
    fr_amp = common.synth_frame("shift_amplitude", n_subjects=14, n_classes=6, per_class=45, seed=4)
    XA, _ = common.basis(fr_amp, AMP)
    XS, _ = common.basis(fr_amp, SHAPE)
    subj = fr_amp.subject.to_numpy()
    mA = _inter_subject_mmd(XA, subj, 4)[0]
    mS = _inter_subject_mmd(XS, subj, 4)[0]
    check("X5 amplitude-shift: MMD large in amplitude basis", mA > 0.05, f"mmd_amp={mA:.3f}")
    check("X5 amplitude-shift: MMD ~0 in shape basis", mS < 0.4 * mA, f"mmd_shape={mS:.3f} vs {mA:.3f}")

    fr_sh = common.synth_frame("shift_shape", n_subjects=14, n_classes=6, per_class=45, seed=4)
    subj2 = fr_sh.subject.to_numpy()
    mS2 = _inter_subject_mmd(common.basis(fr_sh, SHAPE)[0], subj2, 4)[0]
    check("X5 shape-shift: MMD survives in shape basis", mS2 > 0.03, f"mmd_shape={mS2:.3f}")
