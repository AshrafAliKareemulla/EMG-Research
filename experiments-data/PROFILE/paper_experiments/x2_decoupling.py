"""X2 — Representation decoupling. Buries the shared-representation coupling limitation (§9-2).

Is the MMD-to-pool <-> LOSO correlation a real data property, or mechanical (both live in the SAME
feature space)? We compute the PREDICTOR (MMD-to-pool) in basis A and the TARGET (LOSO-LDA) in a
DISJOINT basis B, and correlate cross-basis in both directions. A cross-basis correlation that
survives cannot be a same-space artifact.

GROUND TRUTH:
  mechanical_only : subject shift lives ONLY in A, class structure ONLY in B, no real hardness ->
                    cross-basis r ~ 0 (the control correctly finds no signal).
  real            : a latent hardness perturbs BOTH blocks -> cross-basis r strongly negative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common

AMP = ["MAV", "WL", "RMS", "MFL"]        # amplitude basis A (present in REPR_BASIS)
SHAPE = ["HJ_MOB", "HJ_COM", "WAMP"]     # shape/threshold basis B (present in REPR_BASIS)


def decouple(frame, basisA=AMP, basisB=SHAPE, seed=42):
    y = frame.label.to_numpy()
    subj = frame.subject.to_numpy()
    XA, colsA = common.basis(frame, basisA)
    XB, colsB = common.basis(frame, basisB)
    if not colsA or not colsB:
        return dict(applicable=False, note="basis A or B has no columns in this frame")
    accA = common.loso_accuracy(XA, y, subj, "lda")
    accB = common.loso_accuracy(XB, y, subj, "lda")
    mmdA = common.mmd_to_pool(XA, subj, seed)
    mmdB = common.mmd_to_pool(XB, subj, seed)
    return dict(
        applicable=True, n_colsA=len(colsA), n_colsB=len(colsB),
        same_A=common.corr_across_subjects(mmdA, accA),   # predictor & target both in A (mechanical baseline)
        same_B=common.corr_across_subjects(mmdB, accB),
        cross_A_to_B=common.corr_across_subjects(mmdA, accB),   # predictor A, target B (decoupled)
        cross_B_to_A=common.corr_across_subjects(mmdB, accA),
        interpretation=("cross-basis r that stays negative -> the predictor tracks a real data "
                        "property, not a same-space artifact; only same-basis r -> substantially "
                        "mechanical, scope the claim."),
    )


def run_one(dataset, seed=42, n_jobs=1):
    with common.timer(f"X2 :: {dataset}"):
        frame = common.build_frame(dataset, seed=seed)
        out = dict(dataset=dataset, **decouple(frame, seed=seed))
    return out


# ------------------------------------------------ ground truth
def _synth_decouple(kind, n_subjects=18, per_class=40, n_classes=8, seed=0):
    """Frame where basis A holds amplitude columns and basis B holds shape columns, with a KNOWN link."""
    rng = np.random.default_rng(seed)
    n_ch = 4
    ampcols = [f"{fb}_c{c}" for fb in AMP for c in range(n_ch)]
    shcols = [f"{fb}_c{c}" for fb in SHAPE for c in range(n_ch)]
    dA, dB = len(ampcols), len(shcols)
    classM = rng.standard_normal((n_classes, dB))       # class structure lives in B
    rowsA, rowsB, subj, lab = [], [], [], []
    for s in range(n_subjects):
        h = rng.uniform(0, 1)
        offA = rng.standard_normal(dA); offA = offA / (np.linalg.norm(offA) + 1e-9) * (3.0 * h)
        # 'real' couples the two blocks through h; 'mechanical_only' does not (h_B independent)
        csep = 2.5 * (1.0 - 0.9 * h) if kind == "real" else 2.5
        for c in range(n_classes):
            rowsA.append(offA + rng.standard_normal((per_class, dA)))          # A: shift only
            rowsB.append(classM[c] * csep + rng.standard_normal((per_class, dB)))  # B: class structure
            subj += [s] * per_class
            lab += [c] * per_class
    A = np.vstack(rowsA)
    B = np.vstack(rowsB)
    cols = {}
    for j, cname in enumerate(ampcols):
        cols[cname] = A[:, j]
    for j, cname in enumerate(shcols):
        cols[cname] = B[:, j]
    df = pd.DataFrame(cols)
    df["subject"] = subj
    df["session"] = 0
    df["repetition"] = 0
    df["label"] = lab
    df.attrs["n_channels"] = n_ch
    df.attrs["dataset"] = f"synth_decouple_{kind}"
    return df


def selftest(check):
    real = decouple(_synth_decouple("real", seed=1), seed=1)
    mech = decouple(_synth_decouple("mechanical_only", seed=1), seed=1)
    check("X2 real: cross-basis r strongly negative (predictor A -> target B)",
          real["cross_A_to_B"]["r"] < -0.4, f"r={real['cross_A_to_B']['r']:.3f}")
    check("X2 mechanical-only: cross-basis r ~ 0 (control finds no real signal)",
          abs(mech["cross_A_to_B"]["r"]) < 0.35, f"r={mech['cross_A_to_B']['r']:.3f}")
