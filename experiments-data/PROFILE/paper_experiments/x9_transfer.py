"""X9 — Cross-dataset transfer accuracy. Buries the unvalidated transfer matrix (§5, transfer.py).

For montage-compatible dataset pairs (same channel count, overlapping label space) it trains on A,
tests on B, and correlates the real transfer accuracy with the compatibility MMD, turning a
shape-only, `validated:false` matrix into a validated (or honestly null) transfer result.

GROUND TRUTH: A->A (same data, subject-split) transfer ~ within-dataset LOSO; two identical synthetic
datasets -> compatibility MMD ~ 0 and transfer ~ within; a deliberately incompatible pair -> large MMD
and transfer ~ chance.
"""
from __future__ import annotations

import numpy as np

from . import common


def transfer_pair(frameA, frameB, seed=42):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # shared feature columns (same montage) and shared labels
    colsA = set(common._cols_for(frameA, common.REPR_BASIS))
    colsB = set(common._cols_for(frameB, common.REPR_BASIS))
    cols = sorted(colsA & colsB)
    labs = sorted(set(frameA.label) & set(frameB.label))
    if len(cols) < 3 or len(labs) < 2:
        return dict(applicable=False, note="incompatible montage or label space")
    mA = frameA.label.isin(labs).to_numpy()
    mB = frameB.label.isin(labs).to_numpy()
    XA = np.nan_to_num(frameA.loc[mA, cols].to_numpy(np.float64))
    XB = np.nan_to_num(frameB.loc[mB, cols].to_numpy(np.float64))
    yA = frameA.loc[mA, "label"].to_numpy()
    yB = frameB.loc[mB, "label"].to_numpy()
    mu, sd = XA.mean(0), XA.std(0) + 1e-9          # scaler fit on the SOURCE only
    ZA, ZB = (XA - mu) / sd, (XB - mu) / sd
    clf = LinearDiscriminantAnalysis().fit(ZA, yA)
    acc = float((clf.predict(ZB) == yB).mean())
    rng = np.random.default_rng(seed)
    return dict(applicable=True, n_shared_labels=len(labs), n_cols=len(cols),
                transfer_accuracy=acc, chance=1.0 / len(labs),
                compatibility_mmd=common.mmd_rbf(ZA, ZB, rng=rng))


def run_pairs(pairs, seed=42):
    """pairs: list of (dsA, dsB). Builds native frames and measures transfer for each compatible pair."""
    rows = []
    frames = {}
    for a, b in pairs:
        for ds in (a, b):
            if ds not in frames:
                frames[ds] = common.build_frame(ds, seed=seed)
        r = transfer_pair(frames[a], frames[b], seed)
        if r.get("applicable"):
            rows.append(dict(source=a, target=b, **r))
    out = dict(n_pairs=len(rows), pairs=rows)
    if len(rows) >= 4:
        r, p, n = common.pearson([x["compatibility_mmd"] for x in rows],
                                 [x["transfer_accuracy"] for x in rows])
        out["mmd_vs_transfer_pearson"] = dict(r=r, p=p, n=n)
    common.atomic_write_json(common.results_dir("x9") / "transfer.json", out)
    return out


# ------------------------------------------------ ground truth
def selftest(check):
    fr = common.synth_frame("separable", n_subjects=8, n_classes=5, per_class=60, seed=7)
    fcols = common._cols_for(fr, common.REPR_BASIS)
    # A->A: same distribution -> high transfer, ~0 compatibility MMD
    self_t = transfer_pair(fr, fr, seed=7)
    check("X9 A->A transfer high & MMD ~ 0 (same distribution)",
          self_t["transfer_accuracy"] > 0.7 and self_t["compatibility_mmd"] < 0.05,
          f"acc={self_t['transfer_accuracy']:.3f} mmd={self_t['compatibility_mmd']:.4f}")
    # a marginal shift -> the MMD DETECTS it (large), regardless of transfer
    frB = fr.copy()
    frB[fcols] = frB[fcols].to_numpy() + 15.0
    frB.attrs["n_channels"] = fr.attrs["n_channels"]
    frB.attrs["dataset"] = "synth_shift"
    check("X9 MMD detects a marginal shift (large)", transfer_pair(fr, frB, seed=7)["compatibility_mmd"] > 0.2,
          f"mmd={transfer_pair(fr, frB, seed=7)['compatibility_mmd']:.3f}")
    # a class-SCRAMBLED target (labels cyclically shifted, a derangement) -> transfer ~ chance
    labs = sorted(fr.label.unique())
    remap = {l: labs[(i + 1) % len(labs)] for i, l in enumerate(labs)}
    frC = fr.copy()
    frC["label"] = fr["label"].map(remap)
    frC.attrs["n_channels"] = fr.attrs["n_channels"]
    frC.attrs["dataset"] = "synth_scramble"
    scr = transfer_pair(fr, frC, seed=7)
    check("X9 class-scrambled target: transfer ~ chance", scr["transfer_accuracy"] < scr["chance"] + 0.1,
          f"acc={scr['transfer_accuracy']:.3f} chance={scr['chance']:.3f}")
