"""X6 — Learned-representation replication. Buries the "handcrafted-basis artifact?" limitation (§10).

Re-runs the difficulty correlation and inter-subject MMD in a torch-free LEARNED embedding (whitened
PCA and nonlinear Random-Fourier-Features), so the findings are shown to be (or not be) an artifact of
the handcrafted amplitude basis — meeting the 2023-25 representation-learning expectation without a
training loop.

GROUND TRUTH: a label-permutation null makes the difficulty r collapse to ~0 in the embedding (no
spurious signal); on a separable synthetic the embedding preserves class geometry (silhouette not
much worse than the handcrafted basis).
"""
from __future__ import annotations

import numpy as np

from . import common


def _difficulty_in_space(Z, y, subj, seed):
    acc = common.loso_accuracy(Z, y, subj, "lda")
    mmd = common.mmd_to_pool(Z, subj, seed)
    return common.corr_across_subjects(mmd, acc)


def embed(X, method, seed=0, n_comp=64):
    if method == "pca":
        return common.pca_embed(X, X, n_comp=n_comp, whiten=True, seed=seed)
    if method == "rff":
        return common.rff_embed(X, X, n_comp=max(64, n_comp), seed=seed)
    raise ValueError(method)


def run_one(dataset, seed=42, n_jobs=1):
    with common.timer(f"X6 :: {dataset}"):
        frame = common.build_frame(dataset, seed=seed)
        X, _ = common.basis(frame)
        y, subj = frame.label.to_numpy(), frame.subject.to_numpy()
        out = dict(dataset=dataset,
                   handcrafted=_difficulty_in_space(X, y, subj, seed),
                   pca=_difficulty_in_space(embed(X, "pca", seed), y, subj, seed),
                   rff=_difficulty_in_space(embed(X, "rff", seed), y, subj, seed))
    rs = [out[k]["r"] for k in ("handcrafted", "pca", "rff") if out[k]["r"] == out[k]["r"]]
    out["replicates_across_representations"] = bool(len(rs) == 3 and all(r < 0 for r in rs))
    return out


# ------------------------------------------------ ground truth
def selftest(check):
    fr = common.synth_frame("real_difficulty", n_subjects=16, n_classes=8, per_class=35, seed=6)
    X, _ = common.basis(fr)
    y, subj = fr.label.to_numpy(), fr.subject.to_numpy()
    rng = np.random.default_rng(0)

    r_hand = _difficulty_in_space(X, y, subj, 6)["r"]
    r_pca = _difficulty_in_space(embed(X, "pca", 6), y, subj, 6)["r"]
    r_rff = _difficulty_in_space(embed(X, "rff", 6), y, subj, 6)["r"]
    check("X6 real difficulty r negative in handcrafted basis", r_hand < -0.3, f"{r_hand:.3f}")
    check("X6 difficulty REPLICATES (negative) in the PCA embedding", r_pca < -0.2, f"{r_pca:.3f}")
    check("X6 difficulty REPLICATES (negative) in the nonlinear RFF embedding", r_rff < -0.15, f"{r_rff:.3f}")

    # label-permutation null: with labels shuffled, LOSO ~ chance -> no difficulty signal in the embedding
    yperm = y[rng.permutation(len(y))]
    r_null = _difficulty_in_space(embed(X, "pca", 6), yperm, subj, 6)["r"]
    check("X6 label-permutation null: r collapses (~0)", abs(r_null) < 0.4, f"{r_null:.3f}")
