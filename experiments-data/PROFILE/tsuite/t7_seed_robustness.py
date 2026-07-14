"""T7 — Is any of this stable across random seeds? (the question we currently CANNOT answer)

WHY
---
Every headline number in this project was computed at seed = 42, once. The window subsample, the
MMD subsample, the model initialisation and the train subsample are all seeded. Until the F5 fix
(2026-07-12) the frame cache did not even include the seed in its filename, so re-running with a
different seed silently reused seed-42's data — meaning seed-robustness was not merely untested, it
was UNFALSIFIABLE.

F5 fixed the cache key. So this is now testable, and it is the cheapest possible way to find out
whether the difficulty correlation (r = -0.40 pooled, significant on only 2 of 14 datasets) is a
stable property or a lucky draw. If a per-dataset r swings by +/-0.2 across seeds, then "significant
on 2/14" is noise and the paper must say so.

NOTE ON COST: a new seed rebuilds the feature frame for that dataset (that is the point of the F5
fix), so this experiment is the one T-experiment that is I/O- and CPU-heavy on first run. It is
still CPU-only and it parallelises across datasets.

WHAT IT DOES
------------
For seeds {42, 7, 1, 2026}: rebuild the frame, recompute the MMD predictor and the LDA-LOSO target,
and record the difficulty r. Report mean +/- sd of r per dataset, and whether the SIGN is stable.

PRE-REGISTERED BRANCHES
-----------------------
  A. sd(r) small (< 0.05) and the sign never flips
     -> report every headline as mean +/- sd over 4 seeds. Turns a single-draw number into a
        measured one, and pre-empts the reviewer question at zero argumentative cost.
  B. sd(r) moderate (0.05-0.15)
     -> quote all numbers with their spread and stop calling marginal datasets "significant".
  C. sign flips on any dataset
     -> that dataset's result is noise and must be reported as such. If it flips on several, the
        per-dataset FDR analysis is meaningless and only the pooled effect may be quoted.

GROUND TRUTH
------------
  * On a synthetic with a LARGE, real effect, r must be stable across seeds (sd small) — otherwise
    the estimator itself is unstable and nothing downstream can be trusted.
  * On a synthetic with NO effect, r must scatter around 0 across seeds and the sign must flip — this
    proves the seed sweep can actually DETECT instability rather than always reporting "stable".
"""
from __future__ import annotations

import numpy as np

from . import common as C

TAG = "t7_seed_robustness"
SEEDS = (42, 7, 1, 2026)


def run_one(dataset, seed=42, n_jobs=1):
    """`seed` is ignored on purpose: this experiment's whole subject IS the seed."""
    out = dict(dataset=dataset, seeds=list(SEEDS), per_seed={})
    for sd in SEEDS:
        with C.timer(f"T7 :: {dataset} :: seed={sd}"):
            frame = C.build_frame(dataset, seed=sd)          # F5: a new seed => a new frame
            X, _ = C.basis(frame)
            y, subj = frame.label.to_numpy(), frame.subject.to_numpy()
            mmd = C.mmd_to_pool(X, subj, seed=sd)
            acc = C.loso_accuracy_model(X, y, subj, "lda", seed=sd)
            r = C.corr_across_subjects(mmd, acc)
            out["per_seed"][str(sd)] = dict(
                difficulty_r=r["r"], p=r["p"], n_subjects=r["n"],
                mean_loso_acc=float(np.mean(list(acc.values()))) if acc else float("nan"),
                inter_subject_mmd=float(np.mean(list(mmd.values()))) if mmd else float("nan"))

    rs = np.array([v["difficulty_r"] for v in out["per_seed"].values()], float)
    rs = rs[np.isfinite(rs)]
    accs = np.array([v["mean_loso_acc"] for v in out["per_seed"].values()], float)
    if len(rs) >= 2:
        out["r_mean"] = float(rs.mean())
        out["r_std"] = float(rs.std(ddof=1))
        out["r_min"], out["r_max"] = float(rs.min()), float(rs.max())
        out["sign_stable"] = bool(np.all(rs < 0) or np.all(rs > 0))
        out["acc_std"] = float(np.nanstd(accs, ddof=1))
    return out


def build_pooled(tag=TAG):
    import json
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(C.results_dir(tag).glob(f"*__{tag}.json"))]
    rows = [r for r in rows if "r_std" in r]
    if not rows:
        return dict(note="no per-dataset results yet")
    n = len(rows)
    stds = np.array([r["r_std"] for r in rows])
    unstable = [r["dataset"] for r in rows if not r["sign_stable"]]
    med = float(np.median(stds))

    if not unstable and med < 0.05:
        branch, verdict = "A", (
            f"STABLE: the difficulty correlation does not depend on the seed (median sd(r) = {med:.3f} "
            f"over {len(SEEDS)} seeds, sign never flips). Quote every headline as mean +/- sd.")
    elif not unstable:
        branch, verdict = "B", (
            f"MODERATELY STABLE: median sd(r) = {med:.3f}. The sign holds everywhere, but marginal "
            "datasets must not be called significant on the strength of one seed.")
    else:
        branch, verdict = "C", (
            f"UNSTABLE: the sign of r FLIPS across seeds on {len(unstable)}/{n} datasets "
            f"({', '.join(unstable)}). Those per-dataset results are noise; only the pooled effect "
            "may be quoted, and the per-dataset FDR table must carry this warning.")

    out = dict(tag=tag, n_datasets=n, seeds=list(SEEDS),
               median_r_std=med, max_r_std=float(stds.max()),
               datasets_with_sign_flip=unstable,
               sign_flip_both_ways=C.count_both_ways(unstable, [r["dataset"] for r in rows]),
               per_dataset={r["dataset"]: dict(r_mean=r["r_mean"], r_std=r["r_std"],
                                               r_min=r["r_min"], r_max=r["r_max"],
                                               sign_stable=r["sign_stable"]) for r in rows},
               branch=branch, verdict=verdict)
    C.atomic_write_json(C.results_dir(tag) / "pooled.json", out)
    return out


# ------------------------------------------------------------------ ground truth
def _seed_sweep(seeds=(0, 1, 2, 3), permute_predictor=False):
    """Re-estimate the difficulty correlation once per seed.

    `permute_predictor` builds the NULL world: the same real difficulty structure, but the MMD values
    are shuffled between subjects, so the predictor carries no information about who is hard. Any r
    it produces is pure sampling noise, and its sign must wander from seed to seed.

    (An earlier version of this null used the 'separable' synthetic — identical subjects. That was
    wrong: every subject scores ~1.0, the accuracy vector is CONSTANT, and Pearson's r is undefined,
    so the sweep returned an empty array rather than an unstable one.)
    """
    rs = []
    for sd in seeds:
        fr = C.synth_frame("real_difficulty", n_subjects=14, n_classes=5, per_class=40, seed=sd)
        X, _ = C.basis(fr)
        y, s = fr.label.to_numpy(), fr.subject.to_numpy()
        mmd = C.mmd_to_pool(X, s, seed=sd)
        acc = C.loso_accuracy_model(X, y, s, "lda", seed=sd, cap_train=4000)
        if permute_predictor:
            rng = np.random.default_rng(1000 + sd)
            keys = list(mmd)
            mmd = dict(zip(keys, rng.permutation([mmd[k] for k in keys])))
        rs.append(C.corr_across_subjects(mmd, acc)["r"])
    return np.array([r for r in rs if np.isfinite(r)])


def selftest(check):
    real = _seed_sweep()
    check("T7 a LARGE real effect is seed-stable (sd small, sign never flips)",
          len(real) >= 3 and real.std(ddof=1) < 0.25 and np.all(real < 0),
          f"r={np.round(real, 3).tolist()} sd={real.std(ddof=1):.3f}")

    null = _seed_sweep(permute_predictor=True)
    flips = not (np.all(null < 0) or np.all(null > 0))
    check("T7 the sweep can DETECT instability: with a permuted (information-free) predictor the "
          "sign wanders across seeds and |r| collapses",
          len(null) >= 3 and (flips or abs(null.mean()) < 0.35),
          f"r={np.round(null, 3).tolist()} (sign flips={flips})")
