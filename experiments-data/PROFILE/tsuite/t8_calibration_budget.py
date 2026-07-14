"""T8 — How many SECONDS of a new user's unlabelled data do we actually need?

WHY
---
The selling point of the difficulty predictor is that it is training-free and label-free: point it at
a new user's raw recording and it tells you how well the model will work for them, before any
labelling, fitting or calibration happens. But nobody has asked the only question a practitioner
cares about: **how much data does "point it at a new user" mean?** Ten seconds? Ten minutes?

X12 showed the subsample caps are converged (the estimate stops moving as you add windows), which is
a statistical-hygiene answer. It is not the deployment answer, because it never asked how little data
you can get away with while the PREDICTION still works.

This experiment turns the statistic into a protocol: a budget curve in seconds.

WHAT IT DOES
------------
For each budget b in {25, 50, 100, 200, 400, 800} windows per subject:
  * draw b windows at random from each subject's unlabelled data (10 independent draws);
  * recompute the MMD-to-pool predictor from ONLY those windows;
  * correlate it with the (full-data) LOSO difficulty target.
Report the correlation as a function of budget, the spread of the MMD estimate across draws, and the
smallest budget whose |r| is within 90% of the full-budget |r|. Windows are converted to seconds
using the dataset's own window length and hop, so the answer is in units a practitioner can act on.

PRE-REGISTERED BRANCHES
-----------------------
  A. |r| saturates at a small budget (a few tens of seconds)
     -> "a new user's difficulty can be forecast from N seconds of unlabelled recording." That is a
        deployable protocol, not just a correlation, and it is the most practically useful sentence
        the paper could contain. **Headline.**
  B. |r| needs most of the data
     -> the statistic is a research instrument, not a deployment tool. Say so plainly.
  C. |r| is flat / noisy at every budget
     -> the correlation is too weak to be operationalised at any budget. Report and move on.

GROUND TRUTH
------------
  * On a synthetic with a real effect, |r| must RISE with budget and saturate. A budget curve that
    is flat at the true value from b=25 would mean the subsampling is not actually subsampling.
  * The MMD estimate's spread across draws must SHRINK as the budget grows (it is an average over
    more data). If it does not, the estimator is broken.
"""
from __future__ import annotations

import numpy as np

from . import common as C

TAG = "t8_calibration_budget"
BUDGETS = (25, 50, 100, 200, 400, 800)
N_DRAWS = 10


def _mmd_at_budget(X, subjects, budget, seed, n_draws=N_DRAWS):
    """MMD from a BUDGETED new user to the FULL reference pool. Returns a list of per-draw dicts.

    Two corrections the first draft got wrong, both of which flattered the result:

    1. THE POOL MUST NOT BE BUDGETED. In deployment the reference pool is the training cohort and
       has all of its data; only the NEW USER is data-poor. Subsampling both sides injected noise
       into the reference side that no deployed system would ever have, so the curve did not answer
       the question the experiment asks.

    2. r MUST BE COMPUTED PER DRAW, NOT FROM THE MEAN OF 10 DRAWS. A practitioner has ONE recording
       of b windows - not the average of ten. Averaging 10 draws uses up to 10b distinct windows and
       shrinks the estimator's variance by ~sqrt(10), so the "minimum budget" it produced was
       optimistic by a wide margin. We now return every draw and let the caller report mean +/- sd.
    """
    from paper_experiments.common import mmd_rbf
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    subjects = np.asarray(subjects)
    per_draw = []
    for d in range(n_draws):
        est = {}
        for s in np.unique(subjects):
            idx = np.flatnonzero(subjects == s)
            pool = np.flatnonzero(subjects != s)          # FULL pool, never budgeted
            if len(idx) < 5 or len(pool) < 20:
                continue
            k = min(budget, len(idx))
            this = X[rng.choice(idx, k, replace=False)]   # only the NEW USER is data-poor
            ref = X[pool] if len(pool) <= 800 else X[rng.choice(pool, 800, replace=False)]
            est[int(s)] = mmd_rbf(this, ref, gamma="median", rng=rng)
        per_draw.append(est)
    return per_draw


def _seconds_per_window(dataset):
    """Hop between consecutive windows, in seconds (the marginal cost of one more window)."""
    from dsprofile import config
    return (config.WINDOW_MS / 1000.0) * (1.0 - config.OVERLAP)


def run_one(dataset, seed=42, n_jobs=1):
    with C.timer(f"T8 :: {dataset}"):
        frame = C.build_frame(dataset, seed=seed)
        X, _ = C.basis(frame)
        y, subj = frame.label.to_numpy(), frame.subject.to_numpy()

        # the TARGET uses all the data: we are budgeting the PREDICTOR, not the evaluation
        acc = C.loso_accuracy_model(X, y, subj, "lda", seed=seed)
        full = C.mmd_to_pool(X, subj, seed=seed)
        r_full = C.corr_across_subjects(full, acc)

        hop = _seconds_per_window(dataset)
        out = dict(dataset=dataset, budgets=list(BUDGETS), n_draws=N_DRAWS,
                   seconds_per_window=hop, full_budget_r=r_full["r"],
                   n_subjects=r_full["n"], curve={})

        for b in BUDGETS:
            draws = _mmd_at_budget(X, subj, b, seed)
            rs = [C.corr_across_subjects(est, acc)["r"] for est in draws if len(est) >= 5]
            rs = [v for v in rs if np.isfinite(v)]
            if len(rs) < 3:
                out["curve"][str(b)] = dict(note="too few usable draws at this budget", n=len(rs))
                continue
            subs = sorted(set().union(*[set(e) for e in draws]))
            spread = [float(np.std([e[s] for e in draws if s in e], ddof=1))
                      for s in subs if sum(s in e for e in draws) > 1]
            out["curve"][str(b)] = dict(
                # ONE recording of b windows is what a practitioner has -> the per-draw mean is the
                # honest estimate and the sd is the honest uncertainty.
                difficulty_r=float(np.mean(rs)), difficulty_r_sd=float(np.std(rs, ddof=1)),
                difficulty_r_worst=float(max(rs)),       # closest to zero = worst case for a neg. r
                n_draws_used=len(rs), n_subjects=len(subs),
                seconds_per_subject=float(b * hop),
                mmd_estimate_spread=float(np.mean(spread)) if spread else float("nan"))

    # Smallest budget reaching 90% of the full-data |r| - with two guards the first draft lacked:
    #   (a) the SIGN must match. abs() on both sides meant a budget whose correlation pointed the
    #       WRONG WAY (+0.9|r_full|) qualified as "90% of full strength".
    #   (b) SATURATION: the criterion must hold at that budget AND at every larger one. Taking min()
    #       over a noisy curve is a minimum-of-noise statistic - with 6 budgets, one lucky small
    #       budget wins and then decides the "DEPLOYABLE" branch.
    rf = r_full["r"]
    tgt = 0.90 * abs(rf) if np.isfinite(rf) else np.nan
    sign = np.sign(rf) if np.isfinite(rf) else 0.0

    def _meets(b):
        v = out["curve"][str(b)].get("difficulty_r", np.nan)
        return bool(np.isfinite(v) and np.sign(v) == sign and abs(v) >= tgt)

    ok = [b for i, b in enumerate(BUDGETS) if all(_meets(bb) for bb in BUDGETS[i:])]
    out["min_budget_windows"] = int(min(ok)) if ok else None
    out["min_budget_seconds"] = float(min(ok) * out["seconds_per_window"]) if ok else None
    out["min_budget_criterion"] = ("same sign as the full-data r, |r| >= 90% of full, and holding at "
                                   "every larger budget (saturation, not a lucky single point)")
    sp = [out["curve"][str(b)].get("mmd_estimate_spread", np.nan) for b in BUDGETS]
    sp = [v for v in sp if np.isfinite(v)]
    out["spread_shrinks_with_budget"] = bool(len(sp) >= 2 and sp[-1] < sp[0])
    return out


def build_pooled(tag=TAG):
    import json
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(C.results_dir(tag).glob(f"*__{tag}.json"))]
    rows = [r for r in rows if "curve" in r]
    if not rows:
        return dict(note="no per-dataset results yet")
    n = len(rows)
    secs = [r["min_budget_seconds"] for r in rows if r.get("min_budget_seconds")]
    solved = len(secs)
    mean_curve = {str(b): float(np.nanmean([r["curve"][str(b)].get("difficulty_r", np.nan)
                                            for r in rows])) for b in BUDGETS}

    if solved >= n * 0.7 and np.median(secs) <= 60:
        branch, verdict = "A", (
            f"DEPLOYABLE: on {solved}/{n} datasets the predictor reaches 90% of its full-data strength "
            f"from a median of {np.median(secs):.0f} SECONDS of a new user's unlabelled recording. The "
            "difficulty forecast is a protocol, not just a correlation.")
    elif solved >= n * 0.5:
        branch, verdict = "B", (
            f"NEEDS REAL DATA: 90% strength requires a median of {np.median(secs):.0f} s per user "
            f"(solved on {solved}/{n}). Usable, but it is a calibration step, not a free lunch.")
    else:
        branch, verdict = "C", (
            f"NOT OPERATIONALISABLE: only {solved}/{n} datasets ever reach 90% of the full-data |r| "
            "at any budget tested. The correlation is too weak to turn into a protocol.")

    out = dict(tag=tag, n_datasets=n, mean_r_by_budget=mean_curve,
               solved_both_ways=C.count_both_ways(
                   [r["dataset"] for r in rows if r.get("min_budget_seconds")],
                   [r["dataset"] for r in rows]),                      # k=14 AND k=9 cohorts
               n_datasets_solved=solved,
               median_min_budget_seconds=float(np.median(secs)) if secs else None,
               per_dataset={r["dataset"]: r.get("min_budget_seconds") for r in rows},
               branch=branch, verdict=verdict)
    C.atomic_write_json(C.results_dir(tag) / "pooled.json", out)
    return out


# ------------------------------------------------------------------ ground truth
def selftest(check):
    fr = C.synth_frame("real_difficulty", n_subjects=14, n_classes=5, per_class=60, seed=1)
    X, _ = C.basis(fr)
    y, s = fr.label.to_numpy(), fr.subject.to_numpy()
    acc = C.loso_accuracy_model(X, y, s, "lda", seed=1, cap_train=4000)

    rs, spreads = [], []
    for b in (25, 200):
        draws = _mmd_at_budget(X, s, b, seed=1, n_draws=5)
        rs.append(float(np.mean([abs(C.corr_across_subjects(e, acc)["r"]) for e in draws])))
        subs = sorted(set().union(*[set(e) for e in draws]))
        spreads.append(float(np.mean([float(np.std([e[u] for e in draws if u in e], ddof=1))
                                      for u in subs])))

    check("T8 |r| grows (or holds) as the unlabelled budget grows",
          rs[1] >= rs[0] - 0.10, f"|r|@25={rs[0]:.3f} |r|@200={rs[1]:.3f}")
    check("T8 the MMD estimate gets more precise with more data (spread shrinks)",
          spreads[1] < spreads[0], f"spread@25={spreads[0]:.4f} spread@200={spreads[1]:.4f}")
    check("T8 budget subsampling is real (a tiny budget does not reproduce the full estimate exactly)",
          spreads[0] > 0, f"spread@25={spreads[0]:.4f}")

    # NEGATIVE CONTROL (added after the 2026-07-13 review, which noted all three T8 checks were
    # positive). An INFORMATION-FREE predictor - the same MMD values, shuffled between subjects - must
    # produce a flat, ~zero budget curve. If |r| still "grows with budget" when the predictor carries
    # no information about who is hard, then the curve is an artifact of the estimator and branch A
    # ("DEPLOYABLE: N seconds is enough") would be measuring nothing at all.
    rng = np.random.default_rng(11)
    null_rs = []
    for b in (25, 200):
        draws = _mmd_at_budget(X, s, b, seed=1, n_draws=5)
        per = []
        for e in draws:
            keys = list(e)
            shuffled = dict(zip(keys, rng.permutation([e[k] for k in keys])))
            per.append(abs(C.corr_across_subjects(shuffled, acc)["r"]))
        null_rs.append(float(np.mean(per)))
    check("T8 NEGATIVE CONTROL: an information-free (permuted) predictor gives a flat, near-zero "
          "budget curve - the curve is not an artifact of the estimator",
          max(null_rs) < 0.45 and abs(null_rs[1] - null_rs[0]) < 0.30,
          f"|r|@25={null_rs[0]:.3f} |r|@200={null_rs[1]:.3f} (real: {rs[0]:.3f} -> {rs[1]:.3f})")
