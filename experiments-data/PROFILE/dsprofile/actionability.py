"""SDI actionability — does a cheap difficulty score help you SPEND a calibration budget better?

Scenario: you can collect a limited total number of calibration repetitions, distributed across new
users. Policy A ('difficulty-guided') spends them on the users the cheap statistic (MMD-to-pool, the
SDI's dominant term) predicts are HARD, first. Policy B ('random') spends them on random users. We
measure realised mean accuracy across ALL users vs total budget, using each user's TRUE calibration
curve. If guided beats random (higher area under the accuracy-vs-budget curve), the SDI is not just
predictive but USEFUL — a tool for deciding whom to calibrate.

Self-contained; reuses the cached fast frame. Scalable + dataset-agnostic.
"""
from __future__ import annotations

import json

import numpy as np

from . import config, cv, windows
from .module3_shift import _basis
from .module5_difficulty import subject_shift_stats


def per_subject_calibration(frame, kmax=5):
    """acc[subject][k] = LOSO accuracy for that subject with k calibration reps (LDA)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = _basis(frame); y = frame["label"].to_numpy()
    subj = frame["subject"].to_numpy(); rep = frame["repetition"].to_numpy()
    curves = {}
    for s in sorted(np.unique(subj)):
        tmask = subj == s; omask = ~tmask
        if len(np.unique(y[omask])) < 2 or tmask.sum() < 10:
            continue
        treps = sorted(np.unique(rep[tmask]))
        c = {}
        for k in range(0, min(kmax, len(treps) - 1) + 1):
            cal = tmask & np.isin(rep, treps[:k]) if k > 0 else np.zeros_like(tmask)
            test = tmask & np.isin(rep, treps[k:])
            if test.sum() < 5:
                continue
            Xtr = np.vstack([X[omask], X[cal]]) if k > 0 else X[omask]
            ytr = np.concatenate([y[omask], y[cal]]) if k > 0 else y[omask]
            if len(np.unique(ytr)) < 2:
                continue
            clf = LinearDiscriminantAnalysis().fit(Xtr, ytr)
            c[k] = float((clf.predict(X[test]) == y[test]).mean())
        if c:
            curves[int(s)] = c
    return curves


def _acc_at(curve, k):
    ks = sorted(curve)
    kk = min(k, ks[-1])
    while kk not in curve and kk > ks[0]:
        kk -= 1
    return curve.get(kk, curve[ks[0]])


def _mean_acc_vs_budget(curves, order, kmax):
    """Greedy allocation in `order`: fill each subject up to kmax before moving on."""
    subs = list(curves)
    total = len(subs) * kmax
    xs, ys = [], []
    for b in range(0, total + 1):
        alloc = {s: 0 for s in subs}
        rem = b
        for s in order:
            give = min(kmax, rem)
            alloc[s] = give; rem -= give
            if rem <= 0:
                break
        ys.append(float(np.mean([_acc_at(curves[s], alloc[s]) for s in subs])))
        xs.append(b)
    return np.array(xs), np.array(ys)


def run(dataset, seed=42, kmax=5, n_jobs=None):
    config.ensure_dirs()
    outdir = config.RESULTS_DIR / "actionability"
    outdir.mkdir(parents=True, exist_ok=True)
    frame = windows.build_fast_frame(dataset, seed=seed)
    curves = per_subject_calibration(frame, kmax)
    if len(curves) < 4:
        r = dict(dataset=dataset, note="too few subjects with calibration curves")
        (outdir / f"{dataset}__actionability.json").write_text(json.dumps(r, indent=2))
        return r
    preds = subject_shift_stats(_basis(frame), frame["subject"].to_numpy(), seed,
                                n_jobs=n_jobs, trials=cv.trial_ids(frame))
    subs = [s for s in curves if s in preds]

    # difficulty-guided: hardest (highest MMD-to-pool) first
    guided = sorted(subs, key=lambda s: -preds[s]["mmd_to_pool"])
    # random policies averaged over several shuffles
    rng = np.random.default_rng(seed)
    curves_sub = {s: curves[s] for s in subs}
    xg, yg = _mean_acc_vs_budget(curves_sub, guided, kmax)
    rand_ys = []
    for _ in range(20):
        o = list(subs); rng.shuffle(o)
        _, yr = _mean_acc_vs_budget(curves_sub, o, kmax)
        rand_ys.append(yr)
    yr = np.mean(rand_ys, axis=0)
    # oracle upper bound: order by true 0->kmax gain (steepest first)
    oracle = sorted(subs, key=lambda s: -(_acc_at(curves[s], kmax) - _acc_at(curves[s], 0)))
    _, yo = _mean_acc_vs_budget(curves_sub, oracle, kmax)

    auc = lambda y: float(np.trapz(y, xg) / (xg[-1] + 1e-12))
    a_g, a_r, a_o = auc(yg), auc(yr), auc(yo)

    # --- how big COULD any policy be? ------------------------------------------------
    # The oracle is the best achievable ordering. If oracle - random is ~0, then every
    # subject gains about the same from calibration and NO allocation policy can help,
    # however good the difficulty predictor is. Reporting `guided > random` without this
    # ceiling is meaningless. (Observed: ceilings of ~0.5-1.0 accuracy points.)
    ceiling = float(a_o - a_r)
    adv = float(a_g - a_r)

    # --- spread of the random policy: is `adv` distinguishable from shuffle noise? ----
    rand_aucs = np.array([auc(y) for y in rand_ys])
    sd = float(rand_aucs.std(ddof=1)) if len(rand_aucs) > 1 else float("nan")
    z = float(adv / sd) if sd and sd == sd and sd > 0 else float("nan")
    # one-sided empirical p: how often does a random ordering match/beat the guided one?
    p_emp = float((rand_aucs >= a_g).mean())

    # --- the assumption the policy rests on ------------------------------------------
    # "Spend on predicted-hard users first" only helps if predicted-hard users GAIN more
    # from calibration. That was never tested. Correlate MMD-to-pool against each subject's
    # realised 0->kmax calibration gain. A near-zero correlation means the policy is
    # optimising the wrong quantity, which is the more interesting finding.
    from scipy.stats import pearsonr, spearmanr
    mmd = np.array([preds[s]["mmd_to_pool"] for s in subs])
    gain = np.array([_acc_at(curves[s], kmax) - _acc_at(curves[s], 0) for s in subs])
    base = np.array([_acc_at(curves[s], 0) for s in subs])
    if len(subs) >= 4 and gain.std() > 1e-12:
        r_mg, p_mg = pearsonr(mmd, gain)
        rho_mg, _ = spearmanr(mmd, gain)
    else:
        r_mg = p_mg = rho_mg = float("nan")

    result = dict(
        dataset=dataset, n_subjects=len(subs), kmax=kmax, budget_axis=[int(x) for x in xg],
        mean_acc_guided=[float(v) for v in yg],
        mean_acc_random=[float(v) for v in yr],
        mean_acc_oracle=[float(v) for v in yo],
        auc_guided=a_g, auc_random=a_r, auc_oracle=a_o,
        guided_advantage=adv,
        oracle_ceiling=ceiling,
        # what fraction of the achievable headroom the SDI-guided policy actually captures
        fraction_of_ceiling_captured=float(adv / ceiling) if abs(ceiling) > 1e-9 else float("nan"),
        random_auc_std=sd, guided_advantage_z=z, guided_empirical_p=p_emp,
        guided_beats_random=bool(adv > 0),
        guided_beats_random_significantly=bool(p_emp < 0.05 and adv > 0),
        # the untested premise, now tested
        mmd_vs_calibration_gain=dict(pearson_r=float(r_mg), p_value=float(p_mg),
                                     spearman=float(rho_mg)),
        mean_zero_shot=float(base.mean()), mean_calibration_gain=float(gain.mean()),
        calibration_gain_std=float(gain.std()),
        honest_summary=("`guided_advantage` must be read against `oracle_ceiling`: a positive "
                        "advantage is meaningless if the ceiling itself is a fraction of an "
                        "accuracy point. `mmd_vs_calibration_gain` tests the premise that "
                        "predicted-hard subjects benefit most from calibration; if it is ~0, "
                        "difficulty-guided budget allocation cannot work by construction."),
    )
    (outdir / f"{dataset}__actionability.json").write_text(json.dumps(result, indent=2))
    return result
