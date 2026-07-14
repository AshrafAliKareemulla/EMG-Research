"""Experiment A — window-length robustness (100 / 250 / 500 ms).

The whole paper is computed at a 250 ms window. The first question a reviewer asks is whether the
findings are a window-length artifact. This re-computes the load-bearing metrics at 100, 250 and
500 ms and reports how stable each is:

    separability  : knn_trial_cv, knn_loso, silhouette, fisher
    shift         : inter-subject MMD
    difficulty    : MMD-to-pool vs LDA-LOSO correlation  (the money result)

A finding is ROBUST if its sign/conclusion holds across all three windows. We report the value at
each window plus a coefficient of variation and a sign-stability flag per metric.

Cost: builds NEW fast frames at 100 ms and 500 ms for every dataset (the 250 ms frames are already
cached). Fast features only — no entropy. Reuses the exact metric code from modules 2/3/5.

Run on the box:  python exp_A_window_length.py --datasets all --jobs 8
Output: results/experiments/exp_A_window_length.json
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score

from dsprofile import config, cv, windows, progress
from dsprofile.module3_shift import _basis, mmd_rbf, _grouped
from dsprofile.module2_separability import fisher_ratio
from dsprofile.module5_difficulty import loso_lda_accuracy, subject_shift_stats

WINDOWS_MS = [100.0, 250.0, 500.0]


def metrics_for_frame(frame, seed=42, n_jobs=None):
    """All load-bearing metrics for ONE frame (one window length). Testable on a synthetic frame."""
    X = _basis(frame)
    y = frame.label.to_numpy(); subj = frame.subject.to_numpy()
    groups = cv.trial_ids(frame)
    rng = np.random.default_rng(seed)

    knn_trial = cv.knn_trial_cv(X, y, groups, seed=seed)
    knn_xsub = cv.knn_loso(X, y, subj, seed=seed)
    sidx = rng.choice(len(X), min(4000, len(X)), replace=False)
    sil = float(silhouette_score(X[sidx], y[sidx])) if len(np.unique(y)) > 1 else float("nan")
    fish = fisher_ratio(X, y)

    g = _grouped(X, subj, 600, rng, trials=groups)             # {key: (rows, trial_ids)}
    keys = list(g)
    mmds = [mmd_rbf(g[keys[i]][0], g[keys[j]][0], rng=rng)
            for i in range(len(keys)) for j in range(i + 1, len(keys))]
    is_mmd = float(np.mean(mmds)) if mmds else float("nan")

    target = loso_lda_accuracy(X, y, subj)
    preds = subject_shift_stats(X, subj, seed, n_jobs=n_jobs, trials=groups)
    common = sorted(set(target) & set(preds))
    if len(common) >= 5:
        acc = np.array([target[s] for s in common])
        mmd = np.array([preds[s]["mmd_to_pool"] for s in common])
        dr, dp = pearsonr(mmd, acc)
    else:
        dr = dp = float("nan")

    return dict(n_windows=int(len(frame)),
                knn_trial_cv=float(knn_trial), knn_loso=float(knn_xsub),
                within_minus_cross=float(knn_trial - knn_xsub),
                silhouette=sil, fisher=float(fish), inter_subject_mmd=is_mmd,
                mean_loso_acc=float(np.mean(list(target.values()))) if target else float("nan"),
                difficulty_r=float(dr), difficulty_p=float(dp), n_subjects=len(common))


# metrics whose SIGN / ordering carries the paper's conclusion
_SIGN_METRICS = {
    "difficulty_r": "negative",          # MMD-to-pool predicts LOWER accuracy
    "within_minus_cross": "positive",    # within-subject > cross-subject
    "silhouette": None,                  # value only
}


def stability(per_window):
    """Pure function: {window_label: metrics} -> per-metric stability summary. Testable directly."""
    labels = list(per_window)
    metric_names = [k for k in per_window[labels[0]] if isinstance(per_window[labels[0]][k], float)]
    out = {}
    for m in metric_names:
        vals = {w: per_window[w].get(m) for w in labels}
        arr = np.array([v for v in vals.values() if v is not None and np.isfinite(v)])
        if arr.size == 0:
            out[m] = dict(values=vals, note="all non-finite")
            continue
        mean = float(arr.mean())
        cv_ = float(arr.std() / (abs(mean) + 1e-12))
        entry = dict(values={w: (float(v) if v is not None else None) for w, v in vals.items()},
                     min=float(arr.min()), max=float(arr.max()),
                     mean=mean, coef_of_variation=cv_)
        want = _SIGN_METRICS.get(m, "skip")
        if want == "negative":
            entry["sign_stable"] = bool((arr < 0).all())
        elif want == "positive":
            entry["sign_stable"] = bool((arr > 0).all())
        out[m] = entry
    # headline robustness flags
    dr = out.get("difficulty_r", {})
    wc = out.get("within_minus_cross", {})
    return dict(per_metric=out,
                difficulty_negative_at_all_windows=dr.get("sign_stable"),
                within_gt_cross_at_all_windows=wc.get("sign_stable"))


def run_dataset(dataset, seed=42, n_jobs=None, windows_ms=None):
    windows_ms = windows_ms or WINDOWS_MS
    per_window = {}
    orig = config.WINDOW_MS
    try:
        for w in windows_ms:
            config.WINDOW_MS = w                     # changes the cache key -> builds a new frame
            with progress.timer(f"A :: {dataset} @ {int(w)}ms"):
                frame = windows.build_fast_frame(dataset, seed=seed)
                per_window[f"{int(w)}ms"] = metrics_for_frame(frame, seed, n_jobs)
            m = per_window[f"{int(w)}ms"]
            progress.log(f"  {dataset} {int(w)}ms: knn_loso={m['knn_loso']:.3f} "
                         f"diff_r={m['difficulty_r']:+.3f} IS_mmd={m['inter_subject_mmd']:.3f}")
    finally:
        config.WINDOW_MS = orig                       # always restore, even on error
    return dict(dataset=dataset, windows_ms=[int(w) for w in windows_ms],
                per_window=per_window, stability=stability(per_window))


def build_summary(results):
    ok_diff = ok_wc = n = 0
    for ds, r in results.items():
        if "error" in r:
            continue
        st = r.get("stability", {})
        n += 1
        ok_diff += bool(st.get("difficulty_negative_at_all_windows"))
        ok_wc += bool(st.get("within_gt_cross_at_all_windows"))
    return dict(
        windows_ms=[int(w) for w in WINDOWS_MS], n_datasets=n,
        difficulty_negative_at_all_windows=f"{ok_diff}/{n}",
        within_gt_cross_at_all_windows=f"{ok_wc}/{n}",
        per_dataset={ds: r.get("stability", r) for ds, r in results.items()},
        verdict=("headline findings are window-length robust"
                 if n and ok_diff >= 0.8 * n else
                 "some findings shift with window length; report the sensitivity"),
        _console=[f"difficulty r stays negative at all 3 windows: {ok_diff}/{n}",
                  f"within>cross gap holds at all 3 windows      : {ok_wc}/{n}"],
    )


if __name__ == "__main__":
    from addons import common as exp_common
    exp_common.main("A", run_one=run_dataset, build_summary=build_summary,
                    all_datasets=config.ALL14)
