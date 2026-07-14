"""Block D — channels & sensors.

E7  Channel reduction: rank channels by relevance (Fisher on channel RMS) + min-redundancy (mRMR),
    then measure how much class separability (5-fold kNN accuracy) is retained as channels are added
    1..n. Answers "how few channels for within-X% accuracy".
E6  Sampling-rate sufficiency: rebuild the fast frame at decimation 1/2/4 (native, half, quarter fs)
    and compare separability. Answers "is 2 kHz needed, or does 500 Hz suffice".

Scalable + dataset-agnostic; reuses the cached fast frames (and builds decimated variants on demand).
"""
from __future__ import annotations

import json

import numpy as np

from . import config, cv, windows
from .module4_channels import per_channel_fisher, nmi_matrix, greedy_mrmr, _channel_signal_matrix


def _basis_cols_for_channels(frame, chans):
    cols = []
    for fb in config.REPR_BASIS:
        for c in chans:
            cols += [x for x in frame.columns if x == f"{fb}_c{c}"]
    return cols


def _knn_acc(frame, cols, y, seed=0, max_n=5000, groups=None):
    """Trial-grouped 5-fold kNN. `groups` defaults to trial ids so overlapping windows from
    one trial never straddle a fold boundary (the old version shuffled and used plain cv=5)."""
    X = np.nan_to_num(frame[cols].to_numpy(np.float64))
    if groups is None:
        groups = cv.trial_ids(frame)
    return cv.knn_trial_cv(X, y, groups, max_n=max_n, seed=seed)


def channel_reduction(dataset, seed=42):
    """E7. Channel-subset curve under trial-grouped CV, reported within- AND cross-subject.

    K1 ("minimum channels for within-X% of full accuracy") is only a deployment claim if it
    holds when the test subject is unseen, so the LOSO curve is the one to quote.
    """
    frame = windows.build_fast_frame(dataset, seed=seed)
    y = frame["label"].to_numpy()
    groups = cv.trial_ids(frame); subjects = frame["subject"].to_numpy()
    M, _ = _channel_signal_matrix(frame, "RMS")
    C = M.shape[1]
    relevance = per_channel_fisher(frame)
    nmi = nmi_matrix(M)
    ranking = greedy_mrmr(relevance, nmi)

    def acc(cols, grp):
        X = np.nan_to_num(frame[cols].to_numpy(np.float64))
        return (cv.knn_trial_cv(X, y, grp, seed=seed) if grp is groups
                else cv.knn_loso(X, y, grp, seed=seed))

    allcols = _basis_cols_for_channels(frame, list(range(C)))
    full, full_loso = acc(allcols, groups), acc(allcols, subjects)
    curve, curve_loso = {}, {}
    for k in range(1, C + 1):
        cols = _basis_cols_for_channels(frame, ranking[:k])
        curve[k] = float(acc(cols, groups))
        curve_loso[k] = float(acc(cols, subjects))

    # Chance-corrected. "95 % of full accuracy" is meaningless when full accuracy is itself
    # near chance: emaha_db5's LOSO accuracy is 0.142 against a 10-class chance of 0.100, so
    # 0.95 x 0.142 = 0.135 is reached by 2 channels that carry essentially no information.
    # We instead require 95 % of the ABOVE-CHANCE accuracy, and refuse to answer at all when the
    # full model is not meaningfully above chance.
    chance = 1.0 / max(1, len(np.unique(y)))

    def kmin(cu, fl):
        if fl - chance < 0.05:                    # full model barely beats chance
            return None
        target = chance + 0.95 * (fl - chance)
        return int(next((k for k in sorted(cu) if cu[k] >= target), C))

    return dict(n_channels=C,
                full_accuracy=full, full_accuracy_loso=full_loso,
                mrmr_ranking=[int(x) for x in ranking],
                accuracy_vs_k={int(k): v for k, v in curve.items()},
                accuracy_vs_k_loso={int(k): v for k, v in curve_loso.items()},
                chance_level=float(chance),
                min_channels_for_95pct=kmin(curve, full),
                min_channels_for_95pct_loso=kmin(curve_loso, full_loso),
                criterion="k such that acc(k) >= chance + 0.95*(full - chance); None if "
                          "full - chance < 0.05 (model not meaningfully above chance)",
                protocol="trial-grouped 5-fold (within-subject) and subject-grouped (LOSO)")


def sampling_rate_sufficiency(dataset, seed=42, factors=(1, 2, 4)):
    """E6. Is 2 kHz needed, or does 500 Hz suffice?

    Two corrections over the original:
      * decimation is now anti-aliased (`windows._antialias_decimate`). Naive `[::q]` folded
        the 250-500 Hz band back into the passband, corrupting MNF/MDF/SENT/MNP/TTP while
        leaving amplitude features intact — which is why the curve looked flat.
      * datasets that cannot be decimated meaningfully are skipped rather than reported.
        ninapro_db5 / senic (200 Hz) and myobit (176 Hz) were previously pushed to 44-50 Hz,
        where a 250 ms window holds 11-12 samples. Those rows were noise, not evidence.
    """
    fs = windows.dataset_fs(dataset)
    frame0 = windows.build_fast_frame(dataset, seed=seed)
    if fs is None:
        return dict(note="native fs unknown (no manifest); sufficiency not testable",
                    native_fs=None, testable=False)
    if bool(frame0.attrs.get("is_envelope", False)):
        return dict(note="envelope dataset; sampling-rate sufficiency not meaningful",
                    native_fs=fs, testable=False)
    if fs < config.E6_MIN_NATIVE_FS_HZ:
        return dict(note=f"native fs={fs} Hz < {config.E6_MIN_NATIVE_FS_HZ:.0f} Hz: no bandwidth "
                         f"to remove, sufficiency not testable on this dataset.",
                    native_fs=fs, testable=False)
    usable = [d for d in factors if fs / d >= config.E6_MIN_EFFECTIVE_FS_HZ]
    out = dict(native_fs=fs, testable=True, factors_tested=usable,
               factors_skipped=[d for d in factors if d not in usable])
    curves = {}
    for d in usable:
        frame = windows.build_fast_frame(dataset, seed=seed, decimate=d)
        y = frame["label"].to_numpy()
        cols = [c for c in frame.columns if c not in windows.META]
        curves[f"decimate_{d}"] = dict(effective_fs=int(fs // d),
                                       knn_acc=_knn_acc(frame, cols, y, seed),
                                       n_windows=int(len(frame)))
    base = curves.get("decimate_1", {}).get("knn_acc")
    if base:
        for k in curves:
            curves[k]["retained_frac"] = float(curves[k]["knn_acc"] / (base + 1e-12))
    out["curves"] = curves
    return out


def run(dataset, seed=42):
    config.ensure_dirs()
    outdir = config.RESULTS_DIR / "block_d"
    outdir.mkdir(parents=True, exist_ok=True)
    result = dict(dataset=dataset,
                  E7_channel_reduction=channel_reduction(dataset, seed),
                  E6_sampling_rate=sampling_rate_sufficiency(dataset, seed))
    (outdir / f"{dataset}__block_d.json").write_text(json.dumps(result, indent=2))
    return result
