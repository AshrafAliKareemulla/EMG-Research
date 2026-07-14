"""Block A — signal & feature analyses.

A1  Feature reliability: for each feature, what fraction of its within-(subject,class) variance is
    stable across repetitions (vs repetition-to-repetition drift). reliability in [0,1], 1 = stable.
        reliability = mean over (subject,class) groups of  Var_within_rep / Var_total_within_group
    (law of total variance: Var_total = Var_within_rep + Var_between_rep_means).
A2  Does complexity add information beyond amplitude? Residualise each complexity/entropy feature on
    ALL amplitude features (linear regression), then measure mutual information of the RESIDUAL with
    the class. Residual MI > 0 => complexity carries class info not present in amplitude.

Scalable + dataset-agnostic; uses the cached COMPLEX frame (has both amplitude + entropy features).
"""
from __future__ import annotations

import json

import numpy as np

from . import config, windows

AMPLITUDE = ["MAV", "RMS", "WL", "VAR", "IEMG", "SSI", "DASDV", "AAC", "LOG", "LOGRMS", "NLE"]
COMPLEX = ["SAMPEN", "FUZZYEN", "FAPEN", "PERMEN", "HFD", "HJ_MOB", "HJ_COM", "MFL"]


def _feat_cols(frame, bases):
    cols = []
    for fb in bases:
        cols += [c for c in frame.columns if c.startswith(fb + "_c")]
    return cols


def _usable_cols(frame, cols):
    """Drop columns that are entirely NaN (e.g. complexity on a sub-threshold window length).

    Critical: `np.nan_to_num` would turn a masked entropy column into a column of ZEROS, which
    is a perfectly finite, perfectly meaningless feature. It would then be ranked, regressed
    and fed to the MI estimator as though it carried information.
    """
    return [c for c in cols if c in frame.columns and np.isfinite(frame[c].to_numpy(float)).any()]


def feature_reliability(frame):
    """Per-feature reliability across repetitions, averaged over (subject,class) groups."""
    feat_cols = _usable_cols(frame, [c for c in frame.columns if c not in windows.META])
    rel = {}
    grp = frame.groupby(["subject", "label"])
    for col in feat_cols:
        ratios = []
        for _, g in grp:
            if g["repetition"].nunique() < 2 or len(g) < 6:
                continue
            v = g[col].to_numpy(dtype=np.float64)
            tot = v.var()
            if tot < 1e-12:
                continue
            within = np.mean([g.loc[g.repetition == r, col].var(ddof=0)
                              for r in g.repetition.unique() if (g.repetition == r).sum() > 1])
            if not np.isfinite(within):
                continue
            ratios.append(np.clip(within / tot, 0.0, 1.0))
        rel[col.rsplit("_c", 1)[0]] = rel.get(col.rsplit("_c", 1)[0], [])
        if ratios:
            rel[col.rsplit("_c", 1)[0]].append(float(np.mean(ratios)))
    # average across channels -> one reliability per feature name
    return {k: float(np.mean(v)) for k, v in rel.items() if v}


def complexity_adds_info(frame, seed=0):
    """Residual mutual information of complexity features (after removing amplitude) with class."""
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import mutual_info_classif
    amp = _usable_cols(frame, _feat_cols(frame, AMPLITUDE))
    cpx = _usable_cols(frame, _feat_cols(frame, COMPLEX))
    if not amp or not cpx:
        return dict(note="missing amplitude or complexity features",
                    n_amplitude=len(amp), n_complexity=len(cpx))
    # keep only rows where every retained feature is finite (rather than imputing zeros)
    M = frame[amp + cpx].to_numpy(np.float64)
    ok = np.isfinite(M).all(axis=1)
    if ok.sum() < 50:
        return dict(note="too few rows with all features finite", n_rows_finite=int(ok.sum()))
    y = frame["label"].to_numpy()[ok]
    A = frame[amp].to_numpy(np.float64)[ok]
    C = frame[cpx].to_numpy(np.float64)[ok]

    raw_mi = float(np.mean(mutual_info_classif(C, y, random_state=seed)))
    # residualise complexity on amplitude: removes only the LINEAR component of the dependence,
    # so `complexity_residual_mi` is an upper bound on the genuinely independent information.
    resid = C - LinearRegression().fit(A, C).predict(A)
    res_mi = float(np.mean(mutual_info_classif(resid, y, random_state=seed)))
    amp_mi = float(np.mean(mutual_info_classif(A, y, random_state=seed)))
    return dict(amplitude_mi=amp_mi, complexity_raw_mi=raw_mi,
                complexity_residual_mi=res_mi,
                n_amplitude_features=len(amp), n_complexity_features=len(cpx),
                n_rows_used=int(ok.sum()),
                fraction_retained=float(res_mi / (raw_mi + 1e-12)),
                complexity_adds_info=bool(res_mi > 0.02 * amp_mi),
                threshold_note="complexity_adds_info := residual MI > 2% of amplitude MI",
                linearity_caveat="only the linear dependence on amplitude is removed")


def run(dataset, seed=42):
    config.ensure_dirs()
    outdir = config.RESULTS_DIR / "block_a"
    outdir.mkdir(parents=True, exist_ok=True)
    frame = windows.build_complex_frame(dataset, seed=seed)
    ent_ok = windows.entropy_valid(dataset)
    nsamp = windows.window_samples(dataset)
    rel = {k: v for k, v in feature_reliability(frame).items() if np.isfinite(v)}
    excluded = sorted(set(config.SLOW_COMPLEX) - set(rel)) if not ent_ok else []
    result = dict(
        dataset=dataset,
        entropy_valid=bool(ent_ok),
        window_samples=(int(nsamp) if nsamp is not None else None),
        entropy_min_samples=int(config.ENT["min_samples"]),
        excluded_features=excluded,
        exclusion_reason=(None if ent_ok else
                          f"{config.WINDOW_MS:.0f} ms window holds {nsamp} samples "
                          f"< {config.ENT['min_samples']}; entropy/fractal features are undefined "
                          f"and were masked to NaN rather than reported"),
        feature_reliability=dict(sorted(rel.items(), key=lambda kv: -kv[1])),
        most_reliable=sorted(rel, key=lambda k: -rel[k])[:5],
        least_reliable=sorted(rel, key=lambda k: rel[k])[:5],
        complexity_adds_info=complexity_adds_info(frame, seed),
    )
    (outdir / f"{dataset}__block_a.json").write_text(json.dumps(result, indent=2))
    return result
