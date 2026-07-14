"""Experiment C — cross-session (temporal) difficulty prediction.

The money result predicts cross-SUBJECT difficulty from MMD-to-pool. This is its temporal analog:
does the SAME shift statistic predict cross-SESSION (day) difficulty?

For each subject with >= 2 sessions, and each of that subject's sessions treated as the held-out
"test day":
    cross_session_acc = LDA trained on the subject's OTHER sessions, tested on this session
    mmd_to_other      = MMD(this session's windows, the subject's other sessions' windows)
Both are z-scored WITHIN each subject (removing the subject's own baseline, so we isolate the
session effect, exactly as the SDI does for subjects), then pooled and correlated.

A negative correlation means: the more a day drifts from a user's other days, the harder that day
is to decode — the temporal counterpart of the cross-subject result (scientific questions A3/A4).

Only datasets with real repeated sessions qualify: grabmyo + its two flow variants (3 sessions),
and senic (whose "sessions" are electrode-shift / fatigue CONDITIONS, not days — flagged, reported
separately). Everything else is single-session and skipped.

Run on the box:  python exp_C_cross_session.py --datasets grabmyo,grabmyo_flow_dynamic,grabmyo_flow_static,senic --jobs 8
Output: results/experiments/exp_C_cross_session.json
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from dsprofile import config, windows, progress
from dsprofile.module5_difficulty import _basis
from dsprofile.module3_shift import mmd_rbf

MULTISESSION = ["grabmyo", "grabmyo_flow_dynamic", "grabmyo_flow_static", "senic"]


def _cross_session_records(X, y, subj, sess, seed=42, min_per=50, cap=600):
    """One record per (subject, held-out session): (subject, session, acc, mmd_to_other_sessions).

    Testable on a synthetic frame. Uses each subject's OTHER sessions as training, so the split is
    strictly within-subject and cross-session (no subject leakage; no window leakage across the
    session boundary because sessions are disjoint recordings).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    rng = np.random.default_rng(seed)
    recs = []
    for s in np.unique(subj):
        sm = subj == s
        subj_sessions = np.unique(sess[sm])
        if len(subj_sessions) < 2:
            continue
        for ts in subj_sessions:
            te = sm & (sess == ts)
            tr = sm & (sess != ts)
            if te.sum() < min_per or tr.sum() < min_per or len(np.unique(y[tr])) < 2:
                continue
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            clf = LinearDiscriminantAnalysis().fit((Xtr - mu) / sd, ytr)
            acc = float((clf.predict((Xte - mu) / sd) == yte).mean())
            # cap both sides for the MMD estimate (bounds cost; mmd_rbf also subsamples to 400)
            A = Xte if te.sum() <= cap else Xte[rng.choice(te.sum(), cap, replace=False)]
            B = Xtr if tr.sum() <= cap else Xtr[rng.choice(tr.sum(), cap, replace=False)]
            mmd = mmd_rbf(A, B, rng=rng)
            recs.append((int(s), int(ts), acc, float(mmd)))
    return recs


def _within_demean(recs):
    """Fixed-effects within-transformation: subtract each subject's OWN mean from acc and mmd.

    This is the correct panel-data 'within' estimator (Frisch-Waugh-Lovell). We do NOT z-score
    per subject: dividing by the std of 2-3 sessions is unstable, and a correlation is already
    scale-free; demeaning removes the subject fixed effect, which is all we need to isolate the
    session effect. A subject with < 2 usable sessions, or zero variance in acc or mmd, is
    skipped (nothing to explain within it).

    Returns (mmd_demeaned, acc_demeaned, n_subjects_used) with points pooled across subjects.
    Each retained subject's demeaned values sum to zero by construction -> this is what costs a
    degree of freedom per subject, handled in `_within_corr`.
    """
    by_subj = {}
    for s, ts, acc, mmd in recs:
        by_subj.setdefault(s, []).append((acc, mmd))
    md, ad, used = [], [], 0
    for s, lst in by_subj.items():
        if len(lst) < 2:
            continue
        acc = np.array([r[0] for r in lst], float)
        mmd = np.array([r[1] for r in lst], float)
        if acc.std() < 1e-9 or mmd.std() < 1e-9:
            continue
        ad.append(acc - acc.mean())
        md.append(mmd - mmd.mean())
        used += 1
    if not md:
        return np.array([]), np.array([]), 0
    return np.concatenate(md), np.concatenate(ad), used


def _within_corr(md, ad, n_subjects):
    """Pearson correlation of the within-demeaned values, with the CORRECT degrees of freedom.

        r  = sum(md*ad) / sqrt(sum(md^2) * sum(ad^2))
        df = N_points - n_subjects - 1      (one df lost per removed subject mean, one for slope)
        t  = r * sqrt(df / (1 - r^2)),   two-sided p from Student-t on df

    Using the naive Pearson df (N-2) would OVERSTATE significance because the demeaned points
    within a subject are not independent. Returns (r, p, df).
    """
    from scipy.stats import t as student_t
    n = len(md)
    df = n - n_subjects - 1
    denom = np.sqrt((md ** 2).sum() * (ad ** 2).sum())
    if df < 1 or denom < 1e-30:
        return float("nan"), float("nan"), int(max(df, 0))
    r = float((md * ad).sum() / denom)
    r = max(-0.999999, min(0.999999, r))
    tstat = r * np.sqrt(df / (1.0 - r ** 2))
    p = float(2.0 * student_t.sf(abs(tstat), df))
    return r, p, int(df)


def _per_subject_rank_check(recs):
    """Robustness: within each subject, Spearman(session mmd, session acc); average across
    subjects and test the mean against 0 (one-sample t). Rank-based, so immune to per-subject
    variance differences. With 2-3 sessions each subject's rho is coarse, but the AVERAGE is
    informative and does not assume the fixed-effects pooling."""
    by_subj = {}
    for s, ts, acc, mmd in recs:
        by_subj.setdefault(s, []).append((acc, mmd))
    rhos = []
    for s, lst in by_subj.items():
        if len(lst) < 2:
            continue
        acc = np.array([r[0] for r in lst], float)
        mmd = np.array([r[1] for r in lst], float)
        if acc.std() < 1e-9 or mmd.std() < 1e-9:
            continue
        rho, _ = spearmanr(mmd, acc)
        if rho == rho:
            rhos.append(float(rho))
    if len(rhos) < 3:
        return dict(n_subjects=len(rhos), note="too few subjects for the rank cross-check")
    rhos = np.array(rhos)
    from scipy.stats import ttest_1samp, wilcoxon
    _, p_t = ttest_1samp(rhos, 0.0)
    try:
        _, p_w = wilcoxon(rhos)
    except ValueError:
        p_w = float("nan")
    return dict(n_subjects=len(rhos), mean_per_subject_spearman=float(rhos.mean()),
                median_per_subject_spearman=float(np.median(rhos)),
                frac_negative=float((rhos < 0).mean()),
                ttest_p=float(p_t), wilcoxon_p=float(p_w))


def cross_session(frame, dataset="", seed=42):
    """Full C for one dataset frame: within-subject (fixed-effects) correlation of session shift
    vs session difficulty, with a rank-based cross-check."""
    X = _basis(frame)
    y = frame.label.to_numpy(); subj = frame.subject.to_numpy(); sess = frame.session.to_numpy()
    if len(np.unique(sess)) < 2:
        return dict(applicable=False, note="single-session dataset")

    recs = _cross_session_records(X, y, subj, sess, seed)
    if len(recs) < 6:
        return dict(applicable=False, note=f"only {len(recs)} (subject,session) records")

    md, ad, n_subj = _within_demean(recs)
    if md.size < 6 or n_subj < 3:
        return dict(applicable=False,
                    note=f"only {md.size} within-subject points from {n_subj} subjects")

    r, p, df = _within_corr(md, ad, n_subj)
    rank = _per_subject_rank_check(recs)

    accs = np.array([rr[2] for rr in recs]); mmds = np.array([rr[3] for rr in recs])
    caveat = None
    if dataset == "senic":
        caveat = ("senic 'sessions' are electrode-shift / rotation / fatigue CONDITIONS, not days; "
                  "this is a shift-robustness axis, not a true temporal one. Report separately.")
    return dict(
        applicable=True,
        n_records=len(recs), n_subjects_used=n_subj, n_points_pooled=int(md.size),
        mean_cross_session_acc=float(accs.mean()), std_cross_session_acc=float(accs.std()),
        mean_session_mmd=float(mmds.mean()),
        within_subject_fixedeffects_r=r, p_value=p, df=df,
        rank_cross_check=rank,
        interpretation=("session shift predicts session difficulty (temporal analog of the "
                        "cross-subject result)" if (r == r and r < 0 and p < 0.05) else
                        "no significant within-subject session shift->difficulty relationship"),
        caveat=caveat,
    )


def run_dataset(dataset, seed=42, n_jobs=None):
    """n_jobs accepted for the uniform driver signature; C has no internal joblib. C only reads
    the existing 250 ms cache, so it is safe to run alongside A/B in another terminal."""
    with progress.timer(f"C :: {dataset}"):
        frame = windows.build_fast_frame(dataset, seed=seed)
        r = dict(dataset=dataset, **cross_session(frame, dataset, seed))
    if r.get("applicable"):
        progress.log(f"  {dataset}: within-subject r(shift,acc)={r['within_subject_fixedeffects_r']:+.3f} "
                     f"(p={r['p_value']:.3f}, {r['n_points_pooled']} pts / {r['n_subjects_used']} subj)")
    else:
        progress.log(f"  {dataset}: {r.get('note')}")
    return r


def build_summary(results):
    # true-day datasets only for the headline; senic is reported but excluded (shift-condition axis)
    applicable = {ds: v for ds, v in results.items()
                  if v.get("applicable") and ds != "senic"}
    n_sig = sum(1 for v in applicable.values()
                if v.get("within_subject_fixedeffects_r", 0) < 0 and v.get("p_value", 1) < 0.05)
    return dict(
        n_true_day_datasets=len(applicable), n_significant_negative=n_sig,
        senic=results.get("senic", {}).get("within_subject_fixedeffects_r"),
        per_dataset=results,
        verdict=(f"session shift predicts session difficulty on {n_sig}/{len(applicable)} "
                 "true-day datasets -> the difficulty predictor extends to the temporal axis"
                 if applicable else "no true-day dataset yielded enough records"),
        _console=[f"significant (negative) on {n_sig}/{len(applicable)} true-day datasets "
                  "(senic reported separately as a shift-condition axis)"],
    )


if __name__ == "__main__":
    from addons import common as exp_common
    exp_common.main("C", run_one=run_dataset, build_summary=build_summary,
                    all_datasets=MULTISESSION)
