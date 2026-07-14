"""T6 — Does per-subject centring break down when the new user's data is IMBALANCED?

WHY
---
N5 (per-subject mean-centring, +3.6 pp on 13/14) carries a caveat the project raised itself and then
could not test: a subject whose calibration data is class-imbalanced has a class-BIASED mean, so
subtracting it does not remove a nuisance shift — it removes signal.

X13 tried to test this and returned "untestable": 8 of the 14 datasets are perfectly balanced
(median imbalance ratio = 1.0), so the correlation between imbalance and centring benefit is
literally NaN. The committed result reports a caveat as unresolved.

An unmeasurable caveat is not a limitation of the data — it is a limitation of the experiment. If
the panel will not supply imbalance, MAKE the imbalance: subsample each subject's classes to a
controlled ratio and measure the centring benefit as that ratio grows. That converts "we don't know"
into a curve, and the curve is directly actionable: it tells a deployment engineer how skewed a new
user's calibration data may be before this trick starts to hurt.

DESIGN
------
For each dataset, for ratio in {1 (balanced control), 2, 5, 10, 20}:
  * induce the imbalance PER SUBJECT (the majority class keeps everything, the minority keeps 1/ratio,
    the class order permuted per subject so the imbalance belongs to the subject, not to a class id);
  * re-run the baseline vs centred LOSO contrast on exactly those rows;
  * record the centring benefit (mean accuracy delta) at that ratio.
Ratio 1 must reproduce the committed exp_B/X4 number for that dataset — that is the built-in control
that the induction code did not silently change something else.

PRE-REGISTERED BRANCHES
-----------------------
  A. Benefit stays flat or decays only at extreme ratios (>=10)
     -> N5 is robust and its caveat is DISCHARGED with a number: "centring survives imbalance up to
        ~Nx; beyond that, balance the calibration set first." A cheap, practical deployment rule.
  B. Benefit decays steadily and goes NEGATIVE by ratio ~5
     -> the caveat is REAL and quantified. N5 must be stated with a precondition. Still a good result:
        the honest version of a claim is more useful than the loud version.
  C. Benefit is unrelated to imbalance
     -> the caveat was hypothetical. Say so and close it.

GROUND TRUTH
------------
  * `induce_imbalance` must actually produce (approximately) the requested ratio, and must leave a
    balanced dataset untouched at ratio 1.
  * On a synthetic with a pure per-subject mean offset and BALANCED classes, centring must help.
  * On the same synthetic made severely imbalanced, the class-biased mean must measurably degrade
    the benefit — proving the mechanism exists at all before we look for it in the real data.
"""
from __future__ import annotations

import numpy as np

from . import common as C

TAG = "t6_imbalance_induced"
RATIOS = (1, 2, 5, 10, 20)


def _baseline_vs_center(X, y, subj, seed=42):
    """Mean LOSO accuracy delta of per-subject centring, on exactly the rows given."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    base, cent = {}, {}
    for s in sorted(np.unique(subj)):
        tr, te = subj != s, subj == s
        if len(np.unique(y[tr])) < 2 or te.sum() < 5:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        try:
            base[int(s)] = float((LinearDiscriminantAnalysis().fit(Ztr, y[tr]).predict(Zte) == y[te]).mean())
            Ctr = C.per_subject_center(Ztr, subj[tr])
            cent[int(s)] = float((LinearDiscriminantAnalysis().fit(Ctr, y[tr])
                                  .predict(Zte - Zte.mean(0)) == y[te]).mean())
        except Exception:
            continue
    shared = sorted(set(base) & set(cent))
    if len(shared) < 5:
        return dict(n=len(shared), note="too few subjects")
    a = np.array([base[s] for s in shared]); b = np.array([cent[s] for s in shared])
    from scipy.stats import wilcoxon
    d = b - a
    try:
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(b, a, alternative="greater")[1])
    except ValueError:
        p = 1.0
    return dict(n=len(shared), mean_baseline=float(a.mean()), mean_centered=float(b.mean()),
                center_benefit=float(d.mean()), n_improved=int((d > 0).sum()),
                wilcoxon_p=p, helps=bool(d.mean() > 0 and p < 0.05))


def run_one(dataset, seed=42, n_jobs=1):
    with C.timer(f"T6 :: {dataset}"):
        frame = C.build_frame(dataset, seed=seed)
        X, _ = C.basis(frame)
        y = frame.label.to_numpy()
        subj = frame.subject.to_numpy()

        out = dict(dataset=dataset, ratios=list(RATIOS),
                   native_imbalance_ratio=C.imbalance_ratio(y, subj), curve={})
        for r in RATIOS:
            mask = C.induce_imbalance(y, subj, r, seed=seed)
            res = _baseline_vs_center(X[mask], y[mask], subj[mask], seed)
            res["achieved_ratio"] = C.imbalance_ratio(y[mask], subj[mask])
            res["n_rows"] = int(mask.sum())
            out["curve"][str(r)] = res

    b = {r: out["curve"][str(r)].get("center_benefit", float("nan")) for r in RATIOS}
    out["benefit_balanced"] = b[1]
    out["benefit_at_ratio_5"] = b[5]
    out["benefit_at_ratio_20"] = b[20]
    finite = [(r, v) for r, v in b.items() if np.isfinite(v)]
    if len(finite) >= 3:
        rr = np.log([r for r, _ in finite]); vv = np.array([v for _, v in finite])
        out["slope_benefit_vs_log_ratio"] = float(np.polyfit(rr, vv, 1)[0])
        # Only ratios > 1 count. Including ratio 1 (the NATIVE distribution, which for 6 of the 14
        # datasets is not even balanced) meant a dataset where centring simply never helps was
        # counted as a dataset where IMBALANCE broke centring, and `first_negative_ratio` could come
        # back as 1 - turning "centring does not help here" into "proof of the imbalance caveat".
        induced = [(r, v) for r, v in finite if r > 1]
        neg = [r for r, v in induced if v < 0]
        out["benefit_turns_negative"] = bool(neg)
        out["first_negative_ratio"] = int(min(neg)) if neg else None
        if np.isfinite(b[1]):
            out["benefit_relative_to_native"] = {str(r): float(v - b[1]) for r, v in induced}
    return out


def build_pooled(tag=TAG):
    import json
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(C.results_dir(tag).glob(f"*__{tag}.json"))]
    rows = [r for r in rows if "curve" in r]
    if not rows:
        return dict(note="no per-dataset results yet")
    n = len(rows)
    all_ds = [r["dataset"] for r in rows]
    slopes = [r["slope_benefit_vs_log_ratio"] for r in rows if "slope_benefit_vs_log_ratio" in r]
    turns_neg = [r for r in rows if r.get("benefit_turns_negative")]
    firsts = [r["first_negative_ratio"] for r in turns_neg if r.get("first_negative_ratio")]
    # FDR across the 5 ratios x 14 datasets = 70 Wilcoxon tests (post-hoc; wilcoxon_p is stored).
    tests = [(r["dataset"], rt, r["curve"][str(rt)].get("center_benefit"),
              r["curve"][str(rt)].get("wilcoxon_p"))
             for r in rows for rt in RATIOS
             if r["curve"].get(str(rt), {}).get("wilcoxon_p") is not None]
    flags, _q = C.helps_flags_fdr([t[3] for t in tests], [t[2] for t in tests])
    helps_at_native = [t[0] for t, ok in zip(tests, flags) if ok and t[1] == 1]
    helps_at_20 = [t[0] for t, ok in zip(tests, flags) if ok and t[1] == 20]
    cohorts_native = C.count_both_ways(helps_at_native, all_ds)
    cohorts_20 = C.count_both_ways(helps_at_20, all_ds)
    turns_neg_both = C.count_both_ways([r["dataset"] for r in turns_neg], all_ds)
    mean_curve = {str(rt): float(np.nanmean([r["curve"][str(rt)].get("center_benefit", np.nan)
                                             for r in rows])) for rt in RATIOS}

    mean_slope = float(np.mean(slopes)) if slopes else float("nan")
    # Branch B must require an actual DECAY (negative slope), not merely the existence of a negative
    # benefit somewhere: a panel where centring never helps would otherwise be reported as proof of
    # the imbalance caveat.
    if len(turns_neg) <= n // 4 and (not slopes or mean_slope > -0.01):
        branch, verdict = "A", (
            f"CAVEAT DISCHARGED: per-subject centring survives induced imbalance. The benefit is "
            f"{mean_curve['1']:+.3f} balanced and {mean_curve['20']:+.3f} at a 20x class skew; it "
            f"turns negative on only {len(turns_neg)}/{n} datasets. X13's 'untestable' is now a "
            "measured curve.")
    elif firsts and np.isfinite(mean_slope) and mean_slope < 0:
        branch, verdict = "B", (
            f"CAVEAT IS REAL AND QUANTIFIED: the centring benefit decays with imbalance (mean slope "
            f"{np.mean(slopes):+.4f} per log-ratio) and turns NEGATIVE on {len(turns_neg)}/{n} "
            f"datasets, first at a skew of ~{int(np.median(firsts))}x. N5 must be stated with the "
            "precondition that the new user's calibration data is roughly balanced.")
    else:
        branch, verdict = "C", ("Imbalance does not systematically change the centring benefit; the "
                                "caveat was hypothetical.")

    out = dict(tag=tag, n_datasets=n, mean_benefit_by_ratio=mean_curve,
               mean_slope_vs_log_ratio=float(np.mean(slopes)) if slopes else float("nan"),
               centring_helps_at_native_distribution=cohorts_native,   # FDR-corrected, k=14 and k=9
               centring_helps_at_20x_skew=cohorts_20,
               benefit_turns_negative=turns_neg_both,
               multiplicity=f"Benjamini-Hochberg FDR across all {len(tests)} ratio x dataset tests",
               n_datasets_benefit_turns_negative=len(turns_neg),
               median_first_negative_ratio=(int(np.median(firsts)) if firsts else None),
               branch=branch, verdict=verdict)
    C.atomic_write_json(C.results_dir(tag) / "pooled.json", out)
    return out


# ------------------------------------------------------------------ ground truth
def selftest(check):
    rng = np.random.default_rng(0)
    n_subj, n_cls, per, d = 8, 4, 80, 5
    centers = rng.standard_normal((n_cls, d)) * 2.0
    X, y, s = [], [], []
    for u in range(n_subj):
        off = rng.standard_normal(d) * 5.0                    # pure per-subject MEAN offset
        for c in range(n_cls):
            X.append(centers[c] + off + rng.standard_normal((per, d)) * 0.7)
            y += [c] * per; s += [u] * per
    X = np.vstack(X); y = np.array(y); s = np.array(s)

    check("T6 balanced data has ratio ~1", abs(C.imbalance_ratio(y, s) - 1.0) < 0.01)
    check("T6 ratio=1 leaves every row in place", C.induce_imbalance(y, s, 1, 0).all())
    m10 = C.induce_imbalance(y, s, 10, seed=0)
    got = C.imbalance_ratio(y[m10], s[m10])
    check("T6 induce_imbalance(10) actually produces ~10x skew", 5.0 < got < 20.0, f"achieved={got:.1f}")

    bal = _baseline_vs_center(X, y, s, 0)["center_benefit"]
    check("T6 mechanism exists: on a pure mean-offset synthetic, centring helps when BALANCED",
          bal > 0.02, f"benefit={bal:+.3f}")
    imb = _baseline_vs_center(X[m10], y[m10], s[m10], 0)["center_benefit"]
    check("T6 mechanism exists: the SAME data made 10x imbalanced degrades the centring benefit "
          "(a class-biased mean is a bad mean)", imb < bal, f"imbalanced={imb:+.3f} vs balanced={bal:+.3f}")

    # NEGATIVE CONTROL / CONFOUND CHECK (added after the 2026-07-13 review).
    # `induce_imbalance` keeps the majority class whole and shrinks the rest, so a higher ratio ALWAYS
    # means FEWER ROWS. Any decay in the centring benefit is therefore confounded with sample size,
    # and `imb < bal` above cannot tell the two apart. Here we build a BALANCED subsample with the
    # SAME number of rows as the 10x-imbalanced one: if the benefit survives at equal n, the decay we
    # attribute to imbalance is really about imbalance and not about data volume.
    rng2 = np.random.default_rng(7)
    keep = np.zeros(len(y), bool)
    per_subj_rows = int(m10.sum() // len(np.unique(s)))
    for u in np.unique(s):
        rows_u = np.flatnonzero(s == u)
        cls = np.unique(y[rows_u])
        per_cls = max(2, per_subj_rows // len(cls))
        for c in cls:                                        # balanced: equal rows per class
            idx = rows_u[y[rows_u] == c]
            keep[rng2.choice(idx, min(per_cls, len(idx)), replace=False)] = True
    eq = _baseline_vs_center(X[keep], y[keep], s[keep], 0)["center_benefit"]
    check("T6 CONFOUND CONTROL: a BALANCED subsample of the same size keeps the centring benefit, so "
          "the decay is about imbalance and not merely about having less data",
          eq > imb, f"equal-n balanced={eq:+.3f} vs 10x imbalanced={imb:+.3f} "
                    f"(n={int(keep.sum())} vs {int(m10.sum())})")
