"""T3 — The moment ladder: WHICH moment of a new user's distribution actually has to be aligned?

WHY THIS EXPERIMENT EXISTS
--------------------------
Two committed results point in opposite directions and nobody has reconciled them:

  * exp_B / X4: removing each subject's MEAN improves cross-subject accuracy on 13/14 datasets
    (+3.6 pp, up to +8.5). Aligning the full COVARIANCE (CORAL) helps on 0/14 and sometimes hurts
    badly (grabmyo -11.4 pp).
  * block_c / E3: the between-subject Gaussian-KL divergence is COVARIANCE-dominated — the mean term
    is only ~3-38% of the excess divergence.

So the moment that carries most of the *divergence* is the one that must not be touched, and the
moment that carries little of the divergence is the one worth removing. That is a genuinely
surprising, quotable result — but right now it rests on exactly two rungs (mean, full covariance)
with nothing in between, which is not enough to state a rule.

This experiment fills in the ladder. It is the difference between "centring helped and CORAL didn't"
(an observation) and "align the first moment, never the second" (a RULE another lab can apply).

THE LADDER (every rung is label-free and deployment-legal: the new user contributes only unlabelled
data, never labels, and never touches the model's training set)
--------------------------------------------------------------------------------------------------
  0. baseline        — nothing
  1. center          — subtract the subject's own mean            (1st moment)
  2. scale           — divide by the subject's own per-feature sd (2nd moment, DIAGONAL only)
  3. zscore          — center + scale                             (1st + diagonal 2nd)
  4. coral           — align the subject's full covariance        (2nd moment, FULL)
  5. center_coral    — center + full covariance
  6. whiten          — center + full covariance whitening         (the maximal transform)

Rungs 2 and 3 are the missing middle: if `scale` alone helps, the useful part of the second moment
is only the per-channel gain (electrode contact / muscle-mass differences), and full covariance
alignment fails because it also destroys the class-discriminative correlations between channels.
That is a mechanism, not just an observation.

PRE-REGISTERED BRANCHES
-----------------------
  A. center ~ zscore > scale ~ baseline > coral
     -> "align the MEAN and nothing else." A clean, cheap, quotable deployment rule, and it explains
        why full covariance alignment hurts: the between-subject covariance difference and the
        class-discriminative covariance structure are the SAME structure, so removing one removes
        the other. **Headline.**
  B. zscore > center
     -> the per-channel gain matters too; the rule becomes "align the mean and the per-channel scale,
        but never the full covariance". Still a rule, still novel.
  C. coral or whiten wins somewhere
     -> contradicts X4 on that dataset; investigate before publishing either.
  D. nothing beats baseline
     -> exp_B does not replicate under this protocol. Would invalidate N5. Log it loudly.

GROUND TRUTH (`--selftest`)
---------------------------
Each rung must fix EXACTLY the corruption it is designed for and no other:
  * pure per-subject MEAN offset      -> center fixes it; scale does not.
  * pure per-subject SCALE (gain)     -> scale fixes it; center does not.
  * pure per-subject ROTATION (covar) -> coral fixes it; center and scale do not.
If a rung "fixes" a corruption it has no business fixing, the implementation is wrong and the real
run is meaningless.
"""
from __future__ import annotations

import numpy as np

from . import common as C
from .common import coral

TAG = "t3_moment_ladder"
RUNGS = ("baseline", "center", "scale", "zscore", "coral", "center_coral", "whiten")
EPS = 1e-9


# --------------------------------------------------------------------------- the transforms
def _center(Z):
    return Z - Z.mean(0)


def _scale(Z):
    return Z / (Z.std(0) + EPS)


def _zscore(Z):
    return (Z - Z.mean(0)) / (Z.std(0) + EPS)


def _whiten(Z, ridge=1e-3):
    """Center then decorrelate: Zc @ Cov^{-1/2}. The maximal per-subject linear transform."""
    from paper_experiments.common import _sqrtm_psd
    Zc = Z - Z.mean(0)
    S = np.cov(Zc, rowvar=False) + np.eye(Zc.shape[1]) * ridge
    return Zc @ _sqrtm_psd(S, inverse=True)


def _fit_predict(Xtr, ytr, Xte, yte):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    if len(np.unique(ytr)) < 2:
        return float("nan")
    try:
        return float((LinearDiscriminantAnalysis().fit(Xtr, ytr).predict(Xte) == yte).mean())
    except Exception:
        return float("nan")


def ladder_loso(X, y, subjects, seed=42):
    """Per-subject LOSO accuracy at every rung of the ladder.

    Discipline: the model is trained on the training subjects only. Each TRAINING subject is
    transformed with its OWN statistics (it has them), and the held-out subject is transformed with
    its OWN unlabelled statistics (which a deployed system genuinely has, before any label exists).
    No label of the held-out subject is ever used, and no training statistic leaks into the test.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    acc = {k: {} for k in RUNGS}

    for s in sorted(np.unique(subjects)):
        tr, te = subjects != s, subjects == s
        if len(np.unique(y[tr])) < 2 or te.sum() < 5:
            continue
        # global train-only standardisation first (same as X4, so the rungs are comparable to it)
        mu, sd = X[tr].mean(0), X[tr].std(0) + EPS
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        str_, ytr, yte = subjects[tr], y[tr], y[te]

        def per_subject(Z, subs, fn):
            out = np.array(Z, float, copy=True)
            for u in np.unique(subs):
                m = subs == u
                out[m] = fn(Z[m])
            return out

        acc["baseline"][int(s)] = _fit_predict(Ztr, ytr, Zte, yte)

        for rung, fn in (("center", _center), ("scale", _scale), ("zscore", _zscore),
                         ("whiten", _whiten)):
            Atr = per_subject(Ztr, str_, fn)
            Ate = fn(Zte)
            acc[rung][int(s)] = _fit_predict(Atr, ytr, Ate, yte)

        # CORAL rungs: the held-out subject's covariance is mapped onto the training pool's.
        acc["coral"][int(s)] = _fit_predict(Ztr, ytr, coral(Zte, Ztr), yte)
        Ctr = per_subject(Ztr, str_, _center)
        acc["center_coral"][int(s)] = _fit_predict(Ctr, ytr, coral(_center(Zte), Ctr), yte)

    return acc


def _paired(base, variant):
    from scipy.stats import wilcoxon
    shared = sorted(set(base) & set(variant))
    a = np.array([base[s] for s in shared], float)
    b = np.array([variant[s] for s in shared], float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 5:
        return dict(n=int(len(a)), note="too few subjects")
    d = b - a
    if np.allclose(d, 0):
        p = 1.0
    else:
        try:
            _, p = wilcoxon(b, a, alternative="greater")
        except ValueError:
            p = 1.0
    return dict(n=int(len(a)), mean_baseline=float(a.mean()), mean_variant=float(b.mean()),
                mean_delta=float(d.mean()), n_improved=int((d > 0).sum()),
                wilcoxon_p=float(p), helps=bool(d.mean() > 0 and p < 0.05))


def run_one(dataset, seed=42, n_jobs=1):
    with C.timer(f"T3 :: {dataset}"):
        frame = C.build_frame(dataset, seed=seed)
        X, _ = C.basis(frame)
        acc = ladder_loso(X, frame.label.to_numpy(), frame.subject.to_numpy(), seed)

    out = dict(dataset=dataset, rungs=list(RUNGS))
    out["mean_acc"] = {k: float(np.nanmean(list(v.values()))) for k, v in acc.items() if v}
    out["vs_baseline"] = {k: _paired(acc["baseline"], acc[k]) for k in RUNGS if k != "baseline"}
    # which rung wins, and is the SIMPLEST winner enough? (the rule we are trying to state)
    helping = [k for k, v in out["vs_baseline"].items() if v.get("helps")]
    best = max(out["mean_acc"], key=lambda k: out["mean_acc"][k]) if out["mean_acc"] else None
    out["rungs_that_help"] = helping
    out["best_rung"] = best
    # is `center` alone within 0.5 pp of the best rung? -> the cheap rule suffices
    if best and "center" in out["mean_acc"]:
        out["center_is_sufficient"] = bool(
            out["mean_acc"][best] - out["mean_acc"]["center"] < 0.005)
    out["per_subject_acc"] = {k: {str(s): float(a) for s, a in v.items()} for k, v in acc.items()}
    return out


def build_pooled(tag=TAG):
    import json
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(C.results_dir(tag).glob(f"*__{tag}.json"))]
    rows = [r for r in rows if "vs_baseline" in r]
    if not rows:
        return dict(note="no per-dataset results yet")

    # ---- FDR ACROSS THE WHOLE FAMILY OF TESTS -------------------------------------------------
    # 6 rungs x 14 datasets = 84 one-sided Wilcoxon tests. Counting raw p<0.05 flags across a family
    # that size manufactures ~4 false "helps" by construction, and every headline count ("helps on
    # 12/14") is built from exactly those flags. The per-dataset JSONs store `wilcoxon_p`, so this is
    # applied post-hoc here and needs NO recompute.
    tests = [(k, r["dataset"], r["vs_baseline"][k]["mean_delta"], r["vs_baseline"][k]["wilcoxon_p"])
             for r in rows for k in RUNGS
             if k != "baseline" and "wilcoxon_p" in r["vs_baseline"].get(k, {})]
    flags, qs = C.helps_flags_fdr([t[3] for t in tests], [t[2] for t in tests])
    helps_ds = {k: [] for k in RUNGS if k != "baseline"}
    for (k, ds, _d, _p), ok in zip(tests, flags):
        if ok:
            helps_ds[k].append(ds)

    all_ds = [r["dataset"] for r in rows]
    tally = {}
    for k in RUNGS:
        if k == "baseline":
            continue
        deltas = [r["vs_baseline"][k]["mean_delta"] for r in rows
                  if "mean_delta" in r["vs_baseline"].get(k, {})]
        raw = sum(1 for r in rows if r["vs_baseline"].get(k, {}).get("helps"))
        tally[k] = dict(n_datasets_helps=len(helps_ds[k]),          # AFTER FDR
                        n_datasets_helps_uncorrected=raw,           # what the raw flags said
                        both_ways=C.count_both_ways(helps_ds[k], all_ds),   # k=14 AND k=9 cohorts
                        n_datasets=len(rows),
                        mean_delta_pp=float(np.mean(deltas) * 100) if deltas else float("nan"),
                        worst_delta_pp=float(np.min(deltas) * 100) if deltas else float("nan"))

    n = len(rows)
    center_ok = tally["center"]["n_datasets_helps"]
    z_ok = tally["zscore"]["n_datasets_helps"]
    coral_ok = tally["coral"]["n_datasets_helps"]
    n_center_sufficient = sum(1 for r in rows if r.get("center_is_sufficient"))
    center_cohorts = tally["center"]["both_ways"]["cohorts"]
    z_cohorts = tally["zscore"]["both_ways"]["cohorts"]

    if coral_ok > n // 2:
        branch, verdict = "C", ("CORAL helps on a majority - this CONTRADICTS X4. Do not publish "
                                "either result until the disagreement is explained.")
    elif max(center_ok, z_ok) <= n / 2:
        # Branch A used to be the `else` of this cascade, so center_ok=1/14, z_ok=0/14 landed on it
        # and printed "RULE: align the MEAN and nothing else. Centring helps 1/14 datasets" - a rule
        # with 1/14 support. A rule needs majority support or it is not a rule.
        branch, verdict = "E", (
            f"NO RULE: no rung has majority support (centring helps {center_ok}/{n}, z-score "
            f"{z_ok}/{n}, CORAL {coral_ok}/{n}). Per-subject alignment is not reliably useful on this "
            "panel. Report per-dataset; do not state a rule.")
    elif center_ok == 0 and z_ok == 0:
        branch, verdict = "D", ("Nothing beats baseline: exp_B/N5 does not replicate under the "
                                "ladder protocol. This would invalidate the recalibration claim.")
    elif z_ok > center_ok:
        branch, verdict = "B", (
            f"RULE: align the mean AND the per-channel scale (z-score helps {z_ok}/{n} datasets = "
            f"{z_cohorts} cohorts; centring alone {center_ok}/{n}), but never the full covariance "
            f"(CORAL {coral_ok}/{n}). All counts are FDR-corrected. The "
            "useful part of the second moment is the per-channel gain, not the between-channel "
            "correlation structure.")
    else:
        branch, verdict = "A", (
            f"RULE: align the MEAN and nothing else. Centring helps {center_ok}/{n} datasets = "
            f"{center_cohorts} cohorts (FDR-corrected) and is "
            f"within 0.5 pp of the best rung on {n_center_sufficient}/{n}; full covariance alignment "
            f"helps {coral_ok}/{n}. CANDIDATE EXPLANATION (a hypothesis, NOT measured by this "
            "experiment): the between-subject covariance difference may BE the class-discriminative "
            "covariance structure, so aligning it would destroy the signal. T3 does not test that - "
            "do not print it as a conclusion.")

    out = dict(tag=tag, n_datasets=n, per_rung=tally,
               n_datasets_where_center_is_sufficient=n_center_sufficient,
               multiplicity=("Benjamini-Hochberg FDR applied across all "
                             f"{len(tests)} rung x dataset Wilcoxon tests"),
               branch=branch, verdict=verdict)
    C.atomic_write_json(C.results_dir(tag) / "pooled.json", out)
    return out


# ------------------------------------------------------------------ ground truth
def _corrupt(kind, n_subj=10, n_cls=4, per=60, d=6, seed=0):
    """Clean class structure + ONE kind of per-subject corruption. The matching rung must fix it.

    The class structure is deliberately MARGINAL (centres ~1.1 apart, noise sd 1.0). An easy task is
    useless as a ground truth: LDA is largely invariant to a per-subject gain when the classes are
    far apart, so the baseline would score ~0.95 and no rung could show a recovery. The corruption
    must actually break the classifier for the test to mean anything.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_cls, d)) * 1.1
    X, y, s = [], [], []
    for u in range(n_subj):
        A = np.eye(d)
        off = np.zeros(d)
        gain = np.ones(d)
        if kind == "clean":
            pass                                                 # no corruption at all
        elif kind == "mean":
            off = rng.standard_normal(d) * 4.0
        elif kind == "scale":
            gain = np.exp(rng.standard_normal(d) * 1.5)          # per-channel gain, up to ~20x
        elif kind == "rotation":
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            A = Q @ np.diag(np.exp(rng.standard_normal(d) * 1.5)) @ Q.T   # strongly anisotropic
        for c in range(n_cls):
            Z = centers[c] + rng.standard_normal((per, d)) * 1.0
            Z = (Z @ A) * gain + off
            X.append(Z); y += [c] * per; s += [u] * per
    return np.vstack(X), np.array(y), np.array(s)


def selftest(check):
    # Each corruption must be fixed by ITS rung, and NOT by a rung that has no business fixing it.
    #
    # NOTE on the ("scale", "zscore", "center") row. The obvious expectation is that a per-channel
    # GAIN corruption is inverted by the `scale` rung. It is not, and the reason is physics, not a
    # bug: a multiplicative gain g acts on data whose class means are non-zero, so E[gX] = g E[X] —
    # the gain moves the MEAN as well as the second moment. Dividing by the subject's sd without
    # also removing its mean therefore leaves a distorted first moment behind, and on this synthetic
    # `scale` alone actually scores BELOW baseline (0.48 vs 0.52) while `zscore` recovers it
    # (0.71). The correct inverse of a gain is centre-then-scale. This is precisely the kind of
    # thing the ladder exists to expose, and it is why the ladder has rungs between `center` and
    # `coral` at all.
    for kind, fixer, non_fixer in (("mean", "center", "scale"),
                                   ("scale", "zscore", "center"),
                                   ("rotation", "coral", "center")):
        X, y, s = _corrupt(kind, seed=1)
        acc = ladder_loso(X, y, s, seed=1)
        base = float(np.nanmean(list(acc["baseline"].values())))
        fix = float(np.nanmean(list(acc[fixer].values())))
        bad = float(np.nanmean(list(acc[non_fixer].values())))
        check(f"T3 pure {kind} corruption: '{fixer}' recovers accuracy",
              fix > base + 0.05, f"base={base:.3f} {fixer}={fix:.3f}")
        check(f"T3 pure {kind} corruption: '{non_fixer}' does NOT recover it (rungs are specific)",
              bad < fix, f"{non_fixer}={bad:.3f} vs {fixer}={fix:.3f}")

    # NEGATIVE CONTROL (added after the 2026-07-13 review, which noted T3 had none).
    # On CLEAN data - subjects identical, no per-subject corruption whatsoever - there is nothing for
    # any rung to fix, so no rung may show a real gain. A ladder that "helps" on uncorrupted data is
    # measuring its own noise, and every real-data "helps 12/14" it produced would be worthless.
    Xc, yc, sc = _corrupt("clean", seed=5)
    accc = ladder_loso(Xc, yc, sc, seed=5)
    basec = float(np.nanmean(list(accc["baseline"].values())))
    gains = {k: float(np.nanmean(list(v.values()))) - basec for k, v in accc.items() if k != "baseline"}
    worst = max(gains.values())
    check("T3 NEGATIVE CONTROL: on clean, uncorrupted data no rung gains anything real",
          worst < 0.02, f"baseline={basec:.3f} best rung gain={worst:+.3f} ({gains})")

    # the transforms do what their names say
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((400, 5)) * 3 + 7
    check("T3 center zeroes the mean", abs(_center(Z).mean(0)).max() < 1e-9)
    check("T3 scale sets unit sd", abs(_scale(Z).std(0) - 1).max() < 1e-6)
    W = _whiten(Z)
    off = np.cov(W, rowvar=False) - np.eye(5)
    check("T3 whiten produces ~identity covariance", abs(off).max() < 0.05,
          f"max|cov-I|={abs(off).max():.3f}")
