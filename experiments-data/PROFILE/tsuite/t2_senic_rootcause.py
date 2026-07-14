"""T2 — Why does senic go the WRONG WAY? (the one dataset that reverses the headline)

WHY THIS EXPERIMENT EXISTS
--------------------------
On 12 of 14 datasets, a subject who is far from the pool (high MMD) is a subject the model does
badly on: r < 0. On senic the sign FLIPS: r = +0.31. Being far from everyone else makes you EASIER.
That is not a small blemish — an unexplained sign reversal in 1/14 of your panel is the first thing
a reviewer will point at, and "we exclude it as an outlier" is not an answer.

Four candidate explanations exist and nobody has separated them. Each makes a different, testable
prediction, so one experiment can arbitrate all four.

  H1  TARGET MIS-SPECIFICATION. The difficulty target is LDA accuracy. On senic, and on senic alone,
      a random forest reaches 0.415 while LDA reaches 0.286 — a 13-point gap, far larger than on any
      other dataset (elsewhere RF and LDA are within ~2 points). So the LDA target may simply be the
      wrong measure of "hard" for this data. PREDICTION: with a non-linear target the reversal
      weakens or disappears. (The committed robust_difficulty numbers already hint at this: the
      reversal falls from +0.30 under LDA to +0.12 under RF.)

  H2  SESSION-COUNT CONFOUND. senic's subjects are wildly uneven: 22 subjects have 1 session, 8 have
      3, and 6 have 10. A subject with 10 sessions has more (and more varied) data, which changes
      BOTH its distance to the pool AND its accuracy. PREDICTION: the reversal lives in the
      multi-session subjects and vanishes when we restrict to the 22 single-session subjects.
      (The committed senic_probe returns "inconclusive" on exactly this question — it tested the
      confound at the whole-dataset level but never re-fit the correlation on a clean subset.)

  H3  CONDITION AXIS. senic's "sessions" are not days — they are electrode-shift and fatigue
      CONDITIONS. Pooling them may mix two different populations of windows per subject.
      PREDICTION: within a single condition the reversal disappears.

  H4  IT IS REAL. senic genuinely has the property that distinctive subjects are easy — e.g. because
      its 7 classes are few and coarse, so an idiosyncratic subject is idiosyncratic in a way that
      SEPARATES its classes rather than moving it away from the decision boundary.
      PREDICTION: the reversal survives every control above.

PRE-REGISTERED BRANCHES
-----------------------
  A. H1 confirmed (reversal dies under a non-linear target)
     -> the fix is scientific, not cosmetic: the difficulty TARGET must be model-agnostic. This
        promotes T1 from a robustness check to a load-bearing part of the method, and senic stops
        being an outlier. **Best case, and it makes the paper stronger.**
  B. H2 or H3 confirmed
     -> senic is excluded for a STATED, MEASURED reason (uneven sessions / condition mixing), not
        because it was inconvenient. That is a legitimate, defensible exclusion.
  C. H4 (reversal survives everything)
     -> report it as a genuine counter-example and say plainly that the predictor's sign is not
        universal. Costly but honest; the pooled effect already excludes senic in a sensitivity
        analysis, so the paper survives.

This experiment runs on senic (where the effect is) and on THREE control datasets with the normal
sign (emaha_db1, ninapro_db2, grabmyo). The controls matter: if a control ALSO flips sign under a
non-linear target, then H1 is not about senic at all — it is about the target, everywhere.
"""
from __future__ import annotations

import numpy as np

from . import common as C

TAG = "t2_senic_rootcause"
DATASETS = ["senic", "emaha_db1", "ninapro_db2", "grabmyo"]     # 1 case + 3 controls
TARGET_MODELS = ("lda", "rf", "svm_rbf", "hgb")


def _corr(mmd, acc):
    r = C.corr_across_subjects(mmd, acc)
    return dict(r=r["r"], p=r["p"], n=r["n"])


def run_one(dataset, seed=42, n_jobs=1):
    with C.timer(f"T2 :: {dataset}"):
        frame = C.build_frame(dataset, seed=seed)
        X, _ = C.basis(frame)
        y = frame.label.to_numpy()
        subj = frame.subject.to_numpy()
        sess = frame.session.to_numpy() if "session" in frame.columns else None

        out = dict(dataset=dataset, n_subjects=int(len(np.unique(subj))),
                   n_classes=int(len(np.unique(y))))

        # ---- H1: does the sign depend on the TARGET MODEL? -------------------------------------
        mmd = C.mmd_to_pool(X, subj, seed=seed)
        h1 = {}
        jobs = [(C.loso_accuracy_model, (X, y, subj, m, seed, 15000, 5000)) for m in TARGET_MODELS]
        for m, acc in zip(TARGET_MODELS, C.maybe_parallel(jobs, n_jobs)):
            if len(acc) >= 5:
                h1[m] = dict(mean_acc=float(np.mean(list(acc.values()))), **_corr(mmd, acc))
        out["H1_target_model"] = h1
        signs = [v["r"] for v in h1.values() if np.isfinite(v["r"])]
        out["H1_sign_flips_with_target"] = bool(
            signs and (min(signs) < 0 < max(signs)))

        # ---- H2: is it the uneven session count? -----------------------------------------------
        if sess is not None:
            n_sess = {int(s): int(len(np.unique(sess[subj == s]))) for s in np.unique(subj)}
            out["session_counts"] = {str(k): v for k, v in sorted(n_sess.items())}
            single = np.array([s for s, k in n_sess.items() if k == 1])
            if len(single) >= 8:
                keep = np.isin(subj, single)
                mmd_s = C.mmd_to_pool(X[keep], subj[keep], seed=seed)
                acc_s = C.loso_accuracy_model(X[keep], y[keep], subj[keep], "lda", seed)
                out["H2_single_session_subjects_only"] = dict(
                    n_subjects=int(len(single)), **_corr(mmd_s, acc_s))
            else:
                out["H2_single_session_subjects_only"] = dict(
                    note=f"only {len(single)} single-session subjects; not testable")

            # ---- H3: within ONE condition/session index -----------------------------------------
            per_cond = {}
            for c in np.unique(sess):
                keep = sess == c
                if len(np.unique(subj[keep])) < 8:
                    continue
                mmd_c = C.mmd_to_pool(X[keep], subj[keep], seed=seed)
                acc_c = C.loso_accuracy_model(X[keep], y[keep], subj[keep], "lda", seed)
                if len(acc_c) >= 8:
                    per_cond[str(int(c))] = _corr(mmd_c, acc_c)
            out["H3_within_condition"] = per_cond
            rs = [v["r"] for v in per_cond.values() if np.isfinite(v["r"])]
            out["H3_mean_r_within_condition"] = float(np.mean(rs)) if rs else float("nan")
        else:
            out["H2_single_session_subjects_only"] = dict(note="dataset has no session column")
            out["H3_within_condition"] = {}

    # ---- verdict per dataset ------------------------------------------------------------------
    r_lda = h1.get("lda", {}).get("r", float("nan"))
    r_non = [h1[m]["r"] for m in ("rf", "svm_rbf", "hgb") if m in h1 and np.isfinite(h1[m]["r"])]
    out["r_lda"] = r_lda
    out["r_nonlinear_mean"] = float(np.mean(r_non)) if r_non else float("nan")
    out["reversed_under_lda"] = bool(np.isfinite(r_lda) and r_lda > 0)
    out["reversal_survives_nonlinear_target"] = bool(
        out["reversed_under_lda"] and np.isfinite(out["r_nonlinear_mean"])
        and out["r_nonlinear_mean"] > 0.10)
    return out


def build_pooled(tag=TAG):
    import json
    rows = {}
    for f in sorted(C.results_dir(tag).glob(f"*__{tag}.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows[d["dataset"]] = d
    if "senic" not in rows:
        return dict(note="senic result missing; nothing to arbitrate")

    s = rows["senic"]
    controls = {k: v for k, v in rows.items() if k != "senic"}
    ctrl_flip = [k for k, v in controls.items() if v.get("reversed_under_lda")]

    # BRANCH 0 — the premise. `reversal_survives_nonlinear_target` is False in TWO different worlds:
    # (i) senic reversed under LDA and the reversal died under a non-linear target (genuine H1), and
    # (ii) senic never reversed under LDA at all in this protocol. Reporting (ii) as "H1 CONFIRMED —
    # the reversal is a target artifact" while quoting a NEGATIVE r_lda is exactly the class of error
    # CLAUDE.md 8d forbids. Check the premise before arbitrating the hypotheses.
    if not s.get("reversed_under_lda"):
        branch, verdict = "0", (
            f"PREMISE VOID - senic does NOT reverse under this protocol (r_lda={s['r_lda']:+.3f}). "
            "The committed legacy value was +0.31; this run does not reproduce it. Reconcile the two "
            "protocols BEFORE drawing any conclusion about senic. Nothing about H1/H2/H3/H4 may be "
            "concluded from this run.")
    elif ctrl_flip:
        branch, verdict = "X", (
            f"CONTROL CONTAMINATION - the control dataset(s) {ctrl_flip} ALSO reverse under LDA. The "
            "effect is then about the TARGET or the protocol, not about senic. Read T1 before "
            "concluding anything about senic.")
    elif not s.get("reversal_survives_nonlinear_target"):
        branch, verdict = "A", (
            f"H1 CONFIRMED — senic's reversal is a TARGET artifact. Under LDA r={s['r_lda']:+.3f}; "
            f"averaged over non-linear targets r={s['r_nonlinear_mean']:+.3f}. The difficulty target "
            "must be model-agnostic (see T1); with a proper target senic is no longer an outlier.")
    else:
        h2 = s.get("H2_single_session_subjects_only", {})
        h3 = s.get("H3_mean_r_within_condition", float("nan"))
        if np.isfinite(h2.get("r", np.nan)) and h2["r"] < 0:
            branch, verdict = "B", (
                f"H2 CONFIRMED — the reversal is driven by senic's uneven session counts. Restricted "
                f"to its {h2['n']} single-session subjects the sign returns to normal (r={h2['r']:+.3f}). "
                "senic may be excluded for a measured reason.")
        elif np.isfinite(h3) and h3 < 0:
            branch, verdict = "B", (
                f"H3 CONFIRMED — pooling senic's electrode-shift CONDITIONS creates the reversal; "
                f"within a single condition the sign is normal (mean r={h3:+.3f}).")
        else:
            branch, verdict = "C", (
                f"H4 — the reversal SURVIVES every control (non-linear target r="
                f"{s['r_nonlinear_mean']:+.3f}). senic is a genuine counter-example: on this dataset "
                "distinctive subjects really are easier. Report it; do not hide it.")

    out = dict(tag=tag, senic=dict(r_lda=s.get("r_lda"), r_nonlinear=s.get("r_nonlinear_mean")),
               controls={k: dict(r_lda=v.get("r_lda"), r_nonlinear=v.get("r_nonlinear_mean"))
                         for k, v in controls.items()},
               controls_that_also_reverse=ctrl_flip,
               control_warning=("A CONTROL dataset also reverses -> the effect is about the TARGET, "
                                "not about senic. Re-read T1 before concluding anything about senic."
                                if ctrl_flip else "controls behave normally (r<0), as expected"),
               branch=branch, verdict=verdict)
    C.atomic_write_json(C.results_dir(tag) / "pooled.json", out)
    return out


# ------------------------------------------------------------------ ground truth
def selftest(check):
    """We can BUILD a dataset whose sign reverses, and check the machinery detects the reversal
    and correctly attributes it. Without this, a 'reversal' could be a bug in the correlation code."""
    rng = np.random.default_rng(0)
    n_subj, n_cls, per, d = 14, 4, 60, 6
    # Class information lives ONLY in dims 0-1. Dims 2..5 are NUISANCE directions carrying no label
    # information, so a classifier puts ~no weight on them.
    centers = np.zeros((n_cls, d))
    centers[:, :2] = rng.standard_normal((n_cls, 2)) * 1.6
    X, y, s = [], [], []
    for u in range(n_subj):
        t = u / (n_subj - 1)                         # 0..1; high t = "distinctive"
        # INVERTED difficulty BY CONSTRUCTION (the H4 world). Two ingredients, and BOTH are needed:
        #   1. the subject's distance from the pool is an offset along the NUISANCE directions only,
        #      so it raises MMD WITHOUT moving the class boundary. (My first attempt offset along the
        #      DISCRIMINATIVE direction, which makes a far subject HARD — that is the normal world,
        #      not the reversed one, and the check correctly failed with r = -0.79.)
        #   2. the same subjects have TIGHTER clusters, so they are genuinely EASIER to classify.
        noise = 1.6 - 1.3 * t
        direction = rng.standard_normal(d)
        direction[:2] = 0.0                          # nuisance subspace only
        direction /= np.linalg.norm(direction) + 1e-9
        off = direction * (6.0 * t)
        for c in range(n_cls):
            Z = np.zeros((per, d))
            Z[:, :2] = centers[c, :2]
            Z += rng.standard_normal((per, d)) * noise + off
            X.append(Z); y += [c] * per; s += [u] * per
    X = np.vstack(X); y = np.array(y); s = np.array(s)
    mmd = C.mmd_to_pool(X, s, seed=0)
    acc = C.loso_accuracy_model(X, y, s, "lda", seed=0)
    r = C.corr_across_subjects(mmd, acc)["r"]
    check("T2 machinery detects a CONSTRUCTED sign reversal (far == easy => r > 0)",
          r > 0.3, f"r={r:+.3f}")

    # and on a normal (real-difficulty) synthetic it must NOT report a reversal
    fr = C.synth_frame("real_difficulty", n_subjects=12, n_classes=5, per_class=40, seed=2)
    Xf, _ = C.basis(fr)
    mmd2 = C.mmd_to_pool(Xf, fr.subject.to_numpy(), seed=2)
    acc2 = C.loso_accuracy_model(Xf, fr.label.to_numpy(), fr.subject.to_numpy(), "lda", seed=2,
                                 cap_train=4000)
    r2 = C.corr_across_subjects(mmd2, acc2)["r"]
    check("T2 machinery reports NO reversal on a normal synthetic", r2 < 0, f"r={r2:+.3f}")
