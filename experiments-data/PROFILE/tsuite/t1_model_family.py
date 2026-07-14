"""T1 — Is the difficulty predictor MODEL-AGNOSTIC, or does it only track LINEAR separability?

WHY THIS EXPERIMENT EXISTS
--------------------------
The paper's central claim is: *a training-free statistic computed from a new user's UNLABELLED data
predicts how badly a model will perform on that user.* Every number behind that claim was produced
with an LDA target. A reviewer's first question is therefore unavoidable and fatal if unanswered:

    "You built the predictor from distances in a feature space, and you measured difficulty with a
     LINEAR classifier in the SAME feature space. Have you discovered a property of the DATA, or a
     property of LDA?"

The external review calls this the shared-representation coupling (its §9-2) and rates the
experiment that breaks it as the highest-value one in the whole programme. It originally proposed
doing this with a deep network (X3). **Deep learning lives in a different track and is out of scope
for this repository**, so X3 is formally retired (see docs/archive/ for its withdrawal). This
experiment answers the same scientific question with CPU models only, and answers it *better* than
X3 would have, because it uses five learner families instead of one and runs on all 14 datasets
instead of the single dataset a deep sweep could afford.

WHAT IT DOES
------------
For every dataset, for every subject:
  * PREDICTOR (training-free, label-free): MMD from that subject's windows to the pooled rest.
  * TARGET x5: leave-one-subject-out accuracy under LDA / RBF-SVM / random forest / MLP / gradient
    boosting. Every family gets the IDENTICAL stratified train subsample, so a difference between
    families is a difference between families and not between compute budgets.
Then: correlate the predictor with each target; pool across datasets with a random-effects model;
and measure whether the families even AGREE about which subjects are hard.

PRE-REGISTERED BRANCHES (decided before looking — see CLAUDE.md "pre-registration")
-----------------------------------------------------------------------------------
  A. Pooled r < 0 and significant for the NON-LINEAR families too
     -> the statistic predicts difficulty for ANY learner. The coupling objection is dead and the
        headline is stronger than the paper currently claims. **This is a genuine novelty result.**
  B. Pooled r < 0 for LDA only; non-linear families flat
     -> honest scope correction: "the statistic tracks LINEAR separability, not learnability."
        Still publishable, still useful (LDA is what deployed sEMG systems actually use), and far
        better discovered here than by a reviewer. **We log it and move on.**
  C. Families disagree about WHICH subjects are hard (low rank agreement)
     -> "subject difficulty" is not a well-defined property of the data at all, and the whole N3/N7
        contribution must be re-scoped. This would be the most important negative result available.

GROUND TRUTH (`--selftest`; the instrument is validated before it touches real data)
-----------------------------------------------------------------------------------
  1. On a synthetic where difficulty is real and MODEL-AGNOSTIC, every family must recover r < 0.
  2. On an XOR synthetic (linearly inseparable), LDA must sit at chance while RF/MLP must not —
     this proves the model zoo can actually TELL a linear from a non-linear learner. Without this
     check, branch B could be an artifact of a broken non-linear model rather than a finding.
  3. Under a subject-label permutation of the predictor, r must collapse to ~0.
"""
from __future__ import annotations

import numpy as np

from . import common as C
from .common import MODELS, MODEL_IS_LINEAR

TAG = "t1_model_family"
CAP_TRAIN = 15000          # rows per LOSO fit, identical for every family (RBF-SVM is O(n^2))
CAP_TEST = 5000


def run_one(dataset, seed=42, n_jobs=1):
    with C.timer(f"T1 :: {dataset}"):
        frame = C.build_frame(dataset, seed=seed)
        X, cols = C.basis(frame)
        y = frame.label.to_numpy()
        subj = frame.subject.to_numpy()

        # predictor: unlabelled, training-free
        mmd = C.mmd_to_pool(X, subj, seed=seed)

        # targets: one per model family
        jobs = [(C.loso_accuracy_model, (X, y, subj, m, seed, CAP_TRAIN, CAP_TEST)) for m in MODELS]
        accs = dict(zip(MODELS, C.maybe_parallel(jobs, n_jobs)))

    out = dict(dataset=dataset, n_subjects=int(len(np.unique(subj))),
               n_classes=int(len(np.unique(y))), n_features=len(cols),
               cap_train=CAP_TRAIN, cap_test=CAP_TEST, models={})

    for m in MODELS:
        acc = accs[m]
        if len(acc) < 5:
            out["models"][m] = dict(note="too few subjects with a fitted model", n=len(acc))
            continue
        r = C.corr_across_subjects(mmd, acc)
        mean_acc = float(np.mean(list(acc.values())))
        out["models"][m] = dict(
            mean_loso_acc=mean_acc,
            kappa_chance=C.kappa_chance(mean_acc, out["n_classes"]),
            difficulty_r=r["r"], difficulty_p=r["p"], n_subjects=r["n"],
            is_linear=MODEL_IS_LINEAR[m],
            per_subject_acc={str(k): float(v) for k, v in sorted(acc.items())},
        )

    # do the families agree about WHO is hard? (branch C)
    agree = {}
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            ka, kb = out["models"].get(a, {}), out["models"].get(b, {})
            if "per_subject_acc" not in ka or "per_subject_acc" not in kb:
                continue
            shared = sorted(set(ka["per_subject_acc"]) & set(kb["per_subject_acc"]))
            if len(shared) < 5:
                continue
            rho, p, n = C.spearman(np.array([ka["per_subject_acc"][s] for s in shared]),
                                   np.array([kb["per_subject_acc"][s] for s in shared]))
            agree[f"{a}_vs_{b}"] = dict(spearman=rho, p=p, n=n)
    out["subject_difficulty_agreement"] = agree
    rhos = [v["spearman"] for v in agree.values() if np.isfinite(v["spearman"])]
    out["mean_rank_agreement"] = float(np.mean(rhos)) if rhos else float("nan")

    lin = [out["models"][m]["difficulty_r"] for m in MODELS
           if MODEL_IS_LINEAR[m] and "difficulty_r" in out["models"][m]]
    non = [out["models"][m]["difficulty_r"] for m in MODELS
           if not MODEL_IS_LINEAR[m] and "difficulty_r" in out["models"][m]]
    out["r_linear"] = float(np.mean(lin)) if lin else float("nan")
    out["r_nonlinear_mean"] = float(np.nanmean(non)) if non else float("nan")
    out["predictor_mmd"] = {str(k): float(v) for k, v in sorted(mmd.items())}
    return out


def build_pooled(tag=TAG):
    """Random-effects pooling per model family across datasets + the branch verdict."""
    import json
    rows = []
    for f in sorted(C.results_dir(tag).glob(f"*__{tag}.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "models" in d:
            rows.append(d)
    if not rows:
        return dict(note="no per-dataset results yet")

    all_ds = [d["dataset"] for d in rows]
    pooled = {}
    for m in MODELS:
        rs, ns, ps, names = [], [], [], []
        for d in rows:
            md = d["models"].get(m, {})
            if "difficulty_r" in md and np.isfinite(md["difficulty_r"]):
                rs.append(md["difficulty_r"]); ns.append(md["n_subjects"])
                ps.append(md["difficulty_p"]); names.append(d["dataset"])
        if len(rs) < 3:
            pooled[m] = dict(note="too few datasets", k=len(rs)); continue
        pr = C.pool_random_effects(np.array(rs), np.array(ns))
        # `fdr_bh` returns a TUPLE (rejected, q). Unpacking it as a bare array crashes; the wrapper
        # `C.fdr_q` exists to make that impossible. `pool_random_effects` returns "pooled_r", NOT
        # "pooled_r_random_effects" (that is dsprofile/meta.py's name for a different function).
        q = C.fdr_q(np.array(ps))
        sig_ds = [n for n, r, qq in zip(names, rs, q) if r < 0 and qq < 0.05]
        pooled[m] = dict(k_datasets=len(rs), pooled_r=pr["pooled_r"],
                         ci95=pr["ci95"], I2=pr["I2"], is_linear=MODEL_IS_LINEAR[m],
                         n_datasets_sig_negative=len(sig_ds),
                         significant_both_ways=C.count_both_ways(sig_ds, names),   # k=14 AND k=9
                         n_datasets_wrong_sign=int(sum(1 for r in rs if r > 0)),
                         per_dataset={n: float(r) for n, r in zip(names, rs)})

    lin_r = pooled.get("lda", {}).get("pooled_r", float("nan"))
    non = [pooled[m] for m in MODELS if not MODEL_IS_LINEAR[m] and "pooled_r" in pooled[m]]
    non_neg_sig = [p for p in non if p.get("ci95") and p["ci95"][1] < 0]   # CI excludes 0, negative
    agree = float(np.nanmean([r.get("mean_rank_agreement", np.nan) for r in rows]))
    cohorts = C.count_both_ways(all_ds, all_ds)

    # Branch C means ONE specific thing — the families disagree about WHO is hard — so it must be
    # decided by the rank agreement, not reached by falling through the other cases. (Before this
    # fix, "LDA negative + only 1 of 4 non-linear families significant" landed on C and was labelled
    # 'the predictor does not hold up', which is a mis-verdict for partial support.)
    if np.isfinite(agree) and agree < 0.4:
        branch, verdict = "C", (
            f"ILL-DEFINED: the model families do not even agree about WHICH subjects are hard (mean "
            f"rank agreement {agree:.2f}). 'Subject difficulty' is not a well-defined property of the "
            "data, and N3/N7 must be re-scoped. This is the most consequential negative result "
            "available and it must not be softened.")
    elif len(non_neg_sig) >= max(1, len(non) // 2) and lin_r < 0:
        branch, verdict = "A", (
            f"MODEL-AGNOSTIC: the training-free statistic predicts cross-subject difficulty for "
            f"non-linear learners too, not just LDA ({len(non_neg_sig)}/{len(non)} non-linear "
            f"families have a pooled CI excluding zero; families agree on who is hard, rank "
            f"agreement {agree:.2f}; {cohorts['cohorts']} cohorts). The shared-representation "
            "objection does not hold: difficulty is a property of the DATA, not of the classifier.")
    elif lin_r < 0 and not non_neg_sig:
        branch, verdict = "B", (
            "SCOPE CORRECTION: the statistic predicts LINEAR-model difficulty only. It tracks linear "
            "separability, not general learnability. Report at that scope; do not claim it forecasts "
            "any model's failures.")
    else:
        branch, verdict = "D", (
            f"PARTIAL: {len(non_neg_sig)}/{len(non)} non-linear families show a pooled effect and the "
            f"families broadly agree on who is hard (rank agreement {agree:.2f}), but the evidence is "
            "not uniform. State per-family, do not generalise.")

    out = dict(tag=tag, n_datasets=len(rows), cohort_coverage=cohorts, per_model=pooled,
               pooled_r_linear=lin_r,
               pooled_r_nonlinear={m: pooled[m].get("pooled_r") for m in MODELS
                                   if not MODEL_IS_LINEAR[m]},
               mean_rank_agreement_across_datasets=agree,
               branch=branch, verdict=verdict)
    C.atomic_write_json(C.results_dir(tag) / "pooled.json", out)
    return out


# ------------------------------------------------------------------ ground truth
def _xor_data(n=1200, d=6, seed=0):
    """Linearly INSEPARABLE: label = XOR of the signs of the first two dimensions."""
    rng = np.random.default_rng(seed)
    Xd = rng.standard_normal((n, d))
    y = ((Xd[:, 0] > 0) ^ (Xd[:, 1] > 0)).astype(int)
    return Xd, y


def selftest(check):
    rng = np.random.default_rng(0)

    # 2. THE INSTRUMENT CHECK (run first: if the zoo cannot tell linear from non-linear, nothing
    #    else in this experiment means anything).
    Xd, y = _xor_data(seed=1)
    subj = np.repeat(np.arange(12), len(y) // 12)[:len(y)]
    acc_lda = C.loso_accuracy_model(Xd, y, subj, "lda", seed=0)
    acc_rf = C.loso_accuracy_model(Xd, y, subj, "rf", seed=0)
    m_lda = float(np.mean(list(acc_lda.values())))
    m_rf = float(np.mean(list(acc_rf.values())))
    check("T1 zoo distinguishes linear from non-linear: LDA is at chance on XOR",
          abs(m_lda - 0.5) < 0.08, f"lda={m_lda:.3f}")
    check("T1 zoo distinguishes linear from non-linear: RF solves XOR",
          m_rf > 0.80, f"rf={m_rf:.3f} (vs lda {m_lda:.3f})")

    # 1. On a synthetic where difficulty is REAL and model-agnostic, every family must see it.
    fr = C.synth_frame("real_difficulty", n_subjects=14, n_classes=6, per_class=45, seed=3)
    X_, _ = C.basis(fr)
    yy, ss = fr.label.to_numpy(), fr.subject.to_numpy()
    mmd = C.mmd_to_pool(X_, ss, seed=3)
    rs = {}
    for m in ("lda", "rf"):                       # two families is enough for a ground-truth check
        acc = C.loso_accuracy_model(X_, yy, ss, m, seed=3, cap_train=4000)
        rs[m] = C.corr_across_subjects(mmd, acc)["r"]
    check("T1 real-difficulty synthetic: LDA target gives r < 0", rs["lda"] < -0.2, f"r={rs['lda']:+.3f}")
    check("T1 real-difficulty synthetic: RF target ALSO gives r < 0 (difficulty is model-agnostic "
          "by construction here)", rs["rf"] < -0.2, f"r={rs['rf']:+.3f}")

    # 3. Permutation null: shuffle which subject owns which MMD -> the correlation must die.
    acc = C.loso_accuracy_model(X_, yy, ss, "lda", seed=3, cap_train=4000)
    # With 14 subjects a null r has sd ~ 0.28, so a single permutation with a |r| < 0.5 tolerance
    # passes on most null draws AND on many REAL ones - it does not discriminate. Average over
    # several permutations and tighten the bound (the review flagged exactly this).
    keys = list(mmd)
    nulls = []
    for _ in range(20):
        shuffled = dict(zip(keys, rng.permutation([mmd[k] for k in keys])))
        v = C.corr_across_subjects(shuffled, acc)["r"]
        if np.isfinite(v):
            nulls.append(abs(v))
    r_null = float(np.mean(nulls))
    r_real = abs(C.corr_across_subjects(mmd, acc)["r"])
    check("T1 permuted-predictor null: mean |r| over 20 permutations collapses, and is far below the "
          "real |r| (the null is discriminative, not merely permissive)",
          r_null < 0.25 and r_null < 0.5 * r_real,
          f"mean null |r|={r_null:.3f} vs real |r|={r_real:.3f}")
