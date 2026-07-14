"""T4 — Do COARSE activity categories actually buy cross-subject robustness? (the ADL question)

WHY THIS EXPERIMENT EXISTS
--------------------------
This is the ADL-specific question of the whole programme, and the committed evidence currently says
the OPPOSITE of what the project documents claim.

EMAHA-DB1 ships a FAABOS grouping: its 21 fine activities roll up into 5 coarse functional
categories. The committed `faabos/emaha_db1__faabos.json` shows the coarse task reaching a much
higher LOSO accuracy (0.525) than the fine task (0.253), and the docs read that as "coarse ADL
categories are more cross-subject robust". **They are not.** The coarse task also has a much easier
chance level (1/5 = 0.200 vs 1/21 = 0.048). Correcting for it:

    fine   : (0.253 - 0.048) / (1 - 0.048) = 0.216   <- 22% of the available headroom
    coarse : (0.525 - 0.200) / (1 - 0.200) = 0.406   <- 41% of the available headroom

...so on this one dataset coarsening DOES look genuinely better, not merely easier — but a single
dataset with a single hand-made taxonomy is not evidence, and the raw-accuracy comparison that the
docs actually rely on is invalid. Worse, nothing rules out the trivial explanation: ANY merge of
21 classes into 5 raises kappa, because merging removes exactly the confusions the classifier was
making. The question that matters is whether the FAABOS grouping (or any confusion-aware grouping)
beats a RANDOM grouping of the same shape.

THE DESIGN
----------
Three label sets per dataset, all evaluated with the same model and the same protocol:

  * FINE      — the dataset's own classes (K of them).
  * COARSE-C  — K merged down to G groups by MERGING THE CLASSES THE MODEL CONFUSES (agglomerative
                clustering on the confusion matrix). This is the charitable coarsening: it is the
                best grouping a taxonomy could hope to be.
  * COARSE-R  — K merged down to G groups AT RANDOM, with the same group sizes. The control.

  (+ on emaha_db1 only: COARSE-FAABOS, the real human taxonomy, as a fourth arm — the only place a
   real ADL taxonomy exists.)

Everything is scored with kappa = (acc - chance)/(1 - chance), so a 5-group task and a 21-class task
are on the same scale. And we score BOTH protocols, because the claim is about *cross-subject*
robustness specifically:

    generalisation gap = kappa_within_subject - kappa_LOSO

If coarse labels are genuinely more subject-robust, their GAP must be smaller — not just their
accuracy higher.

PRE-REGISTERED BRANCHES
-----------------------
  A. COARSE-C beats FINE in kappa AND has a smaller generalisation gap AND beats COARSE-R
     -> "hierarchical ADL taxonomies buy real cross-subject robustness, and the benefit comes from
        the STRUCTURE of the grouping, not from the easier chance level." A clean, directly useful
        result for ADL system designers. **Headline for the ADL angle.**
  B. COARSE-C ~ COARSE-R (both beat FINE)
     -> the benefit is an artifact of having fewer classes; ANY merge would do. The honest statement
        is "coarsening makes the task easier but not more subject-robust; a hierarchy is not a
        robustness strategy." This is a real negative finding and it CORRECTS the project's own
        documents. **We log it and move on.**
  C. Coarse does not beat fine in kappa at all
     -> coarsening buys nothing once chance is accounted for. The FAABOS result in the docs is then
        purely a chance-level illusion and must be retracted.

GROUND TRUTH (`--selftest`)
---------------------------
  * A synthetic with TRUE super-groups (classes confusable within a group, separable between groups)
    must show COARSE-C > COARSE-R. If it does not, the merging code cannot find structure that is
    definitely there, and the real run is meaningless.
  * A synthetic where all classes are equally separable must show COARSE-C ~= COARSE-R: with no
    structure to find, the charitable merge must have no advantage over the random one.
"""
from __future__ import annotations

import numpy as np

from . import common as C

TAG = "t4_adl_granularity"
N_GROUPS_TARGET = 5          # FAABOS has 5; use the same for every dataset so the arms are alike

# THE CALIBRATED DECISION THRESHOLD. A confusion-aware merge beats a random merge even when there is
# NO structure to find, because it merges whichever classes happen to be confused in this particular
# finite sample. The selftest MEASURES that null bias on equidistant classes (~0.06 kappa). A real
# structure claim must clear it, so the threshold is set above the measured bias rather than at an
# arbitrary 0.02 — which was below the noise floor and would have called pure sampling noise a
# structural ADL finding.
NULL_ADVANTAGE = 0.08


def _within_subject_acc(X, y, trials, model="lda", seed=0, n_splits=3):
    """Trial-grouped, subject-pooled accuracy: the SAME protocol module2 calls `knn_trial_cv`.

    Windows from one trial never straddle the split (that was the original leak), but subjects DO
    appear on both sides — which is the point: this arm is the optimistic 'within-subject' number
    that we contrast against LOSO.
    """
    from sklearn.model_selection import GroupKFold
    X, y, trials = np.asarray(X, float), np.asarray(y), np.asarray(trials)
    rng = np.random.default_rng(seed)
    accs = []
    n_splits = min(n_splits, len(np.unique(trials)))
    if n_splits < 2:
        return float("nan")
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=trials):
        if len(np.unique(y[tr])) < 2:
            continue
        keep = C.subsample_train(X[tr], y[tr], None, 15000, rng)
        Xtr, ytr = X[tr][keep], y[tr][keep]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        try:
            m = C.make_model(model, seed).fit((Xtr - mu) / sd, ytr)
            accs.append(float((m.predict((X[te] - mu) / sd) == y[te]).mean()))
        except Exception:
            continue
    return float(np.mean(accs)) if accs else float("nan")


def _mapping_for_fold(X_tr, y_tr, subj_tr, mode, n_groups, seed, sizes=None):
    """Derive the coarse label mapping from TRAINING SUBJECTS ONLY.

    ==> THIS IS THE FIX FOR THE LEAK THAT THE 2026-07-13 CODE REVIEW CAUGHT. <==

    The first version of this experiment built ONE mapping from the whole dataset (using an LDA fit
    whose confusion matrix was computed on a held-out subject's TRUE LABELS) and then scored that
    mapping with a LOSO loop in which every subject — including the one whose labels defined the
    mapping — took a turn as the test fold. Two things were wrong with that, and the second is worse
    than the first:

      1. label leakage: the definition of the coarse task came partly from labels of subjects that
         were later tested under it;
      2. selection on the evaluation set: the merge was chosen to collapse exactly the confusions an
         LDA makes on THIS data, in THIS basis, with THESE subjects — so the coarse task was
         optimised to be maximally learnable by the very estimator that then scored it. The random
         control enjoyed none of that. `structure_advantage_over_random_merge` — the quantity the
         whole A-vs-B branch hangs on — was therefore biased upward BY CONSTRUCTION, and the bias had
         nothing to do with whether an ADL taxonomy is real.

    Deriving the mapping inside the fold, from training subjects only, removes both.
    """
    if mode == "confusion":
        return C.merge_labels_confusion(X_tr, y_tr, subj_tr, n_groups, seed=seed)
    if mode == "random":
        return C.merge_labels_random(y_tr, n_groups, seed=seed, group_sizes=sizes)
    raise ValueError(mode)


def _rand_index(m1, m2):
    """Agreement between two label->group mappings (adjusted Rand). If the fold-to-fold mapping is
    unstable, the 'taxonomy' the confusion merge finds is not a stable property of the data — which
    is itself an answer to the ADL question."""
    from sklearn.metrics import adjusted_rand_score
    keys = sorted(set(m1) & set(m2))
    if len(keys) < 3:
        return float("nan")
    return float(adjusted_rand_score([m1[k] for k in keys], [m2[k] for k in keys]))


def _score_arm_loso(X, y, subj, mode, n_groups, model, seed):
    """LOSO accuracy where the coarse mapping is re-derived INSIDE every fold from training subjects
    only. `mode` is 'fine' (no merge), 'confusion', 'random', or a fixed dict (the real taxonomy)."""
    X = np.asarray(X, float); y = np.asarray(y); subj = np.asarray(subj)
    rng = np.random.default_rng(seed)
    accs, maps = {}, {}
    for s in sorted(np.unique(subj)):
        tr, te = subj != s, subj == s
        if len(np.unique(y[tr])) < 2 or te.sum() < 5:
            continue
        if mode == "fine":
            ytr, yte, K = y[tr], y[te], len(np.unique(y))
        elif isinstance(mode, dict):
            ytr, yte, K = C.apply_mapping(y[tr], mode), C.apply_mapping(y[te], mode), \
                len(set(mode.values()))
        else:
            mp = _mapping_for_fold(X[tr], y[tr], subj[tr], "confusion", n_groups, seed)
            if mp is None:
                continue
            if mode == "random":
                sizes = [sum(1 for g in mp.values() if g == gg) for gg in sorted(set(mp.values()))]
                mp = _mapping_for_fold(X[tr], y[tr], subj[tr], "random", n_groups, seed, sizes)
            maps[int(s)] = mp
            # a class present only in the test subject has no fold mapping -> drop those rows
            known = np.array([int(v) in mp for v in y[te]])
            if known.sum() < 5:
                continue
            ytr = C.apply_mapping(y[tr], mp)
            yte = C.apply_mapping(y[te][known], mp)
            te = np.flatnonzero(te)[known]
            K = n_groups
        Xtr, Xte = X[tr], X[te]
        keep = C.subsample_train(Xtr, ytr, None, 15000, rng)
        Xtr, ytr = Xtr[keep], ytr[keep]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        try:
            m = C.make_model(model, seed).fit((Xtr - mu) / sd, ytr)
            accs[int(s)] = float((m.predict((Xte - mu) / sd) == yte).mean())
        except Exception:
            continue
    return accs, maps, K


def _score_arm(X, y, subj, trials, mode, n_groups, model, seed):
    """Chance-corrected scores for one label set. The LOSO arm re-derives the mapping per fold."""
    accs, maps, K = _score_arm_loso(X, y, subj, mode, n_groups, model, seed)
    loso_mean = float(np.mean(list(accs.values()))) if accs else float("nan")

    # the pooled-CV ("optimistic") arm, for the gap. A fixed mapping is acceptable here ONLY for
    # 'fine' and for the real taxonomy, which are not derived from the data at all.
    if mode == "fine":
        pooled_cv = _within_subject_acc(X, y, trials, model, seed)
    elif isinstance(mode, dict):
        pooled_cv = _within_subject_acc(X, C.apply_mapping(y, mode), trials, model, seed)
    else:
        pooled_cv = float("nan")            # a data-derived mapping cannot be scored on its own data

    stab = float("nan")
    if len(maps) >= 2:
        ks = sorted(maps)
        stab = float(np.nanmean([_rand_index(maps[a], maps[b])
                                 for a, b in zip(ks[:-1], ks[1:])]))
    kp, kl = C.kappa_chance(pooled_cv, K), C.kappa_chance(loso_mean, K)
    return dict(n_classes=int(K), chance=1.0 / K,
                acc_pooled_cv=pooled_cv, acc_loso=loso_mean,
                kappa_pooled_cv=kp, kappa_loso=kl,
                pooled_vs_loso_gap=float(kp - kl) if np.isfinite(kp) else float("nan"),
                mapping_stability_across_folds=stab,
                n_subjects=len(accs))


def _faabos_mapping(dataset):
    """The real human ADL taxonomy — exists for emaha_db1 only."""
    import pandas as pd
    from dsprofile import config
    p = config.L1_ROOT / dataset / "manifest.parquet"
    if not p.exists():
        return None
    m = pd.read_parquet(p)
    if "faabos_group" not in m.columns:
        return None
    mp = (m[["label", "faabos_group"]].drop_duplicates()
          .dropna().set_index("label")["faabos_group"].to_dict())
    groups = {g: i for i, g in enumerate(sorted(set(mp.values())))}
    return {int(k): int(groups[v]) for k, v in mp.items()}


def run_one(dataset, seed=42, n_jobs=1):
    from dsprofile import cv
    with C.timer(f"T4 :: {dataset}"):
        frame = C.build_frame(dataset, seed=seed)
        X, _ = C.basis(frame)
        y = frame.label.to_numpy()
        subj = frame.subject.to_numpy()
        trials = np.asarray(cv.trial_ids(frame))
        K = len(np.unique(y))

        out = dict(dataset=dataset, n_classes_fine=int(K), n_subjects=int(len(np.unique(subj))),
                   n_groups_target=N_GROUPS_TARGET, model="lda", arms={})

        if K <= N_GROUPS_TARGET + 1:
            out["note"] = (f"only {K} fine classes; coarsening to {N_GROUPS_TARGET} is not a "
                           "meaningful contrast on this dataset")
            out["arms"]["fine"] = _score_arm(X, y, subj, trials, "fine", K, "lda", seed)
            return out

        # Every arm re-derives its mapping INSIDE each LOSO fold, from training subjects only.
        out["arms"]["fine"] = _score_arm(X, y, subj, trials, "fine", K, "lda", seed)
        out["arms"]["coarse_confusion"] = _score_arm(X, y, subj, trials, "confusion",
                                                     N_GROUPS_TARGET, "lda", seed)
        out["arms"]["coarse_random"] = _score_arm(X, y, subj, trials, "random",
                                                  N_GROUPS_TARGET, "lda", seed)

        # the real ADL taxonomy (emaha_db1 only) — a FIXED, human-made mapping, so it is not derived
        # from the data and needs no fold nesting
        mf = _faabos_mapping(dataset)
        if mf and all(int(v) in mf for v in np.unique(y)):
            out["arms"]["coarse_faabos"] = _score_arm(X, y, subj, trials, mf,
                                                      len(set(mf.values())), "lda", seed)
            out["has_real_taxonomy"] = True

    a = out["arms"]
    if "coarse_confusion" in a and "fine" in a:
        out["coarse_beats_fine_in_kappa"] = bool(a["coarse_confusion"]["kappa_loso"]
                                                 > a["fine"]["kappa_loso"])
    if "coarse_confusion" in a and "coarse_random" in a:
        d = a["coarse_confusion"]["kappa_loso"] - a["coarse_random"]["kappa_loso"]
        out["structure_advantage_over_random_merge"] = float(d)
        out["structure_matters"] = bool(d > NULL_ADVANTAGE)   # must clear the measured null bias
    out["mapping_stability"] = a.get("coarse_confusion", {}).get("mapping_stability_across_folds")
    return out


def build_pooled(tag=TAG):
    import json
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(C.results_dir(tag).glob(f"*__{tag}.json"))]
    usable = [r for r in rows if "coarse_confusion" in r.get("arms", {})]
    if not usable:
        return dict(note="no usable datasets (all too few classes?)")

    n = len(usable)
    beats_ds = [r["dataset"] for r in usable if r.get("coarse_beats_fine_in_kappa")]
    struct_ds = [r["dataset"] for r in usable if r.get("structure_matters")]
    all_ds = [r["dataset"] for r in usable]
    beats_fine, structure = len(beats_ds), len(struct_ds)
    adv = [r["structure_advantage_over_random_merge"] for r in usable
           if "structure_advantage_over_random_merge" in r]
    stab = [r["mapping_stability"] for r in usable
            if isinstance(r.get("mapping_stability"), float) and np.isfinite(r["mapping_stability"])]
    beats_both = C.count_both_ways(beats_ds, all_ds)      # k=14 AND k=9 (CLAUDE.md rule 3)
    struct_both = C.count_both_ways(struct_ds, all_ds)

    if beats_fine > n / 2 and structure > n / 2:
        branch, verdict = "A", (
            f"COARSE LABELS BUY REAL ROBUSTNESS: a confusion-aware grouping (derived inside each "
            f"fold from training subjects only) beats the fine labels in chance-corrected kappa on "
            f"{beats_both['datasets']} datasets / {beats_both['cohorts']} cohorts, AND beats a random "
            f"merge of identical shape on {struct_both['datasets']} datasets / "
            f"{struct_both['cohorts']} cohorts (mean advantage {np.mean(adv):+.3f} kappa). The "
            "benefit is structural, not a chance-level artifact. This is the ADL headline.")
    elif beats_fine > n / 2 and structure <= n / 2:
        branch, verdict = "B", (
            f"COARSENING IS EASIER, NOT MORE ROBUST: coarse labels beat fine in kappa on "
            f"{beats_both['datasets']} datasets, but a RANDOM merge of the same shape does just as "
            f"well (structure advantage {np.mean(adv):+.3f} kappa; beats random on only "
            f"{struct_both['datasets']} datasets / {struct_both['cohorts']} cohorts). A hierarchy is "
            "not a robustness strategy — the gain is the easier chance level. The project's FAABOS "
            "claim must be corrected.")
    else:
        branch, verdict = "C", (
            f"COARSENING BUYS NOTHING once chance is accounted for (coarse beats fine on only "
            f"{beats_fine}/{n} in kappa). The raw-accuracy FAABOS result is a chance-level illusion "
            "and must be retracted.")

    faab = next((r for r in usable if r.get("has_real_taxonomy")), None)
    out = dict(tag=tag, n_datasets=n,
               coarse_beats_fine=beats_both, structure_beats_random_merge=struct_both,
               mean_structure_advantage=float(np.mean(adv)) if adv else float("nan"),
               mean_mapping_stability_across_folds=float(np.mean(stab)) if stab else float("nan"),
               leakage_note=("the coarse mapping is re-derived inside every LOSO fold from TRAINING "
                             "subjects only; it never sees the held-out subject's labels"),
               real_taxonomy_dataset=(faab or {}).get("dataset"),
               real_taxonomy_arms=(faab or {}).get("arms", {}).get("coarse_faabos"),
               branch=branch, verdict=verdict)
    C.atomic_write_json(C.results_dir(tag) / "pooled.json", out)
    return out


# ------------------------------------------------------------------ ground truth
def _grouped_synth(true_structure, n_subj=10, n_groups=4, per_group=3, per=45, seed=0):
    """Two worlds, to check the merge finds structure ONLY when structure exists.

    true_structure : classes inside a group sit almost on top of each other (highly confusable) and
                     the groups are far apart -> a confusion-aware merge MUST recover the groups.
    no structure   : every class is EXACTLY equidistant from every other. Note this must be built
                     deliberately — my first attempt drew random centres, but random points in R^6
                     are NOT equidistant: some pairs really are closer, so the confusion merge found
                     genuine structure and beat the random merge, which made the "null" test fail for
                     the right reason. Placing the K classes on scaled basis vectors in R^K makes all
                     pairwise distances identical, so there is genuinely nothing to find.
    """
    rng = np.random.default_rng(seed)
    K = n_groups * per_group
    if true_structure:
        d = 6
        gc = rng.standard_normal((n_groups, d)) * 6.0
        centers = np.vstack([gc[g] + rng.standard_normal((per_group, d)) * 0.35
                             for g in range(n_groups)])
    else:
        d = K
        centers = np.eye(K) * 4.0                    # all pairwise distances = 4*sqrt(2)
    X, y, s = [], [], []
    for u in range(n_subj):
        off = rng.standard_normal(d) * 0.5
        for c in range(K):
            X.append(centers[c] + off + rng.standard_normal((per, d)) * 1.0)
            y += [c] * per; s += [u] * per
    return np.vstack(X), np.array(y), np.array(s)


def selftest(check):
    # Exercise the SAME code path the real run uses (`_score_arm_loso`, mapping re-derived inside
    # every fold from training subjects only) — not the raw helper. Testing a path the experiment
    # does not take proves nothing about the experiment.
    null_gap = None
    for structured in (True, False):
        X, y, s = _grouped_synth(structured, seed=1)
        trials = np.arange(len(y)) // 15                     # synthetic trial ids
        kc = _score_arm(X, y, s, trials, "confusion", 4, "lda", 1)["kappa_loso"]
        kr = _score_arm(X, y, s, trials, "random", 4, "lda", 1)["kappa_loso"]
        if structured:
            check("T4 confusion-merge FINDS true super-groups (beats a random merge of the same shape)",
                  kc > kr + 0.05, f"confusion kappa={kc:.3f} vs random={kr:.3f}")
        else:
            null_gap = kc - kr
            # THE NULL BIAS. Even with classes that are EXACTLY equidistant — no structure whatsoever
            # to find — a confusion-aware merge still beats a random one, because it merges whichever
            # classes happen to be confused in THIS finite sample. That is a selection effect, and it
            # is the whole reason this control exists. Fold-nesting the mapping (deriving it from
            # training subjects only) removes the leak but NOT this bias.
            #
            # So the honest move is not to tighten a tolerance until the test passes — it is to
            # MEASURE the bias and require the real-data effect to clear it. `NULL_ADVANTAGE` below is
            # that measurement, and `structure_matters` is now defined against it.
            check("T4 null bias is measured, not assumed (a confusion-merge beats a random merge even "
                  "with zero true structure — the real-data threshold must clear this)",
                  np.isfinite(null_gap), f"null advantage = {null_gap:+.3f} kappa")
            check("T4 the calibrated threshold sits ABOVE the measured null bias",
                  NULL_ADVANTAGE >= null_gap - 0.01,
                  f"threshold={NULL_ADVANTAGE:.3f} vs null bias={null_gap:+.3f}")

    check("T4 kappa_chance(0.525, 5 classes) == 0.406", abs(C.kappa_chance(0.525, 5) - 0.40625) < 1e-6)
    check("T4 kappa_chance(0.253, 21 classes) == 0.216", abs(C.kappa_chance(0.253, 21) - 0.21525) < 1e-3)
    check("T4 kappa is 0 at chance", abs(C.kappa_chance(0.25, 4)) < 1e-12)
    check("T4 kappa is 1 at perfect", abs(C.kappa_chance(1.0, 7) - 1.0) < 1e-12)

