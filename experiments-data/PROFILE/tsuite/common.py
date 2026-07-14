"""T-suite shared layer — everything the T experiments need that the X-suite did not already have.

DESIGN RULE (see CLAUDE.md): this module adds NOTHING that `paper_experiments/common.py` already
implements. MMD, the median-bandwidth heuristic, CORAL, per-subject centring, LOSO accuracy, the
statistics (Pearson/Spearman/FDR/random-effects pooling/cluster bootstrap/permutation), the atomic
dataset runner and the synthetic frame generator are all imported from there, unchanged, so a number
computed by a T experiment and a number computed by an X experiment come from the SAME code.

What is genuinely new here, and why:

  1. MODEL ZOO (`fit_predict`, `MODELS`) — the X-suite could only fit LDA, LinearSVC and a small RF.
     T1 needs a *non-linear* family (RBF-SVM, MLP, gradient boosting) to answer "does the cheap
     statistic predict difficulty for ANY learner, or only for a linear one?".

  2. FAIR-COST PROTOCOL (`subsample_train`) — an RBF-SVM is O(n^2); ninapro_db2 has 360k windows.
     Every model family in a comparison is given the SAME stratified subsample, so a difference
     between families is a difference between families, not between budgets.

  3. CHANCE CORRECTION (`kappa_chance`) — comparing a 21-class task with a 5-class task by raw
     accuracy is meaningless: the 5-class task starts at 0.20 and the 21-class task at 0.048.
     kappa = (acc - chance) / (1 - chance) is the fraction of the *available* headroom that was
     captured, and it is what T4 needs to test the ADL-granularity claim honestly.

  4. LABEL COARSENING (`merge_labels_confusion`, `merge_labels_random`) — T4 needs coarse class
     groupings for the 13 datasets that have no FAABOS column, plus a RANDOM-merge control so that
     "coarse is easier" cannot be mistaken for "coarse is more subject-robust".

  5. IMBALANCE INDUCTION (`induce_imbalance`) — X13 could not test whether centring hurts imbalanced
     subjects because 8/14 datasets are perfectly balanced. Rather than report "untestable", T6
     MAKES the imbalance at a controlled ratio and measures the effect.
"""
from __future__ import annotations

import numpy as np

# The X-suite library is the single source of mathematical truth. Do not re-implement anything here
# that already exists there.
from paper_experiments import common as X
from paper_experiments.common import (                      # noqa: F401  (re-exported on purpose)
    atomic_write_json, basis, build_frame, clip_corr, cluster_bootstrap, coral,
    corr_across_subjects, fdr_bh, log, maybe_parallel, mmd_to_pool, pearson, per_subject_center,
    per_subject_zscore, permutation_corr, pool_random_effects, results_dir, run_over_datasets,
    spearman, synth_frame, timer, zscore,
)

EPS = 1e-9


# =====================================================================================
# 0. Honest counting: cohorts, and multiplicity
# =====================================================================================
# The 14 datasets are only NINE independent cohorts (CLAUDE.md §9): the four EMAHA sets are one
# cohort, grabmyo_flow_static/_dynamic are one, ninapro_db4/db5 are one. A headline of "helps on
# 12/14 datasets" that is really "helps in 7/9 cohorts" is a materially weaker claim, and reporting
# only the first overstates the evidence. Every pooled verdict in the T-suite must report both.
def cohort_of(dataset):
    from dsprofile import config
    return config.COHORTS.get(dataset, dataset)


def count_both_ways(datasets_that_qualify, all_datasets):
    """-> dict(datasets='12/14', cohorts='7/9', ...). The number you are allowed to quote is BOTH."""
    dq, da = list(datasets_that_qualify), list(all_datasets)
    cq = {cohort_of(d) for d in dq}
    ca = {cohort_of(d) for d in da}
    return dict(n_datasets=len(dq), n_datasets_total=len(da),
                n_cohorts=len(cq), n_cohorts_total=len(ca),
                datasets=f"{len(dq)}/{len(da)}", cohorts=f"{len(cq)}/{len(ca)}",
                cohorts_qualifying=sorted(cq))


def fdr_q(pvals):
    """q-values only. `paper_experiments.common.fdr_bh` returns a TUPLE (rejected, q) — unpacking it
    wrongly is a real bug that reached the first draft of T1, so this wrapper exists to make the
    mistake impossible."""
    pvals = np.asarray(pvals, float)
    if pvals.size == 0:
        return np.array([])
    _rejected, q = X.fdr_bh(pvals)
    return np.asarray(q, float)


def helps_flags_fdr(pvals, deltas, alpha=0.05):
    """A 'helps' flag per test, AFTER Benjamini-Hochberg correction within the family of tests.

    Why: T3 runs 6 rungs x 14 datasets = 84 one-sided Wilcoxon tests and T6 runs 5 ratios x 14 = 70.
    Counting raw p<0.05 flags across a family that size manufactures ~4 false positives by
    construction, and the pooled counts ("helps on 12/14") are built from exactly those flags.
    """
    q = fdr_q(pvals)
    return [bool(d > 0 and qq < alpha) for d, qq in zip(deltas, q)], q


# =====================================================================================
# 1. Model zoo
# =====================================================================================
# Fixed hyper-parameters, chosen once and never tuned per dataset: any per-dataset tuning would
# make the cross-dataset comparison meaningless (and would tune on the test subject).
MODELS = ("lda", "svm_rbf", "rf", "mlp", "hgb")

MODEL_IS_LINEAR = {"lda": True, "svm_rbf": False, "rf": False, "mlp": False, "hgb": False}


def make_model(name, seed=0):
    """Build one classifier. Linear (lda) + four non-linear families."""
    if name == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis()
    if name == "svm_rbf":
        from sklearn.svm import SVC
        # gamma='scale' = 1/(d * var(X)); with train-only standardisation var~1 so gamma ~ 1/d.
        return SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True,
                             n_iter_no_change=10, random_state=seed)
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    raise ValueError(f"unknown model: {name}")


def subsample_train(X_tr, y_tr, subj_tr, cap_rows, rng):
    """Stratified subsample of the training rows to at most `cap_rows`, preserving class balance.

    Applied IDENTICALLY to every model family so the comparison is fair. Returns the kept indices.
    """
    n = len(y_tr)
    if cap_rows is None or n <= cap_rows:
        return np.arange(n)
    classes, counts = np.unique(y_tr, return_counts=True)
    # proportional allocation, at least 2 rows per class so every class survives
    quota = np.maximum(2, np.floor(cap_rows * counts / n).astype(int))
    keep = []
    for c, q in zip(classes, quota):
        idx = np.flatnonzero(y_tr == c)
        keep.append(idx if len(idx) <= q else rng.choice(idx, q, replace=False))
    return np.sort(np.concatenate(keep))


def loso_accuracy_model(X_, y, subjects, model="lda", seed=0, cap_train=15000, cap_test=5000,
                        min_test=5):
    """Per-subject LOSO accuracy for one model family. Returns {subject: accuracy}.

    Leakage discipline (identical to `paper_experiments.common.loso_accuracy`):
      * the held-out subject contributes NOTHING to the fit — not even its standardisation statistics;
      * standardisation uses TRAIN-ONLY mean/sd;
      * the train subsample is drawn before the fit and never touches the test subject.
    """
    X_ = np.asarray(X_, float)
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    rng = np.random.default_rng(seed)
    out = {}
    for s in sorted(np.unique(subjects)):
        tr, te = subjects != s, subjects == s
        if len(np.unique(y[tr])) < 2 or te.sum() < min_test:
            continue
        Xtr, ytr, Xte, yte = X_[tr], y[tr], X_[te], y[te]
        keep = subsample_train(Xtr, ytr, subjects[tr], cap_train, rng)
        Xtr, ytr = Xtr[keep], ytr[keep]
        if len(Xte) > cap_test:
            sel = rng.choice(len(Xte), cap_test, replace=False)
            Xte, yte = Xte[sel], yte[sel]
        mu, sd = Xtr.mean(0), Xtr.std(0) + EPS          # TRAIN-only statistics
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        try:
            m = make_model(model, seed).fit(Xtr, ytr)
            out[int(s)] = float((m.predict(Xte) == yte).mean())
        except Exception as e:                          # a family that cannot fit is recorded, not hidden
            log(f"    [model-fail] {model} subject={s}: {type(e).__name__}: {e}")
            continue
    return out


# =====================================================================================
# 2. Chance-corrected accuracy
# =====================================================================================
def kappa_chance(acc, n_classes):
    """Fraction of the ABOVE-CHANCE headroom that was captured: (acc - c) / (1 - c), c = 1/K.

    Why this exists: raw accuracy cannot compare tasks with different class counts. A 5-class model
    at 0.52 (chance 0.20) captured 40% of its headroom; a 21-class model at 0.25 (chance 0.048)
    captured 21%. Ranges (-c/(1-c)) .. 1; 0 = chance, 1 = perfect, negative = worse than chance.
    """
    c = 1.0 / float(n_classes)
    return (float(acc) - c) / (1.0 - c)


def chance_level(y):
    return 1.0 / float(len(np.unique(y)))


# =====================================================================================
# 3. Label coarsening (T4)
# =====================================================================================
def merge_labels_confusion(X_, y, subjects, n_groups, seed=0, cap_train=8000):
    """Coarsen the label set into `n_groups` by merging the classes a model CONFUSES.

    Method: fit one within-subject-pooled model under a subject-grouped split, build the confusion
    matrix, symmetrise it into a similarity S = (C + C.T)/2, and agglomeratively merge the most
    confusable classes (average linkage on the distance 1 - S/S.max()).

    This is the *charitable* coarsening — it gives the "coarse labels are better" hypothesis the
    best possible chance, because it merges exactly the classes the model cannot tell apart.
    Compare against `merge_labels_random` (same group sizes, arbitrary members) to separate
    "coarse is genuinely more robust" from "coarse is merely easier".
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import confusion_matrix

    y = np.asarray(y)
    classes = np.unique(y)
    if n_groups >= len(classes) or n_groups < 2:
        return None
    rng = np.random.default_rng(seed)
    # a cheap, leakage-safe confusion estimate: hold out one subject, fit on the rest
    subs = np.unique(subjects)
    held = subs[0]
    tr, te = subjects != held, subjects == held
    keep = subsample_train(X_[tr], y[tr], subjects[tr], cap_train, rng)
    mu, sd = X_[tr][keep].mean(0), X_[tr][keep].std(0) + EPS
    m = make_model("lda", seed).fit((X_[tr][keep] - mu) / sd, y[tr][keep])
    pred = m.predict((X_[te] - mu) / sd)
    C = confusion_matrix(y[te], pred, labels=classes).astype(float)
    C = C / np.maximum(C.sum(1, keepdims=True), 1.0)     # row-normalise: P(pred | true)
    S = (C + C.T) / 2.0                                  # symmetric confusability
    np.fill_diagonal(S, S.max() if S.max() > 0 else 1.0)  # a class is maximally "like" itself
    D = 1.0 - S / (S.max() + EPS)
    np.fill_diagonal(D, 0.0)
    lab = AgglomerativeClustering(n_clusters=n_groups, metric="precomputed",
                                  linkage="average").fit_predict(D)
    return {int(c): int(g) for c, g in zip(classes, lab)}


def merge_labels_random(y, n_groups, seed=0, group_sizes=None):
    """Control for `merge_labels_confusion`: same number of groups (and optionally the same group
    SIZES), but membership assigned at random. Any advantage the confusion-merge has over this is
    the part that is about structure rather than about the easier chance level."""
    rng = np.random.default_rng(seed)
    classes = np.unique(np.asarray(y))
    if n_groups >= len(classes) or n_groups < 2:
        return None
    perm = rng.permutation(classes)
    if group_sizes is None:
        assign = np.array_split(perm, n_groups)
    else:
        assign, i = [], 0
        for sz in group_sizes:
            assign.append(perm[i:i + sz]); i += sz
    return {int(c): int(g) for g, members in enumerate(assign) for c in members}


def apply_mapping(y, mapping):
    return np.array([mapping[int(v)] for v in np.asarray(y)])


# =====================================================================================
# 4. Imbalance induction (T6)
# =====================================================================================
def induce_imbalance(y, subjects, ratio, seed=0):
    """Return a row mask that makes EACH SUBJECT's class distribution imbalanced by `ratio`.

    For every subject we draw a geometric-ish profile over its classes: the most frequent class
    keeps all its rows, the least frequent keeps 1/ratio of them, with the rest log-spaced in
    between. `ratio=1` returns everything (the balanced control). The class ORDER is permuted per
    subject, so the imbalance is a property of the subject, not of the class id — which is exactly
    the situation exp_B's caveat worried about (a subject whose class-biased mean is a bad estimate
    of its true mean).
    """
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    rng = np.random.default_rng(seed)
    if ratio <= 1.0:
        return np.ones(len(y), bool)
    mask = np.zeros(len(y), bool)
    for s in np.unique(subjects):
        rows = np.flatnonzero(subjects == s)
        classes = rng.permutation(np.unique(y[rows]))
        fracs = np.geomspace(1.0, 1.0 / ratio, len(classes))
        for c, f in zip(classes, fracs):
            idx = rows[y[rows] == c]
            k = max(2, int(round(len(idx) * f)))
            mask[rng.choice(idx, min(k, len(idx)), replace=False)] = True
    return mask


def imbalance_ratio(y, subjects):
    """Observed max/min class count per subject (median across subjects). 1.0 = perfectly balanced."""
    y, subjects = np.asarray(y), np.asarray(subjects)
    rs = []
    for s in np.unique(subjects):
        _, cnt = np.unique(y[subjects == s], return_counts=True)
        if len(cnt) > 1 and cnt.min() > 0:
            rs.append(cnt.max() / cnt.min())
    return float(np.median(rs)) if rs else float("nan")
