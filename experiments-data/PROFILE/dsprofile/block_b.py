"""Block B — class structure.

B1  Per-class difficulty within a dataset: one-vs-rest kNN-LOO accuracy per class -> which classes
    are intrinsically confusable; report the spread and the hardest/easiest classes.
B2  Subject-invariant vs idiosyncratic classes: how much does a class's feature centroid move across
    subjects? Low across-subject centroid dispersion = subject-invariant ("anchor") class.

Scalable + dataset-agnostic; uses the cached fast frame (z-scored features).
"""
from __future__ import annotations

import json

import numpy as np

from . import config, cv, windows
from .module3_shift import _basis


def _recall_from(y_true, y_pred):
    return {int(c): float((y_pred[y_true == c] == c).mean()) for c in np.unique(y_true)}


def per_class_difficulty(frame, seed=0, max_n=6000):
    """Per-class recall under BOTH honest protocols.

    Previously this shuffled rows and ran a plain 5-fold, so a window's 50 %-overlapping
    sibling from the same trial sat in the training fold — the class ranking was read off a
    leaked classifier. We now group folds on trial (within-subject) and additionally report
    the subject-disjoint ranking, which is the one an ADL deployment cares about.
    """
    X = _basis(frame); y = frame["label"].to_numpy()
    groups = cv.trial_ids(frame); subjects = frame["subject"].to_numpy()

    yt, yp = cv.knn_predict_trial_cv(X, y, groups, max_n=max_n, seed=seed)
    recall = _recall_from(yt, yp)
    ys, yps = cv.knn_predict_trial_cv(X, y, subjects, max_n=max_n, seed=seed)   # subject-grouped
    recall_loso = _recall_from(ys, yps)

    from scipy.stats import spearmanr
    common = sorted(set(recall) & set(recall_loso))
    rho = float(spearmanr([recall[c] for c in common], [recall_loso[c] for c in common])[0]) \
        if len(common) >= 3 else float("nan")
    vals = np.array(list(recall.values()))
    lv = np.array(list(recall_loso.values()))
    return dict(protocol="GroupKFold on trial (within-subject); *_loso = GroupKFold on subject",
                per_class_recall=recall,
                per_class_recall_loso=recall_loso,
                mean_recall=float(vals.mean()), std_recall=float(vals.std()),
                mean_recall_loso=float(lv.mean()) if lv.size else float("nan"),
                # is the class-difficulty ORDER the same within- and cross-subject?
                rank_stability_trial_vs_loso_spearman=rho,
                hardest_classes=[int(c) for c in sorted(recall, key=lambda k: recall[k])[:5]],
                easiest_classes=[int(c) for c in sorted(recall, key=lambda k: -recall[k])[:5]],
                hardest_classes_loso=[int(c) for c in sorted(recall_loso, key=lambda k: recall_loso[k])[:5]],
                easiest_classes_loso=[int(c) for c in sorted(recall_loso, key=lambda k: -recall_loso[k])[:5]])


def class_subject_invariance(frame):
    """For each class, dispersion of its per-subject centroid (mean pairwise distance between the
    class centroids of different subjects). Low = subject-invariant. Reported normalised by the
    overall between-class centroid spread so it is comparable across datasets."""
    X = _basis(frame); y = frame["label"].to_numpy(); subj = frame["subject"].to_numpy()
    classes = np.unique(y)
    # reference scale: spread of global class centroids
    gcent = np.stack([X[y == c].mean(0) for c in classes])
    ref = float(np.mean([np.linalg.norm(gcent[i] - gcent[j])
                         for i in range(len(classes)) for j in range(i + 1, len(classes))]) + 1e-12)
    disp = {}
    for c in classes:
        cents = [X[(y == c) & (subj == s)].mean(0)
                 for s in np.unique(subj) if ((y == c) & (subj == s)).sum() >= 10]
        if len(cents) < 2:
            continue
        cents = np.stack(cents)
        d = [np.linalg.norm(cents[i] - cents[j])
             for i in range(len(cents)) for j in range(i + 1, len(cents))]
        disp[int(c)] = float(np.mean(d) / ref)
    return dict(class_subject_dispersion=disp,
                most_invariant=[int(c) for c in sorted(disp, key=lambda k: disp[k])[:5]],
                most_idiosyncratic=[int(c) for c in sorted(disp, key=lambda k: -disp[k])[:5]])


def run(dataset, seed=42):
    config.ensure_dirs()
    outdir = config.RESULTS_DIR / "block_b"
    outdir.mkdir(parents=True, exist_ok=True)
    frame = windows.build_fast_frame(dataset, seed=seed)
    result = dict(dataset=dataset,
                  B1_per_class_difficulty=per_class_difficulty(frame, seed),
                  B2_class_subject_invariance=class_subject_invariance(frame))
    (outdir / f"{dataset}__block_b.json").write_text(json.dumps(result, indent=2))
    return result
