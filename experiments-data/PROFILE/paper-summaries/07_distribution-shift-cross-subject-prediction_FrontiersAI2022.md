# Paper Summary — Estimating Distribution Shifts to Predict Cross-Subject Generalization

| | |
|---|---|
| **File** | `data-science-related-papers/frai-05-992732.pdf` |
| **Citation** | Albuquerque I, Monteiro J, Rosanne O, Falk T H, *"Estimating distribution shifts for predicting cross-subject generalization in EEG-based mental workload assessment"*, **Frontiers in Artificial Intelligence** 5:992732, 2022 (open access). |
| **Maps to** | **Module 5 (THE method backbone)** + Module 3. Answers scientific question A9 (subject-difficulty predictor). |

---

## 1. Why this is the single most important method paper for Paper 2
Module 5's headline — "predict who a classifier fails on, before training, from cheap statistics" — is *this
paper's exact method*, and it was demonstrated on **only one EEG dataset (18 subjects)**. Extending it across
**6+ sEMG ADL datasets** is a concrete, defensible novelty for us. It fully specifies two estimators that are
CPU-cheap and implementable in scikit-learn, and it empirically links one of them to the LOSO generalization gap.

## 2. The two estimators we implement (exact recipe)
**(a) Conditional shift — disparity matrix D (labeling-function mismatch across subjects).** For each ordered
pair (i, j): train a **kNN classifier** `f̃_j` on subject j; compute disagreement `μ_ij = (1/N)Σ 1[f_i(x) ≠ f̃_j(x)]`
on subject i's data; set `d_ij = min(μ_ij, μ_ji)`. Assemble symmetric **M×M matrix D** (diagonal = within-subject
disparity via disjoint train/test). Aggregate = **rescaled Frobenius norm ‖D‖_F ∈ [0,1]** = the dataset's
conditional-shift scalar.

**(b) Marginal shift — H-divergence.** `d_H(D_S, D_T) = 2(1 − 2ε)`, where ε = error of a **binary classifier
(Random Forest, 20 trees, 5-fold CV)** trained to distinguish subject-i vs subject-j samples. Build **Hermitian
matrix H** of pairwise errors; aggregate via rescaled Frobenius norm. Higher marginal shift = subjects more
distinguishable = higher cross-subject variability.

**Protocol:** 30 repetitions, ~300 points/subject randomly sampled; RandomForest 30 trees for the LOSO
workload classifier; scikit-learn, fixed seed. **Generalization gap = |train_acc − test_acc| under LOSO.**

## 3. Key results we lean on
- **z-score (whitening) normalization reduced conditional shift AND gave the best LOSO accuracy (~70%)** with
  *no* calibration data — independent confirmation of our `Normalizer(mode="global")` choice.
- **Conditional shift correlates with the LOSO generalization gap** across subjects → the estimator predicts
  difficulty. (They explicitly call for multi-dataset validation — which is our Paper 2.)
- Marginal shift and conditional shift capture *different* things (a subject can be marginally far but
  conditionally close) → report both.

## 4. Methodology we adopt / caveats
- Implement `disparity_matrix()` (kNN conditional shift) and `h_divergence_matrix()` (RF marginal shift) as
  Module-3/5 core functions; regress each subject's Paper-1 LOSO accuracy on their row-aggregates.
- **Caveat:** it is EEG/binary-workload; we transplant the *method*, not the numbers, to multi-class sEMG. Use
  our train-only global z-score to avoid leakage. Use enough points/subject that kNN/RF errors are stable.
