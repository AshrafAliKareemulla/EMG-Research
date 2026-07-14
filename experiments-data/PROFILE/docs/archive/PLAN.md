# PROFILE — Data-Science Paper (Paper 2) — PLAN (code NOT built yet)

> **Status: PLAN ONLY.** No code exists in this folder yet. The user will supply 7 PDFs
> (see §5) next session; fold their exact parameter recipes into the module specs, THEN
> get an explicit go-ahead before writing code. This is the companion paper to
> `experiments-dl/BENCH-LOSO/` (Paper 1); the two share loaders + Track-1 features only.

## 0. What this paper is (and is NOT)
**Thesis:** *A statistical & mathematical characterization of sEMG ADL datasets — what the
data itself tells us before any model is trained, and which properties predict cross-subject
difficulty.*

- **No accuracy axis** → it cannot "look bad" the way a LOSO number can. Contribution is
  understanding + a predictive tool, not SOTA.
- **CPU-only, runs in PARALLEL with Paper 1** (zero extra GPU wall-clock). This is the
  deliberate risk hedge: even if Paper-1's Phase-2 gains are modest, Paper 2 still stands.
- Built on the existing L1 loaders (`semg/`) + Track-1 handcrafted features. No new data.
- Only link to Paper 1: Module 5 consumes Paper-1's per-subject LOSO accuracies as its
  prediction target.

## 1. The five modules (cross-validated against multiple papers — see §4)

**Module 1 — Signal-level characterization** (per subject / channel / class)
- Amplitude/energy: RMS, MAV, variance, dynamic range; active-vs-rest RMS ratio (generalize
  the rest-bug computation into an SNR proxy).
- Spectral: mean/median frequency, spectral entropy, bandwidth, power-band ratios.
- Complexity / non-stationarity: sample entropy, permutation entropy, fuzzy entropy, Higuchi
  fractal dimension, multiscale entropy. (These exist BECAUSE sEMG is nonlinear/non-stationary;
  they capture what linear features miss. Fatigue lowers complexity — a known validation.)

**Module 2 — Class structure & separability** (per dataset)
- Cheap filters: Fisher discriminant ratio, Davies-Bouldin index, silhouette, kNN-LOO.
- ★ Revision from lit review: the comparative sEMG study (PMC11175337) found **wrapper /
  MI methods (RFE, feature-importance, mutual information) beat filter separability indices**,
  and mRMR can HURT. So include BOTH families, and report which predicts accuracy best rather
  than assuming Fisher/DBI is the answer.
- Intrinsic dimensionality: PCA-95%-var dim + TwoNN estimator vs channel count.

**Module 3 — Distribution shift, quantified** (the core novelty)
- Inter-subject vs inter-day vs inter-class distances: MMD, Wasserstein / Earth-Mover's,
  and H-divergence (pairwise subject-classifier error) on feature embeddings.
- ★ Directly answers scientific question A4 ("is inter-day variability larger than
  inter-subject?"). Multiple sources note within-subject temporal drift can EXCEED
  between-subject variation — a counterintuitive, measurable, citable result.

**Module 4 — Channel & sensor analysis**
- Channel redundancy: normalized mutual information (NMI) / correlation between channels
  (the HD-EMG poor-channel literature uses exactly NMI) → minimal channel subset.
- Sampling-rate sufficiency: re-estimate separability after downsampling → is 2 kHz needed?

**Module 5 — ★ Subject-difficulty predictor** (the money result; bridges to Paper 1)
- Regress each subject's ACTUAL LOSO accuracy (from Paper-1's full EMAHA sweep) on cheap
  statistics from a short calibration window (distribution distance to the pool, entropy,
  SNR proxy).
- Method after the Frontiers recipe (PMC9576998): per-subject H-divergence / MMD to the
  training pool, correlate with the generalization gap. **If a cheap statistic predicts who a
  model will fail on BEFORE training → headline result + the one link between the two papers.**

## 2. Deliverables
Per-dataset "data cards" (tables + figures), a cross-dataset comparison matrix, the
distribution-shift heatmaps, and the difficulty-predictor correlation plot.

## 3. Open decisions (need user answer before coding)
- **(a)** Does Module 5 wait for the full EMAHA LOSO sweep (it needs those per-subject
  accuracies as its target)? — likely yes; Modules 1–4 can start immediately.
- **(b)** Profile all 14 canonical L1 datasets, or just the Paper-1 six (emaha_db1, fors_emg,
  grabmyo, ninapro_db1/db2/db5)? — recommend the six first, expand later.

## 4. Cross-checked references (accessible = I read them)
- **Frontiers / PMC9576998** — predicting cross-subject generalization from distribution
  shift (H-divergence + kNN conditional shift). Core method for Module 5. [read]
- **PMC11175337** — comparative sEMG feature-evaluation methods; wrapper > filter finding. [read]
- **PMC10221262** — poor-quality HD-EMG channel detection via NMI. Module 4. [read]
- **arXiv 2508.21278** — detecting domain shifts in myoelectric activations (drift analysis). [read]
- **arXiv 1803.10753** — Higuchi fractal dimension vs sample entropy of sEMG. Module 1. [read]
- **arXiv 2409.07484** — FORS-EMG dataset (one of our datasets). [read]
- Muscle-synergy low-dimensional structure (NMF) — PMC3342930; EMG dimensionality-reduction
  for object weight — PLOS One pone.0255926. Module 5 intrinsic-dim grounding. [open]

## 5. PDFs the user will supply NEXT SESSION (paywalled / blocked to the agent)
Fold each one's exact parameter recipe into the relevant module before coding.
1. Fuzzy & Permutation Entropy in EMG gesture characterization (params m, r, delay) —
   https://www.mdpi.com/2076-3417/10/20/7144  (blocked 403) → Module 1
2. Multiscale Fuzzy Approximate Entropy of biceps sEMG —
   https://pubmed.ncbi.nlm.nih.gov/26738119/  (abstract only) → Module 1
3. Fuzzy Approximate Entropy for muscle fatigue (EMG) —
   https://link.springer.com/article/10.1007/s10439-010-9933-5  → Module 1
4. Phinyomark — Feature Reduction and Selection for EMG Classification (canonical
   separability/redundancy ref) —
   https://www.sciencedirect.com/science/article/abs/pii/S0957417412001200  → Module 2
5. Inter-Subject Variance Transfer Learning for EMG (Bayesian) —
   https://www.researchgate.net/publication/391954427  → Module 3
6. Nonlinear muscle synergies from sEMG (multi-model) —
   https://www.researchgate.net/publication/336331506  → Module 5 (intrinsic dim)
7. From zero- to few-shot: wrist EMG cross-user gesture recognition —
   https://www.researchgate.net/publication/395661441  → Module 5 (difficulty/calibration)

## 6. STATUS LOG (append-only)
- **2026-07-09 — Plan written + literature cross-checked (~15 papers across 5 modules).**
  Module 2 revised (wrapper/MI, not only Fisher/DBI). 7 paywalled PDFs to be supplied next
  session. Awaiting user go-ahead + answers to §3 (a)/(b) before building code.
