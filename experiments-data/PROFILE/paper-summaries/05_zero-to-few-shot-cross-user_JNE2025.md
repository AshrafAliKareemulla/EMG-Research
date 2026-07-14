# Paper Summary — From Zero- to Few-Shot: Cross-User Wrist EMG

| | |
|---|---|
| **File** | `data-science-related-papers/From_zero-_to_few-shot_deep_temporal_learning_of_w.pdf` |
| **Citation** | Botros F S, Williams H E, Phinyomark A, Scheme E J, *"From zero- to few-shot: deep temporal learning of wrist EMG enables scalable cross-user gesture recognition"*, **Journal of Neural Engineering** 22(5), 2025, 056018 (**open access, CC-BY**). https://doi.org/10.1088/1741-2552/ae08eb |
| **Maps to** | **Module 5** (subject-difficulty / calibration curve) + bridge to **Paper 1** (few-shot recovery). Also touches Module 2 (feature set choice) and normalisation. |
| **PLAN §5 item** | #7 |

---

## 1. What the paper is
A rigorous cross-user study (33 subjects; wrist **and** forearm EMG recorded simultaneously at 1 kHz;
17 gestures reduced to 5 + rest) comparing **8 ML/DL architectures** (LDA, MLP, GRU, LSTM, BiLSTM, TCN,
TCN-LSTM, TCN-BiLSTM) under three regimes: **zero-shot** (no target data), **one-shot** (1 calibration rep),
**few-shot** (2–7 reps). It maps out the full **calibration curve** — accuracy vs number of target reps —
and shows a novel **TCN-BiLSTM** with a z-score-normalised **inter-day feature set (IDFS)** is best.

## 2. Why it matters to our use case
This is the **most directly aligned modern paper** we have for Module 5. Module 5's "money result" is a
**calibration/difficulty curve** — predicting how much a new subject costs to onboard. This paper *is* that
curve, done properly, in a top-tier journal (JNE), by the Scheme/Phinyomark group. It gives us: (a) the
target metric shape (zero→one→few-shot), (b) a strong DL baseline for Paper-1's Phase-2 recovery experiment,
(c) empirical proof that **z-score normalisation + an inter-day/complexity feature set** is the winning
representation for cross-user — which is *exactly* what Paper-2 Modules 1–3 are characterising. Note the
IDFS feature set literally contains our Module-1 complexity features (SampEn, Hjorth complexity, max fractal
length), tying the two papers together.

## 3. What we take (recipes + numbers)
**Preprocessing (clean, modern, reusable):** 3rd-order Butterworth **high-pass 20 Hz**, **60 Hz notch**
(Q=50), active-segment onset via RMS threshold, windows **150 ms / 75 ms overlap (50%)**.

**Feature set — IDFS (inter-day feature set, 14-dim, the winner):**
ASM, AR (4th order), **HC (Hjorth complexity)**, HNSM (Hjorth normalised spectral moment), KURT, **MFL (max
fractal length)**, MNF, **SampEn**, SKEW, SSC, WLR. → several are Module-1 complexity/nonlinear features →
**strong external validation of Module 1's design.**

**Normalisation:** **z-score is best** (beats min-max and raw), consistently. For zero-shot, μ/σ are the
**average over all training users**; as calibration reps arrive, μ/σ become personalised. (Mirrors our
`Normalizer(mode="global")` train-only convention.)

**Calibration curve numbers (cross-user, wrist EMG, TCN-BiLSTM):**
- Zero-shot **78.2%** → one-shot (1 rep) **91.6%** → few-shot rising to **98.3%** (7 reps).
- Forearm EMG lags wrist at every point (zero-shot 71.6%). LDA baseline gains little from *more training
  users*; DL benefits — a clean ML-vs-DL trade-off point (theme B).

**LDA transfer via Bayesian μ/σ adaptation** (relevant to the missing PLAN §5 #5 paper):
`μ̃ = (1−τ)μ₁ + τμ₂`, `σ̃ = (1−λ)σ₁ + λσ₂`, with τ=0.75, λ=0.9 (Bayesian-optimised). Useful as a cheap,
classical calibration baseline in Module 5.

## 4. What mattered / what didn't
- **Mattered:** the **calibration-curve framing** — this is the exact deliverable shape for Module 5 and
  Paper-1 Phase 2. We copy the "accuracy vs #target reps" axis.
- **Mattered:** IDFS proves complexity/nonlinear + spectral features win cross-user → validates spending
  Module-1 effort on entropy/fractal features rather than only amplitude features.
- **Mattered:** z-score-global normalisation is the right default (we already use it) — independent
  confirmation from a strong paper.
- **★ Mattered most for Module 5's framing:** **the amount of calibration data dominated the feature-set
  choice.** Adding calibration reps or more training users moved accuracy *far more* than optimising the
  feature set (compare their Fig 4 vs Fig 5). Implication: Module 5's story should centre on **"how much
  calibration does a new subject need"** (the curve), not on squeezing the feature set — the calibration
  budget is the real lever.
- **Mattered (validates our design choices):** (a) feeding **raw EMG to the DL models was tested and gave
  longer training + worse accuracy** than the feature-based IDFS → supports Paper-2/Track-1's feature-first
  stance; (b) training temporal models on **steady-state-only windows (excluding ramp-up transitions) was
  worse than the LDA baseline** → transition/contraction dynamics carry information, a caution for how we
  window and for Paper-1's segmentation.
- **IDFS's own authors group its 14 features into functional buckets** — amplitude/power (ASM, WLR),
  time-series model (AR), **nonlinear complexity (HC, MFL, SampEn)**, frequency (MNF, SSC), unique
  (HNSM, KURT, SKEW) — the *same "one representative per mathematical group"* logic as Phinyomark
  (summary 04). Two independent papers converging on that principle strengthens our Module-2 basis choice.
- **Didn't matter for us directly:** wrist-vs-forearm electrode-location comparison (our datasets are mostly
  forearm); the specific TCN-BiLSTM architecture is Paper-1's concern, not Paper-2's (Paper 2 is CPU/no-DL).

## 5. Novelty it lends to *our* Paper 2
It gives Module 5 a **published, credible target to predict**: instead of only regressing on Paper-1's LOSO
accuracies, we can frame the difficulty predictor as "estimate where on this calibration curve a new subject
lands, from cheap statistics of a short calibration window." If a Module-3 distribution-distance statistic
(MMD/H-divergence of the new subject to the training pool) predicts the subject's zero-shot / one-shot
accuracy, that is the unifying result — and this paper supplies the *ground-truth curve shape* to validate
against. It also lets us state the ML-vs-DL cross-user trade-off with a real reference.

## 6. Methodology we adopt
- Adopt the **calibration-curve deliverable** (accuracy vs k-shot) as Module 5's output format, aligned with
  Paper-1's `--k_shot` sweep so the two papers share one x-axis.
- Include IDFS's complexity features (SampEn, Hjorth complexity, MFL) in Module 1 explicitly; report their
  cross-subject distribution shift in Module 3.
- Use the Bayesian μ/σ LDA adaptation as a **cheap classical calibration baseline** to contrast with the DL
  curve — a clean ML/DL comparison inside Module 5.
- Confirm z-score-global as the normalisation for all cross-subject distance metrics.

## 7. Caveats / what NOT to borrow
- Their cross-user protocol is **leave-one-user-out** with an explicit target-calibration budget — this is
  the transductive/calibration-based regime, **not** calibration-free zero-shot in the strict sense (zero-
  shot here still uses pooled-train μ/σ). Keep the distinction sharp in our writing (ties to the
  `Normalizer.calibrate()` "transductive, not calibration-free" note in `splitter.py`).
- Don't import the DL architecture into Paper 2 — Paper 2 stays CPU/no-training. TCN-BiLSTM belongs to
  Paper-1 Phase 2 as the recovery baseline.
- Their 5-gesture reduced set is HCI-oriented; our datasets have many more classes and a rest class — the
  absolute accuracy numbers won't transfer, only the *curve shape* and *feature/normalisation choices*.
- **Reproducibility caveat:** the dataset is **not public** ("available on reasonable request"), and data
  collection was tightly controlled (single session, fixed posture) — so limb-position and cross-day effects
  are *not* covered here. We rely on our own datasets (e.g. SeNic's position/shift metadata) for those.
