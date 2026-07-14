# PROFILE (Paper 2) — Supplied-Paper Summaries Index

One summary per supplied PDF. Each explains: why the paper matters to Paper 2, the exact
recipe/parameters we take, what mattered vs didn't, the novelty it lends us, and the caveats.

| # | Summary file | Paper (short) | Venue / year | Maps to |
|---|--------------|---------------|--------------|---------|
| 01 | `01_fuzzy-permutation-entropy_ApplSci2020.md` | Fuzzy Entropy (FEn) & Permutation Entropy (PEn) for gesture characterization | Appl. Sci. 2020 | **Module 1** (FEn/PEn params) |
| 02 | `02_multiscale-fuzzy-ApEn-biceps_EMBC2015.md` | Multiscale Fuzzy ApEn of biceps sEMG (Navaneethakrishna) | IEEE EMBC 2015 | **Module 1** (multiscale + fatigue validation) |
| 03 | `03_fuzzy-ApEn-fatigue_AnnBiomedEng2010.md` | Fuzzy Approximate Entropy — foundational method (Xie) | Ann. Biomed. Eng. 2010 | **Module 1** (fApEn definition + "why fuzzy") |
| 04 | `04_phinyomark-feature-reduction_ESWA2012.md` | Feature reduction & selection for EMG (Phinyomark) | Expert Syst. Appl. 2012 | **Module 2** (feature redundancy/separability) |
| 05 | `05_zero-to-few-shot-cross-user_JNE2025.md` | Zero- to few-shot cross-user wrist EMG (Botros/Scheme) | J. Neural Eng. 2025 | **Module 5** (calibration curve) + Paper-1 bridge |
| 06 | `06_nonlinear-muscle-synergies-MRD_Dwivedi.md` | Nonlinear muscle synergies via MRD (Dwivedi & Shibata) | conf. | **Module 5** (intrinsic-dim grounding) |
| 07 | `07_distribution-shift-cross-subject-prediction_FrontiersAI2022.md` | Estimating distribution shifts to predict cross-subject generalization (Albuquerque) | Front. Artif. Intell. 2022 | **Module 5 backbone** + Module 3 |
| 08 | `08_distance-MI-feature-channel-selection_BSPC2016.md` | Distance & MI feature/channel subset selection (Al-Angari) | Biomed. Signal Process. Control 2016 | **Module 2** + **Module 4** |
| 09 | `09_domain-shift-detection-stream-learning_arXiv2025.md` | Detecting domain shifts in myoelectric activations (Sun) | arXiv 2025 | **Module 3** (Gaussian-KL drift) |
| 10 | `10_deep-TL-inter-subject-inter-day_BSPC2024.md` | Deep TL for inter-subject & inter-day HGR (Li) | Biomed. Signal Process. Control 2025 | **Module 3** (A4 answer) |
| 11 | `11_inter-subject-variance-TL-bayesian_arXiv2025.md` | Inter-subject variance TL, Bayesian (Yoneda & Furui) | arXiv 2025 | **Module 3 + 5** (mean-vs-covariance) |
| 12 | `12_eval-recognition-algorithms_MBEC2020.md` | Evaluation of sEMG recognition algorithms, 44-feature catalog (Abbaspour) | Med. Biol. Eng. Comput. 2020 | **Module 1** (fractal/Hjorth formulas) |
| 13 | `13_channel-selection-MCCSP_BMEO2014.md` | MCCSP channel selection for HD-EMG (Geng) | BioMed. Eng. OnLine 2014 | **Module 4** (channel selection) |
| 14 | `14_electrode-placement-gesture-category_FrontiersNeuro2025.md` | Electrode placement × gesture category (Qiu) | Front. Neurosci. 2025 | Robustness + Module 3/4 |

## Shared entropy core (Module 1) — one implementation serves 01/02/03
`m = 2`, Chebyshev distance, baseline-removed embedding, Gaussian/exponential membership
`exp(−dⁿ / r)`, tolerance `r = k·std(x)` (k ≈ 0.25–0.3). Exponent `n`: FEn sweeps {1,2,5}
(best 5); fApEn fixes 2. Multiscale wrapper (scales 1–10) → MSfApEn {MED, LS-MED, HS-MED}.
Also compute PEn (d≤5, τ=1, `d! ≪ N`) and SampEn. Entropy needs ≥~200 samples/window.

## Missing from PLAN §5
- **§5 item 5 — Inter-Subject Variance TL (Bayesian):** not supplied as a PDF. Located open-access:
  *"Inter-Subject Variance Transfer Learning for EMG Pattern Classification Based on Bayesian Inference"*,
  **arXiv:2505.15381** — https://arxiv.org/abs/2505.15381 . Its Bayesian μ/σ-transfer idea also appears
  inside summary 05 (Botros LDA-TL: τ=0.75, λ=0.9). Attach the PDF if you want a dedicated summary.
