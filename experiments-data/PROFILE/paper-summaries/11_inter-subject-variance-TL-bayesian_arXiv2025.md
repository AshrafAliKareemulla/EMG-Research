# Paper Summary — Inter-Subject Variance Transfer Learning (Bayesian)

| | |
|---|---|
| **File** | `data-science-related-papers/2505.15381v1.pdf` |
| **Citation** | Yoneda S, Furui A, *"Inter-subject Variance Transfer Learning for EMG Pattern Classification Based on Bayesian Inference"*, arXiv:2505.15381, 2025. (This is PLAN §5 item 5, previously missing.) |
| **Maps to** | **Module 3 + Module 5** — the mean-vs-variance decomposition of distribution shift. |

---

## 1. Why it matters (the key conceptual lever)
It states and validates the hypothesis that reframes our whole distribution-shift story: **the *means* of EMG
features vary greatly across subjects, but their *variances/covariances* exhibit similar patterns across
subjects.** This tells Module 3 *how to decompose* inter-subject shift, and tells Module 5 *what a new subject
actually needs to calibrate* (mostly its mean, cheaply).

## 2. What we take
- **Decomposition principle:** split marginal/distribution shift into a **mean (location) component** and a
  **covariance (shape) component**. The Gaussian-KL (summary 09) does this analytically:
  mean term `(μ_i−μ_j)ᵀΣ⁻¹(μ_i−μ_j)` vs covariance term `tr(Σ_j⁻¹Σ_i)+ln(detΣ_j/detΣ_i)`. Report both per
  dataset — a clean, novel, citable figure ("inter-subject shift is dominated by the mean, not the covariance").
- **Bayesian variance-transfer / Adaptive-LDA** as a *cheap classical calibration baseline* for Module 5:
  Adaptive-LDA `μ̃ = τμ_cal + (1−τ)μ_src`, `Σ̃ = λΣ_cal + (1−λ)Σ_src`. (Botros used τ=0.75, λ=0.9.)

## 3. Key findings we cite
- **Variance transfer > no transfer > mean transfer**: transferring the *mean* from source subjects actually
  *hurts* (feature means are too subject-idiosyncratic), while sharing the *precision/covariance* helps with
  minimal (1-trial) calibration. Directly supports the decomposition and the "re-estimate the mean" implication.
- Gaussian Classification Model with Gaussian-Wishart + Dirichlet conjugate priors; weight `w^s` controls
  transfer amount (peaks near source/calibration ratio ≈ 1).

## 4. Methodology we adopt / caveats
- Make the **mean-vs-covariance shift decomposition** a first-class Module-3 output; use it to explain *why*
  z-score-global normalisation (which equalises first/second moments) helps cross-subject.
- **Caveat:** the full Bayesian GCM is heavier than Paper 2 needs — we implement only the *decomposition* and the
  lightweight Adaptive-LDA baseline, not the variational GCM. Features here are 1 Hz envelopes; we compute on our
  standard feature windows instead.
