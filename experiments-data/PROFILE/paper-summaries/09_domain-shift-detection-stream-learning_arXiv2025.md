# Paper Summary — Detecting Domain Shifts in Myoelectric Activations (Stream Learning)

| | |
|---|---|
| **File** | `data-science-related-papers/2508.21278v1.pdf` |
| **Citation** | Sun Y, Lim N, Cassales G W, Gomes H M, Pfahringer B, Bifet A, Dwivedi A, *"Detecting Domain Shifts in Myoelectric Activations: Challenges and Opportunities in Stream Learning"*, arXiv:2508.21278, 2025. Data: NinaPro **DB6**. |
| **Maps to** | **Module 3** (temporal / inter-day drift, a cheap distribution-distance) + inter-subject shift evidence. |

---

## 1. Why it matters
It defines **domain = unique (subject, day, time-slot)** and gives a **cheap, closed-form distribution-distance**
for temporal drift that we can drop straight into Module 3, plus direct evidence that subjects differ enormously
in intrinsic divergence — the seed of Module 5's difficulty story.

## 2. What we take
- **Closed-form KL divergence between two multivariate Gaussians** fit on feature windows:
  `D_KL = ½[ tr(Σ₁⁻¹Σ₀) + (μ₁−μ₀)ᵀΣ₁⁻¹(μ₁−μ₀) − d + ln(detΣ₁/detΣ₀) ]`. Cheap, batch-computable → Module-3
  inter-day / temporal-drift metric (reference-window vs sliding-window). **Note the KL naturally splits into a
  covariance term `tr(...)+ln(det ratio)` and a mean term `(μ₁−μ₀)ᵀΣ⁻¹(μ₁−μ₀)`** — the exact mean-vs-covariance
  decomposition Yoneda (summary 12) motivates.
- Mahalanobis-distance-on-slope drift score; RMS 200 ms/20 ms preprocessing; zero-channel elimination (DB6 has
  dead channels) — a reminder to screen dead/near-silent channels before any distance metric.

## 3. Key findings
- **Subjects differ hugely in baseline KL divergence** (their subject 1 low, subject 6 high): "different people
  activate muscles in relatively different manners even doing similar movements" → supports Module-3 inter-subject
  shift and Module-5 (divergent subjects are intrinsically harder).
- Off-the-shelf **drift detectors (CUSUM, Page-Hinckley, ADWIN, HDDM, SEED, ABCD) all perform poorly** (F1 < 0.41)
  on EMG → streaming drift detection is hard. *Context only — Paper 2 is batch, not streaming.*

## 4. Methodology we adopt / caveats
- Add closed-form Gaussian-KL as a Module-3 distance (complements MMD/energy-distance/Wasserstein), and use its
  mean/covariance split as a headline decomposition figure.
- **Caveat:** don't chase the streaming drift-detector angle — it's out of scope and shown not to work well.
  Guard the KL against singular Σ (regularise/shrink covariance; screen dead channels first).
