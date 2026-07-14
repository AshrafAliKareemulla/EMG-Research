# Paper Summary — Phinyomark: Feature Reduction and Selection for EMG Classification

| | |
|---|---|
| **File** | `data-science-related-papers/1-s2.0-S0957417412001200-main.pdf` |
| **Citation** | Phinyomark A, Phukpattaranont P, Limsakul C, *"Feature reduction and selection for EMG signal classification"*, **Expert Systems with Applications** 39(8), 2012, 7420–7431. https://doi.org/10.1016/j.eswa.2012.01.102 |
| **Maps to** | **Module 2** (feature redundancy & separability — canonical reference) |
| **PLAN §5 item** | #4 |

---

## 1. What the paper is
The **canonical redundancy audit** of EMG features: 37 features (26 time-domain + 11 frequency-domain)
evaluated by (a) 2-D scatter plots for visual class separation and (b) LDA accuracy (5-fold CV, 20 subjects,
6 upper-limb movements). Its core result: **most time-domain features are mathematically redundant**, and
they cluster into **four groups by mathematical property** — (1) energy/complexity, (2) frequency
information, (3) prediction model, (4) time-dependence — with one representative per group being enough.

## 2. Why it matters to our use case
Module 2 is about **class structure & separability**, and Module 5's difficulty predictor needs a *compact,
non-redundant* feature basis to compute distribution distances on. This paper tells us **which features are
duplicates** so we don't feed 20 collinear amplitude features into MMD/Fisher and get distances dominated by
redundancy. It also gives an authoritative **separability ranking** we can reproduce on our datasets as a
sanity check, and — importantly — a *counter-intuitive nuance* about frequency-domain features that our
plan already flagged.

## 3. What we take (the redundancy map — fold into Module 2)
**Four TD groups and their representatives (paper's recommendation):**
- **Energy / complexity** — IEMG, MAV, MAV1, MAV2, SSI, VAR, RMS, v-Order, LOG (energy subclass) + WL, AAC,
  DASDV (complexity subclass). → keep **MAV** (energy) and **WL** (complexity); the rest are redundant.
- **Frequency information (TD proxies)** — ZC, MYOP, WAMP, SSC. → keep **WAMP**.
- **Prediction model** — AR (4th order) and Cepstral coeffs (CC). → keep **AR4** (best TD group,
  ~92% alone; CC ~91.5%).
- **Time-dependence** — MAVS, MHW, MTW, HIST. → **poor separability, not recommended.**

**Frequency-domain (PSD-based) verdict:** MNF, MDF, PKF, MNP, TTP, SM1–3, VCF, FR, PSR all show **poor class
separability** and are **not better than TD features** for *classification* (even though they track fatigue
well — a domain distinction we must keep). Among them **MNF > MDF**; PKF/VCF/FR/PSR are weakest.

**Headline combinations:** MAV+WL+WAMP ≈ **94.2%**; adding AR4 → ≈ **97%**. Normalisation for the scatter
analysis was **min-max per channel**.

## 4. What mattered / what didn't
- **Mattered:** the group-by-mathematical-property idea. We use it to **de-duplicate our feature bank before
  computing separability / distribution distances**, so Module 2/3/5 metrics reflect information, not
  collinearity.
- **Mattered (the nuance our plan already caught):** FD features are great for *fatigue/spectral* description
  but weak for *class separability*. This is a scientific-question C7/C8 point (does high-freq content carry
  independent class info?) — we can quantify it on our datasets rather than assert it.
- **Didn't matter for us as-is:** the specific 94%/97% numbers (only 6 movements, healthy, within-subject).
  We treat them as a **within-subject ceiling reference**, not a target — our story is cross-subject (LOSO),
  where this ranking may *not* hold. Testing whether the ranking survives LOSO is itself a finding (theme L4).

## 5. Novelty it lends to *our* Paper 2
Two levers. (1) It lets Module 2 **report a de-duplicated separability ranking on many datasets** and check
whether Phinyomark's within-subject ordering is *stable across datasets and under subject-disjoint eval* —
directly answering "is method/feature ranking dataset-specific noise?" (2) It grounds our claim that a small
representative feature set (MAV, WL, WAMP, AR4 + our complexity features from summaries 01–03) is a
principled, non-redundant basis for the whole paper — not an arbitrary pick.

## 6. Methodology we adopt
- Use MAV, WL, WAMP, AR4 as the **representative classical basis** and *augment* it with the complexity
  features (FEn, PEn, fApEn, MSfApEn) — this mirrors summary 01's finding that complexity features add
  information *on top of* the classical set.
- In Module 2, compute Fisher ratio / silhouette / kNN-LOO **and** wrapper/MI importance (per the plan's
  revision), then check the ranking against Phinyomark's group structure to confirm our pipeline reproduces
  a known result before we trust it on novel questions.
- Report the FD-features "good for fatigue, weak for class separability" split explicitly.

## 7. Caveats / what NOT to borrow
- **Do NOT treat their ranking as ground truth for cross-subject.** It is within-subject, 6-class, healthy,
  20 subjects. Our LOSO setting can reorder it — verifying that is part of the contribution, not a
  formality.
- Their scatter-plot "separability" is qualitative; we replace it with quantitative indices (Fisher/DBI/
  silhouette/kNN-LOO + MI/RFE) so results are numeric and dataset-comparable.
- min-max normalisation is fine for their 2-D viz, but for cross-subject distance metrics we use the
  train-only **z-score `Normalizer(mode="global")`** already in `semg/splits/splitter.py` to avoid leakage.
