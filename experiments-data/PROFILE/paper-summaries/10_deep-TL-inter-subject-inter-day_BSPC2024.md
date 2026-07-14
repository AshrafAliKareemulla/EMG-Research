# Paper Summary — Deep End-to-End Transfer Learning for Inter-Subject & Inter-Day HGR

| | |
|---|---|
| **File** | `data-science-related-papers/1-s2.0-S1746809424009509-main.pdf` |
| **Citation** | Li J, Jiang X, Fan J, Geng Y, Jia F, Dai C, *"Deep end-to-end transfer learning for robust inter-subject and inter-day hand gesture recognition using surface EMG"*, **Biomedical Signal Processing and Control** 100, 2025, 106892. HD-sEMG, 256 ch, 36 subjects, 2 days. |
| **Maps to** | **Module 3** — the empirical answer to scientific question A4 (inter-subject vs inter-day). |

---

## 1. Why it matters
It gives a **direct, same-dataset comparison of inter-subject vs inter-day** on 36 subjects × 2 days — the exact
A4 question Module 3 exists to answer — and a mechanistic reason (covariance alignment / CORAL helps), consistent
with Yoneda's "variance is shared."

## 2. Key findings we cite
- **Baseline statistical ML (KNN/LDA/SVM/RF) is significantly better inter-day than inter-subject (p<0.001)** →
  **inter-subject variation > inter-day variation** for sEMG. (This *contradicts* some literature claiming
  temporal drift can exceed between-subject variance — so the field is genuinely split, and our multi-dataset
  Module-3 quantification is a real contribution, not a foregone conclusion.)
- **LDA works inter-day (linear boundary suffices) but fails inter-subject** (needs nonlinear) → because
  inter-subject shift is larger and more structured.
- **Deep CORAL (second-order/covariance alignment) TL** lifted CNN inter-subject 85.4→91.4% and inter-day
  82.2→91.4%. CORAL aligning covariance working = evidence covariance is the transferable structure.
- Monitoring insight: without TL, same-gesture (intra-domain) mismatch shrinks during training while
  source↔target (inter-domain) mismatch grows — "clustering gestures at the expense of domain mismatch."

## 3. Methodology we adopt / caveats
- Adopt A4 as a headline Module-3 result and report the inter-subject-vs-inter-day ratio per dataset; explicitly
  contrast our finding with the mixed literature.
- Preprocessing/window reference: BPF 10–500 Hz, 250 ms/125 ms windows (matches our defaults).
- **Caveat:** their datasets that support inter-day are HD-sEMG with 2 days; of *our* six, only grabmyo (3
  sessions) and senic (uneven sessions) support an inter-day axis — so we answer A4 on that subset and treat the
  single-session datasets as inter-subject-only.
