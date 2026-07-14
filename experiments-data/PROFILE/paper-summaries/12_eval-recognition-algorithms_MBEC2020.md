# Paper Summary — Evaluation of sEMG Recognition Algorithms (44-feature catalog)

| | |
|---|---|
| **File** | `data-science-related-papers/s11517-019-02073-z.pdf` |
| **Citation** | Abbaspour S, Lindén M, Gholamhosseini H, Naber A, Ortiz-Catalan M, *"Evaluation of surface EMG-based recognition algorithms for decoding hand movements"*, **Medical & Biological Engineering & Computing** 58, 2020, 83–100. Dataset: **BioPatRec** (our family), 20 subjects, 4 ch, 2 kHz. |
| **Maps to** | **Module 1** (exact formulas for fractal/Hjorth/complexity features) + Module 2 (feature ranking). |

---

## 1. Why it matters
It is the **formula reference** for the nonlinear/complexity features we want in Module 1 that aren't in the
`semg/` registry yet — Hjorth mobility/complexity, Higuchi fractal dimension (HFD), maximum fractal length
(MFL), skewness, kurtosis, percentile, inter-channel correlation, DASDV — all with equations, and it's validated
on **BioPatRec**, the user's own dataset family.

## 2. What we take (exact formulas — fold into Module 1)
- **Maximum fractal length** `MFL = log₁₀( sqrt( Σ (x_{i+1}−x_i)² ) )` (eq. 11).
- **Higuchi fractal dimension** `HFD = [log FDim(1) − log FDim(10)] / [log 10 − log 1]` with the curve-length
  `FDim(k)` of eq. 10.
- **Hjorth mobility** `√(Var(dx/dt)/Var(x))` (eq. 16); **Hjorth complexity** `Mobility(dx/dt)/Mobility(x)` (eq. 17).
- Skewness `M₃/M₂^{3/2}`, kurtosis `M₄/M₂²`, 75th **percentile**, inter-channel **correlation coefficient**,
  DASDV, multi-channel **energy ratio**. Window 200 ms/100 ms overlap; BPF 10–500 Hz + 50 Hz notch; trim 15% each end.

## 3. Key findings we cite
- **Best compact feature set = WL + Correlation coefficient + Hjorth parameters** → beats Hudgins by 5.5–6.3%
  across LDA/KNN/MLE/SVM. Adds Hjorth + inter-channel correlation to our Module-2 representative basis.
- **LogRMS and Normalized Logarithmic Energy (NLE) = best single features (>95%)** — cheap, strong; include them.
- Confirms information-content sweet spot at **window length 100–300 ms**.

## 4. Methodology we adopt / caveats
- Add MFL, HFD, Hjorth mobility/complexity, skewness, kurtosis, percentile, inter-channel correlation, LogRMS/NLE
  to Module 1's feature bank (these + the entropy family = the "complexity/nonlinear" block).
- **Caveat:** rankings are within-subject on BioPatRec; we re-evaluate the ranking under our cross-dataset/LOSO
  framing (Module 2) rather than assuming it holds. HFD's k-range (1..10) and MFL are amplitude-sensitive → compute
  on the train-only z-scored signal for cross-subject comparability.
