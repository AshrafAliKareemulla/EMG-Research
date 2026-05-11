# USER-REQUIREMENTS.md

## Researcher Background
- BTech with Honors — sEMG signal classification for ADL (Activities of Daily Living)
- Experienced with: preprocessing, feature extraction, classical ML classifiers
- Some transformer work on raw sEMG (no feature extraction) — accuracy was low
- Familiar datasets: EMAHA (primary), NinaPro, BioPatRec

## Two Research Tracks

**Track 1 — Classical ML (exhaustive):**
All preprocessing × all feature extraction × all classifiers. Find the single best pipeline. Leave no stone unturned. Include dimensionality reduction (PCA, SFS, tsfresh).

**Track 2 — Deep Learning (exploratory → systematic):**
CNN, BiLSTM, Transformers, hybrids, transfer learning, distillation, wavelet/Fourier+DL, LLM fine-tuning. Both raw-signal and feature-based approaches.

## Cross-Cutting Goals
- LOSO evaluation for both tracks
- Cross-dataset transfer learning
- ADL-Net: one universal model across all ADL datasets
- ML vs DL: when is DL justified?
- See `literature-review-papers/engineering-questions.md` for full list of things to keep in mind while reading complete research paper. Only read this file when you are doing literature review. If not, skip.


## Constraints
- Python (scikit-learn, PyTorch)
- All experiments reproducible (config-driven)
- Consistent evaluation protocol across all experiments
- The improvised solutions must fit 24VRAM Nvidia A5000 GPU.
