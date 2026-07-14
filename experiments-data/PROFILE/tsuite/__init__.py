"""T-suite — the experiments designed on 2026-07-13 to close the evidence gaps found in the audit
of the frozen `results/legacy_v1` tree.

Every T experiment obeys the same contract (enforced by CLAUDE.md and by tsuite/run.py):

  1. It states WHY it exists, in terms of a specific gap in the committed evidence.
  2. It PRE-REGISTERS its branches: what a success means, what a failure means, and the fact that a
     failure is logged and published rather than buried.
  3. It carries a synthetic GROUND-TRUTH selftest that must pass before it is allowed to touch real
     data. The selftest validates the INSTRUMENT (can this code detect the effect it is looking for,
     and does it correctly report nothing when nothing is there?).
  4. It runs on ALL 14 datasets (unless the question is dataset-specific, in which case it names the
     datasets and its controls explicitly, as T2 does).
  5. It writes one atomic JSON per dataset plus a `pooled.json` verdict, into
     results/v2/<tag>/, and never into the frozen legacy tree.

  T1  model-family target      — does the predictor work for ANY learner, or only a linear one?
                                 (the CPU-only replacement for the retired, out-of-scope X3)
  T2  senic root cause         — why does one dataset reverse the sign?
  T3  moment ladder            — WHICH moment of a subject's distribution must be aligned?
  T4  ADL granularity          — do coarse ADL categories buy real cross-subject robustness?
  T5  transfer after alignment — does per-subject alignment rescue the failed cross-dataset transfer?
  T6  induced imbalance        — does centring break when the new user's data is skewed?
  T7  seed robustness          — is any of this stable across random seeds?
  T8  calibration budget       — how many SECONDS of unlabelled data does a new user need?
"""
