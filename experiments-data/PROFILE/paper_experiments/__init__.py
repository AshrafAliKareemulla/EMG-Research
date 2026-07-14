"""paper_experiments — the proven, ground-truth-backed experiment suite for Paper 2.

Every experiment that supports a claim in the paper lives here as a small, testable module on top
of `common.py` (hardened math / stats / IO / distances / classifiers / synthetic ground truth).

Design contract for EVERY function here:
  * scalable & shape-agnostic  — guards on n_subjects / n_classes / n_channels / n_windows; degrades
                                 gracefully (returns a `note`) instead of crashing on a small dataset;
  * numerically safe           — guarded division, np.errstate, correlation clipping before arctanh,
                                 zero-variance / rank-deficiency handled explicitly;
  * leakage-safe               — subject/trial-disjoint splits, train-only standardisation;
  * proven                     — a synthetic control with a KNOWN answer in `selftest.py` / tests;
  * parallel- & resume-safe    — atomic per-dataset writes, one result dir per experiment.

Import is cheap and h5-free: heavy dataset loaders (`dsprofile.windows`) are imported lazily inside
the run functions, so the pure functions + synthetic ground truth run on any machine.
"""
__all__ = ["common"]
