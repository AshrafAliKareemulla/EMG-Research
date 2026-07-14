"""Delete exactly the results/caches invalidated by the 2026-07-10 correctness fixes.

`run_phase2.py` skips any experiment whose marker JSON already exists, so without this the
box would happily re-report every stale number. This removes the affected outputs and the
affected feature caches — and NOTHING else, because the expensive `fast`/`complex` frames
(hours of CPU) are still valid.

    python invalidate_stale.py --dry-run      # show what would go
    python invalidate_stale.py                # actually delete

What is invalidated, and why
----------------------------
REBUILD (results only, frames reused):
  module1   complexity_median emitted NaN on the sub-threshold-fs datasets  -> re-run
  module2   knn_loo_acc was a shuffled 5-fold over 50%-overlapping windows  -> re-run
  module3   h_divergence LEAKED (RF memorised trial identity; d_H saturated
            near its max of 2 on every dataset); two keys deprecated        -> re-run
  module5   in-sample R^2, post-hoc best-of-4 predictor, leaked hdiv_to_pool -> re-run
  block_a   masked entropy would otherwise be imputed as ZEROS by nan_to_num -> re-run
  block_b   same kNN leak, drove the per-class hard/easy ranking            -> re-run
  block_c   E3 was affine-invariant (measured nothing); its null floor was
            split by ROW not by TRIAL, under-estimating the noise floor ~14x -> re-run
  block_d   same kNN leak + naive `[::q]` decimation with no anti-alias filter -> re-run
  faabos    used the leaky knn_loo; the paper's only ADL-specific result    -> re-run
  action    no oracle ceiling, untested "hard subjects gain most" premise   -> re-run
  senic     collinear duplicate + unsupported confound verdict              -> re-run
  transfer  each dataset standardised independently before the MMD          -> re-run
  sdi/meta  leave-one-DATASET-out (cohort leak), no FDR, circular predictor -> re-run

ALSO REBUILD (added 2026-07-12, for the F1/F3/F4/F-dec fixes):
  robust_difficulty  CONSUMES mmd_to_pool -> F4's median-gamma bandwidth moves every value.
                     (The old header listed this under "NOT rebuilt". That was correct for the
                     2026-07-10 fixes and WRONG for these — it would have left old-gamma numbers
                     sitting next to new-gamma ones in the same results tree.)
  experiments        add-ons A-E inherit the same MMD.
  x2 x4..x15         the X-suite: x10 (rep-guard) and x11 (interpretation) actually changed; the
                     rest are re-run so every number in the paper comes from ONE code version.
  floor_effect_x1    re-run for the same reason.
  floor_effect, x3   DELETED, not rebuilt: superseded / withdrawn.

NOT rebuilt (verified by grep 2026-07-12 — neither references mmd, mi_symmetric_uncertainty,
nor the entropy features, so no fix can move them):
  module4            channel NMI / relevance / sampling-rate sufficiency.
  calibration        accuracy-vs-k curves use no shift statistic.

DELETE (caches that are now wrong):
  *_dec2.parquet, *_dec4.parquet   built with naive subsampling; the new cache key is
                                   `_dec2v2` / `_dec4v2`, so these are dead weight.

KEEP (still valid, expensive):
  <ds>__fast__...parquet           (no change to fast features)
  <ds>__complex__...parquet        (entropy masked on READ, no rebuild needed)
  <ds>__fast__..._nz-none.parquet  (still used by transfer.py)
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsprofile import config          # noqa: E402

# result subdirectories whose contents must be recomputed
#   module1 : complexity_median emitted NaN on the sub-threshold-fs datasets
#   faabos  : used the leaky knn_loo; it is the paper's only ADL-specific result
STALE_DIRS = ["module1", "module2", "module3", "module5", "block_a", "block_b", "block_c",
              "block_d", "actionability", "senic_probe", "faabos", "transfer",
              "module6_sdi", "meta",
              # the 28 Phase-1 PNGs are rendered from module3/module5 -> also stale
              "figures",
              # ---- added 2026-07-12 (the F1/F3/F4/F-dec fixes) --------------------------
              # robust_difficulty CONSUMES mmd_to_pool. The old header claimed it was
              # "verified unaffected" — that was true of the 2026-07-10 fixes, but F4 changes
              # the MMD bandwidth from gamma=1/d to the median heuristic, so every mmd_to_pool
              # value moves and this MUST be recomputed. Leaving it in KEEP silently mixed
              # old-gamma and new-gamma numbers in one results tree.
              "robust_difficulty",
              # the add-on experiments (A-E) inherit the same MMD
              "experiments",
              # the X-suite: x10 (rep-guard fix) and x11 (interpretation) changed; the rest are
              # re-run so that every number in the paper comes from ONE code version.
              "x2", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
              "floor_effect_x1",
              # superseded by floor_effect_x1 and internally self-contradictory -> must not exist
              "floor_effect", "x3"]
# aggregate + summary artifacts derived from the above
STALE_FILES = ["cross_dataset_matrix.xlsx", "cross_dataset_matrix.csv"]
# Verified by grep 2026-07-12: neither module4 (channel NMI / sampling-rate sufficiency) nor
# calibration (accuracy-vs-k) references mmd, mi_symmetric_uncertainty, or the entropy features,
# so none of F1/F3/F4/F-dec can move their numbers. Genuinely reusable -> hours of CPU saved.
KEEP_DIRS = ["module4", "calibration"]

# obsolete decimated caches (naive subsampling)
STALE_CACHE_GLOBS = ["*_dec2.parquet", "*_dec4.parquet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    a = ap.parse_args()

    root = config.RESULTS_DIR
    cache = config.L3_CACHE
    doomed_dirs, doomed_files = [], []

    for d in STALE_DIRS:
        p = root / d
        if p.exists():
            doomed_dirs.append(p)
    for f in STALE_FILES:
        p = root / f
        if p.exists():
            doomed_files.append(p)
    for g in STALE_CACHE_GLOBS:
        for f in glob.glob(str(cache / g)):
            # `_dec2v2` must not match `_dec2.parquet`
            if Path(f).name.endswith(("_dec2.parquet", "_dec4.parquet")):
                doomed_files.append(Path(f))
    # `synth__*.json` are written by tests/test_blocks.py into the real results tree. The stale
    # dirs get removed wholesale, but the KEPT dirs would otherwise carry them into the paper.
    for d in KEEP_DIRS:
        doomed_files += [Path(f) for f in glob.glob(str(root / d / "synth__*"))]

    print(f"results root : {root}")
    print(f"cache root   : {cache}\n")
    print("WILL DELETE (result dirs, to be recomputed):")
    for p in doomed_dirs:
        n = len(list(p.glob('*')))
        print(f"  {p.relative_to(root)}/   ({n} files)")
    print("\nWILL DELETE (files):")
    for p in doomed_files:
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p
        print(f"  {rel}  ({p.stat().st_size/1e6:.1f} MB)")
    print("\nWILL KEEP (still valid):")
    doomed_set = set(doomed_files)
    for d in KEEP_DIRS:
        p = root / d
        if p.exists():
            # don't count the synth__* files that are already in the DELETE list above
            n = sum(1 for f in p.glob('*') if f not in doomed_set)
            print(f"  {d}/   ({n} files)")
    keep_cache = [f for f in glob.glob(str(cache / "*.parquet"))
                  if not Path(f).name.endswith(("_dec2.parquet", "_dec4.parquet"))]
    tot = sum(os.path.getsize(f) for f in keep_cache) / 1e9
    print(f"  _feature_cache/   ({len(keep_cache)} parquet, {tot:.1f} GB) <- the expensive part")

    if a.dry_run:
        print("\n[dry-run] nothing deleted")
        return
    if not a.yes:
        if input("\nproceed? [y/N] ").strip().lower() != "y":
            print("aborted"); return

    for p in doomed_dirs:
        shutil.rmtree(p)
    for p in doomed_files:
        p.unlink()
    print(f"\ndeleted {len(doomed_dirs)} dirs + {len(doomed_files)} files")
    print("\nRe-run IN THIS ORDER (Phase-1 modules first: block_c/sdi/meta read their outputs):")
    print("  1) python run_profile.py --module 12345 --datasets all --jobs 8")
    print("  2) python run_phase2.py  --exp all      --datasets all --jobs 8")
    print("  3) python validate_results.py        # must exit 0 before any write-up")


if __name__ == "__main__":
    main()
