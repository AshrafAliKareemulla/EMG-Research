"""T-suite runner. Shard by EXPERIMENT across terminals; never split one experiment's datasets.

    python -m tsuite.selftest                                  # ground truth first (must exit 0)
    python -m tsuite.run --exp t1 --datasets all --jobs 6
    python -m tsuite.run --exp t3 t4 --datasets all --jobs 6
    python -m tsuite.run --exp all --datasets all --jobs 4     # everything, serially
    python -m tsuite.run --pooled-only                         # rebuild verdicts from existing JSONs

Every run:
  * re-runs that experiment's ground-truth checks and REFUSES to touch real data if they fail;
  * writes one atomic JSON per dataset into results/v2/<tag>/ (resume-safe: an existing file is
    skipped unless --force);
  * writes a `pooled.json` verdict with the PRE-REGISTERED branch it landed on;
  * appends a row per (experiment, dataset) to the run log in logs/.

Results are immutable (CLAUDE.md): if you change an experiment's logic, bump its TAG to `..._v2`
rather than overwriting the old numbers.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

from dsprofile import config
from paper_experiments import common as X

# name -> (module, mode, datasets_override)
REGISTRY = {
    "t1": ("t1_model_family", "per_dataset_pooled", None),
    "t2": ("t2_senic_rootcause", "per_dataset_pooled", ["senic", "emaha_db1", "ninapro_db2", "grabmyo"]),
    "t3": ("t3_moment_ladder", "per_dataset_pooled", None),
    "t4": ("t4_adl_granularity", "per_dataset_pooled", None),
    "t5": ("t5_transfer_after_alignment", "pairs", None),
    "t6": ("t6_imbalance_induced", "per_dataset_pooled", None),
    "t7": ("t7_seed_robustness", "per_dataset_pooled", None),
    "t8": ("t8_calibration_budget", "per_dataset_pooled", None),
    # ---- batch 2 (2026-07-14). t9/t10 BUILD new frames -> do not run alongside t1-t8.
    "t9": ("t9_feature_families", "per_dataset_pooled", None),
    "t10": ("t10_rest_class_inflation", "per_dataset_pooled", None),
    "t11": ("t11_subject_scaling", "per_dataset_pooled", None),
}

LOG_DIR = config.PROFILE_DIR / "logs"


def _log_row(exp, dataset, status, seconds, note=""):
    """Append-only run log. One line per (experiment, dataset). This is the audit trail that lets
    STATE.md be updated from FACTS instead of from memory."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = dict(timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
               experiment=exp, dataset=dataset, status=status,
               seconds=round(float(seconds), 1), note=note)
    with (LOG_DIR / "tsuite_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _gate(name, mod):
    """Ground truth must pass before real data is touched."""
    from . import selftest as st
    p, t, fails = st.run_module(mod)
    if fails:
        X.log(f"[REFUSED] {name}: ground truth FAILED {fails} — not running on real data")
        return False
    X.log(f"{name}: ground truth held ({p}/{t}) -> running")
    return True


def run_exp(name, datasets, jobs, seed, force=False, pooled_only=False,
            datasets_explicit=False):
    mod_name, mode, override = REGISTRY[name]
    mod = importlib.import_module(f"tsuite.{mod_name}")
    tag = mod.TAG

    if pooled_only:
        v = mod.build_pooled()
        X.log(f"{name} pooled: branch={v.get('branch')} :: {str(v.get('verdict'))[:160]}")
        return v

    if not _gate(name, mod):
        return None

    # An EXPLICIT --datasets wins over an experiment's built-in default list. T2, for instance,
    # defaults to [senic + 3 controls] — sensible for the real run, but it made T2 impossible to
    # smoke-test on one small dataset, which is exactly why T2 was the last experiment never to have
    # touched real data. Explicit beats implicit.
    ds = datasets if datasets_explicit else (override or datasets)
    if mode == "pairs":
        # An explicit --datasets also RESTRICTS the pair list (both ends must be named). Without
        # this, T5 could only ever be run on all four of its heavy pairs at once, which is why it
        # was impossible to smoke-test on a small pair before committing to hours of compute.
        pairs = mod.FINE_PAIRS
        if datasets_explicit:
            pairs = [(s, t) for s, t in pairs if s in datasets and t in datasets]
            if not pairs:
                X.log(f"[SKIP] {name}: no label-compatible pair inside --datasets {datasets}. "
                      f"Available pairs: {mod.FINE_PAIRS}")
                return None
        t0 = time.perf_counter()
        v = mod.run_pairs(pairs=pairs, seed=seed, n_jobs=jobs, force=force)
        _log_row(name, "PAIRS:" + ",".join(f"{s}->{t}" for s, t in pairs), "ok",
                 time.perf_counter() - t0)
    else:
        for d in ds:
            p = X.results_dir(tag) / f"{d}__{tag}.json"
            if p.exists() and not force:
                X.log(f"[SKIP] {name} :: {d} :: already done")
                _log_row(name, d, "skip", 0.0, "existing result reused")
                continue
            t0 = time.perf_counter()
            try:
                r = mod.run_one(d, seed=seed, n_jobs=jobs)
                X.atomic_write_json(p, r)
                dt = time.perf_counter() - t0
                X.log(f"[OK] {name} :: {d} :: {dt:.1f}s")
                _log_row(name, d, "ok", dt)
            except Exception as e:
                traceback.print_exc()
                dt = time.perf_counter() - t0
                X.log(f"[FAIL] {name} :: {d} :: {type(e).__name__}: {e}")
                _log_row(name, d, "fail", dt, f"{type(e).__name__}: {e}")
        v = mod.build_pooled()
        X.log(f"{name} pooled: branch={v.get('branch')} :: {str(v.get('verdict'))[:200]}")
    return v


def main():
    ap = argparse.ArgumentParser(description="T-suite runner")
    ap.add_argument("--exp", nargs="+", default=["all"], help="t1..t8 or 'all'")
    ap.add_argument("--datasets", default="all", help="'all' or a comma list")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="recompute even if a result exists")
    ap.add_argument("--pooled-only", action="store_true",
                    help="rebuild pooled verdicts from existing per-dataset JSONs; no compute")
    a = ap.parse_args()

    datasets_explicit = a.datasets != "all"
    datasets = (list(config.ALL14) if a.datasets == "all"
                else [d.strip() for d in a.datasets.split(",") if d.strip()])
    exps = list(REGISTRY) if a.exp == ["all"] else a.exp
    for e in exps:
        if e not in REGISTRY:
            X.log(f"[unknown experiment] {e} (choices: {', '.join(REGISTRY)})")
            continue
        with X.timer(f"experiment {e}"):
            run_exp(e, datasets, a.jobs, a.seed, a.force, a.pooled_only, datasets_explicit)

    X.log("DONE. Results in results/v2/<tag>/. Next: python -m cli.build_ledger")


if __name__ == "__main__":
    main()
