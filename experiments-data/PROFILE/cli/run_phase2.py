"""CLI for Phase-2 experiments (Blocks A-F beyond the Phase-1 modules).

Dataset-scoped experiments run per dataset; aggregate ones run once over all datasets. Same infra
as Phase 1: resumable (skip finished), parallel (--jobs), progress+timing logs, crash-isolated.

Examples
--------
  python run_phase2.py --exp all --datasets all --jobs 8       # everything
  python run_phase2.py --exp c --datasets all --jobs 8         # just Block C
  python run_phase2.py --exp sdi,meta,transfer                 # aggregate only
  python run_phase2.py --exp a,b --datasets emaha_db1 --smoke  # quick sanity
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from dsprofile import config
from dsprofile import block_a, block_b, block_c, block_d, calibration, faabos, senic_probe
from dsprofile import robust_difficulty, actionability
from dsprofile import sdi, meta, transfer, figures, datacards

# name -> (module, results-subdir, marker-suffix)
DATASET_EXP = {
    "a": (block_a, "block_a", "block_a"),
    "b": (block_b, "block_b", "block_b"),
    "c": (block_c, "block_c", "block_c"),
    "d": (block_d, "block_d", "block_d"),
    "cal": (calibration, "calibration", "calibration"),
    "robust": (robust_difficulty, "robust_difficulty", "robust_difficulty"),
    "action": (actionability, "actionability", "actionability"),
    "faabos": (faabos, "faabos", "faabos"),
    "senic": (senic_probe, "senic_probe", "senic_probe"),
}
AGG_EXP = {
    "sdi": (sdi, "module6_sdi", "sdi"),
    "meta": (meta, "meta", "meta"),
    "transfer": (transfer, "transfer", "transfer"),
}
# `figs` is opt-in only (`--exp figs`) and deliberately NOT part of `--exp all`: figures are a
# write-up concern, and a plotting error must never abort or invalidate a numeric run.
ALL = list(DATASET_EXP) + list(AGG_EXP)


def resolve_datasets(spec):
    if spec == ["all"]:
        return config.ALL14
    if spec == ["six"]:
        return config.SIX
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", help="comma list from " + ",".join(ALL) + " or 'all'")
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.jobs is not None:
        config.N_JOBS = args.jobs
    if args.smoke:
        config.MAX_WINDOWS_PER_CLASS = 40
        config.ENTROPY_MAX_WINDOWS_PER_CLASS = 8

    exps = ALL if args.exp == "all" else [e.strip() for e in args.exp.split(",")]
    datasets = resolve_datasets(args.datasets)
    config.ensure_dirs()
    print(f"[phase2] exps={exps} datasets={datasets} jobs={config.N_JOBS} smoke={args.smoke}", flush=True)

    if "figs" in exps:
        figures.build_all()
        if exps == ["figs"]:
            return

    # dataset-scoped
    for e in [x for x in exps if x in DATASET_EXP]:
        mod, subdir, suf = DATASET_EXP[e]
        for ds in datasets:
            marker = config.RESULTS_DIR / subdir / f"{ds}__{suf}.json"
            if marker.exists() and not args.force and not args.smoke:
                print(f"[SKIP] {e} :: {ds} :: already done", flush=True); continue
            t0 = time.time()
            try:
                mod.run(ds, seed=args.seed)
                print(f"[OK] {e} :: {ds} :: {time.time()-t0:.1f}s", flush=True)
            except Exception as ex:
                print(f"[FAIL] {e} :: {ds} :: {type(ex).__name__}: {ex}", flush=True)
                traceback.print_exc()

    # aggregate (run once)
    for e in [x for x in exps if x in AGG_EXP]:
        mod, subdir, suf = AGG_EXP[e]
        marker = config.RESULTS_DIR / subdir / f"{suf}.json"
        if marker.exists() and not args.force and not args.smoke:
            print(f"[SKIP] {e} (aggregate) :: already done", flush=True); continue
        t0 = time.time()
        try:
            if e == "transfer":
                mod.run(datasets=datasets, seed=args.seed)
            else:
                mod.run()
            print(f"[OK] {e} (aggregate) :: {time.time()-t0:.1f}s", flush=True)
        except Exception as ex:
            print(f"[FAIL] {e} (aggregate) :: {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()

    # Always rebuild the cross-dataset matrix: it reads the corrected keys, and a stale .xlsx
    # sitting next to fresh JSONs is exactly how a wrong number reaches the paper.
    try:
        _, path = datacards.build(datasets)
        print(f"\ncross-dataset matrix -> {path}", flush=True)
    except Exception as ex:
        print(f"[FAIL] datacards :: {type(ex).__name__}: {ex}", flush=True)

    print("\nNEXT: python validate_results.py    (gates the write-up)", flush=True)


if __name__ == "__main__":
    main()
