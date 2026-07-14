"""CLI for the PROFILE (Paper 2) data-science characterization.

Examples
--------
  # smoke test one module on one dataset, tiny subsample:
  python run_profile.py --module 2 --datasets emaha_db1 --smoke

  # full Phase-A (Modules 1-4) on the six datasets:
  python run_profile.py --module 1234 --datasets six

  # everything + cross-dataset matrix + figures:
  python run_profile.py --module all --datasets six --figures
"""
from __future__ import annotations

import argparse
import sys
import traceback

# allow "python run_profile.py" from the PROFILE dir
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import time

from dsprofile import config
from dsprofile import module1_signal, module2_separability, module3_shift
from dsprofile import module4_channels, module5_difficulty, datacards, viz


MODULES = {"1": module1_signal, "2": module2_separability, "3": module3_shift,
           "4": module4_channels, "5": module5_difficulty}

# each module's marker output file (used to SKIP already-finished (dataset, module) on re-run)
OUT = {
    "1": lambda ds: config.RESULTS_DIR / "module1" / f"{ds}__card.json",
    "2": lambda ds: config.RESULTS_DIR / "module2" / f"{ds}__separability.json",
    "3": lambda ds: config.RESULTS_DIR / "module3" / f"{ds}__shift.json",
    "4": lambda ds: config.RESULTS_DIR / "module4" / f"{ds}__channels.json",
    "5": lambda ds: config.RESULTS_DIR / "module5" / f"{ds}__difficulty.json",
}


def resolve_datasets(spec):
    if spec == ["six"]:
        return config.SIX
    if spec == ["all"]:
        return config.ALL14
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="1234",
                    help="digits 1-5 concatenated, or 'all'")
    ap.add_argument("--datasets", nargs="+", default=["all"])   # all 14 by default
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny caps for a fast end-to-end sanity run")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if a (dataset, module) output already exists")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel workers (default all cores; set e.g. 8 alongside GPU work)")
    args = ap.parse_args()

    if args.jobs is not None:
        config.N_JOBS = args.jobs

    if args.smoke:
        config.MAX_WINDOWS_PER_CLASS = 40
        config.ENTROPY_MAX_WINDOWS_PER_CLASS = 8

    mods = list("12345") if args.module == "all" else list(args.module)
    datasets = resolve_datasets(args.datasets)
    print(f"[run] datasets={datasets}  modules={mods}  force={args.force}  smoke={args.smoke}",
          flush=True)

    for ds in datasets:
        for m in mods:
            marker = OUT[m](ds)
            if marker.exists() and not args.force and not args.smoke:
                print(f"[SKIP] module {m} :: {ds} :: already done ({marker.name}) "
                      f"-- use --force to recompute", flush=True)
                continue
            t0 = time.time()
            try:
                res = MODULES[m].run(ds, seed=args.seed)
                key = {k: res[k] for k in list(res)[:6]}
                print(f"[OK] module {m} :: {ds} :: {time.time()-t0:.1f}s :: {key}", flush=True)
            except Exception as e:
                print(f"[FAIL] module {m} :: {ds} :: {time.time()-t0:.1f}s :: "
                      f"{type(e).__name__}: {e}", flush=True)
                traceback.print_exc()

    if args.figures:
        for ds in datasets:
            print("figure:", viz.shift_heatmap(ds), viz.difficulty_scatter(ds))

    df, path = datacards.build(datasets)
    print(f"\ncross-dataset matrix -> {path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
