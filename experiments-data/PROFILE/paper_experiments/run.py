"""Command-line runner — run the experiment suite WITHOUT notebooks (avoids Jupyter-kernel issues).

Use this in the conda env where the .py scripts already work (e.g. `ashraf_dl`), so you never depend
on which kernel Jupyter picked. Reads the 250 ms cache read-only, writes results/<tag>/ atomically.

    conda activate ashraf_dl
    python -m paper_experiments.run --selftest                    # ground truth first (must pass)
    python -m paper_experiments.run --exp all --datasets all --jobs 4
    python -m paper_experiments.run --exp x2 x4 x5 --datasets grabmyo,emaha_db1 --jobs 4
    python -m paper_experiments.run --exp x8 x11 x15              # pooled experiments
    # X1 (floor effect) is its own script:  python floor_effect_x1.py --datasets all --jobs 4
"""
from __future__ import annotations

import argparse
import importlib

from . import common

# name -> (module, mode, datasets_override)
REGISTRY = {
    "x2": ("x2_decoupling", "per_dataset", None),
    "x4": ("x4_recalibration_coral", "per_dataset", None),
    "x5": ("x5_deamplitude", "per_dataset", None),
    "x6": ("x6_learned_repr", "per_dataset", None),
    "x7": ("x7_mmd_sensitivity", "per_dataset", None),
    "x8": ("x8_ood_baselines", "per_dataset_pooled", None),
    "x9": ("x9_transfer", "pairs", None),
    "x10": ("x10_senic", "per_dataset",
            ["senic", "grabmyo", "grabmyo_flow_static", "grabmyo_flow_dynamic"]),
    "x11": ("x11_meta_regression", "pooled_run", None),
    "x12": ("x12_stability", "per_dataset", None),
    "x13": ("x13_imbalance", "per_dataset", None),
    "x14": ("x14_adaptive_lda", "per_dataset", None),
    "x15": ("x15_conformal_difficulty", "pooled_run", None),
}

DEFAULT_PAIRS = [("ninapro_db2", "ninapro_db4"), ("ninapro_db4", "ninapro_db2"),
                 ("grabmyo_flow_static", "grabmyo_flow_dynamic"), ("emaha_db1", "emaha_db4")]


def _run_x1(datasets, jobs, seed):
    """X1 (floor effect) lives in its own script; drive it here so `--exp all` covers it too."""
    from cli import floor_effect_x1 as fx
    if not fx.selftest():
        common.log("[SKIP] x1: ground truth failed — not running on real data")
        return
    outdir = common.results_dir("floor_effect_x1")
    per_dataset, all_rungs = {}, []
    for ds in datasets:
        try:
            out, rungs = fx.run_dataset(ds, target_classes=17, target_acc=0.15,
                                        n_subsets=20, seed=seed, n_jobs=jobs)
        except Exception as e:
            common.log(f"[FAIL] x1 :: {ds} :: {type(e).__name__}: {e}")
            per_dataset[ds] = dict(error=str(e))
            continue
        common.atomic_write_json(outdir / f"{ds}__x1.json", out)
        per_dataset[ds] = out
        all_rungs += rungs
        common.log(f"[OK] x1 :: {ds}")
    pooled = fx.build_pooled(all_rungs, per_dataset, bootstrap=2000, seed=seed)
    common.atomic_write_json(outdir / "pooled.json", pooled)
    common.log(f"x1 pooled verdict: {pooled.get('verdict', {}).get('headline', pooled.get('note', ''))[:90]}")


def run_exp(name, datasets, jobs, seed, force=False):
    if name == "x1":
        return _run_x1(datasets, jobs, seed)
    mod_name, mode, ds_over = REGISTRY[name]
    mod = importlib.import_module(f"paper_experiments.{mod_name}")
    ds = ds_over or datasets
    # ground-truth gate: never run an experiment on real data unless its synthetic checks pass
    from . import selftest as _st
    p, t, fails = _st.run_module(mod)
    if fails:
        common.log(f"[SKIP] {name}: GROUND TRUTH FAILED {fails} — not running on real data")
        return
    common.log(f"{name}: ground truth held ({p}/{t}) -> running")
    if mode in ("per_dataset", "per_dataset_pooled"):
        common.run_over_datasets(ds, mod.run_one, name, seed=seed, n_jobs=jobs, force=force)
        if mode == "per_dataset_pooled":
            common.log(f"{name} pooled: {mod.build_pooled(name)}")
    elif mode == "pooled_run":
        common.log(f"{name}: {mod.run()}")
    elif mode == "pairs":
        common.log(f"{name}: {mod.run_pairs(DEFAULT_PAIRS, seed=seed)}")


def main():
    ap = argparse.ArgumentParser(description="Paper-2 experiment suite runner")
    ap.add_argument("--exp", nargs="+", default=["all"],
                    help="experiment names (x2..x15) or 'all'")
    ap.add_argument("--datasets", default="all", help="'all' or a comma list")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="recompute even if a result file exists")
    ap.add_argument("--selftest", action="store_true", help="run all ground-truth checks and exit")
    args = ap.parse_args()

    if args.selftest:
        import sys
        from . import selftest
        sys.exit(0 if selftest.main() else 1)

    datasets = (list(common.config.ALL14) if args.datasets == "all"
                else [d.strip() for d in args.datasets.split(",") if d.strip()])
    # 'all' runs the fast suite (x2..x15) first, then X1 (the long one) last.
    exps = (list(REGISTRY) + ["x1"]) if args.exp == ["all"] else args.exp
    for e in exps:
        if e != "x1" and e not in REGISTRY:
            common.log(f"[unknown experiment] {e} (choices: x1, {', '.join(REGISTRY)})")
            continue
        with common.timer(f"experiment {e}"):
            run_exp(e, datasets, args.jobs, args.seed, args.force)
    common.log("ALL REQUESTED EXPERIMENTS DONE. Results in results/<tag>/. "
               "Also run: python floor_effect_x1.py --datasets all --jobs 4")


if __name__ == "__main__":
    main()
