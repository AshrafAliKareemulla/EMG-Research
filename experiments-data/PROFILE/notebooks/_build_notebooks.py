"""Generate one parallel-runnable .ipynb per experiment (+ a master ground-truth notebook).

Run once (fast, no data touched):  python notebooks/_build_notebooks.py
Each notebook: a ground-truth GATE cell (must pass) -> the real run -> a summary. All notebooks are
parallel-safe: they READ the existing 250 ms cache read-only and WRITE to their own results/<tag>/ dir
with atomic writes, so you can launch them all at once. Keep the SUM of `JOBS` across the notebooks you
run simultaneously <= your physical core count.
"""
import json
import os

NB_DIR = os.path.dirname(os.path.abspath(__file__))

BOOTSTRAP = [
    "import sys, os\n",
    "# --- L1 dataset location (edit here if your data ever moves) -------------------------\n",
    "os.environ.setdefault('SEMG_L1_ROOT', '/home/honors/Ashraf_Ali_2025_Batch/data/L1')\n",
    "cwd = os.getcwd()\n",
    "cands = [cwd, os.path.dirname(cwd), os.path.join(cwd, 'notebooks')]\n",
    "PROFILE = next((p for p in cands if os.path.isdir(os.path.join(p, 'dsprofile'))), cwd)\n",
    "if PROFILE not in sys.path: sys.path.insert(0, PROFILE)\n",
    "print('PROFILE =', PROFILE)\n",
]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(lines)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(lines)}


def _lines(lines):
    flat = []
    for ln in lines:
        flat.extend(ln.splitlines(keepends=True) if "\n" in ln else [ln])
    # ensure each element ends with newline except possibly the last
    return [(s if s.endswith("\n") else s + "\n") for s in flat[:-1]] + [flat[-1].rstrip("\n")] if flat else []


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def gate(module):
    return code(
        f"from paper_experiments import selftest as _st, {module} as _mod\n",
        "p, t, fails = _st.run_module(_mod)\n",
        "assert not fails, f'GROUND TRUTH FAILED (do NOT trust real numbers): {fails}'\n",
        "print(f'ground truth held: {p}/{t} checks')\n",
    )


PER_DATASET_RUN = [
    "from paper_experiments import common\n",
    "DATASETS = list(common.config.ALL14)   # trim to a subset if you like\n",
    "JOBS = 2                               # sum of JOBS over notebooks run at once <= physical cores\n",
    "res = common.run_over_datasets(DATASETS, _mod.run_one, '{tag}', n_jobs=JOBS)\n",
    "print('done datasets:', [k for k in res if 'error' not in (res[k] or {})])\n",
]

EXPERIMENTS = [
    ("X2_representation_decoupling", "x2_decoupling", "x2",
     "the shared-representation coupling (predictor & target in the same feature space)", "per_dataset", None),
    # X3 (DL-target swap) was RETIRED on 2026-07-13: deep learning belongs to another track and this
    # repository is data-science-only. Its scientific question — does the statistic predict a
    # NON-LINEAR model's failures, not just an LDA's? — is answered by tsuite/t1_model_family.py on
    # CPU, with five learner families across all 14 datasets. (The orphan line left behind when X3
    # was first deleted made this file a SyntaxError; that is why the notebooks could not be
    # regenerated.)
    ("X4_coral_recalibration", "x4_recalibration_coral", "x4",
     "the E3 vs exp_B tension: does covariance (CORAL) alignment beat mean-centring?", "per_dataset", None),
    ("X5_deamplituded_basis", "x5_deamplitude", "x5",
     "'is cross-subject shift just contraction amplitude?'", "per_dataset", None),
    ("X6_learned_representation", "x6_learned_repr", "x6",
     "the 'handcrafted-basis artifact?' concern (PCA + RFF embeddings)", "per_dataset", None),
    ("X7_mmd_kernel_sensitivity", "x7_mmd_sensitivity", "x7",
     "fixed-gamma MMD & the scale-free _frob aggregate", "per_dataset", None),
    ("X8_ood_baseline_bakeoff", "x8_ood_baselines", "x8",
     "'is MMD the best cheap statistic?' vs Mahalanobis/kNN/energy", "per_dataset_pooled", None),
    ("X9_cross_dataset_transfer", "x9_transfer", "x9",
     "the unvalidated (shape-only) transfer matrix", "pairs", None),
    ("X10_senic_electrode_shift", "x10_senic", "x10",
     "the quarantined senic outlier (electrode-shift testbed + within-condition null)", "per_dataset",
     ["senic", "grabmyo", "grabmyo_flow_static", "grabmyo_flow_dynamic"]),
    ("X11_meta_regression", "x11_meta_regression", "x11",
     "the between-dataset floor test (complements X1)", "pooled_run", None),
    ("X12_subsample_stability", "x12_stability", "x12",
     "'are the 40/600 caps adequate?'", "per_dataset", None),
    ("X13_imbalance_stratification", "x13_imbalance", "x13",
     "exp_B's biased-mean caveat (centring vs class imbalance)", "per_dataset", None),
    ("X14_adaptive_lda_calibration", "x14_adaptive_lda", "x14",
     "the missing calibration-budget baseline (Adaptive-LDA curve)", "per_dataset", None),
    ("X15_conformal_difficulty", "x15_conformal_difficulty", "x15",
     "point-only difficulty prediction — adds distribution-free coverage-guaranteed INTERVALS (novel)",
     "pooled_run", None),
]


def build_experiment(name, module, tag, buries, mode, datasets):
    cells = [md(f"# {name}\n", f"\n**Buries:** {buries}\n",
                "\nGround truth is checked first; the real run is trusted only if it passes. "
                "Parallel-safe (reads the cache read-only, writes `results/" + tag + "/`)."),
             code(*BOOTSTRAP), gate(module)]
    if mode in ("per_dataset", "per_dataset_pooled"):
        run = [ln.replace("{tag}", tag) for ln in PER_DATASET_RUN]
        if datasets is not None:
            run[1] = "DATASETS = " + json.dumps(datasets) + "\n"
        cells.append(code(*run))
        if mode == "per_dataset_pooled":
            cells.append(code(f"pooled = _mod.build_pooled('{tag}')\n", "print(pooled.get('ranking', pooled))\n"))
    elif mode == "pooled_run":
        cells.append(code("out = _mod.run()\n", "print({k: out[k] for k in list(out)[:3]})\n"))
    elif mode == "pairs":
        cells.append(code(
            "PAIRS = [('ninapro_db2','ninapro_db4'), ('ninapro_db4','ninapro_db2'),\n",
            "         ('grabmyo_flow_static','grabmyo_flow_dynamic'), ('emaha_db1','emaha_db4')]\n",
            "out = _mod.run_pairs(PAIRS)\n",
            "print('pairs:', out.get('n_pairs'), out.get('mmd_vs_transfer_pearson'))\n"))
    return name, notebook(cells)


def build_x1():
    cells = [
        md("# X1_floor_effect_correct\n",
           "\n**Buries:** the floor-effect confound — replaces the contested 'RESOLVED' single-trend "
           "with three probes (matched class count / matched accuracy / dataset-clustered pooled model)."),
        code(*BOOTSTRAP),
        code("from cli import floor_effect_x1 as fx\n",
             "assert fx.selftest(), 'X1 GROUND TRUTH FAILED'\n"),
        code("from paper_experiments import common\n",
             "DATASETS = list(common.config.ALL14)   # all 14 -> more cohort clusters -> tighter CIs (long run)\n",
             "JOBS = 4\n",
             "per_dataset, all_rungs = {}, []\n",
             "outdir = common.results_dir('floor_effect_x1')\n",
             "for ds in DATASETS:\n",
             "    try:\n",
             "        out, rungs = fx.run_dataset(ds, target_classes=17, target_acc=0.15, n_subsets=20, seed=42, n_jobs=JOBS)\n",
             "    except Exception as e:\n",
             "        print('[FAIL]', ds, e); per_dataset[ds] = {'error': str(e)}; continue\n",
             "    common.atomic_write_json(outdir / f'{ds}__x1.json', out)\n",
             "    per_dataset[ds] = out; all_rungs += rungs; print('[OK]', ds)\n",
             "pooled = fx.build_pooled(all_rungs, per_dataset, bootstrap=2000, seed=42)\n",
             "common.atomic_write_json(outdir / 'pooled.json', pooled)\n",
             "print(pooled.get('verdict', pooled))\n"),
    ]
    return "X1_floor_effect_correct", notebook(cells)


def build_selftest_all():
    cells = [
        md("# 00_SELFTEST_ALL — run this FIRST\n",
           "\nRuns every experiment's synthetic **ground-truth** checks. Only trust the real runs if this "
           "is all-green. No dataset needed."),
        code(*BOOTSTRAP),
        code("from paper_experiments import selftest\n",
             "ok = selftest.main()\n",
             "assert ok, 'GROUND TRUTH FAILURES — see output above'\n",
             "print('\\nALL GROUND TRUTH HELD — the suite is safe to run on real data.')\n"),
    ]
    return "00_SELFTEST_ALL", notebook(cells)


def main():
    made = []
    for nm, nb in [build_selftest_all(), build_x1()] + [
            build_experiment(*e) for e in EXPERIMENTS]:
        path = os.path.join(NB_DIR, nm + ".ipynb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        made.append(nm + ".ipynb")
    print(f"wrote {len(made)} notebooks to {NB_DIR}:")
    for m in made:
        print("  " + m)


if __name__ == "__main__":
    main()
