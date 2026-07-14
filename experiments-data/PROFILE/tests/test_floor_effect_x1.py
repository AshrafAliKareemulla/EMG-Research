"""Ground-truth tests for Experiment X1 (`floor_effect_x1.py`).

Validates the floor-effect analysis against synthetic controls with KNOWN answers, so the code is
trusted before it runs on real datasets — the same discipline as `test_math.py`.

Run:  python tests/test_floor_effect_x1.py     (prints PASS/FAIL; exit 0 iff all pass)

Only needs numpy / pandas / scikit-learn (NOT h5py/semg): `floor_effect_x1` imports the dataset
loaders lazily, so the pure functions + synthetic frames run on any box.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tests never write into a real results tree (CLAUDE.md rule 1)
import os as _os, pathlib as _pl
_os.environ.setdefault("PROFILE_RESULTS_DIR",
                       str(_pl.Path(__file__).resolve().parents[1] / "results" / "_test_sandbox"))

import numpy as np

from cli import floor_effect_x1 as x1

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}   {detail}")


# ------------------------------------------------------------------ pure statistics
def test_partial_corr():
    """x,y correlated ONLY through z -> partial ~ 0; correlated through an extra w -> partial > 0."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal(800)
    x = z + 0.1 * rng.standard_normal(800)
    y = z + 0.1 * rng.standard_normal(800)
    pc = x1.partial_corr(x, y, z)
    check("partial_corr(shared-only-via-z) ~ 0", abs(pc) < 0.2, f"{pc:.3f}")

    w = rng.standard_normal(800)
    x2 = w + z + 0.1 * rng.standard_normal(800)
    y2 = w + z + 0.1 * rng.standard_normal(800)
    pc2 = x1.partial_corr(x2, y2, z)
    check("partial_corr(shared-w-beyond-z) > 0.5", pc2 > 0.5, f"{pc2:.3f}")

    # a raw-correlated pair whose association is FULLY explained by z -> partial collapses to ~0
    x3 = 2.0 * z + 0.05 * rng.standard_normal(800)
    y3 = 3.0 * z + 0.05 * rng.standard_normal(800)
    raw = float(np.corrcoef(x3, y3)[0, 1])
    pc3 = x1.partial_corr(x3, y3, z)
    check("partial_corr removes a z-mediated raw correlation", raw > 0.9 and abs(pc3) < 0.2,
          f"raw={raw:.3f} partial={pc3:.3f}")


# ------------------------------------------------------ clustered bootstrap + verdict
def _rung_table(mode, seed=0):
    """Synthetic (dataset, rung) table with a KNOWN floor structure.

    'floor'   : |r| falls linearly as mean_acc rises (a ceiling/floor effect).
    'const'   : |r| is constant in mean_acc (no floor effect).
    'variance': |r| rises with acc_std, flat in mean_acc (a pure variance effect).
    """
    r = np.random.default_rng(seed)
    rows = []
    for d in range(9):
        for acc in np.linspace(0.10, 0.70, 8):
            spread = 0.05 + 0.02 * r.standard_normal()
            if mode == "floor":
                absr = 0.85 - 1.0 * acc + 0.03 * r.standard_normal()
            elif mode == "variance":
                absr = 0.30 + 2.0 * (spread - 0.05) + 0.03 * r.standard_normal()
            else:
                absr = 0.40 + 0.03 * r.standard_normal()
            rows.append(dict(dataset=f"d{d}", abs_r=float(np.clip(absr, 0, 1)),
                             mean_acc=float(acc), acc_std=float(abs(spread))))
    return rows


def test_cluster_bootstrap_and_verdict():
    # FLOOR: ceiling partial negative, CI excludes 0 -> verdict flags a ceiling effect
    ceil_f = x1.cluster_bootstrap_partial(_rung_table("floor"), "mean_acc", "acc_std", B=500, seed=0)
    var_f = x1.cluster_bootstrap_partial(_rung_table("floor"), "acc_std", "mean_acc", B=500, seed=1)
    check("floor table: ceiling partial < 0 & CI excludes 0",
          ceil_f["partial_r"] < 0 and ceil_f["excludes_zero"],
          f"r={ceil_f['partial_r']:.3f} ci={ceil_f['ci95']}")
    v = x1.floor_verdict(ceil_f, var_f)
    check("floor table: verdict reports a ceiling effect", v["ceiling_present"], v["headline"][:48])

    # CONSTANT: neither effect survives clustering
    ceil_c = x1.cluster_bootstrap_partial(_rung_table("const"), "mean_acc", "acc_std", B=500, seed=0)
    var_c = x1.cluster_bootstrap_partial(_rung_table("const"), "acc_std", "mean_acc", B=500, seed=1)
    vc = x1.floor_verdict(ceil_c, var_c)
    check("constant table: CI includes 0 (no ceiling effect)",
          not ceil_c["excludes_zero"] and not vc["ceiling_present"],
          f"r={ceil_c['partial_r']:.3f} ci={ceil_c['ci95']}")

    # VARIANCE-ONLY: variance partial positive & significant, ceiling not
    ceil_v = x1.cluster_bootstrap_partial(_rung_table("variance"), "mean_acc", "acc_std", B=500, seed=0)
    var_v = x1.cluster_bootstrap_partial(_rung_table("variance"), "acc_std", "mean_acc", B=500, seed=1)
    vv = x1.floor_verdict(ceil_v, var_v)
    check("variance table: verdict reports a variance effect", vv["variance_present"], vv["headline"][:48])


def test_naive_vs_clustered_contrast():
    """The clustered bootstrap must be MORE conservative than the naive pooled correlation on nested,
    dataset-clustered data — this is the whole point of the fix."""
    tab = _rung_table("floor", seed=3)
    from scipy.stats import pearsonr
    naive_r, naive_p = pearsonr([r["abs_r"] for r in tab], [r["mean_acc"] for r in tab])
    clustered = x1.cluster_bootstrap_partial(tab, "mean_acc", "acc_std", B=600, seed=0)
    # The naive p treats all 72 NESTED rungs as independent (absurdly tiny); the clustered bootstrap
    # resamples the 9 DATASETS as the unit -> honest inference at the correct sample size.
    check("clustered bootstrap uses the 9 datasets as the unit, unlike the pseudo-replicated naive p",
          naive_p < 1e-6 and clustered["n_datasets"] == 9 and clustered["excludes_zero"],
          f"naive_p={naive_p:.1e}, n_datasets={clustered['n_datasets']}")


# ------------------------------------------------------------- X1a on synthetic frames
def test_x1a_real_difficulty_survives():
    """Difficulty injected INDEPENDENT of class count -> the negative r must survive a class match.

    Thresholds are set wide of the sampling noise (the injected effect is strong), so a CORRECT
    implementation passes deterministically. The meaningful GT is the CONTRAST with pure-floor below:
    real difficulty gives r << 0 that persists; pure floor does not.
    """
    fr = x1.synth_frame("real_difficulty", n_subjects=20, n_channels=6, n_classes=24,
                        per_class=45, seed=1)   # >=40 windows/class so LOSO-LDA r is not noise-limited
    a = x1.x1a_matched_class_count(fr, target_classes=8, n_subsets=6, seed=1)
    check("X1a real: r_full strongly negative", a.get("r_full", 0.0) < -0.4, f"r_full={a.get('r_full')}")
    check("X1a real: negative r PERSISTS at matched class count (not a class-count artifact)",
          a.get("r_matched_mean", 0.0) < x1.STRONG_NEG,
          f"r_matched={a.get('r_matched_mean')} decision={a.get('decision','')[:30]}")


def test_x1a_pure_floor_no_signal():
    """Pure-floor synthetic has no real per-subject difficulty -> NOT a strong-negative r."""
    ff = x1.synth_frame("pure_floor", n_subjects=24, n_channels=6, n_classes=24,
                        per_class=30, seed=1)
    b = x1.x1a_matched_class_count(ff, target_classes=8, n_subsets=6, seed=1)
    check("X1a pure-floor: NOT a strong-negative r (no real difficulty signal)",
          b.get("r_full", 0.0) > -0.35, f"r_full={b.get('r_full')}")


# ------------------------------------------------------------- X1c rungs + pooled path
def test_x1c_rungs_and_pooled():
    fr = x1.synth_frame("real_difficulty", n_subjects=16, n_channels=6, n_classes=16,
                        per_class=30, seed=2)
    rungs = x1.x1c_rungs(fr, n_subsets=4, seed=2)
    ok_keys = rungs and all({"dataset", "k", "mean_acc", "acc_std", "r", "abs_r"} <= set(r) for r in rungs)
    check("X1c produces well-formed rungs from random channel subsets", ok_keys,
          f"n_rungs={len(rungs)}")

    # build_pooled must run and return a verdict on a floor-structured rung set
    pooled = x1.build_pooled(_rung_table("floor", seed=5), {}, bootstrap=400, seed=0)
    check("build_pooled returns a floor/ceiling verdict on floor rungs",
          "verdict" in pooled and pooled["verdict"]["ceiling_present"],
          pooled.get("verdict", {}).get("headline", "")[:48])
    check("build_pooled keeps the naive pseudo-replicated number (labelled, for contrast)",
          "naive_pseudoreplicated_DO_NOT_QUOTE" in pooled)


def main():
    for fn in [test_partial_corr, test_cluster_bootstrap_and_verdict, test_naive_vs_clustered_contrast,
               test_x1a_real_difficulty_survives, test_x1a_pure_floor_no_signal,
               test_x1c_rungs_and_pooled]:
        try:
            fn()
        except Exception as e:
            import traceback
            check(fn.__name__, False, f"EXCEPTION {type(e).__name__}: {e}")
            traceback.print_exc()
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n==== {n_pass}/{len(RESULTS)} checks passed ====")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
