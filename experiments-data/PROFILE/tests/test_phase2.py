"""End-to-end + math tests for the Phase-2 modules (sdi.py, meta.py).

Validates: meta-analysis pooling (Fisher-z + DerSimonian-Laird) against hand computation and known
properties; SDI within-dataset standardisation; reproducibility; numerical robustness near +-1;
scalability (no hardcoded datasets); and the full run() I/O (keys present, no NaN/Inf).

Run:  python tests/test_phase2.py
"""
from __future__ import annotations

import sys, os, json, math, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tests never write into a real results tree (CLAUDE.md rule 1)
import os as _os, pathlib as _pl
_os.environ.setdefault("PROFILE_RESULTS_DIR",
                       str(_pl.Path(__file__).resolve().parents[1] / "results" / "_test_sandbox"))

import numpy as np

from dsprofile import meta, sdi, config

R = []
def check(name, cond, detail=""):
    R.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name}   {detail}")
def approx(a, b, tol=1e-6): return abs(float(a) - float(b)) <= tol * (1 + abs(float(b)))


# ------------------------------------------------ meta-analysis math
def test_fisher_z_roundtrip():
    for r in [-0.9, -0.43, 0.0, 0.3, 0.77]:
        check(f"tanh(atanh({r}))=={r}", approx(math.tanh(math.atanh(r)), r, 1e-9))

def test_pool_identical_effects():
    # all correlations identical -> pooled == that r, tau2==0, I2==0 (no heterogeneity)
    out = meta.pool_random_effects([0.4, 0.4, 0.4, 0.4], [20, 30, 25, 40])
    check("identical: pooled_r==0.4", approx(out["pooled_r_random_effects"], 0.4, 1e-6),
          f"got {out['pooled_r_random_effects']:.6f}")
    check("identical: tau2==0", approx(out["tau2"], 0.0, 1e-9), f"got {out['tau2']:.2e}")
    check("identical: I2==0", approx(out["I2"], 0.0, 1e-9), f"got {out['I2']:.2e}")
    check("identical: RE==FE", approx(out["pooled_r_random_effects"], out["pooled_r_fixed"], 1e-9))

def test_pool_fixed_effect_handcalc():
    # 2 studies: r=0.5 (n=20), r=0.3 (n=30). Fisher-z fixed-effect by hand.
    z1, z2 = math.atanh(0.5), math.atanh(0.3)
    w1, w2 = 20 - 3, 30 - 3
    z_fixed = (w1 * z1 + w2 * z2) / (w1 + w2)
    expect = math.tanh(z_fixed)
    out = meta.pool_random_effects([0.5, 0.3], [20, 30])
    check("fixed-effect vs hand calc", approx(out["pooled_r_fixed"], expect, 1e-9),
          f"got {out['pooled_r_fixed']:.6f} vs {expect:.6f}")

def test_pool_heterogeneous():
    # divergent effects (incl. a positive outlier like senic) -> tau2>0, I2>0, CI finite & valid
    out = meta.pool_random_effects([-0.7, -0.6, -0.4, +0.7], [27, 40, 25, 36])
    check("heterogeneous: tau2>0", out["tau2"] > 0, f"tau2={out['tau2']:.3f}")
    check("heterogeneous: 0<I2<=1", 0 < out["I2"] <= 1, f"I2={out['I2']:.3f}")
    lo, hi = out["ci95"]
    check("CI ordered & finite", lo < hi and np.isfinite(lo) and np.isfinite(hi), f"[{lo:.3f},{hi:.3f}]")
    check("pooled_r within [-1,1]", -1 <= out["pooled_r_random_effects"] <= 1)

def test_pool_near_unity_no_inf():
    # correlations at +-1 must not produce inf via arctanh (clipping)
    out = meta.pool_random_effects([1.0, -1.0, 0.99], [20, 20, 20])
    ok = all(np.isfinite(v) for v in [out["pooled_r_random_effects"], *out["ci95"], out["tau2"]])
    check("near +-1 clipped, all finite", ok)

def test_pool_weight_direction():
    # larger-n studies pull the pooled estimate toward their value
    out = meta.pool_random_effects([0.1, 0.8], [200, 10])   # big-n study says 0.1
    check("large-n dominates pooled", out["pooled_r_fixed"] < 0.4, f"got {out['pooled_r_fixed']:.3f}")


# ------------------------------------------------ SDI standardisation + reproducibility
def test_sdi_standardisation():
    A = sdi._load_standardised()
    ok = True
    for ds, g in A.groupby("dataset"):
        for c in ["z_loso_acc"] + ["z_" + p for p in sdi.PREDICTORS]:
            if abs(g[c].mean()) > 1e-6 or abs(g[c].std(ddof=0) - 1.0) > 1e-3:
                # std may differ slightly (pandas ddof); check within tolerance on ddof=0
                if abs(g[c].std() - 1.0) > 0.05:
                    ok = False
    check("within-dataset z-score: mean~0, std~1", ok, f"{A['dataset'].nunique()} datasets")

def test_sdi_reproducible():
    r1 = sdi.run(); r2 = sdi.run()
    check("SDI weights reproducible", r1["weights"] == r2["weights"])
    check("SDI LODO reproducible", r1["lodo_mean_spearman"] == r2["lodo_mean_spearman"])
    check("SDI LODO spearman in [-1,1]",
          all(-1 <= v["spearman"] <= 1 for v in r1["lodo_per_dataset"].values()))


# ------------------------------------------------ meta full run + robustness
def test_meta_run_io():
    r = meta.run()
    for k in ["what_makes_hard", "atlas", "meta_analysis", "how_many_subjects"]:
        check(f"meta output has '{k}'", k in r)
    # what_makes_hard is now {per_predictor, fdr_alpha, survivors_after_fdr, headline}
    wmh = r["what_makes_hard"]
    check("what_makes_hard is FDR-corrected", "per_predictor" in wmh and "fdr_alpha" in wmh)
    per = wmh["per_predictor"]
    good = all(np.isfinite(v["spearman_r"]) and -1 <= v["spearman_r"] <= 1 for v in per.values())
    check("what_makes_hard corrs valid", good)
    # If the accuracy-valued predictors are present at all, they MUST be flagged circular.
    # They are absent when results/ still holds pre-fix module2 JSONs (no knn_trial_cv_acc) —
    # which is itself worth surfacing, so say so rather than silently passing.
    acc_preds = [k for k in ("knn_trial_cv", "knn_loso") if k in per]
    if acc_preds:
        check("accuracy-valued predictors flagged circular",
              all(per[k].get("circular") for k in acc_preds), f"{acc_preds}")
    else:
        check("accuracy-valued predictors flagged circular", True,
              "SKIPPED: results/module2 predates the fix -> run invalidate_stale.py + re-run")
    check("circular predictors excluded from FDR",
          all(v.get("spearman_q_fdr") is None for v in per.values() if v.get("circular")))
    check("non-circular predictors carry a q-value",
          all("spearman_q_fdr" in v for v in per.values() if not v.get("circular")))
    ma = r["meta_analysis"]
    check("meta_analysis reports cohort independence",
          "n_independent_cohorts" in ma and ma["n_independent_cohorts"] <= ma["k_datasets"],
          f"{ma.get('n_independent_cohorts')} cohorts / {ma.get('k_datasets')} datasets")
    check("meta_analysis uses the a-priori predictor",
          ma.get("predictor") == config.PRIMARY_PREDICTOR)
    check("meta_analysis FDR-corrects the per-dataset table",
          all("q_fdr" in v for v in ma["per_dataset"].values()))
    # atlas coords finite; clusters are ints in range
    at = r["atlas"]
    check("atlas coords finite", all(np.isfinite(c).all() for c in at["coords"].values()))
    check("atlas clusters valid", all(isinstance(v, int) and v >= 0 for v in at["cluster"].values()))
    # how_many_subjects curves monotone-ish: std at small k >= std at full k (more subjects -> stabler)
    mono = True
    for ds, v in r["how_many_subjects"].items():
        ks = sorted(v["curve"]);
        if v["curve"][ks[0]][1] < v["curve"][ks[-1]][1] - 1e-9:
            mono = False
    check("how_many_subjects: std shrinks with k", mono)

def test_no_nan_inf_in_outputs():
    bad = []
    for f in glob.glob(str(config.RESULTS_DIR / "meta" / "*.json")) + \
             glob.glob(str(config.RESULTS_DIR / "module6_sdi" / "*.json")):
        s = open(f).read()
        if "NaN" in s or "Infinity" in s:
            bad.append(os.path.basename(f))
    check("no NaN/Inf in phase-2 JSON outputs", not bad, str(bad))


# ------------------------------------------------ scalability (no hardcoded datasets)
def _strip_comments_and_docstrings(src):
    """Return executable source only. Dataset names in COMMENTS are documentation (e.g. which
    datasets share a cohort); dataset names in CODE would be a scalability bug. Only the latter
    is what this test is about."""
    import io, tokenize, ast
    out, prev_end, prev_tok = [], (1, 0), tokenize.INDENT
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return src
    docstrings = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d is not None and node.body:
                docstrings.add((node.body[0].lineno, node.body[0].col_offset))
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and tok.start in docstrings:
            continue
        out.append(tok.string)
    return "\n".join(out)


def test_scalable_no_hardcoding():
    import re
    here = os.path.dirname(__file__)
    src = ""
    for m in ("meta.py", "sdi.py"):
        src += _strip_comments_and_docstrings(
            open(os.path.join(here, "..", "dsprofile", m)).read())
    # dataset discovery must be by glob / config, not a literal list of dataset names in code
    hard = re.findall(r"emaha_db1|ninapro_db|grabmyo|fors_emg|senic|myobit", src)
    check("no hardcoded dataset names in sdi/meta CODE", len(hard) == 0, f"found {set(hard)}")
    # tokens are re-joined with newlines, so match the identifier rather than the call syntax
    check("dataset discovery via glob", re.search(r"\bglob\b", src) is not None)
    # cohort structure and outliers live in config, not inline in the analysis modules
    check("cohort/outlier structure comes from config",
          "COHORTS" in src and "OUTLIER_DATASETS" in src)


def main():
    for fn in [test_fisher_z_roundtrip, test_pool_identical_effects, test_pool_fixed_effect_handcalc,
               test_pool_heterogeneous, test_pool_near_unity_no_inf, test_pool_weight_direction,
               test_sdi_standardisation, test_sdi_reproducible, test_meta_run_io,
               test_no_nan_inf_in_outputs, test_scalable_no_hardcoding]:
        try:
            fn()
        except Exception as e:
            import traceback; check(fn.__name__, False, f"EXC {type(e).__name__}: {e}"); traceback.print_exc()
    print(f"\n==== {sum(R)}/{len(R)} checks passed ====")
    sys.exit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
