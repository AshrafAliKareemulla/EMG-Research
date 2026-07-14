"""Ground-truth gate for the T-suite. NOTHING runs on real data until this is green.

    python -m tsuite.selftest          # exit 0 iff every synthetic check passes

Each experiment's `selftest(check)` validates the INSTRUMENT, not the hypothesis:
  * can this code detect the effect it is looking for, on data where we KNOW the effect is present?
  * does it correctly report NOTHING on data where we know the effect is absent?
A test that only does the first is worthless: an experiment that always finds an effect is not an
experiment. Every module here is required to have at least one negative control.
"""
from __future__ import annotations

import sys
import traceback

from . import (t1_model_family, t2_senic_rootcause, t3_moment_ladder, t4_adl_granularity,
               t5_transfer_after_alignment, t6_imbalance_induced, t7_seed_robustness,
               t8_calibration_budget, t9_feature_families, t10_rest_class_inflation,
               t11_subject_scaling)

MODULES = [
    ("T1 model-family target", t1_model_family),
    ("T2 senic root cause", t2_senic_rootcause),
    ("T3 moment ladder", t3_moment_ladder),
    ("T4 ADL granularity", t4_adl_granularity),
    ("T5 transfer after alignment", t5_transfer_after_alignment),
    ("T6 induced imbalance", t6_imbalance_induced),
    ("T7 seed robustness", t7_seed_robustness),
    ("T8 calibration budget", t8_calibration_budget),
    ("T9 feature families", t9_feature_families),
    ("T10 rest-class inflation", t10_rest_class_inflation),
    ("T11 subject scaling", t11_subject_scaling),
]


def run_module(module):
    passed = total = 0
    failures = []

    def check(label, cond, detail=""):
        nonlocal passed, total
        total += 1
        if cond:
            passed += 1
            print(f"    [PASS] {label}   {detail}")
        else:
            failures.append(label)
            print(f"    [FAIL] {label}   {detail}")

    module.selftest(check)
    _check_build_pooled(module, check)
    return passed, total, failures


def _check_build_pooled(module, check):
    """Every experiment's `build_pooled()` must RUN, and must survive an empty results directory.

    ==> ADDED AFTER THE 2026-07-13 CODE REVIEW. <==

    `build_pooled` is where the pre-registered branch is decided — it is the single most consequential
    function in each experiment — and NOTHING was testing it. The review found that T1's would crash
    on its very first call, with two bugs stacked back to back:

        q = C.fdr_bh(np.array(ps))            # returns a TUPLE (rejected, q), not an array
        pr["pooled_r_random_effects"]         # the key is "pooled_r"

    Both would have raised *after* five model families x 14 datasets of compute had finished, and
    `run.py` calls `build_pooled()` outside any try/except, so the entire run would have died at the
    finish line with nothing to show for it.

    A full branch-by-branch test needs synthetic per-dataset JSONs, which is worth doing later. This
    check is the cheap 90%: import it, call it against an empty (or partially populated) results dir,
    and require that it returns a dict instead of exploding. That alone would have caught the crash.
    """
    name = getattr(module, "TAG", module.__name__)
    if not hasattr(module, "build_pooled"):
        return
    try:
        out = module.build_pooled()
        check(f"{name}: build_pooled() runs and returns a dict (the branch verdict is reachable)",
              isinstance(out, dict), f"keys={sorted(out)[:4] if isinstance(out, dict) else out}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(f"{name}: build_pooled() runs without raising", False,
              f"{type(e).__name__}: {e}")


def main():
    all_pass, tp, tt = True, 0, 0
    for title, mod in MODULES:
        print(f"\n=== {title} ===")
        try:
            p, t, fails = run_module(mod)
        except Exception:
            traceback.print_exc()
            print(f"    [ERROR] {title} raised")
            all_pass = False
            continue
        tp += p
        tt += t
        if fails:
            all_pass = False
    print(f"\n==== T-suite ground truth: {tp}/{tt} checks passed ====")
    if not all_pass:
        print("REFUSING to certify: fix the failures before running on real data.")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
