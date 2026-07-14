"""Master ground-truth selftest for the whole suite.

Runs every module's synthetic ground-truth checks (known-answer controls) + the X1 floor experiment.
No dataset / h5 needed. This is the proof that the code is correct BEFORE it runs on real data — run
it first (locally or as notebook 00), and only trust the real numbers if this is all-green.

    python -m paper_experiments.selftest      # exit 0 iff all checks pass
"""
from __future__ import annotations

import sys

from . import (code_fixes, x2_decoupling, x4_recalibration_coral, x5_deamplitude,
               x6_learned_repr, x7_mmd_sensitivity, x8_ood_baselines, x9_transfer, x10_senic,
               x11_meta_regression, x12_stability, x13_imbalance, x14_adaptive_lda,
               x15_conformal_difficulty)

MODULES = [
    ("F1-F4/F-dec code fixes", code_fixes),
    ("X2 representation decoupling", x2_decoupling),
    ("X4 CORAL recalibration", x4_recalibration_coral),
    ("X5 de-amplituded basis", x5_deamplitude),
    ("X6 learned representation", x6_learned_repr),
    ("X7 MMD kernel sensitivity", x7_mmd_sensitivity),
    ("X8 OOD baselines", x8_ood_baselines),
    ("X9 cross-dataset transfer", x9_transfer),
    ("X10 senic electrode-shift", x10_senic),
    ("X11 meta-regression", x11_meta_regression),
    ("X12 subsample stability", x12_stability),
    ("X13 imbalance stratification", x13_imbalance),
    ("X14 adaptive-LDA calibration", x14_adaptive_lda),
    ("X15 conformal difficulty intervals", x15_conformal_difficulty),
]


def run_module(module):
    """Run one module's selftest(check); return (n_pass, n_total, [failures])."""
    res = []

    def check(name, cond, detail=""):
        res.append((name, bool(cond), detail))
        print(f"    [{'PASS' if cond else 'FAIL'}] {name}   {detail}")

    module.selftest(check)
    fails = [n for n, ok, _ in res if not ok]
    return sum(1 for _, ok, _ in res if ok), len(res), fails


def main():
    total_pass = total = 0
    module_fails = []
    for title, mod in MODULES:
        print(f"\n=== {title} ===")
        try:
            p, t, fails = run_module(mod)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    [FAIL] {title} raised {type(e).__name__}: {e}")
            module_fails.append(title)
            continue
        total_pass += p
        total += t
        if fails:
            module_fails.append(f"{title}: {', '.join(fails)}")

    # X1 (its own harness)
    print("\n=== X1 floor-effect (floor_effect_x1.py) ===")
    try:
        sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
            __import__("os").path.abspath(__file__))))
        from cli import floor_effect_x1
        ok_x1 = floor_effect_x1.selftest()
    except Exception as e:
        import traceback
        traceback.print_exc()
        ok_x1 = False
    if not ok_x1:
        module_fails.append("X1 floor-effect")

    print(f"\n{'=' * 60}\nGROUND-TRUTH SUMMARY: {total_pass}/{total} module checks passed; "
          f"X1={'PASS' if ok_x1 else 'FAIL'}")
    if module_fails:
        print("FAILURES:")
        for m in module_fails:
            print("  - " + m)
    ok = (total_pass == total) and ok_x1 and not module_fails
    print(("ALL GROUND TRUTH HELD [OK]" if ok else "GROUND TRUTH FAILURES ABOVE [FAIL]") + f"\n{'=' * 60}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
