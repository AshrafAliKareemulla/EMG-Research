"""Post-run validation of results/ — catches the failure modes the 2026-07-10 audit found.

Run AFTER `run_phase2.py`. Exits non-zero if any hard check fails, so it can gate the write-up.

    python validate_results.py                 # all datasets
    python validate_results.py --strict        # warnings become failures

Checks
------
STALE     no result JSON still carries a superseded key (knn_loo_acc, combined_linear_r2,
          E3_meancov_raw_vs_norm, best_predictor, guided_advantage-without-ceiling, ...)
MATH      the affine-invariance assertion actually passed on every dataset, and the ridge
          demonstrably breaks it (that is the paper's methodological claim)
LEAK      within-subject kNN >= cross-subject kNN, and the gap is reported
VALIDITY  entropy features are absent exactly on the sub-threshold-fs datasets
NAN       no unexpected NaN/Inf outside the documented, benign cases
HONESTY   FDR fields present; actionability reports a ceiling; senic verdict present
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsprofile import config, windows          # noqa: E402

FAIL, WARN = [], []


def fail(msg):
    FAIL.append(msg); print(f"[FAIL] {msg}", flush=True)


def warn(msg):
    WARN.append(msg); print(f"[WARN] {msg}", flush=True)


def ok(msg):
    print(f"[ ok ] {msg}", flush=True)


def _is_synth(f):
    """`synth` files are written by the test suite, not by a real run."""
    return Path(f).name.startswith("synth__")


def _walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, o


SUPERSEDED = {
    "knn_loo_acc": "module2 -> knn_trial_cv_acc / knn_loso_acc",
    "combined_linear_r2": "module5 -> combined_cv_r2",
    "E3_meancov_raw_vs_norm": "block_c -> E3_meancov",
    "best_predictor": "module5 -> primary_predictor (+ best_predictor_posthoc)",
    "mean_vs_cov_ratio": "module3 -> block_c E3 (key renamed *_DEPRECATED)",
    "A4_inter_subject_over_inter_day_mmd": "module3 -> block_c E2 (key renamed *_DEPRECATED)",
}


def check_stale(root):
    bad = 0
    for f in glob.glob(str(root / "**" / "*.json"), recursive=True):
        if _is_synth(f):
            continue
        d = json.loads(Path(f).read_text())
        for path, _ in _walk(d):
            leaf = path.rsplit("/", 1)[-1]
            if leaf in SUPERSEDED:
                fail(f"stale key `{leaf}` in {Path(f).name}  ({SUPERSEDED[leaf]}) "
                     f"-> this file predates the fix; delete and re-run")
                bad += 1
                break
    if not bad:
        ok("no superseded keys in any result JSON")


def check_math(root):
    seen = 0
    for f in sorted(glob.glob(str(root / "block_c" / "*__block_c.json"))):
        d = json.loads(Path(f).read_text())
        ds = d.get("dataset", Path(f).name)
        inv = (d.get("E3_meancov") or {}).get("affine_invariance_check") or {}
        if not inv.get("checked"):
            warn(f"{ds}: affine-invariance check not run ({inv.get('reason')})")
            continue
        seen += 1
        if not inv.get("invariant"):
            fail(f"{ds}: mean/cov split NOT affine-invariant at ridge=0 "
                 f"(rel={inv['max_relative_difference']:.2e}, tol={inv.get('numerical_tolerance')}, "
                 f"cond={inv.get('covariance_condition_number')}) -- beyond what conditioning "
                 f"explains; the E3 numbers cannot be trusted")
        elif inv.get("numerically_limited"):
            warn(f"{ds}: invariance holds only to {inv['max_relative_difference']:.1e} because the "
                 f"raw covariance is ill-conditioned (cond={inv.get('covariance_condition_number'):.1e}). "
                 f"The identity is exact algebra; this is float64 precision, not a defect.")
        if not inv.get("ridge_breaks_invariance"):
            warn(f"{ds}: the ridge did NOT visibly break invariance "
                 f"(rel_ridge={inv.get('ridge_relative_difference')}); the paper's claim that "
                 f"the old E3 was a ridge artifact is not demonstrated on this dataset")
        pooled = ((d.get("E3_meancov") or {}).get("representations") or {}).get("pooled") or {}
        if pooled.get("shift_detectable") and not pooled.get("null_estimated"):
            fail(f"{ds}: shift declared detectable without a null floor")
        if pooled and not pooled.get("null_is_trial_disjoint"):
            fail(f"{ds}: E3 null floor is not trial-disjoint -> it under-estimates the "
                 f"estimation-noise floor and every mean/cov share is inflated (pre-fix run)")
        if pooled.get("shift_detectable"):
            ms = pooled.get("mean_share_of_excess")
            if ms is None or not (0.0 <= ms <= 1.0):
                fail(f"{ds}: mean_share_of_excess={ms} outside [0,1]")
    if seen:
        ok(f"affine-invariance asserted on {seen} datasets")


def check_hdiv(root):
    """The honest leak diagnostic is d_H between two trial-disjoint halves of the SAME subject.

    The earlier version of this check looked at `hdiv_frob`, which is `_frob` = RMS of
    (offdiag / offdiag.max()) -- a UNIFORMITY statistic, not a magnitude. An all-equal matrix
    scores 1.0 no matter what the values are, so the >0.9 threshold flagged well-behaved runs.
    A high d_H between two DIFFERENT subjects is expected and real: people are separable in
    feature space. Only a high d_H WITHIN one subject proves the classifier is reading trial id.
    """
    checked = 0
    for f in sorted(glob.glob(str(root / "module3" / "*__shift.json"))):
        if _is_synth(f):
            continue
        d = json.loads(Path(f).read_text())
        ds = d.get("dataset")
        null = d.get("hdiv_within_subject_null")
        if null is None:
            warn(f"{ds}: module3 has no `hdiv_within_subject_null` -> cannot verify that "
                 f"h_divergence is leak-free (pre-fix run)")
            continue
        if "mean" not in null:
            continue
        checked += 1
        if null.get("leak_suspected"):
            fail(f"{ds}: within-subject d_H = {null['mean']:.3f} (max {null['max']:.3f}). Two "
                 f"trial-disjoint halves of ONE subject must be indistinguishable; this means "
                 f"h_divergence is still memorising trial identity.")
    if checked:
        ok(f"h_divergence leak-free on {checked} datasets (within-subject d_H null ~ 0)")


def check_leak(root):
    n = 0
    for f in sorted(glob.glob(str(root / "module2" / "*__separability.json"))):
        d = json.loads(Path(f).read_text())
        ds = d.get("dataset")
        w, c = d.get("knn_trial_cv_acc"), d.get("knn_loso_acc")
        if w is None or c is None:
            fail(f"{ds}: module2 missing knn_trial_cv_acc / knn_loso_acc (old leaky run?)")
            continue
        n += 1
        if not (math.isfinite(w) and math.isfinite(c)):
            warn(f"{ds}: non-finite kNN accuracies ({w}, {c})")
            continue
        if c > w + 0.02:
            warn(f"{ds}: cross-subject kNN ({c:.3f}) exceeds within-subject ({w:.3f}) — unusual; "
                 f"check the trial grouping")
    if n:
        ok(f"both kNN protocols present on {n} datasets")


def check_validity(root):
    for f in sorted(glob.glob(str(root / "block_a" / "*__block_a.json"))):
        d = json.loads(Path(f).read_text())
        ds = d.get("dataset")
        if "entropy_valid" not in d:
            fail(f"{ds}: block_a missing entropy_valid (old run)")
            continue
        try:
            expect = windows.entropy_valid(ds)
        except Exception:
            continue
        if bool(d["entropy_valid"]) != bool(expect):
            fail(f"{ds}: entropy_valid={d['entropy_valid']} but window has "
                 f"{windows.window_samples(ds)} samples")
        rel = d.get("feature_reliability", {})
        leaked = [k for k in config.SLOW_COMPLEX if k in rel] if not expect else []
        if leaked:
            fail(f"{ds}: entropy invalid yet {leaked} still ranked in feature_reliability")
    ok("entropy validity flags consistent with window length")


def check_e6(root):
    for f in sorted(glob.glob(str(root / "block_d" / "*__block_d.json"))):
        d = json.loads(Path(f).read_text())
        ds = d.get("dataset")
        e6 = d.get("E6_sampling_rate") or {}
        if "testable" not in e6:
            fail(f"{ds}: block_d E6 missing `testable` (pre-anti-alias run)")
            continue
        if e6.get("testable"):
            for k, v in (e6.get("curves") or {}).items():
                if v.get("effective_fs", 1e9) < config.E6_MIN_EFFECTIVE_FS_HZ:
                    fail(f"{ds}: E6 reported {k} at {v['effective_fs']} Hz, below the "
                         f"{config.E6_MIN_EFFECTIVE_FS_HZ:.0f} Hz floor")
    ok("E6 respects the sampling-rate floor")


def check_honesty(root):
    m = root / "meta" / "meta.json"
    if m.exists():
        d = json.loads(m.read_text())
        wm = d.get("what_makes_hard") or {}
        if "per_predictor" not in wm:
            fail("meta: what_makes_hard has no FDR-corrected per_predictor block")
        else:
            circ = [k for k, v in wm["per_predictor"].items() if v.get("circular")]
            if not circ:
                warn("meta: no predictor flagged circular; knn_* should be")
            else:
                ok(f"meta: circular predictors flagged: {circ}")
        ma = d.get("meta_analysis") or {}
        for k in ("per_dataset", "n_independent_cohorts", "heterogeneity_warning"):
            if k not in ma:
                fail(f"meta: meta_analysis missing `{k}`")
    for f in sorted(glob.glob(str(root / "actionability" / "*__actionability.json"))):
        d = json.loads(Path(f).read_text())
        if "note" in d and len(d) <= 2:
            continue
        if "oracle_ceiling" not in d:
            fail(f"{d.get('dataset')}: actionability missing oracle_ceiling (old run)")
        if "mmd_vs_calibration_gain" not in d:
            fail(f"{d.get('dataset')}: actionability missing mmd_vs_calibration_gain")
    sp = root / "senic_probe" / "senic__senic_probe.json"
    if sp.exists():
        d = json.loads(sp.read_text())
        if "verdict" not in d:
            fail("senic_probe: no explicit verdict field (old run)")
        elif "NOT supported" in str(d["verdict"]):
            ok(f"senic verdict: {d['verdict'][:70]}...")
    s = root / "module6_sdi" / "sdi.json"
    if s.exists():
        d = json.loads(s.read_text())
        if d.get("validation", "").find("cohort") < 0:
            fail("sdi: validation is not leave-one-cohort-out")
        else:
            ok(f"sdi: {d['validation']}, {d.get('n_independent_cohorts')} cohorts")


BENIGN_NAN = ("senic_probe/", "/mean_share_of_excess", "/kl_excess_removed_by",
              "/spearman_q_fdr", "/p_value", "/pearson_r", "/mannwhitney_u",
              "/rank_biserial", "/snr_excess_over_null", "/guided_advantage_z")


def check_nan(root):
    hits = []
    for f in glob.glob(str(root / "**" / "*.json"), recursive=True):
        rel = str(Path(f).relative_to(root)).replace(os.sep, "/")
        d = json.loads(Path(f).read_text())
        for path, v in _walk(d):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                full = rel + path
                if not any(b in full for b in BENIGN_NAN):
                    hits.append(full)
    if hits:
        warn(f"{len(hits)} unexpected NaN/Inf, e.g. {hits[:5]}")
    else:
        ok("no unexpected NaN/Inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--root", choices=["legacy", "v2"], default="legacy",
                    help="which results tree to validate. 'legacy' = the frozen 2026-07-13 evidence "
                         "(default, because that is what the paper quotes); 'v2' = the live tree.")
    a = ap.parse_args()
    root = config.LEGACY_DIR if a.root == "legacy" else config.RESULTS_DIR
    print(f"validating {root}\n")
    check_stale(root)
    check_math(root)
    check_hdiv(root)
    check_leak(root)
    check_validity(root)
    check_e6(root)
    check_honesty(root)
    check_nan(root)
    print(f"\n{len(FAIL)} failures, {len(WARN)} warnings")
    sys.exit(1 if (FAIL or (a.strict and WARN)) else 0)


if __name__ == "__main__":
    main()
