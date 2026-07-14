"""Build THE ledger — one Excel file that records every experiment result in the repository.

    python -m cli.build_ledger              # -> results/LEDGER.xlsx

Why this exists: the audit of 2026-07-13 found that the project's own tracking documents misquoted
its own results in five places — not because anyone was careless, but because the numbers lived in
480 JSON files across 30 folders and two code generations, and nobody could hold them in their head.
The ledger removes the need to. It is generated FROM THE FILES, never typed by hand, so it cannot
drift from the evidence. If a number is not in the ledger, it is not a result.

Sheets
------
  INDEX      one row per experiment: what it is, which tree it lives in, how many datasets it
             covers, when it ran, and (for T-suite experiments) which pre-registered branch it landed
             on with the verdict sentence.
  RESULTS    one row per (experiment, dataset): the headline numbers, flattened. This is the sheet
             you filter and sort.
  RUNLOG     the append-only run log (logs/tsuite_runs.jsonl): every run attempt, its status, and how
             long it took. The audit trail behind every row in RESULTS.
  READ_ME    the reading traps: places where a flag in a JSON disagrees with the numbers in the same
             JSON, and which one to trust.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from dsprofile import config  # noqa: E402

# Human-readable name + the question each experiment answers.
CATALOGUE = {
    "module1": "Signal-level characterisation (per dataset/channel/class)",
    "module2": "Class separability — within-subject vs cross-subject (the generalisation gap)",
    "module3": "Distribution shift between subjects (MMD / H-div / Gaussian-KL)",
    "module4": "Channel & sampling-rate analysis (minimal montage, fs sufficiency)",
    "module5": "Subject-difficulty predictor (MMD -> LOSO accuracy)",
    "module6_sdi": "SDI — portable cross-dataset difficulty index (leave-one-cohort-out)",
    "block_a": "Feature reliability + does complexity add information?",
    "block_b": "Per-class difficulty; does class difficulty transfer across subjects?",
    "block_c": "E2 inter-subject vs inter-day; E3 mean-vs-covariance (affine-invariance proof)",
    "block_d": "Channel reduction (E7) + sampling-rate sufficiency (E6)",
    "calibration": "Accuracy vs number of calibration repetitions",
    "robust_difficulty": "Is subject difficulty classifier-agnostic? (LDA/SVM/RF)",
    "actionability": "Can the difficulty signal guide a calibration budget? (oracle ceiling)",
    "faabos": "ADL taxonomy: coarse FAABOS groups vs fine gestures (emaha_db1 only)",
    "senic_probe": "Is senic's sign reversal a session-count confound?",
    "transfer": "Cross-dataset compatibility matrix (MMD between datasets)",
    "meta": "Random-effects meta-analysis + what makes a dataset hard",
    "floor_effect_x1": "X1 — is the predictor real, or just distance from the accuracy floor?",
    "x2": "X2 — is the predictor/target link a shared-representation artifact?",
    "x4": "X4 — CORAL (covariance alignment) vs mean-centring",
    "x5": "X5 — is 'shift' just contraction amplitude?",
    "x6": "X6 — does the result replicate in a learned (PCA/RFF) representation?",
    "x7": "X7 — is the result an MMD-bandwidth artifact?",
    "x8": "X8 — is MMD the best cheap out-of-distribution score?",
    "x9": "X9 — does the compatibility matrix predict real cross-dataset transfer?",
    "x10": "X10 — senic / electrode-shift condition axis",
    "x11": "X11 — which dataset properties moderate the effect? (meta-regression)",
    "x12": "X12 — are the subsample caps converged?",
    "x13": "X13 — does centring hurt imbalanced subjects? (UNTESTABLE on this panel)",
    "x14": "X14 — adaptive-LDA calibration curve (accuracy vs k reps)",
    "x15": "X15 — coverage-guaranteed difficulty intervals (split-conformal)",
    "experiments": "Add-ons A-E (window length / recalibration / cross-session / ranking / permutation)",
    "t1_model_family": "T1 — does the predictor work for ANY learner, or only a linear one?",
    "t2_senic_rootcause": "T2 — WHY does senic reverse the sign?",
    "t3_moment_ladder": "T3 — WHICH moment of a subject's distribution must be aligned?",
    "t4_adl_granularity": "T4 — do coarse ADL categories buy real cross-subject robustness?",
    "t5_transfer_after_alignment": "T5 — does alignment rescue the failed cross-dataset transfer?",
    "t6_imbalance_induced": "T6 — does centring break when the new user's data is skewed?",
    "t7_seed_robustness": "T7 — is any of this stable across random seeds?",
    "t8_calibration_budget": "T8 — how many SECONDS of unlabelled data does a new user need?",
    "t9_feature_families": "T9 — which handcrafted feature families survive CROSS-SUBJECT?",
    "t10_rest_class_inflation": "T10 — how much accuracy does the REST class manufacture?",
    "t11_subject_scaling": "T11 — does collecting MORE USERS buy cross-subject accuracy?",
}

# Places where a JSON's own flag disagrees with its own numbers. Verified 2026-07-13.
READ_ME = [
    ("x5", "shift_is_amplitude_dominated says 13/14, but shape_invariant.difficulty_r stays "
           "negative on only 10/14. Trust the number, not the flag."),
    ("x6", "replicates_across_representations is true on 10/14, not 11/14 as STATE.md claimed."),
    ("x9", "transfer accuracy is below chance on 2 of 4 pairs, not 3 of 4. The r = -0.85 has n=4, "
           "p=0.15 -> uninterpretable, do not quote."),
    ("x15", "coverage is valid (0.908) but conformal_better_calibrated is FALSE — the plain "
            "Gaussian interval does just as well (0.903)."),
    ("x11", "read the coefficients, not the prose. The outcome is Fisher-z of a NEGATIVE r, so a "
            "POSITIVE coefficient means a WEAKER predictor."),
    ("legacy_v1/module4", "produced 2026-07-09, before the F-fixes. Verified independent of them "
                          "(uses sklearn's own NMI), so the numbers stand — but say so if asked."),
    ("legacy_v1/calibration", "produced 2026-07-10, before the F-fixes. Verified independent."),
]


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# Keys whose VALUES are big nested dicts (per-subject, per-seed, per-pair, per-cohort ...). They are
# the raw material, not the summary: including them exploded the RESULTS sheet to 1505 columns, which
# is not a ledger, it is a dump. The JSON remains the source of truth for anything in this list.
_BULK_KEYS = {
    "per_subject_acc", "per_subject", "predictor_mmd", "per_class_recall", "curve", "curves",
    "rows", "per_dataset", "cohort_map", "atlas", "per_seed", "per_pair", "per_model", "per_score",
    "per_cohort", "per_rung", "per_metric", "per_window", "representations", "feature_reliability",
    "mi_su_top", "accuracy_vs_k", "session_counts", "H3_within_condition", "meta_analysis",
    "subject_difficulty_agreement", "lodo_per_dataset", "what_makes_hard", "how_many_subjects",
    "single_predictor_pooled", "combination_lodo_mean_spearman", "predictors", "predictor_correlations",
    "compatibility_mmd", "shape_only_mmd", "mrmr_ranking", "excluded_features", "weights",
    "accuracy_vs_k_loso", "inter_classifier_agreement", "difficulty_prediction_by_classifier",
}


def _flatten(d, prefix="", depth=0, max_depth=2):
    """Scalar leaves only, to a bounded depth: the ledger is a summary, not a dump."""
    out = {}
    if depth > max_depth or not isinstance(d, dict):
        return out
    for k, v in d.items():
        if k in _BULK_KEYS:
            continue
        key = f"{prefix}{k}"
        if _num(v) or isinstance(v, (str, bool)) or v is None:
            out[key] = v
        elif isinstance(v, dict):
            out.update(_flatten(v, f"{key}.", depth + 1, max_depth))
    return out


def scan_tree(root: Path, tree_name: str):
    """One row per (experiment, result file) across a results tree."""
    index, results = [], []
    if not root.exists():
        return index, results
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if sub.name.startswith("_"):
            continue
        files = sorted(sub.glob("*.json"))
        if not files:
            continue
        mtimes = [f.stat().st_mtime for f in files]
        pooled = next((f for f in files if f.name in ("pooled.json", "meta.json", "sdi.json",
                                                      "transfer.json", "conformal.json",
                                                      "meta_regression.json")), None)
        verdict = branch = ""
        if pooled:
            try:
                pj = json.loads(pooled.read_text(encoding="utf-8"))
                verdict = str(pj.get("verdict", ""))[:500]
                branch = str(pj.get("branch", ""))
            except Exception:
                pass
        index.append(dict(
            experiment=sub.name, tree=tree_name,
            question=CATALOGUE.get(sub.name, "(uncatalogued — add it to cli/build_ledger.py)"),
            n_result_files=len(files),
            first_run=dt.datetime.fromtimestamp(min(mtimes)).strftime("%Y-%m-%d %H:%M"),
            last_run=dt.datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M"),
            branch=branch, verdict=verdict,
            path=str(sub.relative_to(config.RESULTS_ROOT)),
        ))
        for f in files:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                results.append(dict(experiment=sub.name, tree=tree_name, file=f.name,
                                    dataset="?", error=f"unreadable: {e}"))
                continue
            if not isinstance(d, dict):
                continue
            ds = d.get("dataset") or d.get("source") or f.stem.split("__")[0]
            row = dict(experiment=sub.name, tree=tree_name, dataset=ds, file=f.name,
                       modified=dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))
            row.update(_flatten(d))
            results.append(row)
    return index, results


def main():
    idx, res = [], []
    for root, name in ((config.LEGACY_DIR, "legacy_v1 (frozen)"), (config.RESULTS_DIR, "v2 (live)")):
        i, r = scan_tree(root, name)
        idx += i
        res += r

    runlog = []
    p = config.PROFILE_DIR / "logs" / "tsuite_runs.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                runlog.append(json.loads(line))
            except Exception:
                continue

    out = config.RESULTS_ROOT / "LEDGER.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        pd.DataFrame(idx).to_excel(xl, sheet_name="INDEX", index=False)
        df = pd.DataFrame(res)
        # stable, readable column order: identity first, then everything else alphabetically
        lead = [c for c in ("experiment", "tree", "dataset", "file", "modified") if c in df.columns]
        df = df[lead + sorted(c for c in df.columns if c not in lead)]
        df.to_excel(xl, sheet_name="RESULTS", index=False)
        pd.DataFrame(runlog or [{"note": "no T-suite runs logged yet"}]).to_excel(
            xl, sheet_name="RUNLOG", index=False)
        pd.DataFrame(READ_ME, columns=["where", "trap"]).to_excel(xl, sheet_name="READ_ME", index=False)

    print(f"ledger -> {out}")
    print(f"  {len(idx)} experiments, {len(res)} result rows, {len(runlog)} logged runs")
    missing = [i["experiment"] for i in idx if i["experiment"] not in CATALOGUE]
    if missing:
        print(f"  [!] uncatalogued experiments (add a one-line question to CATALOGUE): {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
