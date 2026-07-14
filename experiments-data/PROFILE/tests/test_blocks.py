"""End-to-end tests for Phase-2 Block modules (A-F), on a SYNTHETIC frame.

Validates input/output shapes, value ranges, and math properties WITHOUT running the heavy
experiments on real data. A synthetic frame with separable classes + subject variation + repetition
structure is injected via monkeypatching windows.build_fast_frame / build_complex_frame.

Run:  python tests/test_blocks.py
"""
from __future__ import annotations

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tests never write into a real results tree (CLAUDE.md rule 1)
import os as _os, pathlib as _pl
_os.environ.setdefault("PROFILE_RESULTS_DIR",
                       str(_pl.Path(__file__).resolve().parents[1] / "results" / "_test_sandbox"))

import numpy as np
import pandas as pd

from dsprofile import (config, windows, block_a, block_b, block_c, block_d,
                       calibration, transfer, faabos, senic_probe,
                       robust_difficulty, actionability)

R = []
def check(name, cond, detail=""):
    R.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name}   {detail}")

FEATS = sorted(set(config.REPR_BASIS + block_a.AMPLITUDE + block_a.COMPLEX +
                   ["MNF", "MDF"]))
# deterministic per-feature weight (NOT Python hash(), which is randomised per process -> would
# make the synthetic data non-reproducible across machines). Index-based, stable everywhere.
FEAT_W = {f: (i % 5 + 1) / 5.0 for i, f in enumerate(FEATS)}


def synth_frame(n_subj=6, n_class=4, n_rep=3, n_sess=1, per=12, n_ch=2, seed=0):
    """Separable classes (class-dependent feature mean) + subject offsets + rep noise + 2 special
    reliability-probe columns (REL_HIGH = iid across reps; REL_LOW = constant within a rep)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        for sess in range(n_sess):
            for c in range(n_class):
                for r in range(n_rep):
                    for _ in range(per):
                        row = dict(subject=s, session=sess, repetition=r, label=c)
                        for f in FEATS:
                            for ch in range(n_ch):
                                fw = FEAT_W[f]                          # deterministic, reproducible
                                # every feature carries class signal (mult in [1.2,2.0]) so the
                                # REPR_BASIS is cleanly separable; modest subject offset + noise
                                row[f"{f}_c{ch}"] = (c * (1.2 + 0.8 * fw) + s * 0.3 + rng.normal(0, 0.4))
                        for ch in range(n_ch):
                            row[f"REL_HIGH_c{ch}"] = rng.normal(0, 1)             # iid every window
                            row[f"REL_LOW_c{ch}"] = float(r) + 0.001 * rng.normal()  # ~constant per rep
                        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["dataset"] = "synth"; df.attrs["is_envelope"] = False; df.attrs["n_channels"] = n_ch
    return df


def patch(frame):
    windows.build_fast_frame = lambda *a, **k: frame
    windows.build_complex_frame = lambda *a, **k: frame


def test_block_a():
    patch(synth_frame(seed=1))
    r = block_a.run("synth")
    rel = r["feature_reliability"]
    check("A: reliability values in [0,1]", all(0 <= v <= 1 for v in rel.values()))
    check("A: REL_HIGH more reliable than REL_LOW",
          rel.get("REL_HIGH", 0) > rel.get("REL_LOW", 1),
          f"high={rel.get('REL_HIGH'):.2f} low={rel.get('REL_LOW'):.2f}")
    ci = r["complexity_adds_info"]
    check("A: complexity MI values finite & >=0",
          all(np.isfinite(ci[k]) and ci[k] >= 0 for k in ["amplitude_mi", "complexity_raw_mi", "complexity_residual_mi"]))


def test_block_b():
    patch(synth_frame(seed=2))
    r = block_b.run("synth")
    rec = r["B1_per_class_difficulty"]["per_class_recall"]
    check("B: per-class recall in [0,1]", all(0 <= v <= 1 for v in rec.values()))
    # 4 classes -> chance = 0.25; cleanly-separable synth should clear it comfortably (deterministic)
    check("B: separable synth -> mean recall > chance", r["B1_per_class_difficulty"]["mean_recall"] > 0.5,
          f"got {r['B1_per_class_difficulty']['mean_recall']:.2f} (chance 0.25)")
    disp = r["B2_class_subject_invariance"]["class_subject_dispersion"]
    check("B: dispersion >= 0 & finite", all(np.isfinite(v) and v >= 0 for v in disp.values()))


def test_block_c():
    patch(synth_frame(n_sess=2, seed=3))
    r = block_c.run("synth", n_jobs=1)
    a4 = r["E2_a4_fair"]
    check("C/E2: A4 ratio finite & positive",
          np.isfinite(a4["ratio_inter_subject_over_inter_day"]) and a4["ratio_inter_subject_over_inter_day"] > 0)
    mc = r["E3_meancov"]
    reps = mc["representations"]
    check("C/E3: mean & cov terms finite for all three representations",
          all(np.isfinite(reps[k]["mean_term"]) and np.isfinite(reps[k]["cov_term"])
              for k in ("pooled", "subject_center", "subject_zscore")))
    check("C/E3: affine-invariance asserted at ridge=0",
          mc["affine_invariance_check"].get("invariant") is True,
          f"rel={mc['affine_invariance_check'].get('max_relative_difference'):.1e}")
    # Per-subject centering removes the mean SHIFT, but each group is a random subsample, so its
    # raw mean_term still equals the sampling-noise floor. The null-corrected EXCESS is the
    # quantity that must vanish.
    check("C/E3: per-subject centering drives the mean-term EXCESS to 0",
          reps["subject_center"]["mean_term_excess"] == 0.0,
          f"excess={reps['subject_center']['mean_term_excess']:.2e} "
          f"(raw={reps['subject_center']['mean_term']:.3f} ~ null={reps['subject_center']['null_mean_term']:.3f})")
    check("C/E3: synth has subject offsets -> pooled shift is mean-dominated",
          reps["pooled"]["shift_detectable"] and reps["pooled"]["mean_share_of_excess"] > 0.8,
          f"share={reps['pooled']['mean_share_of_excess']:.3f}")
    check("C/E3: centering removes essentially all detectable divergence",
          reps["subject_center"]["shift_detectable"] is False)
    check("C/E3: shares are only reported when a shift is detectable",
          (reps["pooled"]["shift_detectable"]
           and 0 <= reps["pooled"]["mean_share_of_excess"] <= 1)
          or (not reps["pooled"]["shift_detectable"]
              and not np.isfinite(reps["pooled"]["mean_share_of_excess"])))
    cd = r["E4_conditional_disparity"]
    check("C/E4: conditional disparity in [0,1]",
          0 <= cd["conditional_disparity_frob"] <= 1 and 0 <= cd["mean_disagreement"] <= 1)


def test_block_d():
    patch(synth_frame(seed=4))
    r = block_d.run("synth")
    cr = r["E7_channel_reduction"]
    check("D/E7: full accuracy in [0,1]", 0 <= cr["full_accuracy"] <= 1)
    check("D/E7: accuracy_vs_k all in [0,1]", all(0 <= v <= 1 for v in cr["accuracy_vs_k"].values()))
    check("D/E7: min_channels <= n_channels", cr["min_channels_for_95pct"] <= cr["n_channels"])
    check("D/E7: LOSO curve reported alongside the within-subject one",
          "accuracy_vs_k_loso" in cr and "min_channels_for_95pct_loso" in cr)
    check("D/E7: cross-subject accuracy <= within-subject",
          cr["full_accuracy_loso"] <= cr["full_accuracy"] + 0.05,
          f"loso={cr['full_accuracy_loso']:.3f} within={cr['full_accuracy']:.3f}")
    sr = r["E6_sampling_rate"]
    check("D/E6: reports whether sufficiency is testable at all", "testable" in sr)
    if sr.get("testable"):
        check("D/E6: retained_frac present & finite",
              all(np.isfinite(v.get("retained_frac", np.nan)) for v in sr["curves"].values()))
        check("D/E6: never decimates below the effective-fs floor",
              all(v["effective_fs"] >= config.E6_MIN_EFFECTIVE_FS_HZ
                  for v in sr["curves"].values()))
    else:
        check("D/E6: untestable datasets are skipped, not reported", "note" in sr, sr.get("note", "")[:50])


def test_calibration():
    patch(synth_frame(seed=5))
    r = calibration.run("synth")
    cal = r["E8_calibration"]
    vals = [v for v in cal["accuracy_vs_k"].values() if v is not None]
    check("E8: calibration accuracies in [0,1]", all(0 <= v <= 1 for v in vals))
    check("E8: one-shot >= zero-shot (calibration helps)",
          cal["one_shot"] is None or cal["zero_shot"] is None or cal["one_shot"] >= cal["zero_shot"] - 0.05,
          f"zs={cal['zero_shot']} os={cal['one_shot']}")


def test_transfer_unit():
    V = transfer._shared_vector(synth_frame(seed=6))
    check("F/transfer: shared vector shape = (n_windows, |REPR_BASIS|)",
          V.shape[1] == len(config.REPR_BASIS) and np.isfinite(V).all(),
          f"shape {V.shape}")


def test_faabos_graceful():
    patch(synth_frame(seed=7))
    r = faabos.run("synth")   # no manifest on disk for 'synth' -> graceful skip
    check("E9: faabos graceful skip when no faabos column", "note" in r)


def test_robust_difficulty():
    patch(synth_frame(n_subj=6, seed=8))
    r = robust_difficulty.run("synth", n_jobs=1)
    ml = r["mean_loso_acc"]
    check("robust: mean LOSO acc per clf in [0,1]", all(0 <= v <= 1 for v in ml.values()),
          f"{ {k: round(v,2) for k,v in ml.items()} }")
    check("robust: 3 classifiers present", set(ml) == {"lda", "svm", "rf"})
    ag = r["inter_classifier_agreement"]
    check("robust: agreement spearman in [-1,1]",
          all(-1 <= v["spearman"] <= 1 for v in ag.values()))
    dc = r["difficulty_prediction_by_classifier"]
    check("robust: difficulty corr computed per clf, r in [-1,1]",
          all(-1 <= v["pearson_r"] <= 1 for v in dc.values()))


def test_actionability():
    patch(synth_frame(n_subj=8, n_rep=4, seed=9))
    r = actionability.run("synth", kmax=3, n_jobs=1)
    if "note" in r:
        check("action: produced curves", False, r["note"]); return
    for key in ["mean_acc_guided", "mean_acc_random", "mean_acc_oracle"]:
        check(f"action: {key} in [0,1]", all(0 <= v <= 1 for v in r[key]))
    check("action: AUCs finite & in [0,1]",
          all(0 <= r[k] <= 1 for k in ["auc_guided", "auc_random", "auc_oracle"]))
    check("action: oracle AUC >= random AUC (oracle is upper bound)",
          r["auc_oracle"] >= r["auc_random"] - 1e-6,
          f"oracle={r['auc_oracle']:.3f} random={r['auc_random']:.3f}")


def test_shapes_no_nan():
    """Every produced JSON parses; NaN only where it is a deliberate 'undefined' marker.

    `mean_share_of_excess` is NaN by design when no shift is detectable above the estimation-
    noise floor — reporting a number there would be a ratio of noise to noise. That is the one
    sanctioned NaN; anything else is a bug.
    """
    import glob, math

    def nan_paths(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from nan_paths(v, f"{p}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from nan_paths(v, f"{p}[{i}]")
        elif isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            yield p

    ALLOWED = ("/mean_share_of_excess",)
    bad = []
    for f in glob.glob(str(config.RESULTS_DIR / "block_*" / "synth__*.json")) + \
             glob.glob(str(config.RESULTS_DIR / "calibration" / "synth__*.json")) + \
             glob.glob(str(config.RESULTS_DIR / "faabos" / "synth__*.json")):
        d = json.loads(open(f).read())
        for p in nan_paths(d):
            if not any(p.endswith(a) for a in ALLOWED):
                bad.append(f"{os.path.basename(f)}{p}")
    check("outputs: no UNSANCTIONED NaN/Inf in block JSONs", not bad, str(bad[:4]))


def main():
    for fn in [test_block_a, test_block_b, test_block_c, test_block_d, test_calibration,
               test_transfer_unit, test_faabos_graceful, test_robust_difficulty,
               test_actionability, test_shapes_no_nan]:
        try:
            fn()
        except Exception as e:
            import traceback; check(fn.__name__, False, f"EXC {type(e).__name__}: {e}"); traceback.print_exc()
    print(f"\n==== {sum(R)}/{len(R)} checks passed ====")
    sys.exit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
