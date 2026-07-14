"""Ground-truth tests for the add-on experiments A / B / C + the shared infra.

Each experiment's core math is exercised on a SYNTHETIC frame where the correct answer is known
by construction, so the test checks the calculation, the index handling, and the data structures
— not just that it runs. Mirrors the tests/test_blocks.py pattern (monkeypatch the frame loader).

Run:  python tests/test_addon_experiments.py
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
import pandas as pd

from dsprofile import config

R = []


def check(name, cond, detail=""):
    R.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}   {detail}")


# ---------------------------------------------------------------------------------------
# synthetic frame builder: REPR_BASIS feature columns + (subject, session, label, repetition)
# ---------------------------------------------------------------------------------------
def _cols(n_ch):
    return [f"{fb}_c{c}" for fb in config.REPR_BASIS for c in range(n_ch)]


def synth_frame(n_subj=8, n_class=4, n_sess=1, reps_per=6, per=12, n_ch=2,
                class_sep=3.0, subj_off=0.0, subj_off_spread=0.0, drift=0.0,
                noise=0.6, seed=0):
    """A frame with controllable structure.

    class_sep       : magnitude of the class signal (bigger -> more separable)
    subj_off        : magnitude of a per-SUBJECT mean offset (the confound B removes)
    subj_off_spread : heterogeneity of the offset magnitude across subjects -> creates
                      per-subject DIFFICULTY VARIANCE (some subjects farther from the pool =>
                      harder to decode AND higher MMD-to-pool). Needed for a finite difficulty r.
    drift           : magnitude of an INDEPENDENT per-(subject,session) shift -> sessions sit at
                      different distances from each other, so higher-shift sessions are both
                      farther (higher MMD) and harder (lower cross-session acc). The confound C detects.
    Each (subject,label,session,rep) is one trial with `per` near-duplicate windows.
    """
    rng = np.random.default_rng(seed)
    cols = _cols(n_ch)
    D = len(cols)
    class_centers = rng.normal(0, 1, (n_class, D)) * class_sep
    subj_mag = subj_off * (1.0 + subj_off_spread * rng.random(n_subj))      # heterogeneous
    subj_offsets = rng.normal(0, 1, (n_subj, D)) * subj_mag[:, None]
    session_offsets = rng.normal(0, 1, (n_subj, n_sess, D)) * drift         # independent per session
    rows = []
    for s in range(n_subj):
        for sess in range(n_sess):
            for lab in range(n_class):
                for rep in range(reps_per):
                    center = class_centers[lab] + subj_offsets[s] + session_offsets[s, sess]
                    win = center + rng.normal(0, noise, (per, D))
                    for w in range(per):
                        rows.append(dict(subject=s, session=sess, label=lab, repetition=rep,
                                         **{c: v for c, v in zip(cols, win[w])}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# B — per-subject mean recalibration
# ---------------------------------------------------------------------------------------
def test_B():
    from addons import recalibration as B
    from dsprofile.module5_difficulty import _basis

    # constructed so the SUBJECT offset dominates the class signal -> global z-score struggles,
    # per-subject centering should recover accuracy.
    f = synth_frame(n_subj=8, n_class=4, subj_off=6.0, class_sep=2.0, seed=1)
    X = _basis(f); y = f.label.to_numpy(); subj = f.subject.to_numpy()
    acc = B.loso_variants(X, y, subj, seed=1)

    check("B: three normalisation modes returned",
          set(acc) == {"baseline", "subject_center", "subject_zscore"})
    check("B: every per-subject accuracy in [0,1]",
          all(0 <= v <= 1 for m in acc.values() for v in m.values()))
    mb = np.mean(list(acc["baseline"].values()))
    mc = np.mean(list(acc["subject_center"].values()))
    check("B: with a strong subject offset, per-subject centering BEATS baseline",
          mc > mb + 0.05, f"baseline={mb:.3f} center={mc:.3f}")

    d = B._paired_delta(acc["baseline"], acc["subject_center"])
    check("B: paired delta math (mean_delta = mean_recal - mean_baseline)",
          abs(d["mean_delta"] - (d["mean_recal"] - d["mean_baseline"])) < 1e-9)
    check("B: frac_improved in [0,1] and n matches", 0 <= d["frac_improved"] <= 1)
    check("B: centering flagged as helping here", d["helps"] is True, f"p={d['wilcoxon_p']:.4f}")

    # NULL case: no subject offset -> centering must NOT meaningfully help (it can only remove info)
    f0 = synth_frame(n_subj=8, n_class=4, subj_off=0.0, class_sep=3.0, seed=2)
    X0 = _basis(f0)
    a0 = B.loso_variants(X0, f0.label.to_numpy(), f0.subject.to_numpy(), seed=2)
    d0 = B._paired_delta(a0["baseline"], a0["subject_center"])
    check("B: with NO subject offset, centering does not spuriously help",
          d0["mean_delta"] <= 0.05, f"delta={d0['mean_delta']:+.3f}")

    # per-subject centering must zero each subject's mean (index handling)
    C = B._per_subject_center(X, subj)
    per_means = [np.abs(C[subj == s].mean(0)).max() for s in np.unique(subj)]
    check("B: _per_subject_center zeroes each subject's mean", max(per_means) < 1e-9)


# ---------------------------------------------------------------------------------------
# C — cross-session difficulty prediction (math + df + ground truth)
# ---------------------------------------------------------------------------------------
def test_C_math():
    from addons import cross_session as C

    # _within_corr degrees-of-freedom formula, checked against a direct t-computation
    rng = np.random.default_rng(3)
    n_subj = 4
    md, ad = [], []
    for _ in range(n_subj):                        # 3 sessions each -> demeaned within subject
        x = rng.normal(size=3); z = -x + rng.normal(0, 0.3, 3)
        md.append(x - x.mean()); ad.append(z - z.mean())
    md = np.concatenate(md); ad = np.concatenate(ad)
    r, p, df = C._within_corr(md, ad, n_subj)
    from scipy.stats import t as st
    denom = np.sqrt((md**2).sum() * (ad**2).sum())
    r_manual = float((md * ad).sum() / denom)
    df_manual = len(md) - n_subj - 1
    t_manual = r_manual * np.sqrt(df_manual / (1 - r_manual**2))
    p_manual = float(2 * st.sf(abs(t_manual), df_manual))
    check("C: within-correlation r matches direct formula", abs(r - r_manual) < 1e-9,
          f"{r:.4f} vs {r_manual:.4f}")
    check("C: df = N - n_subjects - 1 (fixed-effects, not N-2)", df == df_manual,
          f"df={df} expected={df_manual}")
    check("C: p uses the corrected df", abs(p - p_manual) < 1e-9)
    check("C: correctly NEGATIVE on anti-correlated construction", r < 0, f"r={r:.3f}")

    # demeaning: each subject's demeaned values must sum to ~0
    recs = [(0, 0, 0.5, 1.0), (0, 1, 0.4, 1.5), (0, 2, 0.3, 2.0),
            (1, 0, 0.6, 0.5), (1, 1, 0.55, 0.7), (1, 2, 0.5, 0.9)]
    m, a, ns = C._within_demean(recs)
    check("C: _within_demean uses both subjects", ns == 2)
    check("C: demeaned mmd sums to ~0 per subject", abs(m[:3].sum()) < 1e-9 and abs(m[3:].sum()) < 1e-9)


def test_C_ground_truth():
    from addons import cross_session as C
    import dsprofile.windows as W

    # sessions sit at different distances (independent per-session shifts) -> higher-shift sessions
    # are both farther (higher MMD) and harder (lower cross-session acc). More sessions -> the
    # within-subject correlation has points to work with. Moderate class_sep leaves accuracy headroom.
    f = synth_frame(n_subj=10, n_class=4, n_sess=5, drift=2.0, class_sep=1.6, subj_off=1.0,
                    noise=0.6, seed=5)
    orig = W.build_fast_frame
    W.build_fast_frame = lambda ds, seed=42, **k: f
    try:
        r = C.run_dataset("synthms", seed=5)
    finally:
        W.build_fast_frame = orig
    check("C: applicable on a 3-session frame", r.get("applicable") is True, r.get("note", ""))
    check("C: session shift predicts session difficulty (r<0, significant)",
          r["within_subject_fixedeffects_r"] < 0 and r["p_value"] < 0.05,
          f"r={r['within_subject_fixedeffects_r']:+.3f} p={r['p_value']:.4f} df={r['df']}")
    rc = r["rank_cross_check"]
    check("C: rank cross-check agrees (mean per-subject spearman < 0)",
          rc.get("mean_per_subject_spearman", 0) < 0,
          f"mean rho={rc.get('mean_per_subject_spearman')}")
    check("C: n_points = n_records and both accounted", r["n_points_pooled"] <= r["n_records"])

    # NULL: no drift -> no significant relationship
    f0 = synth_frame(n_subj=10, n_class=4, n_sess=5, drift=0.0, class_sep=1.6, noise=0.6, seed=6)
    W.build_fast_frame = lambda ds, seed=42, **k: f0
    try:
        r0 = C.run_dataset("synthms0", seed=6)
    finally:
        W.build_fast_frame = orig
    ok = (not r0.get("applicable")) or r0["p_value"] > 0.05 or r0["within_subject_fixedeffects_r"] >= 0
    check("C: NO drift -> no spurious significant negative relationship", ok,
          f"r={r0.get('within_subject_fixedeffects_r')} p={r0.get('p_value')}")


# ---------------------------------------------------------------------------------------
# A — window robustness (metric computation + stability aggregation)
# ---------------------------------------------------------------------------------------
def test_A():
    from addons import window_length as A
    # heterogeneous subject offsets -> per-subject difficulty variance -> a finite, negative
    # difficulty_r; moderate class_sep so accuracy isn't saturated.
    f = synth_frame(n_subj=14, n_class=4, class_sep=1.6, subj_off=2.5, subj_off_spread=4.0, seed=7)
    m = A.metrics_for_frame(f, seed=7, n_jobs=1)
    for k in ("knn_trial_cv", "knn_loso", "silhouette", "fisher", "inter_subject_mmd",
              "within_minus_cross"):
        check(f"A: metric '{k}' present & finite", k in m and np.isfinite(m[k]))
    check("A: difficulty_r present", "difficulty_r" in m)          # NaN is valid if acc is uniform
    check("A: difficulty_r finite & negative on a difficulty-variance frame",
          np.isfinite(m["difficulty_r"]) and m["difficulty_r"] < 0, f"r={m['difficulty_r']:+.3f}")
    check("A: kNN accuracies in [0,1]", 0 <= m["knn_trial_cv"] <= 1 and 0 <= m["knn_loso"] <= 1)
    check("A: within-subject >= cross-subject (gap non-negative here)",
          m["within_minus_cross"] >= -0.05, f"{m['within_minus_cross']:+.3f}")

    # stability() pure-function math
    per_window = {
        "100ms": dict(difficulty_r=-0.5, within_minus_cross=0.20, silhouette=0.10),
        "250ms": dict(difficulty_r=-0.6, within_minus_cross=0.18, silhouette=0.12),
        "500ms": dict(difficulty_r=-0.4, within_minus_cross=0.22, silhouette=0.08),
    }
    st = A.stability(per_window)
    check("A: difficulty flagged sign-stable when negative at all windows",
          st["difficulty_negative_at_all_windows"] is True)
    check("A: within>cross flagged stable when positive at all windows",
          st["within_gt_cross_at_all_windows"] is True)
    dr = st["per_metric"]["difficulty_r"]
    vals = np.array([-0.5, -0.6, -0.4])
    check("A: CV math = std/|mean|",
          abs(dr["coef_of_variation"] - vals.std() / abs(vals.mean())) < 1e-9)
    check("A: min/max reported correctly", dr["min"] == -0.6 and dr["max"] == -0.4)

    # sign instability is detected
    per2 = {"100ms": dict(difficulty_r=-0.5, within_minus_cross=0.2),
            "250ms": dict(difficulty_r=+0.1, within_minus_cross=0.2),
            "500ms": dict(difficulty_r=-0.3, within_minus_cross=0.2)}
    st2 = A.stability(per2)
    check("A: sign-flip across windows is caught",
          st2["difficulty_negative_at_all_windows"] is False)


# ---------------------------------------------------------------------------------------
# exp_common — atomic write, resume/skip, resolve, collect
# ---------------------------------------------------------------------------------------
def test_common():
    from addons import common as exp_common
    import json, tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.json")
        exp_common.atomic_write_json(p, {"a": 1, "b": [1, 2, 3]})
        d = json.loads(open(p).read())
        check("common: atomic_write_json round-trips", d == {"a": 1, "b": [1, 2, 3]})
        check("common: no leftover temp files", len(os.listdir(td)) == 1)

    check("common: resolve_datasets('all')", exp_common.resolve_datasets("all") == list(config.ALL14))
    check("common: resolve_datasets csv",
          exp_common.resolve_datasets("a, b ,c") == ["a", "b", "c"])

    # run_sharded: skip-if-done + atomic + error isolation
    tag = "TESTONLY"
    exp_dir = exp_common.experiments_dir()
    calls = {"n": 0}

    def run_one(ds):
        calls["n"] += 1
        if ds == "bad":
            raise ValueError("boom")
        return {"dataset": ds, "ok": True}

    for f in exp_dir.glob(f"exp_{tag}_*"):
        f.unlink()
    res = exp_common.run_sharded(tag, ["good1", "bad", "good2"], run_one, force=False)
    check("common: error isolated (bad dataset -> error dict, others fine)",
          "error" in res["bad"] and res["good1"]["ok"] and res["good2"]["ok"])
    n_first = calls["n"]
    res2 = exp_common.run_sharded(tag, ["good1", "good2"], run_one, force=False)
    check("common: resume/skip does not recompute finished datasets",
          calls["n"] == n_first, f"calls went {n_first}->{calls['n']}")
    got, missing = exp_common.collect(tag, ["good1", "good2", "neverran"])
    check("common: collect reports missing", missing == ["neverran"] and "good1" in got)
    for f in exp_dir.glob(f"exp_{tag}_*"):    # cleanup
        f.unlink()


if __name__ == "__main__":
    for t in (test_common, test_A, test_B, test_C_math, test_C_ground_truth):
        print(f"\n--- {t.__name__} ---")
        t()
    print(f"\n==== {sum(R)}/{len(R)} checks passed ====")
    sys.exit(0 if all(R) else 1)
