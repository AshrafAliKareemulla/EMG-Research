"""Floor-effect experiment — is difficulty PREDICTION, or a floor artifact?

THE QUESTION (STATE.md §5)
--------------------------
The MMD-to-pool predictor works where LOSO accuracy is near chance (ninapro_db1 r=-0.77 at
acc=0.12, ninapro_db2 r=-0.73 at acc=0.14) and fails where accuracy is healthy (grabmyo r=+0.03
at acc=0.70). Across datasets this confounds accuracy with class count, device, subjects, task —
n=13 cannot separate them.

THE DESIGN
----------
Vary accuracy WITHIN a single dataset by using k channels (k = 1, 2, 4, ... C). Everything else —
subjects, task, device, window, feature basis — is held constant. At each k, recompute per
subject: (a) LOSO accuracy (LDA, subject-disjoint), (b) MMD-to-pool on the SAME k-channel basis.
Then correlate the two across subjects.

  * If |r| RISES as mean accuracy FALLS  -> the predictor only works near the floor. A real
    limitation: "cheap statistics predict who a model fails on, but only once it is failing."
  * If r is roughly CONSTANT across the accuracy sweep -> the predictor tracks the DATA, not the
    accuracy regime. The floor correlation across datasets was a confound.

Run on the box (needs the cached fast frames):
    python floor_effect.py --datasets grabmyo,ninapro_db2,fors_emg,emaha_db1 --jobs 8
Output: results/floor_effect/<dataset>__floor.json + a combined floor_effect.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from dsprofile import config, windows, progress
from dsprofile.module3_shift import mmd_rbf
from dsprofile.module5_difficulty import loso_lda_accuracy


def _channel_cols(frame, chans):
    cols = []
    for fb in config.REPR_BASIS:
        for c in chans:
            cols += [x for x in frame.columns if x == f"{fb}_c{c}"]
    return cols


def _standardise(X):
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.where(sd < 1e-12, 1.0, sd)


def _mmd_to_pool(X, subjects, seed, cap=800):
    """Per-subject mean MMD to the rest of the pool, on whatever basis X is."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in np.unique(subjects):
        this = X[subjects == s]; rest = X[subjects != s]
        if len(this) < 20 or len(rest) < 20:
            continue
        if len(this) > cap:
            this = this[rng.choice(len(this), cap, replace=False)]
        if len(rest) > cap:
            rest = rest[rng.choice(len(rest), cap, replace=False)]
        out[int(s)] = mmd_rbf(this, rest, rng=rng)
    return out


def _channel_order(frame):
    """mRMR-style channel order (reuse Block D's ranking so the sweep adds informative channels
    first — the realistic degradation path)."""
    from dsprofile.module4_channels import per_channel_fisher, nmi_matrix, greedy_mrmr, _channel_signal_matrix
    M, _ = _channel_signal_matrix(frame, "RMS")
    relevance = per_channel_fisher(frame)
    nmi = nmi_matrix(M)
    return [int(c) for c in greedy_mrmr(relevance, nmi)], M.shape[1]


def run_dataset(dataset, seed=42):
    from scipy.stats import pearsonr, spearmanr
    frame = windows.build_fast_frame(dataset, seed=seed)
    y = frame["label"].to_numpy(); subj = frame["subject"].to_numpy()
    order, C = _channel_order(frame)

    # geometric-ish sweep of channel counts (skip the trivial 1-channel case if C is large)
    ks = sorted(set([k for k in (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28) if k <= C] + [C]))
    curve = []
    for k in ks:
        chans = order[:k]
        Xk = _standardise(np.nan_to_num(frame[_channel_cols(frame, chans)].to_numpy(np.float64)))
        acc = loso_lda_accuracy(Xk, y, subj)                    # subject -> LOSO accuracy
        mmd = _mmd_to_pool(Xk, subj, seed)                      # subject -> MMD-to-pool
        common = sorted(set(acc) & set(mmd))
        if len(common) < 5:
            continue
        a = np.array([acc[s] for s in common]); m = np.array([mmd[s] for s in common])
        if a.std() < 1e-9 or m.std() < 1e-9:
            r = p = rho = float("nan")
        else:
            r, p = pearsonr(m, a); rho, _ = spearmanr(m, a)
        curve.append(dict(k=int(k), n_subjects=len(common),
                          mean_loso_acc=float(a.mean()), std_loso_acc=float(a.std()),
                          mmd_vs_acc_pearson_r=float(r), p_value=float(p),
                          mmd_vs_acc_spearman=float(rho)))
        progress.log(f"{dataset} k={k:2d}: acc={a.mean():.3f}  r(mmd,acc)={r:+.3f} (p={p:.3f})")

    # Does |r| depend on the accuracy it was measured at? This is the whole experiment.
    accs = np.array([c["mean_loso_acc"] for c in curve])
    rs = np.array([c["mmd_vs_acc_pearson_r"] for c in curve])
    ok = np.isfinite(accs) & np.isfinite(rs)
    trend = {}
    if ok.sum() >= 4:
        # correlation between the accuracy at each rung and the (signed) predictor r at that rung.
        # POSITIVE trend r means: lower accuracy -> more negative predictor r  = floor effect.
        tr, tp = pearsonr(accs[ok], rs[ok])
        trend = dict(acc_vs_predictor_r_pearson=float(tr), p_value=float(tp),
                     interpretation=("floor effect: predictor strengthens as accuracy drops"
                                     if tr > 0.3 and tp < 0.1 else
                                     "no floor effect: predictor ~constant across accuracy"
                                     if abs(tr) < 0.3 else
                                     "reverse: predictor weakens as accuracy drops"))
    return dict(dataset=dataset, n_channels=C, channel_order=order,
                curve=curve, floor_trend=trend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="grabmyo,ninapro_db2,fors_emg,emaha_db1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    if a.jobs is not None:
        config.N_JOBS = a.jobs
    outdir = config.RESULTS_DIR / "floor_effect"
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in a.datasets.split(",")]

    summary = {}
    for ds in datasets:
        with progress.timer(f"floor_effect :: {ds}"):
            res = run_dataset(ds, a.seed)
        (outdir / f"{ds}__floor.json").write_text(json.dumps(res, indent=2))
        summary[ds] = res.get("floor_trend", {})
        print(f"[OK] {ds}: {summary[ds].get('interpretation', 'n/a')}", flush=True)

    (outdir / "floor_effect.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FLOOR-EFFECT VERDICT (per dataset) ===")
    for ds, t in summary.items():
        print(f"  {ds:20} trend r={t.get('acc_vs_predictor_r_pearson', float('nan')):+.3f} "
              f"p={t.get('p_value', float('nan')):.3f}  {t.get('interpretation', '')}")


if __name__ == "__main__":
    main()
