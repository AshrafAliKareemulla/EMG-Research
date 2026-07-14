"""Paper figures, built from results/ only (no frames, no recompute).

Palette: the validated reference categorical ramp (colorblind-safe; verified with the dataviz
validator for both light and dark surfaces). Light-mode yellow carries a contrast "relief"
flag, so every series is direct-labelled or legended — identity is never colour-alone. Series
also differ by marker, which is the secondary encoding the CVD floor requires at n>=4.

Rules honoured: one y-axis per panel (never dual-axis), fixed hue order (never cycled),
sequential = single hue light->dark, diverging = two hues with a NEUTRAL GRAY midpoint,
recessive grid, thin marks, selective labels.

    python -c "from dsprofile import figures; figures.build_all()"
    python run_phase2.py --exp figs
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from . import config

# ---- design tokens (reference palette, light surface) ---------------------------------
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD, CRIT = "#0ca30c", "#d03b3b"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })
    return plt


def _outdir():
    d = config.RESULTS_DIR / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load(sub, pat="*.json"):
    out = {}
    for f in config.find_all(sub, pat):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        out[d.get("dataset") or Path(f).stem.split("__")[0]] = d
    return out


def _one(path):
    p = config.find(*path.split("/"))
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, name):
    p = _outdir() / name
    fig.savefig(p)
    _mpl().close(fig)
    return str(p)


def _short(ds):
    return (ds.replace("ninapro_", "np_").replace("grabmyo_flow_", "gm_flow_")
              .replace("grabmyo", "gm").replace("emaha_", "em_").replace("fors_emg", "fors"))


# =======================================================================================
# 1. The leak figure — within-subject vs cross-subject separability
# =======================================================================================
def fig_leak_gap():
    m2 = _load("module2", "*__separability.json")
    rows = [(ds, d.get("knn_trial_cv_acc"), d.get("knn_loso_acc"))
            for ds, d in m2.items()
            if d.get("knn_trial_cv_acc") is not None and d.get("knn_loso_acc") is not None]
    if not rows:
        return None
    rows.sort(key=lambda r: -(r[1] or 0))
    plt = _mpl()
    ds, w, c = zip(*rows)
    x = np.arange(len(ds)); bw = 0.38
    fig, ax = plt.subplots(figsize=(max(7, .75 * len(ds)), 3.6))
    # 2px surface gap between adjacent bars == a small width inset
    ax.bar(x - bw / 2, w, bw * 0.94, color=SERIES[0], label="within-subject (trial-grouped CV)")
    ax.bar(x + bw / 2, c, bw * 0.94, color=SERIES[1], label="cross-subject (subject-grouped CV)")
    for i, (a, b) in enumerate(zip(w, c)):
        ax.annotate(f"−{a-b:.2f}", (i, max(a, b) + .02), ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([_short(d) for d in ds], rotation=45, ha="right")
    ax.set_ylabel("5-NN accuracy")
    ax.set_title("Separability collapses when the test subject is unseen", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, min(1.0, max(w) * 1.25))
    return _save(fig, "fig01_within_vs_cross_subject.png")


# =======================================================================================
# 2. Forest plot — per-dataset difficulty correlation + pooled effect
# =======================================================================================
def fig_forest():
    meta = _one("meta/meta.json")
    if not meta or "meta_analysis" not in meta:
        return None
    ma = meta["meta_analysis"]
    per = ma.get("per_dataset") or {}
    if not per:
        return None
    items = sorted(per.items(), key=lambda kv: kv[1]["r"])
    n_max = max(v["n"] for _, v in items)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, .32 * len(items) + 2.0))
    ys = np.arange(len(items))
    for i, (ds, v) in enumerate(items):
        r, n = v["r"], v["n"]
        z = np.arctanh(np.clip(r, -.999999, .999999)); se = 1 / np.sqrt(max(n - 3, 1))
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        col = SERIES[0] if v.get("significant_fdr") else MUTED
        ax.plot([lo, hi], [i, i], color=col, lw=1.6, solid_capstyle="round", zorder=2)
        # marker area ~ n (study weight), the forest-plot convention
        ax.plot([r], [i], marker="s", ms=4 + 5 * np.sqrt(n / n_max),
                color=col, zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.2)
        if not v.get("sign_as_expected"):
            ax.annotate("sign reversed", (hi + .03, i), va="center", fontsize=7, color=CRIT)
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    pooled = ma.get("pooled_r_random_effects"); ci = ma.get("ci95")
    if pooled is not None and ci:
        y = len(items) + .6
        ax.add_patch(plt.Polygon([[ci[0], y], [pooled, y + .3], [ci[1], y], [pooled, y - .3]],
                                 closed=True, color=INK, zorder=4))
        ax.annotate(f"pooled {pooled:+.2f}  [{ci[0]:+.2f}, {ci[1]:+.2f}]", (ci[1] + .03, y),
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(list(ys) + [len(items) + .6])
    ax.set_yticklabels([_short(d) for d, _ in items] + ["random-effects"], fontsize=8)
    ax.set_xlabel(f"Pearson r: {ma.get('predictor','MMD-to-pool')} vs LOSO accuracy")
    ax.set_title(f"Difficulty prediction across datasets (I² = {ma.get('I2', float('nan')):.2f})",
                 color=INK, loc="left")
    ax.annotate("filled = survives FDR", (.02, .02), xycoords="axes fraction",
                fontsize=7, color=MUTED)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-1, len(items) + 1.5)
    return _save(fig, "fig02_forest_difficulty.png")


# =======================================================================================
# 3. E3 — mean vs covariance excess, with the estimation-noise floor
# =======================================================================================
def fig_meancov():
    bc = _load("block_c", "*__block_c.json")
    rows = []
    for ds, d in bc.items():
        p = ((d.get("E3_meancov") or {}).get("representations") or {}).get("pooled") or {}
        if p.get("shift_detectable"):
            rows.append((ds, p["mean_term_excess"], p["cov_term_excess"],
                         p["mean_share_of_excess"], p.get("uncorrected_mean_share")))
    if not rows:
        return None
    rows.sort(key=lambda r: -r[3])
    plt = _mpl()
    ds, me, ce, sh, unc = zip(*rows)
    x = np.arange(len(ds)); bw = .38
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(7, .75 * len(ds)), 6),
                                 gridspec_kw=dict(hspace=.45))
    # Grouped, not stacked: the two terms span orders of magnitude, and stacked segments on a
    # log/symlog axis encode length non-additively — the eye cannot read the sum.
    a1.bar(x - bw / 2, me, bw * .94, color=SERIES[0], label="mean term (excess over null)")
    a1.bar(x + bw / 2, ce, bw * .94, color=SERIES[1], label="covariance term (excess over null)")
    a1.set_yscale("log")
    a1.set_ylabel("null-corrected KL (log)")
    a1.set_title("Between-subject divergence decomposes into mean and covariance",
                 color=INK, loc="left")
    a1.legend(frameon=False, fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels([_short(d) for d in ds], rotation=45, ha="right")

    a2.plot(x, sh, marker=MARKERS[0], color=SERIES[0], label="null-corrected mean share")
    if any(u is not None for u in unc):
        a2.plot(x, [u if u is not None else np.nan for u in unc], marker=MARKERS[1],
                color=SERIES[2], ls="--", label="uncorrected (noise-inflated)")
    a2.axhline(.5, color=AXIS, lw=1, ls=":")
    a2.set_ylim(0, 1); a2.set_ylabel("mean share of divergence")
    a2.set_xticks(x); a2.set_xticklabels([_short(d) for d in ds], rotation=45, ha="right")
    a2.legend(frameon=False, fontsize=8)
    a2.set_title("Subtracting the within-subject estimation floor changes the conclusion",
                 color=INK, loc="left")
    return _save(fig, "fig03_mean_vs_covariance.png")


# =======================================================================================
# 4. A4 — inter-subject vs inter-day
# =======================================================================================
def fig_a4():
    bc = _load("block_c", "*__block_c.json")
    rows = []
    for ds, d in bc.items():
        e = d.get("E2_a4_fair") or {}
        if e.get("applicable"):
            rows.append((ds, e["inter_subject_within_session_mmd"], e["inter_subject_mmd_std"],
                         e["inter_day_within_subject_mmd"], e["inter_day_mmd_std"],
                         e.get("p_value"), bool(e.get("caveat"))))
    if not rows:
        return None
    plt = _mpl()
    ds, a, asd, b, bsd, pv, cav = zip(*rows)
    x = np.arange(len(ds)); bw = .36
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(ds)), 3.8))
    ax.bar(x - bw / 2, a, bw * .94, yerr=asd, capsize=3, color=SERIES[0],
           error_kw=dict(ecolor=MUTED, lw=1), label="inter-subject (within a session)")
    ax.bar(x + bw / 2, b, bw * .94, yerr=bsd, capsize=3, color=SERIES[1],
           error_kw=dict(ecolor=MUTED, lw=1), label="inter-day (within a subject)")
    for i, (aa, bb, p, c) in enumerate(zip(a, b, pv, cav)):
        ax.annotate(f"×{aa/(bb+1e-12):.1f}" + ("*" if (p or 1) < .05 else ""),
                    (i, max(aa, bb) + .02), ha="center", fontsize=8, color=INK2)
        if c:
            ax.annotate("†", (i, -.04), ha="center", fontsize=10, color=CRIT,
                        annotation_clip=False)
    ax.set_xticks(x); ax.set_xticklabels([_short(d) for d in ds], rotation=20, ha="right")
    ax.set_ylabel("mean pairwise MMD")
    ax.set_title("Inter-subject shift exceeds inter-day shift", color=INK, loc="left")
    ax.legend(frameon=False)
    ax.annotate("* Mann-Whitney p<0.05   † see caveat (not independent / not true days)",
                (.0, -.32), xycoords="axes fraction", fontsize=7, color=MUTED)
    return _save(fig, "fig04_a4_intersubject_vs_interday.png")


# =======================================================================================
# 5. Robustness — is difficulty classifier-agnostic?
# =======================================================================================
def fig_robustness():
    rb = _load("robust_difficulty", "*__robust_difficulty.json")
    rows = []
    for ds, d in rb.items():
        dc = d.get("difficulty_prediction_by_classifier") or {}
        if len(dc) >= 3 and ds != "synth":
            rows.append((ds, [dc[c]["pearson_r"] for c in ("lda", "svm", "rf")]))
    if not rows:
        return None
    rows.sort(key=lambda r: np.mean(r[1]))
    plt = _mpl()
    ds = [r[0] for r in rows]; R = np.array([r[1] for r in rows])
    x = np.arange(len(ds)); bw = .26
    fig, ax = plt.subplots(figsize=(max(7, .8 * len(ds)), 3.8))
    for k, (name, col, mk) in enumerate(zip(["LDA", "linear SVM", "random forest"],
                                            SERIES[:3], MARKERS[:3])):
        ax.bar(x + (k - 1) * bw, R[:, k], bw * .9, color=col, label=name)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([_short(d) for d in ds], rotation=45, ha="right")
    ax.set_ylabel("r: MMD-to-pool vs LOSO accuracy")
    ax.set_title("Who is hard does not depend on the classifier", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    return _save(fig, "fig05_robustness_by_classifier.png")


# =======================================================================================
# 6. Calibration + actionability
# =======================================================================================
def fig_actionability():
    ac = _load("actionability", "*__actionability.json")
    items = [(ds, d) for ds, d in ac.items()
             if "budget_axis" in d and ds != "synth"]
    if not items:
        return None
    items.sort(key=lambda kv: -(kv[1].get("oracle_ceiling") or 0))
    items = items[:6]
    plt = _mpl()
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.0), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (ds, d) in zip(axes, items):
        x = d["budget_axis"]
        for y, name, col, mk in [(d["mean_acc_oracle"], "oracle", SERIES[3], None),
                                 (d["mean_acc_guided"], "SDI-guided", SERIES[0], None),
                                 (d["mean_acc_random"], "random", SERIES[2], None)]:
            ax.plot(x, y, color=col, lw=1.8, label=name)
        ax.set_title(f"{_short(ds)}\nceiling {d['oracle_ceiling']*100:.2f} pp",
                     fontsize=8, color=INK)
        ax.set_xlabel("calibration budget")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("mean accuracy")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Difficulty-guided calibration: the achievable headroom is <1 accuracy point",
                 color=INK, x=.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .93))
    return _save(fig, "fig06_actionability.png")


def fig_calibration_premise():
    """The premise test: do predicted-hard subjects actually gain more from calibration?"""
    ac = _load("actionability", "*__actionability.json")
    rows = [(ds, (d.get("mmd_vs_calibration_gain") or {}).get("pearson_r"),
             (d.get("mmd_vs_calibration_gain") or {}).get("p_value"))
            for ds, d in ac.items() if d.get("mmd_vs_calibration_gain") and ds != "synth"]
    rows = [r for r in rows if r[1] is not None and np.isfinite(r[1])]
    if not rows:
        return None
    rows.sort(key=lambda r: r[1])
    plt = _mpl()
    ds, r, p = zip(*rows)
    fig, ax = plt.subplots(figsize=(6.2, .3 * len(ds) + 1.8))
    cols = [SERIES[0] if (pp or 1) < .05 else MUTED for pp in p]
    ax.barh(np.arange(len(ds)), r, .62, color=cols)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(np.arange(len(ds))); ax.set_yticklabels([_short(d) for d in ds], fontsize=8)
    ax.set_xlabel("r: MMD-to-pool vs realised calibration gain")
    ax.set_title("The premise behind guided allocation, tested", color=INK, loc="left")
    ax.annotate("guided allocation can only work if this is reliably positive",
                (.02, .02), xycoords="axes fraction", fontsize=7, color=MUTED)
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig07_calibration_premise.png")


# =======================================================================================
# 7. Channels & sampling rate
# =======================================================================================
def fig_channels_fs():
    bd = _load("block_d", "*__block_d.json")
    plt = _mpl()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    k = 0
    for ds, d in sorted(bd.items()):
        e7 = d.get("E7_channel_reduction") or {}
        cur = e7.get("accuracy_vs_k_loso") or {}
        if not cur or k >= 6:
            continue
        ks = sorted(int(x) for x in cur)
        full = e7.get("full_accuracy_loso") or 1
        a1.plot(ks, [cur[str(i)] / (full + 1e-12) for i in ks], marker=MARKERS[k % 6],
                color=SERIES[k % 6], label=_short(ds), ms=4)
        k += 1
    a1.axhline(.95, color=AXIS, ls=":", lw=1)
    a1.annotate("95% of full", (.55, .955), xycoords=("axes fraction", "data"),
                fontsize=7, color=MUTED)
    a1.set_xlabel("# channels (mRMR order)"); a1.set_ylabel("fraction of full LOSO accuracy")
    a1.set_title("Channel reduction (subject-disjoint)", color=INK, loc="left")
    a1.legend(frameon=False, fontsize=7, ncol=2)

    k = 0
    for ds, d in sorted(bd.items()):
        e6 = d.get("E6_sampling_rate") or {}
        if not e6.get("testable"):
            continue
        cur = e6.get("curves") or {}
        pts = sorted((v["effective_fs"], v["retained_frac"]) for v in cur.values()
                     if "retained_frac" in v)
        if len(pts) < 2 or k >= 6:
            continue
        a2.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS[k % 6],
                color=SERIES[k % 6], label=_short(ds), ms=4)
        k += 1
    a2.axhline(1.0, color=AXIS, ls=":", lw=1)
    a2.set_xlabel("effective sampling rate (Hz, anti-aliased)")
    a2.set_ylabel("accuracy retained vs native")
    a2.set_title("Sampling-rate sufficiency", color=INK, loc="left")
    if k:
        a2.legend(frameon=False, fontsize=7)
    else:
        a2.annotate("no dataset qualifies\n(native fs too low to decimate)", (.5, .5),
                    xycoords="axes fraction", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    return _save(fig, "fig08_channels_and_sampling_rate.png")


# =======================================================================================
# 8. Feature reliability heatmap (sequential = ONE hue, light->dark)
# =======================================================================================
def fig_reliability():
    ba = _load("block_a", "*__block_a.json")
    if not ba:
        return None
    feats = sorted({f for d in ba.values() for f in (d.get("feature_reliability") or {})})
    dsets = sorted(ba)
    if not feats:
        return None
    M = np.full((len(feats), len(dsets)), np.nan)
    for j, ds in enumerate(dsets):
        rel = ba[ds].get("feature_reliability") or {}
        for i, f in enumerate(feats):
            if f in rel:
                M[i, j] = rel[f]
    order = np.argsort(-np.nanmean(M, axis=1))
    M, feats = M[order], [feats[i] for i in order]
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    cmap.set_bad("#f0efec")
    fig, ax = plt.subplots(figsize=(.55 * len(dsets) + 3, .24 * len(feats) + 2))
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=np.nanmin(M), vmax=np.nanmax(M))
    ax.set_xticks(range(len(dsets)))
    ax.set_xticklabels([_short(d) for d in dsets], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=7)
    ax.grid(visible=False)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cb.set_label("reliability across repetitions", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title("Feature reliability (gray = undefined: window too short for entropy)",
                 color=INK, loc="left", fontsize=9)
    return _save(fig, "fig09_feature_reliability.png")


# =======================================================================================
# 9. Dataset atlas
# =======================================================================================
def fig_atlas():
    meta = _one("meta/meta.json")
    at = (meta or {}).get("atlas")
    if not at:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ev = at.get("explained_variance", [0, 0])
    for ds, (x, y) in at["coords"].items():
        c = at["cluster"][ds]
        ax.scatter(x, y, s=70, color=SERIES[c % 6], marker=MARKERS[c % 6],
                   edgecolor=SURFACE, linewidth=1.4, zorder=3)
        ax.annotate(_short(ds), (x, y), textcoords="offset points", xytext=(7, 3),
                    fontsize=7, color=INK2)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}% var)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}% var)")
    ax.set_title("Atlas of sEMG datasets (colour+marker = cluster)", color=INK, loc="left")
    ax.annotate("axes mix device, fs and channel count with population — see caveats",
                (.0, -.14), xycoords="axes fraction", fontsize=7, color=MUTED)
    return _save(fig, "fig10_dataset_atlas.png")


# =======================================================================================
# 10. SDI leave-one-cohort-out
# =======================================================================================
def fig_sdi():
    s = _one("module6_sdi/sdi.json")
    if not s:
        return None
    per = s.get("lodo_per_dataset") or {}
    if not per:
        return None
    items = sorted(per.items(), key=lambda kv: kv[1]["spearman"])
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.2, .3 * len(items) + 2))
    r = [v["spearman"] for _, v in items]
    cols = [SERIES[0] if v.get("significant_fdr") else MUTED for _, v in items]
    ax.barh(np.arange(len(items)), r, .62, color=cols)
    ax.axvline(0, color=AXIS, lw=1)
    m = s.get("lodo_mean_spearman")
    if m is not None:
        ax.axvline(m, color=INK, ls="--", lw=1.2)
        ax.annotate(f"mean {m:.2f}", (m, len(items) - .3), fontsize=8, color=INK)
    ax.set_yticks(np.arange(len(items)))
    ax.set_yticklabels([_short(d) for d, _ in items], fontsize=8)
    ax.set_xlabel("Spearman: SDI-predicted vs actual LOSO accuracy")
    ax.set_title("SDI under leave-one-cohort-out (filled = survives FDR)", color=INK, loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig11_sdi_lodo.png")


# =======================================================================================
# 11. Transfer compatibility heatmap
# =======================================================================================
def fig_transfer():
    t = _one("transfer/transfer.json")
    if not t or "compatibility_mmd" not in t:
        return None
    ds = t["datasets"]
    M = np.array([[t["compatibility_mmd"][a][b] for b in ds] for a in ds])
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(.5 * len(ds) + 3.2, .5 * len(ds) + 2.6))
    im = ax.imshow(M, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(ds))); ax.set_yticks(range(len(ds)))
    ax.set_xticklabels([_short(d) for d in ds], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([_short(d) for d in ds], fontsize=7)
    ax.grid(visible=False)
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.02)
    cb.set_label("MMD (native scale)", fontsize=8); cb.outline.set_visible(False)
    ax.set_title("Cross-dataset compatibility — EXPLORATORY, no transfer accuracy measured",
                 color=INK, loc="left", fontsize=9)
    return _save(fig, "fig12_transfer_heatmap.png")


ALL = [fig_leak_gap, fig_forest, fig_meancov, fig_a4, fig_robustness, fig_actionability,
       fig_calibration_premise, fig_channels_fs, fig_reliability, fig_atlas, fig_sdi,
       fig_transfer]


def build_all():
    made, skipped = [], []
    for f in ALL:
        try:
            p = f()
        except Exception as e:
            skipped.append(f"{f.__name__}: {type(e).__name__}: {e}")
            continue
        (made if p else skipped).append(p or f"{f.__name__}: inputs missing")
    for m in made:
        print(f"[fig] {m}")
    for s in skipped:
        print(f"[skip] {s}")
    return made, skipped
