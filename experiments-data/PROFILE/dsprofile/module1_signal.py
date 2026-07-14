"""Module 1 — signal-level characterization (amplitude, spectral, complexity/non-stationarity).

Consumes a COMPLEX frame (FAST + SLOW features on a per-(subject,class) subsample) and emits a
per-dataset data card: per-feature mean and cross-subject coefficient of variation (CV), plus the
complexity block per class. Complexity features validate against physiology (fatigue -> lower
complexity; summaries 02/03) — here we simply report their distributions across subjects/classes.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, windows


def _per_channel_mean(frame, feat_base):
    """Average a feature across its channel columns -> one value per window."""
    cols = [c for c in frame.columns if c.startswith(feat_base + "_c")]
    if not cols:
        return None
    return frame[cols].mean(axis=1)


def run(dataset, seed=42):
    config.ensure_dirs()
    frame = windows.build_complex_frame(dataset, seed=seed)
    is_env = frame.attrs["is_envelope"]

    all_feats = config.FAST_TIME + ([] if is_env else config.FAST_FREQ) + \
        config.SLOW_COMPLEX + ["MSFAPEN_MED", "MSFAPEN_LS", "MSFAPEN_HS"]

    # collapse each feature to one column per window (channel-averaged)
    collapsed = {"subject": frame.subject, "label": frame.label}
    for fb in all_feats:
        col = _per_channel_mean(frame, fb)
        if col is not None:
            collapsed[fb] = col
    cdf = pd.DataFrame(collapsed)

    # per-subject mean of each feature, then cross-subject summary
    per_subj = cdf.groupby("subject").mean(numeric_only=True).drop(columns=["label"], errors="ignore")
    summary = pd.DataFrame({
        "mean": per_subj.mean(),
        "std_across_subjects": per_subj.std(),
        "cv_across_subjects": per_subj.std() / (per_subj.mean().abs() + 1e-12),
    })

    # complexity block per class (mean across subjects)
    complex_feats = [f for f in ["SAMPEN", "FUZZYEN", "FAPEN", "PERMEN", "HFD",
                                 "MSFAPEN_MED", "MSFAPEN_LS", "MSFAPEN_HS"] if f in cdf.columns]
    per_class = cdf.groupby("label")[complex_feats].mean() if complex_feats else pd.DataFrame()

    outdir = config.RESULTS_DIR / "module1"
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(outdir / f"{dataset}__feature_summary.parquet")
    if not per_class.empty:
        per_class.to_parquet(outdir / f"{dataset}__complexity_by_class.parquet")

    # Complexity is undefined when the window is shorter than ENT["min_samples"] (fs < 800 Hz at
    # 250 ms). Those columns are NaN by design; omit them from the card rather than emitting a
    # `NaN` that downstream consumers (meta's `sampen`/`hfd` predictors) might treat as a value.
    ent_ok = windows.entropy_valid(dataset)
    med = {}
    for f in complex_feats:
        col = cdf[f].to_numpy(dtype=float)
        if np.isfinite(col).any():
            med[f] = float(np.nanmedian(col))

    info = dict(dataset=dataset, is_envelope=bool(is_env), n_windows=int(len(frame)),
                n_subjects=int(cdf.subject.nunique()), n_classes=int(cdf.label.nunique()),
                features=list(summary.index),
                entropy_valid=bool(ent_ok),
                window_samples=windows.window_samples(dataset),
                complexity_median=med,
                complexity_note=(None if ent_ok else
                                 "complexity features omitted: window shorter than "
                                 f"{config.ENT['min_samples']} samples"))
    (outdir / f"{dataset}__card.json").write_text(json.dumps(info, indent=2))
    return info
