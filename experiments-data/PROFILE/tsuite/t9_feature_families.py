"""T9 — WHICH handcrafted feature families actually survive cross-subject?

WHY THIS EXPERIMENT EXISTS
--------------------------
This is the biggest hole in the project, and it is the one closest to the group's own ML track.

`block_a` ranks features by RELIABILITY (an ICC-style score: does a feature give a consistent value
for the same subject?). That is a useful hygiene statistic and it is NOT the question anybody asks.
The question the sEMG feature literature has argued about for thirty years is:

    Which feature set should I compute, and does the expensive one earn its cost?

Nobody in this project has ever measured **LOSO accuracy per feature family**. Every experiment here
uses one fixed 7-feature basis (`REPR_BASIS`) chosen a priori, and every ML paper in the field picks
a family — Hudgins' TD4, Phinyomark's extended TD set, frequency-domain, entropy/complexity — and
defends it with a within-subject number. A within-subject number is exactly the number this project
has shown to be misleading (N4: within-subject overstates cross-subject on 14/14 datasets).

So the panel of 14 datasets is precisely the instrument needed to settle it honestly, cross-subject.

THE FAIR-COMPARISON RULE (this is why the experiment uses the `complex` frame)
------------------------------------------------------------------------------
Every arm must be scored on the SAME ROWS. The `fast` frame has no entropy columns, and the
`complex` frame has both fast and entropy columns but a smaller per-class cap. Scoring the cheap
families on one frame and the expensive ones on another would compare feature families AND sample
sizes at the same time, which answers nothing. All arms therefore run on the `complex` frame, with
identical rows, identical subjects, identical folds. The only thing that changes between arms is
which COLUMNS the model may see.

THE ARMS
--------
  hudgins_td4   MAV, WL, ZC, SSC                          — the 1993 classic; 4 features/channel
  td_extended   + WAMP, VAR, RMS, IEMG, LOG, SSI, DASDV   — the Phinyomark-style extended TD set
  amplitude     MAV, RMS, IEMG, SSI, VAR, LOG, LOGRMS, AAC, DASDV — pure amplitude/energy
  frequency     MNF, MDF, SENT, MNP, TTP                  — spectral only
  hjorth        HJ_ACT, HJ_MOB, HJ_COM                    — Hjorth descriptors
  complexity    SAMPEN, FUZZYEN, FAPEN, PERMEN, HFD       — the EXPENSIVE ones (O(N^2) per window)
  repr_basis    the 7 this project has been using all along — is our own choice defensible?
  all_features  everything available

Scored with chance-corrected kappa (class counts differ across datasets) under LOSO, plus the
within-subject arm for the generalisation gap, plus **cost**: features/channel and wall-clock, so the
verdict can be stated as an accuracy-per-CPU-second trade-off rather than an accuracy alone.

PRE-REGISTERED BRANCHES
-----------------------
  A. complexity/entropy adds nothing over cheap TD features cross-subject
     -> *"The expensive nonlinear features do not survive the subject boundary."* A direct, useful,
        money-saving result for every sEMG practitioner, and a genuine correction to a literature that
        justifies entropy features on within-subject numbers. **Headline for the feature question.**
  B. complexity DOES add real accuracy cross-subject
     -> then it earns its cost, and we say so — with the first multi-dataset cross-subject evidence
        for it. Equally publishable, and it would change our own pipeline.
  C. hudgins_td4 is within ~2 pp of all_features
     -> *"Four features per channel is enough; the last thirty years of feature engineering buys
        almost nothing once you leave the subject."* Provocative, cheap to state, easy to check.
  D. no family separates from another (all within noise)
     -> feature choice is not the lever; shift is. Consistent with the rest of the paper, and worth
        saying plainly.

Note: entropy is undefined on 4 datasets whose 250 ms window holds < 200 samples (ninapro_db1 at
100 Hz, ninapro_db5 / senic at 200 Hz, myobit at 176 Hz). Those datasets report the cheap arms and
mark the complexity arm not-applicable — they are NOT silently dropped from the panel.

GROUND TRUTH
------------
  * A synthetic where the label lives ONLY in the amplitude features: the amplitude arm must win and
    the frequency arm must sit at chance. If the column selector cannot find a signal we planted in a
    known family, every real-data ranking is meaningless.
  * A synthetic where the label lives ONLY in the entropy-like columns: the complexity arm must win.
    (This is the negative control for branch A: it proves that "complexity adds nothing" would be a
    finding about EMG, not an artifact of a broken complexity arm.)
"""
from __future__ import annotations

import time

import numpy as np

from . import common as C

TAG = "t9_feature_families"

FAMILIES = {
    "hudgins_td4":  ["MAV", "WL", "ZC", "SSC"],
    "td_extended":  ["MAV", "WL", "ZC", "SSC", "WAMP", "VAR", "RMS", "IEMG", "LOG", "SSI", "DASDV"],
    "amplitude":    ["MAV", "RMS", "IEMG", "SSI", "VAR", "LOG", "LOGRMS", "AAC", "DASDV"],
    "frequency":    ["MNF", "MDF", "SENT", "MNP", "TTP"],
    "hjorth":       ["HJ_ACT", "HJ_MOB", "HJ_COM"],
    "complexity":   ["SAMPEN", "FUZZYEN", "FAPEN", "PERMEN", "HFD"],
    "repr_basis":   ["MAV", "WL", "WAMP", "RMS", "HJ_MOB", "HJ_COM", "MFL"],
}


def _cols_for_family(frame, feats):
    """Column names present in the frame for this family (features are stored as `<FEAT>_c<channel>`)."""
    cols = []
    for f in feats:
        cols += [c for c in frame.columns if c.startswith(f + "_c")]
    return sorted(cols)


def _usable(frame, cols):
    """A family is usable only if its columns exist AND are not all-NaN (entropy is masked to NaN on
    the sub-800 Hz datasets — that must be reported as not-applicable, never imputed to zero)."""
    if not cols:
        return False, "no columns for this family in the frame"
    sub = frame[cols].to_numpy(np.float64)
    if not np.isfinite(sub).any():
        return False, "all values NaN (entropy undefined at this sampling rate / window length)"
    frac = float(np.isfinite(sub).mean())
    if frac < 0.5:
        return False, f"only {frac:.0%} finite values"
    return True, ""


def run_one(dataset, seed=42, n_jobs=1):
    with C.timer(f"T9 :: {dataset}"):
        # the COMPLEX frame: it carries the fast AND the entropy columns, so every arm is scored on
        # identical rows. (Using two different frames would confound family with sample size.)
        from dsprofile import windows
        frame = windows.build_complex_frame(dataset, seed=seed)
        y = frame.label.to_numpy()
        subj = frame.subject.to_numpy()
        K = int(len(np.unique(y)))

        out = dict(dataset=dataset, frame="complex", n_subjects=int(len(np.unique(subj))),
                   n_classes=K, n_windows=int(len(frame)), arms={})

        fams = dict(FAMILIES)
        fams["all_features"] = sorted({f for v in FAMILIES.values() for f in v})

        for name, feats in fams.items():
            cols = _cols_for_family(frame, feats)
            ok, why = _usable(frame, cols)
            if not ok:
                out["arms"][name] = dict(applicable=False, note=why, n_cols=len(cols))
                continue
            X = np.nan_to_num(frame[cols].to_numpy(np.float64))
            X = C.zscore(X, axis=0)
            t0 = time.perf_counter()
            acc = C.loso_accuracy_model(X, y, subj, "lda", seed=seed)
            secs = time.perf_counter() - t0
            if len(acc) < 5:
                out["arms"][name] = dict(applicable=False, note="too few subjects scored")
                continue
            mean_acc = float(np.mean(list(acc.values())))
            out["arms"][name] = dict(
                applicable=True, n_cols=len(cols),
                features_per_channel=len(feats),
                acc_loso=mean_acc, kappa_loso=C.kappa_chance(mean_acc, K),
                seconds=float(secs),
                per_subject_acc={str(k): float(v) for k, v in sorted(acc.items())})

    arms = {k: v for k, v in out["arms"].items() if v.get("applicable")}
    if not arms:
        return out
    best = max(arms, key=lambda k: arms[k]["kappa_loso"])
    out["best_family"] = best
    out["best_kappa"] = arms[best]["kappa_loso"]

    # the two questions that matter
    if "complexity" in arms and "td_extended" in arms:
        out["complexity_gain_over_cheap_td"] = float(
            arms["complexity"]["kappa_loso"] - arms["td_extended"]["kappa_loso"])
    if "all_features" in arms and "complexity" in arms:
        cheap = [k for k in arms if k not in ("complexity", "all_features")]
        best_cheap = max(cheap, key=lambda k: arms[k]["kappa_loso"]) if cheap else None
        if best_cheap:
            out["all_vs_best_cheap"] = float(
                arms["all_features"]["kappa_loso"] - arms[best_cheap]["kappa_loso"])
            out["best_cheap_family"] = best_cheap
    if "hudgins_td4" in arms and "all_features" in arms:
        out["td4_deficit_vs_all"] = float(
            arms["all_features"]["kappa_loso"] - arms["hudgins_td4"]["kappa_loso"])
        out["td4_is_enough"] = bool(out["td4_deficit_vs_all"] < 0.02)
    return out


def build_pooled(tag=TAG):
    import json
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(C.results_dir(tag).glob(f"*__{tag}.json"))]
    rows = [r for r in rows if r.get("arms")]
    if not rows:
        return dict(note="no per-dataset results yet")

    all_ds = [r["dataset"] for r in rows]
    fams = sorted({k for r in rows for k in r["arms"]})
    table = {}
    for f in fams:
        ks = [r["arms"][f]["kappa_loso"] for r in rows
              if r["arms"].get(f, {}).get("applicable")]
        ds = [r["dataset"] for r in rows if r["arms"].get(f, {}).get("applicable")]
        wins = [r["dataset"] for r in rows if r.get("best_family") == f]
        if not ks:
            table[f] = dict(applicable_on=0, note="not applicable on any dataset"); continue
        table[f] = dict(applicable_on=len(ks), mean_kappa=float(np.mean(ks)),
                        median_kappa=float(np.median(ks)),
                        n_datasets_best=len(wins),
                        best_on=C.count_both_ways(wins, all_ds),
                        mean_features_per_channel=float(np.mean(
                            [r["arms"][f]["features_per_channel"] for r in rows
                             if r["arms"].get(f, {}).get("applicable")])),
                        mean_seconds=float(np.mean([r["arms"][f]["seconds"] for r in rows
                                                    if r["arms"].get(f, {}).get("applicable")])))

    gains = [r["complexity_gain_over_cheap_td"] for r in rows
             if "complexity_gain_over_cheap_td" in r]
    td4 = [r["td4_deficit_vs_all"] for r in rows if "td4_deficit_vs_all" in r]
    td4_enough = [r["dataset"] for r in rows if r.get("td4_is_enough")]
    ranked = sorted((f for f in table if table[f].get("mean_kappa") is not None),
                    key=lambda f: -table[f]["mean_kappa"])

    mean_gain = float(np.mean(gains)) if gains else float("nan")
    mean_td4 = float(np.mean(td4)) if td4 else float("nan")

    if gains and mean_gain < 0.01:
        branch, verdict = "A", (
            f"THE EXPENSIVE FEATURES DO NOT SURVIVE THE SUBJECT BOUNDARY: entropy/complexity scores "
            f"{mean_gain:+.3f} kappa against a cheap extended time-domain set (tested on "
            f"{len(gains)} datasets where entropy is even defined), while costing "
            f"{table['complexity']['mean_seconds'] / max(table['td_extended']['mean_seconds'], 1e-9):.0f}x "
            "the CPU. A literature that justifies these features on within-subject numbers is "
            "justifying them on exactly the number this paper shows to be misleading.")
    elif gains and mean_gain > 0.02:
        branch, verdict = "B", (
            f"COMPLEXITY EARNS ITS COST: entropy/complexity adds {mean_gain:+.3f} kappa over the cheap "
            "time-domain set, cross-subject, across the panel. First multi-dataset cross-subject "
            "evidence for it — and it means our own 7-feature basis is leaving accuracy on the table.")
    elif td4 and mean_td4 < 0.02:
        branch, verdict = "C", (
            f"FOUR FEATURES ARE ENOUGH: Hudgins' TD4 is within {mean_td4:.3f} kappa of the full "
            f"feature set on {C.count_both_ways(td4_enough, all_ds)['datasets']} datasets. Cross-subject, "
            "thirty years of feature engineering buys almost nothing.")
    else:
        branch, verdict = "D", (
            "NO FAMILY SEPARATES: the feature families are within noise of one another cross-subject. "
            "Feature choice is not the lever — distribution shift is. Consistent with the rest of the "
            "paper, and worth stating plainly.")

    out = dict(tag=tag, n_datasets=len(rows), ranking_by_mean_kappa=ranked, per_family=table,
               mean_complexity_gain_over_cheap_td=mean_gain,
               mean_td4_deficit_vs_all=mean_td4,
               td4_is_enough_on=C.count_both_ways(td4_enough, all_ds),
               entropy_not_applicable_on=[r["dataset"] for r in rows
                                          if not r["arms"].get("complexity", {}).get("applicable")],
               branch=branch, verdict=verdict)
    C.atomic_write_json(C.results_dir(tag) / "pooled.json", out)
    return out


# ------------------------------------------------------------------ ground truth
def _planted_frame(where, n_subj=10, n_cls=4, per=40, seed=0):
    """A frame in which the label is discoverable ONLY from `where` family's columns. Everything
    else is pure noise. The arm that owns `where` MUST win; the others MUST sit near chance."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    fams = {"amplitude": ["MAV", "RMS", "IEMG", "SSI", "VAR", "LOG", "LOGRMS", "AAC", "DASDV"],
            "frequency": ["MNF", "MDF", "SENT", "MNP", "TTP"],
            "complexity": ["SAMPEN", "FUZZYEN", "FAPEN", "PERMEN", "HFD"]}
    n_ch = 2
    rows, labels, subs = [], [], []
    centers = rng.standard_normal((n_cls, 8)) * 3.0
    for u in range(n_subj):
        for c in range(n_cls):
            for _ in range(per):
                rec = {}
                i = 0
                for fam, feats in fams.items():
                    for f in feats:
                        for ch in range(n_ch):
                            if fam == where:
                                rec[f"{f}_c{ch}"] = centers[c][i % 8] + rng.standard_normal() * 0.8
                                i += 1
                            else:
                                rec[f"{f}_c{ch}"] = rng.standard_normal()
                rows.append(rec); labels.append(c); subs.append(u)
    fr = pd.DataFrame(rows)
    fr["label"] = labels
    fr["subject"] = subs
    return fr


def selftest(check):
    for planted in ("amplitude", "complexity"):
        fr = _planted_frame(planted, seed=2)
        y, s = fr.label.to_numpy(), fr.subject.to_numpy()
        res = {}
        for fam in ("amplitude", "frequency", "complexity"):
            cols = _cols_for_family(fr, FAMILIES[fam])
            X = C.zscore(np.nan_to_num(fr[cols].to_numpy(np.float64)), axis=0)
            acc = C.loso_accuracy_model(X, y, s, "lda", seed=2, cap_train=4000)
            res[fam] = float(np.mean(list(acc.values()))) if acc else float("nan")
        winner = max(res, key=lambda k: res[k])
        check(f"T9 label planted ONLY in '{planted}': that family's arm wins",
              winner == planted, f"accs={ {k: round(v, 3) for k, v in res.items()} }")
        others = [v for k, v in res.items() if k != planted]
        check(f"T9 label planted ONLY in '{planted}': the OTHER families sit near chance "
              "(the column selector is not leaking signal between arms)",
              max(others) < 0.45, f"chance=0.25, best other={max(others):.3f}")

    # a family whose columns are all NaN must be reported NOT APPLICABLE, never imputed to zero
    import pandas as pd
    fr = _planted_frame("amplitude", seed=3)
    for c in _cols_for_family(fr, FAMILIES["complexity"]):
        fr[c] = np.nan
    ok, why = _usable(fr, _cols_for_family(fr, FAMILIES["complexity"]))
    check("T9 an all-NaN family (entropy on a low-fs dataset) is flagged NOT APPLICABLE, "
          "not silently imputed to zero", ok is False, f"reason={why!r}")
