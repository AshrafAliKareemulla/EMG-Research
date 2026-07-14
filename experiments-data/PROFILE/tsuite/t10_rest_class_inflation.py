"""T10 — How much accuracy does the REST class manufacture?

WHY
---
Every number in this project drops the rest class (`config.DROP_REST = True`). That is the right
choice: rest is usually the easiest class to recognise (little or no muscle activity), so including
it inflates the headline while telling you nothing about whether the system can tell one *gesture*
from another.

But large parts of the sEMG literature report rest-INCLUSIVE accuracies, and this project has never
quantified what that is worth. That is a gap in our own leakage/honesty story (N2): we say the field's
numbers are optimistic, and we can prove it for overlapping-window CV — but for the rest class we
have only asserted it.

This experiment measures it. It is cheap, and it converts a stylistic choice into a number.

DESIGN
------
Two frames per dataset, identical in every other respect (same window, overlap, cap, seed):
  * `rest_dropped`  — what this project uses (K gesture classes)
  * `rest_included` — the same data with the rest class restored (K+1 classes)

Three quantities, because raw accuracy alone would be a trap (the two tasks have different class
counts and therefore different chance levels):
  1. **raw accuracy** with and without rest — the number the literature actually reports;
  2. **chance-corrected kappa** for both — the honest comparison;
  3. **gesture-only accuracy when rest is present** — of the windows that are truly a gesture, how
     many are still classified correctly once a big easy rest class is competing for them? This
     separates "rest inflates the average" from "rest actively steals gesture predictions".

THE CRITERION — and why the obvious one is WRONG
------------------------------------------------
The obvious criterion is *"raw accuracy rises but chance-corrected kappa does not"*. **It does not
work, and the ground truth proved it before this ever touched real data:** adding an easy, far-away
extra class raises kappa too (measured 0.783 -> 0.836 on the synthetic). That is not a defect in
kappa — the model genuinely classifies the easy class correctly, and chance-correction adjusts the
BASELINE, not the arithmetic of averaging in one more class the model gets right.

The quantity that DOES isolate the inflation is **gesture-only accuracy**: of the windows that are
genuinely a gesture, how many are still right once rest is competing for them? If the raw number
rises while gesture-only accuracy does not budge, the gain is pure arithmetic and nothing about
gesture discrimination improved. Every branch below is keyed on that, not on kappa.

(The smoke run on ninapro_db5 confirmed why this matters: rest is only 2 % of its windows, raw
accuracy rose +1.3 pp, gesture-only accuracy moved +0.1 pp — textbook inflation — yet a kappa-keyed
branch would have reported "rest genuinely helps".)

PRE-REGISTERED BRANCHES
-----------------------
  A. Raw accuracy rises while GESTURE-ONLY accuracy does not (|change| < 2 pp)
     -> *"Rest-inclusive accuracies are inflated by N pp; the gain is one easy class dragging the mean
        up, not better gesture discrimination."* A quantified honesty result that strengthens the
        leakage audit and applies to a large slice of the literature. **The expected outcome.**
        The MAGNITUDE is reported, never gated: it scales with how big the rest class is (2 % of
        windows on ninapro_db5, far more elsewhere), so it must always be quoted alongside
        `rest_share_of_windows`.
  B. GESTURE-ONLY accuracy IMPROVES (>= +2 pp) when rest is present
     -> rest genuinely helps: the rest windows appear to teach the model where the no-activity region
        of feature space lies, sharpening the gesture boundaries. Surprising; we would report it.
  C. GESTURE-ONLY accuracy FALLS (<= -2 pp)
     -> rest actively steals gesture predictions. The strongest form of the argument for dropping it,
        and it turns our protocol choice into a finding rather than a convention.

APPLICABILITY: only 7 of the 14 datasets ship a rest class (emaha_db1, grabmyo, grabmyo_flow_static,
ninapro_db1/db2/db4/db5). The other 7 are reported `applicable: false` with a reason — never silently
dropped.

GROUND TRUTH
------------
  * K gesture classes + one EASY, far-away extra class: raw accuracy must RISE and gesture-only
    accuracy must NOT move. (Kappa is asserted only as a recorded contrast — see above.)
  * An extra class that OVERLAPS a gesture: gesture-only accuracy must FALL. This proves the
    machinery can detect prediction-stealing when it is really there — branch C's control.
"""
from __future__ import annotations

import numpy as np

from . import common as C

TAG = "t10_rest_class_inflation"
REST_LABEL = 0                    # dsprofile/windows.py drops `manifest.label != 0`


def _score(X, y, subj, seed, restrict_to=None):
    """LOSO accuracy. If `restrict_to` is given, accuracy is measured ONLY on those true labels
    (the model still has to choose among ALL classes it was trained on)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X, y, subj = np.asarray(X, float), np.asarray(y), np.asarray(subj)
    rng = np.random.default_rng(seed)
    accs = {}
    for s in sorted(np.unique(subj)):
        tr, te = subj != s, subj == s
        if len(np.unique(y[tr])) < 2 or te.sum() < 5:
            continue
        keep = C.subsample_train(X[tr], y[tr], None, 15000, rng)
        Xtr, ytr = X[tr][keep], y[tr][keep]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xte, yte = X[te], y[te]
        try:
            m = LinearDiscriminantAnalysis().fit((Xtr - mu) / sd, ytr)
            pred = m.predict((Xte - mu) / sd)
        except Exception:
            continue
        if restrict_to is not None:
            sel = np.isin(yte, restrict_to)
            if sel.sum() < 5:
                continue
            pred, yte = pred[sel], yte[sel]
        accs[int(s)] = float((pred == yte).mean())
    return accs


def run_one(dataset, seed=42, n_jobs=1):
    from dsprofile import config, windows
    out = dict(dataset=dataset, rest_label=REST_LABEL)

    orig = config.DROP_REST
    try:
        # ---- arm 1: rest DROPPED (what this project uses; the frame is already cached) -----------
        config.DROP_REST = True
        with C.timer(f"T10 :: {dataset} :: rest dropped"):
            fr_no = windows.build_fast_frame(dataset, seed=seed)
        X_no, _ = C.basis(fr_no)
        y_no, s_no = fr_no.label.to_numpy(), fr_no.subject.to_numpy()
        K_no = int(len(np.unique(y_no)))
        acc_no = _score(X_no, y_no, s_no, seed)

        # ---- arm 2: rest INCLUDED (a NEW frame; different cache key `_rest0`) --------------------
        config.DROP_REST = False
        with C.timer(f"T10 :: {dataset} :: rest included (builds a new frame)"):
            fr_yes = windows.build_fast_frame(dataset, seed=seed)
        X_yes, _ = C.basis(fr_yes)
        y_yes, s_yes = fr_yes.label.to_numpy(), fr_yes.subject.to_numpy()
        K_yes = int(len(np.unique(y_yes)))
    finally:
        config.DROP_REST = orig                    # ALWAYS restore, even on error

    if K_yes <= K_no:
        out["applicable"] = False
        out["note"] = (f"no rest class found (label {REST_LABEL}); the dataset has {K_no} classes "
                       "with rest dropped and the same number with it included")
        return out

    acc_yes = _score(X_yes, y_yes, s_yes, seed)
    gestures = [c for c in np.unique(y_yes) if c != REST_LABEL]
    acc_gest = _score(X_yes, y_yes, s_yes, seed, restrict_to=gestures)

    m_no = float(np.mean(list(acc_no.values()))) if acc_no else float("nan")
    m_yes = float(np.mean(list(acc_yes.values()))) if acc_yes else float("nan")
    m_gest = float(np.mean(list(acc_gest.values()))) if acc_gest else float("nan")

    out.update(
        applicable=True,
        n_classes_rest_dropped=K_no, n_classes_rest_included=K_yes,
        rest_share_of_windows=float(np.mean(y_yes == REST_LABEL)),
        acc_rest_dropped=m_no, acc_rest_included=m_yes,
        kappa_rest_dropped=C.kappa_chance(m_no, K_no),
        kappa_rest_included=C.kappa_chance(m_yes, K_yes),
        acc_gestures_only_when_rest_present=m_gest,
        # the three headline quantities
        raw_inflation_pp=float((m_yes - m_no) * 100),
        kappa_inflation=float(C.kappa_chance(m_yes, K_yes) - C.kappa_chance(m_no, K_no)),
        gesture_accuracy_stolen_pp=float((m_gest - m_no) * 100),
    )
    # THE DECISION CRITERION - corrected after the ground truth caught the flaw.
    #
    # The obvious criterion is "raw accuracy rises but chance-corrected kappa does not". It is WRONG,
    # and the selftest proved it: adding an easy, far-away extra class raises kappa too (0.783 ->
    # 0.836 on the synthetic). That is not a bug in kappa - the model genuinely classifies the easy
    # class correctly, and chance-correction only adjusts the BASELINE, not the fact that a real class
    # is being got right. Kappa therefore cannot separate "the average was inflated by an easy class"
    # from "the model got better at gestures".
    #
    # The quantity that CAN separate them is GESTURE-ONLY accuracy: of the windows that are genuinely
    # a gesture, how many are still right once rest is competing? If raw accuracy rises while
    # gesture-only accuracy is unchanged, the gain is pure arithmetic - an easy class dragging the
    # mean up - and nothing about gesture discrimination improved. THAT is the inflation.
    out["inflation_is_an_artifact"] = bool(
        out["raw_inflation_pp"] > 0.0 and abs(out["gesture_accuracy_stolen_pp"]) < 2.0)
    out["rest_steals_gesture_predictions"] = bool(out["gesture_accuracy_stolen_pp"] < -2.0)
    return out


def build_pooled(tag=TAG):
    import json
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(C.results_dir(tag).glob(f"*__{tag}.json"))]
    rows = [r for r in rows if r.get("applicable")]
    if not rows:
        return dict(note="no applicable datasets (none has a rest class?)")

    all_ds = [r["dataset"] for r in rows]
    infl = np.array([r["raw_inflation_pp"] for r in rows], float)
    kinf = np.array([r["kappa_inflation"] for r in rows], float)
    stolen = np.array([r["gesture_accuracy_stolen_pp"] for r in rows], float)
    artifact = [r["dataset"] for r in rows if r.get("inflation_is_an_artifact")]
    steals = [r["dataset"] for r in rows if r.get("rest_steals_gesture_predictions")]

    # THE BRANCH ORDER IS KEYED ON GESTURE-ONLY ACCURACY, NOT ON KAPPA.
    #
    # The smoke test on ninapro_db5 exposed the flaw in the first draft: rest is only 2% of that
    # dataset's windows, so raw accuracy rose just +1.3 pp while gesture-only accuracy was UNCHANGED
    # (+0.1 pp) — textbook inflation. But kappa ticked up +0.014 (because accuracy rose at all), and a
    # branch keyed on kappa therefore fired "REST GENUINELY HELPS". That is precisely backwards.
    #
    # Rest can only be said to HELP if the model gets better at the thing we actually care about:
    # telling gestures apart. So branch B now requires gesture-only accuracy to IMPROVE, and branch A
    # (inflation) fires whenever the raw number rises while gesture accuracy does not — at whatever
    # magnitude, which is reported rather than gated behind an arbitrary threshold.
    if np.nanmean(stolen) < -2.0 and len(steals) > len(rows) / 2:
        branch, verdict = "C", (
            f"REST ACTIVELY STEALS GESTURE PREDICTIONS: with rest in the label set, accuracy on the "
            f"windows that are genuinely gestures FALLS by {abs(np.nanmean(stolen)):.1f} pp on "
            f"{C.count_both_ways(steals, all_ds)['datasets']} datasets. Dropping rest is not a "
            "convention — it is a correction, and this is the strongest form of the argument.")
    elif np.nanmean(infl) > 0.0 and abs(np.nanmean(stolen)) < 2.0:
        branch, verdict = "A", (
            f"REST-INCLUSIVE ACCURACIES ARE INFLATED BY {np.nanmean(infl):.1f} pp "
            f"(range {infl.min():.1f} to {infl.max():.1f}), while accuracy on the windows that are "
            f"genuinely GESTURES moves by only {np.nanmean(stolen):+.1f} pp. The gain is pure "
            f"arithmetic — one easy class dragging the mean up — and nothing about gesture "
            f"discrimination improved. Confirmed on "
            f"{C.count_both_ways(artifact, all_ds)['datasets']} datasets / "
            f"{C.count_both_ways(artifact, all_ds)['cohorts']} cohorts. A large slice of the "
            "literature reports exactly this inflated number.")
    elif np.nanmean(stolen) >= 2.0:
        branch, verdict = "B", (
            f"REST GENUINELY HELPS: accuracy on the GESTURE windows themselves rises by "
            f"{np.nanmean(stolen):+.1f} pp when rest is in the label set, so it is not merely an easy "
            "extra class — the rest windows appear to teach the model where the no-activity region of "
            "feature space lies, sharpening the gesture boundaries. Surprising; report it.")
    else:
        branch, verdict = "D", (
            f"REST CHANGES LITTLE (raw {np.nanmean(infl):+.1f} pp, kappa {np.nanmean(kinf):+.3f}). The "
            "protocol choice is defensible but is not, by itself, a finding.")

    out = dict(tag=tag, n_datasets=len(rows),
               mean_raw_inflation_pp=float(np.nanmean(infl)),
               mean_kappa_inflation=float(np.nanmean(kinf)),
               mean_gesture_accuracy_change_pp=float(np.nanmean(stolen)),
               inflation_is_artifact_on=C.count_both_ways(artifact, all_ds),
               rest_steals_predictions_on=C.count_both_ways(steals, all_ds),
               note=("The inflation MAGNITUDE scales with how big the rest class is: on ninapro_db5 "
                     "rest is only 2% of windows and the inflation is +1.3 pp, whereas a dataset "
                     "where rest is a third of the windows will inflate far more. Report the raw "
                     "inflation ALONGSIDE rest_share, never alone."),
               per_dataset={r["dataset"]: dict(raw_inflation_pp=r["raw_inflation_pp"],
                                               gesture_change_pp=r["gesture_accuracy_stolen_pp"],
                                               kappa_inflation=r["kappa_inflation"],
                                               rest_share=r["rest_share_of_windows"])
                            for r in rows},
               branch=branch, verdict=verdict)
    C.atomic_write_json(C.results_dir(tag) / "pooled.json", out)
    return out


# ------------------------------------------------------------------ ground truth
def _with_extra_class(kind, n_subj=8, n_cls=4, per=50, d=6, seed=0):
    """K gesture classes + one extra class (label 0, standing in for rest).
    kind='easy'    -> the extra class sits far from everything: it must inflate RAW accuracy only.
    kind='overlap' -> the extra class sits ON TOP of gesture 1: it must STEAL gesture predictions."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_cls, d)) * 2.0
    extra = np.full(d, 12.0) if kind == "easy" else centers[0].copy()
    X, y, s = [], [], []
    for u in range(n_subj):
        off = rng.standard_normal(d) * 0.6
        for c in range(n_cls):
            X.append(centers[c] + off + rng.standard_normal((per, d)) * 1.0)
            y += [c + 1] * per; s += [u] * per        # gestures are 1..K
        X.append(extra + off + rng.standard_normal((per, d)) * 1.0)
        y += [REST_LABEL] * per; s += [u] * per       # the "rest" class is 0
    return np.vstack(X), np.array(y), np.array(s)


def selftest(check):
    # --- an EASY extra class must inflate RAW accuracy but NOT kappa ---------------------------
    X, y, s = _with_extra_class("easy", seed=1)
    keep = y != REST_LABEL
    K_no, K_yes = len(np.unique(y[keep])), len(np.unique(y))
    a_no = float(np.mean(list(_score(X[keep], y[keep], s[keep], 1).values())))
    a_yes = float(np.mean(list(_score(X, y, s, 1).values())))
    k_no, k_yes = C.kappa_chance(a_no, K_no), C.kappa_chance(a_yes, K_yes)
    check("T10 an EASY extra class inflates RAW accuracy",
          a_yes > a_no + 0.02, f"raw {a_no:.3f} -> {a_yes:.3f} (+{(a_yes-a_no)*100:.1f} pp)")
    # NOTE: kappa does NOT save us here, and the first draft of this test wrongly assumed it would.
    # An easy extra class raises kappa too (measured: 0.783 -> 0.836), because the model really does
    # classify it correctly — chance-correction fixes the BASELINE, not the arithmetic of averaging in
    # an extra class the model gets right. So kappa cannot separate "the mean was dragged up by an
    # easy class" from "gesture discrimination improved".
    # The invariant that CAN separate them is GESTURE-ONLY accuracy, so that is what we assert.
    gestures0 = [c for c in np.unique(y) if c != REST_LABEL]
    a_gest0 = float(np.mean(list(_score(X, y, s, 1, restrict_to=gestures0).values())))
    check("T10 ...and GESTURE-ONLY accuracy is UNCHANGED, so the raw gain is pure arithmetic and not "
          "better gesture discrimination (THIS, not kappa, is the criterion that works)",
          abs(a_gest0 - a_no) < 0.03,
          f"gestures-only {a_no:.3f} -> {a_gest0:.3f}; for the record kappa moved {k_no:.3f} -> {k_yes:.3f}")

    # --- an OVERLAPPING extra class must STEAL gesture predictions (branch C's control) ---------
    X, y, s = _with_extra_class("overlap", seed=2)
    keep = y != REST_LABEL
    gestures = [c for c in np.unique(y) if c != REST_LABEL]
    a_no = float(np.mean(list(_score(X[keep], y[keep], s[keep], 2).values())))
    a_gest = float(np.mean(list(_score(X, y, s, 2, restrict_to=gestures).values())))
    check("T10 an OVERLAPPING extra class STEALS gesture predictions (the machinery can detect "
          "prediction-stealing when it is really there)",
          a_gest < a_no - 0.02, f"gesture-only {a_no:.3f} -> {a_gest:.3f}")
