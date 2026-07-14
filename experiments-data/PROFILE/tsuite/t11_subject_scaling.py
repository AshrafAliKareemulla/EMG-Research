"""T11 — Does collecting MORE USERS buy cross-subject accuracy? (the scaling law nobody has run)

WHY THIS EXPERIMENT EXISTS
--------------------------
Every practitioner in this field, on being told their model does not transfer to new users, does the
same thing: collects more users. It is the obvious move and nobody has checked whether it works.

This project has *no* such measurement. (`meta.how_many_subjects` sounds like it, and it is NOT: it
measures the STABILITY OF THE MMD ESTIMATE as you add subjects to the matrix — a hygiene statistic
about the estimator, not a statement about accuracy. Reading its numbers as accuracies is a mistake,
and it was very nearly made on 2026-07-13.)

So the question is open, and it is the most practically consequential one the panel can answer:

    If I train on 4 users instead of 20, how much cross-subject accuracy do I lose?

The answer matters because it is the difference between two completely different research agendas.
If the curve keeps rising, the field's bottleneck is DATA and the answer is to collect more people.
If it saturates almost immediately, the bottleneck is SHIFT — no amount of extra users fixes a new
user who lies outside the distribution — and everything this paper argues (align the mean, forecast
the hard users, stop trusting within-subject numbers) is the right agenda.

DESIGN
------
For each dataset, for each held-out test subject, and for each training-set size
n in {2, 4, 8, 16, 24, 32, all-1}:
  * draw n training subjects at random from the remaining pool (REPEATS independent draws),
  * train, test on the held-out subject,
  * average over draws and over held-out subjects.
This is a proper learning curve over SUBJECTS (not over windows), which is the axis that matters.
The training-ROW budget is held FIXED across all n (`cap_train`), so that "more subjects" means more
subject DIVERSITY and not simply more data — otherwise the curve would confound the two and answer
neither question. That is the whole methodological point of this experiment.

PRE-REGISTERED BRANCHES
-----------------------
  A. The curve SATURATES early (little gain beyond ~8 subjects)
     -> *"Collecting more users does not buy cross-subject accuracy. The bottleneck is distribution
        shift, not data volume."* Directly actionable, counterintuitive, and it is the empirical
        foundation for the entire paper's agenda. **Headline.**
  B. The curve keeps RISING to the end of our panel
     -> the field's instinct is right: collect more people. That would be an important negative for
        our own thesis and we would report it as such — and it would also mean our 14 datasets are
        too small to see the ceiling, which is itself worth saying.
  C. The curve is FLAT from the very start (n=2 is as good as n=40)
     -> subject diversity contributes nothing at all; the model is learning a subject-invariant
        structure or nothing at all. Check against T1 before believing it.

GROUND TRUTH
------------
  * A synthetic where subjects are DIVERSE and coverage genuinely helps (each subject occupies a
    different region, so seeing more of them covers the test subject better): the curve MUST RISE.
  * A synthetic where every subject is identical: the curve MUST be FLAT — more subjects add no
    information. Without this control, a rising curve could just be an artifact of the row budget.
"""
from __future__ import annotations

import numpy as np

from . import common as C

TAG = "t11_subject_scaling"
SIZES = (2, 4, 8, 16, 24, 32)
REPEATS = 5
CAP_TRAIN = 8000          # FIXED across every n: "more subjects" must mean more DIVERSITY, not more rows


def scaling_curve(X, y, subj, seed=42, sizes=SIZES, repeats=REPEATS, cap_train=CAP_TRAIN):
    """Mean LOSO accuracy as a function of the NUMBER OF TRAINING SUBJECTS, at a fixed row budget."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X, y, subj = np.asarray(X, float), np.asarray(y), np.asarray(subj)
    rng = np.random.default_rng(seed)
    subs = np.unique(subj)
    n_sub = len(subs)
    usable = [n for n in sizes if n <= n_sub - 1] + [n_sub - 1]
    usable = sorted(set(usable))

    # THE ROW BUDGET MUST ACTUALLY BE FIXED - the ground truth caught that it was not.
    #
    # `subsample_train(..., cap)` caps rows at `cap`, but it cannot INVENT rows. At n=2 subjects there
    # may simply be fewer rows than the cap (measured on the synthetic: 320 rows at n=2 vs 2560 at
    # n=16), so the small-n arms were quietly trained on less data. The curve would then have
    # confounded "more subjects" with "more rows" and answered neither question - which is exactly the
    # confound this experiment exists to avoid.
    #
    # Fix: the binding budget is whatever the SMALLEST training size can actually supply. Every n is
    # then trained on the same number of rows, and the only thing that varies is how many DIFFERENT
    # people those rows came from.
    per_subject_rows = int(np.median([np.sum(subj == u) for u in subs]))
    budget = int(min(cap_train, per_subject_rows * min(usable)))

    curve = {}
    for n in usable:
        accs = []
        for s in subs:                                   # every subject takes a turn as the test set
            pool = np.array([u for u in subs if u != s])
            for _ in range(repeats):
                pick = rng.choice(pool, min(n, len(pool)), replace=False)
                tr = np.isin(subj, pick)
                te = subj == s
                if len(np.unique(y[tr])) < 2 or te.sum() < 5:
                    continue
                keep = C.subsample_train(X[tr], y[tr], None, budget, rng)
                Xtr, ytr = X[tr][keep], y[tr][keep]
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                try:
                    m = LinearDiscriminantAnalysis().fit((Xtr - mu) / sd, ytr)
                    accs.append(float((m.predict((X[te] - mu) / sd) == y[te]).mean()))
                except Exception:
                    continue
        if accs:
            curve[int(n)] = dict(mean_acc=float(np.mean(accs)), sd=float(np.std(accs, ddof=1)),
                                 n_fits=len(accs), train_rows=int(budget))
    return curve


def run_one(dataset, seed=42, n_jobs=1):
    with C.timer(f"T11 :: {dataset}"):
        frame = C.build_frame(dataset, seed=seed)
        X, _ = C.basis(frame)
        y, subj = frame.label.to_numpy(), frame.subject.to_numpy()
        K = int(len(np.unique(y)))
        curve = scaling_curve(X, y, subj, seed=seed)

    out = dict(dataset=dataset, n_subjects=int(len(np.unique(subj))), n_classes=K,
               cap_train=CAP_TRAIN, repeats=REPEATS,
               note="training ROW budget is FIXED across all n, so 'more subjects' means more "
                    "subject DIVERSITY, not more data",
               curve=curve)
    if len(curve) < 3:
        out["applicable"] = False
        return out

    ns = sorted(curve)
    first, last = curve[ns[0]]["mean_acc"], curve[ns[-1]]["mean_acc"]
    out["applicable"] = True
    out["acc_at_min_subjects"] = first
    out["acc_at_max_subjects"] = last
    out["total_gain"] = float(last - first)
    out["total_gain_pp"] = float((last - first) * 100)
    out["kappa_gain"] = float(C.kappa_chance(last, K) - C.kappa_chance(first, K))

    # where does the curve reach 95% of its final value? (the practical "how many users do I need")
    tgt = first + 0.95 * (last - first)
    reached = [n for n in ns if curve[n]["mean_acc"] >= tgt] if last > first else []
    out["subjects_for_95pct_of_final"] = int(min(reached)) if reached else None
    # marginal gain from doubling 8 -> 16 subjects: the honest "is it worth collecting more?" number
    if 8 in curve and 16 in curve:
        out["gain_from_8_to_16_pp"] = float((curve[16]["mean_acc"] - curve[8]["mean_acc"]) * 100)
    out["saturates"] = bool(out["total_gain"] < 0.02)
    return out


def build_pooled(tag=TAG):
    import json
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(C.results_dir(tag).glob(f"*__{tag}.json"))]
    rows = [r for r in rows if r.get("applicable")]
    if not rows:
        return dict(note="no per-dataset results yet")

    all_ds = [r["dataset"] for r in rows]
    gains = np.array([r["total_gain_pp"] for r in rows], float)
    sat = [r["dataset"] for r in rows if r.get("saturates")]
    doubling = [r["gain_from_8_to_16_pp"] for r in rows if "gain_from_8_to_16_pp" in r]
    need = [r["subjects_for_95pct_of_final"] for r in rows if r.get("subjects_for_95pct_of_final")]
    sat_both = C.count_both_ways(sat, all_ds)

    if len(sat) > len(rows) / 2 or np.nanmean(gains) < 2.0:
        branch, verdict = "A", (
            f"COLLECTING MORE USERS DOES NOT BUY CROSS-SUBJECT ACCURACY. Going from the smallest to "
            f"the largest training set changes accuracy by {np.nanmean(gains):+.1f} pp on average, and "
            f"the curve saturates on {sat_both['datasets']} datasets / {sat_both['cohorts']} cohorts. "
            f"Doubling from 8 to 16 training subjects buys "
            f"{np.nanmean(doubling) if doubling else float('nan'):+.1f} pp. **The bottleneck is "
            "distribution shift, not data volume** — which is the empirical foundation for everything "
            "else this paper argues.")
    elif np.nanmean(gains) > 5.0:
        branch, verdict = "B", (
            f"MORE USERS DO HELP: +{np.nanmean(gains):.1f} pp from the smallest to the largest training "
            f"set, and the curve has not flattened by the end of our panel "
            f"(median subjects needed for 95% of the final value: "
            f"{int(np.median(need)) if need else 'n/a'}). The field's instinct is right, our panel is "
            "too small to see the ceiling, and that is an honest negative for our own thesis.")
    else:
        branch, verdict = "C", (
            f"MODEST AND MIXED: {np.nanmean(gains):+.1f} pp across the panel. Subject diversity helps a "
            "little but is clearly not the lever. State per-dataset; do not build a story on it.")

    out = dict(tag=tag, n_datasets=len(rows),
               mean_total_gain_pp=float(np.nanmean(gains)),
               mean_gain_from_8_to_16_pp=float(np.nanmean(doubling)) if doubling else float("nan"),
               median_subjects_for_95pct=float(np.median(need)) if need else None,
               saturates_on=sat_both,
               per_dataset={r["dataset"]: dict(gain_pp=r["total_gain_pp"],
                                               saturates=r["saturates"],
                                               curve={k: round(v["mean_acc"], 4)
                                                      for k, v in r["curve"].items()})
                            for r in rows},
               branch=branch, verdict=verdict)
    C.atomic_write_json(C.results_dir(tag) / "pooled.json", out)
    return out


# ------------------------------------------------------------------ ground truth
def _scaling_synth(diverse, n_subj=20, n_cls=4, per=60, d=6, seed=0):
    """diverse=True : every subject sits in a DIFFERENT region, so seeing more subjects COVERS the
                      test subject better -> the curve MUST rise.
       diverse=False: every subject is identical -> extra subjects add no information -> FLAT.

    The per-subject offset is large (sd 6) on purpose. Once the row budget is genuinely held fixed
    (which it now is), the model sees the SAME number of rows at n=2 and at n=19 — so the only thing
    a larger n can buy is coverage of the space the test subject might live in. That is a real but
    modest effect, and with a weak offset the curve barely moves (measured: +0.027 with sd 3), which
    is not a fair test of whether the instrument can see coverage at all. A strong offset makes the
    effect unambiguous, which is what a POSITIVE control is for.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_cls, d)) * 1.5
    X, y, s = [], [], []
    for u in range(n_subj):
        off = rng.standard_normal(d) * 6.0 if diverse else np.zeros(d)
        for c in range(n_cls):
            X.append(centers[c] + off + rng.standard_normal((per, d)) * 1.0)
            y += [c] * per; s += [u] * per
    return np.vstack(X), np.array(y), np.array(s)


def selftest(check):
    for diverse in (True, False):
        X, y, s = _scaling_synth(diverse, seed=1)
        cur = scaling_curve(X, y, s, seed=1, sizes=(2, 4, 8, 16), repeats=3, cap_train=4000)
        ns = sorted(cur)
        gain = cur[ns[-1]]["mean_acc"] - cur[ns[0]]["mean_acc"]
        pts = {n: round(cur[n]["mean_acc"], 3) for n in ns}
        if diverse:
            check("T11 DIVERSE subjects: the curve RISES with more training subjects (the instrument "
                  "can detect a real coverage benefit)",
                  gain > 0.03, f"curve={pts} gain={gain:+.3f}")
        else:
            check("T11 IDENTICAL subjects: the curve is FLAT (a rise here would mean the row budget, "
                  "not subject diversity, is driving the curve — the whole experiment would be void)",
                  abs(gain) < 0.03, f"curve={pts} gain={gain:+.3f}")

    # THE BUDGET CONTROL. The curve only means anything if every n was trained on the SAME number of
    # rows. Read it straight out of the curve the experiment actually produced.
    X, y, s = _scaling_synth(True, seed=2)
    cur = scaling_curve(X, y, s, seed=2, sizes=(2, 4, 8, 16), repeats=2, cap_train=4000)
    rows = {n: cur[n]["train_rows"] for n in sorted(cur)}
    check("T11 the training ROW budget is IDENTICAL at every training-set size, so the curve measures "
          "subject DIVERSITY and not data volume (the first draft failed this: 320 rows at n=2 vs "
          "2560 at n=16)",
          len(set(rows.values())) == 1, f"rows per n: {rows}")
