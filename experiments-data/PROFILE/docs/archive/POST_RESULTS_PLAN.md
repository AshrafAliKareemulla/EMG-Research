> # ⚠ PARTLY SUPERSEDED (2026-07-10)
>
> Stage 1's checklist of "things to confirm" is still the right idea, but the specific
> expectations are wrong: the E3 mean/cov flip it asks you to confirm **cannot occur** (the
> estimator is affine-invariant), and the senic confound it asks you to confirm **is not
> supported** by the probe's own criterion. Stage 1 also assumes `knn_loo` and `hdiv` are valid;
> both were leaked.
>
> Use **`NEXT_STEPS.md`** (rewritten 2026-07-10) as the operative plan, and `CORRECTIONS.md`
> for what changed. The Stage-2 figure list and Stage-3 write-up guidance here remain useful.

# PROFILE — What to do once the Phase-2 `results/` come back

All experiment code is built + tested (104 checks: 46 feature-math + 31 sdi/meta + 27 blocks).
This file is the roadmap for turning the raw `results/` into the finished data-science paper.
Three stages: **(1) review & consolidate → (2) figures → (3) write-up.**

---

## Stage 0 — bring the results back
On the box, copy back `experiments-data/PROFILE/results/` **excluding** `results/_feature_cache/`
(that's the large intermediate). Expected new Phase-2 subfolders:
`block_a/ block_b/ block_c/ block_d/ calibration/ robust_difficulty/ actionability/ faabos/
senic_probe/ module6_sdi/ meta/ transfer/` — plus the Phase-1 `module1..5/`.

Completeness check (I'll run this): every per-dataset experiment has 14 JSONs (faabos = emaha only,
senic_probe = senic only), aggregate ones (sdi, meta, transfer) have 1 each, no `[FAIL]` rows, no NaN/Inf.

---

## Stage 1 — review & consolidate (I do this, like the Phase-1 review)
Per-block sanity check + assemble cross-dataset tables. **Specific things to confirm** (these validate
the Phase-1 caveats + the new depth experiments):

- **Block C / E2 (A4-fix):** does `inter_subject_within_session_mmd > inter_day_within_subject_mmd`
  now hold cleanly? (Phase-1's conflated version wrongly suggested the opposite.) Expected: inter-subject > inter-day.
- **Block C / E3 (mean/cov on raw):** on RAW features the **mean term should dominate** (Yoneda); on
  z-scored features the **covariance term dominates** → explains why z-score helps. Confirm the flip.
- **Block E / robust_difficulty:** do LDA/SVM/RF **agree on who is hard** (inter-classifier Spearman > ~0.5),
  and does MMD-to-pool predict difficulty for **all three** (not just LDA)? If yes → difficulty is
  classifier-agnostic and the SDI headline is bulletproof.
- **Block E / actionability:** is `auc_guided > auc_random` on most datasets (guided_advantage > 0)?
  If yes → the SDI is a *useful tool*, not just a predictor.
- **senic_probe (E10):** confirm the positive-correlation anomaly is a session-imbalance confound
  (loso & mmd both fall with session count) → report senic separately as an outlier.
- **SDI (E1):** re-check LODO mean/median Spearman; report **with and without senic**.
- **Meta (Block F):** what-makes-hard (separability dominates), pooled meta-analysis r + 95% CI, atlas
  clusters, how-many-subjects (~20–30). Apply **FDR / multiple-comparison correction** to the many
  per-dataset correlations here.

Deliverables of Stage 1: one **cross-dataset comparison table per block** (parquet/xlsx) + a short
`RESULTS_REVIEW_PHASE2.md` mirroring the Phase-1 review.

---

## Stage 2 — figures (I build a `figures.py`, once numbers are seen)
Not yet coded (viz.py only has the 2 Phase-1 figures). Build the paper figures:
1. **Dataset atlas** — PCA scatter of the 14 datasets, coloured by cluster (Block F).
2. **Meta-analysis forest plot** — per-dataset difficulty r + 95% CI + pooled diamond.
3. **SDI scatter** — predicted difficulty vs actual LOSO accuracy (pooled, standardised), LODO.
4. **Calibration curves** — accuracy vs #calibration reps (E8), + the **actionability** guided-vs-random
   -vs-oracle budget curves (the "SDI is useful" figure).
5. **Robustness panel** — difficulty correlation per classifier × seed (bars ± std).
6. **Distribution-shift heatmaps** — inter-subject MMD (have Phase-1 code) + the mean-vs-cov decomposition bars.
7. **Channel-reduction & sampling-rate curves** — accuracy retained vs #channels / vs fs (Block D).
8. **Feature-reliability ranking** — bar chart across datasets (Block A) + complexity-adds-info bars.
9. **Transfer-compatibility heatmap** — dataset × dataset MMD, best-source annotations (Block F).

All theme-aware, colour-blind-safe, saved to `results/figures/`.

---

## Stage 3 — write-up (largest remaining effort; partly yours)
Assemble into the 6-section paper (Blocks A–F), leading with the two novelty pillars:
- **Pillar 1 — the SDI** (portable, classifier-agnostic, LODO-validated, *actionable*).
- **Pillar 2 — the science-of-datasets meta-analysis** (what makes a dataset hard; the atlas; pooled effect).
Answer the scientific questions (A4, A6, A8, A9, C1/C5, K1/K2, L4, I2). Be honest about:
- **ADL framing** — most datasets are gesture; EMAHA is the ADL one → "sEMG datasets *including* ADL",
  lean on the FAABOS (E9) result for the ADL-specific angle.
- **Cross-dataset confounds** — datasets differ in device/fs/channels; the atlas mixes them; acknowledge.
- **Target** — difficulty = self-computed classical LOSO (self-contained; BENCH-LOSO/DL set aside).
- **Multiple comparisons** — report FDR-corrected significance.

---

## Optional (only if a reviewer/venue wants more)
- Window-length robustness (re-run key blocks at 100 ms; config already supports it).
- Add entropy/complexity summaries as extra SDI predictors (does it beat MMD+KL alone?).
These are small; not required for a complete paper.

---

## STATUS
- Experiment code: **100% built + tested** (21 experiments, 14 datasets, 104 checks green).
- Pending: run on box → Stage 1 review → Stage 2 figures → Stage 3 write-up.
