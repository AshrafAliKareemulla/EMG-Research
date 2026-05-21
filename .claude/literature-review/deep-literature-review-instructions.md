# sEMG Literature Review — Paper Review Protocol & In-depth detailing

**Purpose** This markdown file is the operating manual for any agent (LLM or human reviewer) understanding structured information from a research paper in the surface EMG (sEMG) literature. It tells the agent *what* to read, *how* to read it, *what* to record, and *what not to do*.

**Extraction** The extraction pipeline follows 2 step approach.
First step is to use the existing MinerU extraction output as your starting point. We have compared MinerU, PaddleOCR and opendataloader. Out of these MinerU is working best for extracting structured output from our research papers. So we chose mineru and we proceed the review with the markdown files present in the "mineru_output" sub-folder in each conference and journal folder present at @"literature-review-papers/conferences-and-journals-complete-list". The markdown files given by mineru contains structured information like text, tables, maths equations and references. The images in the paper are ignored and only the title and other text part around the image is extracted. For this style of literature review, I presume that images are not as important for our research work. So, we go through each and every text, tables and maths equations in detail for our review. There can be few minor issues in some numbering, maths equations, unicode characters, parts of some text etc, this is completely fine and can be ignored unless the markdown file provided is not at all readable/corrupted or wrong extraction or partial extraction or the text completely contradicts the numbers present in the table all over paper. In these type of cases, fall back to your own extraction techniques (this is the second step of our approach) and use the pdf paper directly which is present in the sub-folder "papers". Flag such inappropriate discrepancies.
The naming convention of each pdf paper and each mineru markdown file is of the format: "ieee_<some_number>.pdf" for pdf files and "ieee_<some_number>.md" for mineru markdown files. The file names is the same for pdfs and extracted markdown files.

**Output** You are requested to construct 33 columns of data into an excel sheet (json formatted data). Just prepare a JSON file with keys as column names (given below) and the values of each key will be a string appropriately explaining that key (which is also given below on what is the description of each column) in a proper JSON formatted file, so that after the completion of all papers review in a particular folder, the parent agent will run a small python script which takes all these jsons and create an excel sheet with proper formating and styles. You need not create the excel sheet but just create json files for the same and keep in the "_shared" folder present in each conference and journal folder.



## 0. Literature Review Goal

- My experstise and my background are given in the file @.claude/user-background.md
- **Domain:** The problem statement domain revolves around Surface electromyography (sEMG) signal processing, primarily for gesture / movement / intent recognition, prosthetic control, rehabilitation and other similar areas of research — with the specific research focus on classification of **Activities of Daily Living (ADL)** activities.
- **Goal of the literature review:** Two major approaches to our problem - one is purely machine learning (extract the features manually + model a machine learning algorithm for the same ), second is purely deep learning based technologies (or machine learning + deep learning). So, to address what type of algorithms, signal processing, features, etc tools and techniques are used, we build a comparable, evidence-grounded view across many papers so that downstream decisions — model choice, feature choice, evaluation protocol, robustness experiments, addressing scientific questions, etc — are informed rather than guessed. The goal is basically setup our own research preview of a paper is to help us in such a way that we can present our research in a good way, address atleast few of the very scientific questions we are asking while doing the literature review (list of these questions are given below, there are separate columns for such questions, refer them later in this section) by comparing different methods/results with existing works and answering the scientific questions on why people should read our papers and is it worth reading it and what it brings to the sEMG community.



## 1. Agent Operating Rules (Hard Rules — Do Not Violate)

1. **Read the entire Extracted Markdown File/PDF.** Including abstract, introduction, literature review, related work, methods, all experimental subsections, ablations, discussion, limitations, appendices/supplementary, datasets, math equations, tables, etc. Do **not** rely only on the abstract or conclusion and do not omit any section or sub-section. Read the complete extracted paper or pdf.
2. **Every field gets a concrete answer or the literal string `N/A — not discussed`.** Do not leave fields blank. Do not invent. Do not paraphrase the abstract when the paper actually doesn't say.
3. **Quote sparingly, cite specifically.** When a fact is precise (e.g. "bandpass 20–450 Hz"), include the section/figure/table reference: `(Sec. 3.2)` or `(Table 2)`. Keep direct quotes under 15 words and use them only when wording is itself the point.
4. **Distinguish "the authors claim X" from "X is true."** Use language like *"Authors report ..."*, *"Authors claim ..."*, *"Paper states ..."*. Do not assert claims as fact.
5. **Numbers are not negotiable.** Sampling rate, window length, accuracy, subject count, channel count → record the exact number with units. If a range is given, record the range. If the paper is silent, write `N/A — not discussed`.
6. **Never confuse the proposed dataset with comparison datasets.** Some papers introduce a new dataset and benchmark on existing ones — list all, label each.
7. **Flag inconsistencies.** If the abstract says one number and the results table says another, record both and add a `⚠ Inconsistency:` note.
8. **Do not summarize the related-work section as if it were the paper's own contribution.** Describe pecifically for what the *authors* claim is novel relative to prior work.
9. **Keep limitations honest.** List both author-acknowledged limitations and limitations the agent infers from gaps in the methodology. Label each.
10. **Code & data triage matters.** If the paper has a public code repo and a concrete improvement idea is identifiable, raise the `🚩 HIGH-VALUE FOLLOW-UP` flag.
11. **One record per paper — no per-dataset / per-model splitting.** When the paper uses multiple datasets, list them as bullets *inside* the relevant field (e.g. "Datasets used: • NinaPro DB1 — train+test • CapgMyo dbA — external validation"). When the paper evaluates multiple models, list them as bullets inside the model field. Do **not** create one extraction (json) file per dataset or per model. Summarize the accuracy results and focus more on details of the dataset, methodology, evaluation, findings, limitations, gaps in their research work, what we can infer or use from their research work, is it answering our scientifc questions in any way and similar focus area.



## 2. Reading Protocol (Suggested Order)

This is the order that minimises re-reads:
1. **Title + abstract** — first-pass orientation only.
2. **Tables, Math Equations and their captions** — most paper's real story lives here.
3. **Methods / Approach** section — for preprocessing, segmentation, features, models.
4. **Datasets / Experimental Setup** section — for sampling rate, channels, subjects, gestures, splits.
5. **Results** + **Ablations** — for headline numbers, per-dataset breakdowns, what each component contributes.
6. **Related Work** + **Introduction** (last paragraph) — for novelty claims.
7. **Limitations / Discussion / Future Work** — for the authors' own caveats.
8. **Appendix / Supplementary** — for hyperparameters, hardware, additional plots.
Only after going through the entire document twice, fill in below Section 3 details in order.



## 3. sEMG ADL Literature Review — Excel Extraction Template

The template has three column groups:
- **Core descriptive (Columns 1–17):** factual extraction from the paper.
- **Critical analysis (Columns 18–20):** the agent's *reasoning* about the paper — what's weak, what's reusable, what's worth experimenting with.
- **Scientific questions tracking (Columns 21–34):** one column per scientific-question. Each holds a structured Y/N/Partial fill per sub-question, so you can filter the sheet by "papers that address LOSO", "papers that test SSL" etc.
A general rule for the agent: for every non-trivial claim, either quote a short phrase, or write `Not stated`. Never infer typical pipeline values. Distinguish *what authors say* from *what you (the agent) inferred*. There are separate columns for these and you will find this in the later parts of this section.



## 3.1 Column Group 1 — Core descriptive columns

### Column 1. Paper Title
Enter the title of the paper as is.
**Example:** EMGTTL: Transformers-Based TL for ADL

### Column 2. Paper PDF Link
Enter the link of the pdf paper, you can find it in either the .csv file present in the journal or conference folder or yo can take it from the abstract_screening excel sheet present in that folder.
**Example:** https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9657777

### Column 3. Aim & Contribution Type
Claim of the paper in one detailed sentenced + a tag from a fixed taxonomy. Forces categorization over rambling.
**Taxonomy tags (pick one or more):** `new_method | new_architecture | new_dataset | transfer_learning | domain_adaptation | self_supervised | foundation_model | augmentation_GAN | feature_engineering | benchmark_survey | clinical_study | edge_deployment | interpretability | survey | pure_machine_learning | deep_learning | manual_feature_extraction` or create a unique tag if present outside of this list.
**Example:** `Proposes raw-signal Transformer with cross-dataset TL to bypass handcrafted features for ADL. Tags: new_architecture, deep_learning transfer_learning.`

### Column 4. Datasets used
One line per dataset with key specs. List All datasets if multiple. Also specify if dataset was collected by the author itself and what other datasets he has compared his results to.
**Per dataset capture:** name, # subjects (healthy/impaired split), # channels, sampling rate, # classes/activities, intra-day vs inter-day, public/private, ADL vs gesture vs both or any other.
**Example fill:**
`Own Dataset: Yes/No (in brackets secify the name of the dataset if given or state no name)`
`Existing dataset names the results are compared with: (specify the names of all the datasets used to compare the results)`
`EMAHA-DB1 — 25 healthy, 5ch Noraxon, 2 kHz, 22 ADLs (FAABOS), intra-day, public.`
`EMAHA-DB4 — 10 healthy, 5ch, 1.5 kHz, 8 ADLs, intra-day, public.`

### Column 5. Subjects & acquisition
Subject demographics + electrode/sensor specifics.
**Capture:** N healthy / N impaired (type: amputee/stroke/SCI/etc.), age range, sex split, handedness; electrode placement (forearm flexor/extensor, full-arm, etc.), electrode type (wet gel / dry / HD-grid), device brand if named, skin-prep protocol.
**Why this matters:** Lets you filter "amputee-validated" papers vs healthy-only — critical for real ADL deployment claims.
**Example fill:** `25 healthy adults (22M/3F, age 22–35); 5 wireless wet-gel sensors on forearm flexor + extensor + brachioradialis; Noraxon Ultium; skin-prep not stated.`

### Column 6. Preprocessing pipeline
Every preprocessing step done in order, with parameters.
**Look for:** "Methods → Signal processing" or "Preprocessing" section. Check supplementary if main is vague.
**What to capture in order:**
- Notch filter (50 or 60 Hz)
- Bandpass (filter type, order, cutoff frequencies, zero-phase vs causal)
- Rectification (full-wave / half-wave / none)
- RMS smoothing window
- MVC (max voluntary contraction) normalization
- Per-channel normalization (z-score, min-max, per-trial vs per-session)
- Denoising (wavelet, VMD, EMD, ICA)
- ECG / heartbeat removal
- Bad-channel exclusion criteria
- Motion-artifact handling
- Also notedown the preprocessingstep used if it doesn't match with the any of the above names or procedures.
**If missing:** write `Not stated` — never infer typical defaults.
**Example fill:**
`Notch 50 Hz → 4th-order Butterworth bandpass 20–500 Hz (zero-phase, filtfilt) → full-wave rectification → per-channel z-score (per session). No MVC norm, no ICA, no explicit ECG removal, bad-channel handling not stated.`

### Column 7. Segmentation & windowing
Window length, overlap, decision rate, windowing function, causal vs centered, post-processing, etc related to segmentation of the signals. If there are multiple datasets and different segmentation measurements are used for different datasets, list all of them in order for each dataset.
**Why detail matters:** Shorter windows = lower latency but lower accuracy. This trade-off is core to real-time claims. So this info is vital.
**Example fill:** `(Dataset - 1) 250 ms windows, 50% overlap (causal sliding), no window function specified, no majority-vote post-processing. Decision rate ≈ 8 Hz.`

### Column 8. Handcrafted or Manual features
All handcrafted features by domain. If raw end-to-end signal is used, note it down clearly, write `None — raw signal`. Any type of manual features needs to be listed here like time domain, frequency domain, wavelet domain, etc or any other domain, please list it down clearly. This is one of the most important information in entire paper. And if different features are extracted for different datasets, mention it clearly for each dataset. This column description must be detailed and do not miss any vital info regarding the manual features.
**Categories to fill:**
- *Time-domain:* MAV, RMS, WL (waveform length), ZC (zero crossings), SSC (slope sign changes), VAR, IAV, Hudgins TD4/TD5, Phinyomark TD9, autoregressive coefficients, Willison amplitude.
- *Frequency-domain:* MNF, MDF, PSD bands, peak frequency, spectral entropy, total power.
- *Time-frequency:* STFT, CWT (wavelet family + # levels), DWT, EMD/VMD modes, scalogram.
- *Image-like:* spectrogram, scalogram, EMG-image, Gramian angular field, recurrence plot.
- *Per-channel total count* (e.g., "5ch × 9 features = 45-dim").
**Example fill:** `Time: MAV, RMS, WL, ZC, SSC (Hudgins TD5). Frequency: MNF, MDF. Time-frequency: 4-level DWT (Daubechies db4, energy of each level). Total: 5ch × 8 features = 40-dim per window.`

### Column 9. Learned features / encoder
What type of architecture or algorithm or probabilistic model or any other paradigm used for automatic feature extraction. Here, sometimes the author extracts both manual features and pass it through some deep learning or anyother model to further process them, in that case, mention the manual features explicitly in column 8's description and continue the further extraction of features in this column. Mention clearly if it is a mix of both or exactly one only and this has be done for each dataset discussed in the paper. If the described approach is implemented across all datasets, you can just simply mention `(All datasets)` in brackets like this while describing this column or anyother column in general. If no automatic features, write `None`
**Capture:** family (CNN / RNN / Transformer / hybrid / autoencoder / SSL), exact architecture (layers, hidden dims, kernel sizes, attention heads), input format (raw signal / spectrogram / handcrafted features), parameter count, etc with any other useful information.
**Example fill:** `1D CNN encoder: 4 conv blocks (64→128→256→256 channels, kernel size 7, BatchNorm, ReLU), global average pooling → 256-dim embedding. Followed by 2 Transformer encoder layers (4 heads, model dim 256, FFN 1024). Input: raw 250 ms × 5 channels. Parameters: ~1.2M.`

### Column 10. Feature Selection / Dimensionality Reduction
Whether and how features were reduced, with impact on accuracy. Sometimes the extracted features (both manual features + learned features) are huge in length, so people use some sort of dimensionality reduction methods as follows: `PCA | LDA | ICA | mRMR | ReliefF | SFS / SBS | Boruta | tsfresh + filter | autoencoder bottleneck | attention-based selection | Lasso / L1` or any other method. Capture all the details completely and report it across datasets if given.
**Capture:** method, original dim → reduced dim, accuracy delta, justification given.
**Example fill:** `PCA reducing 40 → 15 dims (explains 92% variance); accuracy drops 0.3pp — suggests heavy feature redundancy. No automated FS method tested.` **Or:** `None — full feature vector used; no analysis of redundancy provided.`

### Column 11. Data Sufficiency & Augmentation
Total samples, per-class balance, if augmentation techniques used or not, whether augmentation actually helped. Note down the data augmentation details if given and across datasets. Note down clearly by what method (gan | gaussian | probabilistic approaches, etc) has been used for data augmentation and clearly write down its steps in order and in-detail if given. Look for "Data" section + any "Augmentation" ablation section.
**Example fill:** `~55k windows total, ~2.5k/class, near-balanced (max/min ratio 1.3). Augmentations: Gaussian noise (σ=0.05), magnitude warp, time warp. Ablation shows +2.1pp gain over no-aug. No synthetic data (GAN/VAE/diffusion).`

### Column 12. Task type
Classification / Regression / Proportional Control / Continuous Decoding, etc. Note if multi-task.
**Example fill:** `Classification: 22 ADL classes (EMAHA-DB1) and 8 ADL classes (DB4). Also reports 5-class FAABOS-category accuracy as auxiliary metric.`

### Column 13. Methodology & Training
Full pipeline including model + training recipe. If multiple models compared, list all baselines + proposed, across datasets.
**Capture:**
- Model family + architecture (one-line summary)
- Loss function
- Optimizer + LR + schedule
- Batch size, # epochs
- Regularization (dropout, weight decay, label smoothing)
- Early stopping criterion + validation source
- Hyperparameter-search method (grid / random / Bayesian / none)
- Model parameter count
- Or any other useful information related to methodology and training
**Example fill:**
`Proposed: 1D-CNN encoder + 2-layer Transformer → MLP classifier. Cross-entropy loss, Adam optimizer, LR 1e-3 with cosine decay, batch 64, 100 epochs, dropout 0.2, weight decay 1e-4, early stop on val loss (patience 10). Hyperparams via grid search on fold-1 val. ~1.2M params. Baselines: SVM-RBF, LDA, Random Forest, vanilla 1D-CNN, BiLSTM — all on identical splits.`

### Column 14. Evaluation protocol ⭐ *(critical — where most loopholes hide)*
**What it captures:** Split strategy + CV scheme + whether test set was touched during model selection. Clearly note down the details in-detail and look for the obvious choices the author has made.
**Tags:** `random_k-fold | within-subject_k-fold | LOSO | LOTO (leave-trial-out) | LOSeO (leave-session-out) | leave-day-out | TSTS (time-series train-test split) | cross-dataset | held-out_subjects` or any similar tags
**Also note:** validation source for hyperparameter selection, # random seeds, mean ± std reporting, statistical test used.
**Example fill:** `5-fold CV within-subject pooled across all subjects (NOT subject-disjoint). Hyperparams tuned via grid search on fold-1 val set. 3 random seeds, mean ± std reported. No LOSO, no cross-day, no statistical significance test.`
**Loophole flag:** `Random split across subjects' samples; inter-subject performance unverified — reported accuracy may not reflect deployment to new users.`

### Column 15. Novelty Positioning
Clearly specify what gap the authors claim (state the authors findings in their literature review (related work section) and what actually they are trying to solve), what's actually new, whether novelty is real or rhetorical.
**Three sub-questions for the agent:**
1. What gap do they claim to fill (paraphrased from intro/related work) ?
2. What is genuinely novel — architecture, training scheme, dataset, combination or any other ?
3. Is the novelty real or rhetorical (does the claimed gap actually apply to the same datasets/protocols they evaluate on) ?
**Example fill:** `Claim: "first to use Transformer + cross-dataset TL on raw EMG for ADL." Real novelty: TL setup between EMAHA-DB1 ↔ DB4. Architectural novelty modest — Transformer on raw EMG has prior art (e.g., Burrello et al. 2024). Rhetorical part: cite limitations of "handcrafted features" but compare against weak handcrafted baselines.` Remember these example filles are only for your understanding and not the format to be followed, you can use any format as long as the answers are noted down properly according to each description of the column.

### Column 16. Main results
Headline numbers, one line per dataset and per protocol. Include metric used and Δ over the strongest baseline.
**Format:** `Dataset (protocol): metric = X% — Δ vs strongest baseline.`
**Example fill:**
`EMAHA-DB1 (5-fold WS): Transformer 64.5%, +TL from DB4 → 66.8% (+2.3pp over Transformer alone, +4.1pp over best ML baseline SVM-RBF).`
`EMAHA-DB4 (5-fold WS): Transformer 68.8%, +TL from DB1 → 71.0% (+2.2pp / +5.0pp).`
`Metric: top-1 accuracy. No LOSO, no F1, no statistical test.`

### Column 17. Reproducibility
Quick scorecard of what's reproducible.
**Fields (each Y/N):** code released, pretrained weights, data splits, hyperparameter file, random seeds, container/env spec, etc related information.
**Example fill:** `Code: No. Weights: No. Splits: Not provided (custom k-fold). Hyperparams: listed in paper. Seeds: not disclosed. Env: Python + PyTorch (versions not stated). Score: 1/6.`


## Group 2 — Critical analysis columns (the agent's *thinking* columns)
These three columns are where the agent reasons rather than extracts. They should read like the agent's own analyst notes — opinionated, specific, actionable.

### Column 18. Limitations, gaps & proposed improvements
A genuine analytical paragraph where the agent identifies what's weak in the paper *and* proposes specific improvements that could be tested.
**Three sub-parts the agent must address:**
1. **What the authors missed or under-addressed**, especially when their own related-work section identified gaps they failed to deliver on.
2. **What specific experiment, methodology change, or hyperparameter tweak would address it** — concrete, not vague.
3. **Why this improvement matters** — expected gain, what it would prove, why a reviewer would care.
**Inferred-limitation checklist for the agent to run through:**
- No LOSO or cross-day evaluation?
- Small or homogeneous subject pool?
- Healthy subjects only — no clinical population?
- No statistical significance test?
- No code / weights / splits released?
- Hyperparameters likely tuned on test set?
- Baselines copied from prior papers vs re-implemented?
- No real-time latency or edge-deployment analysis?
- No failure-mode / confusion-matrix discussion?
- No robustness tests (electrode shift, position, fatigue)?
- No augmentation ablation?
- Did the authors' own literature review promise something they didn't deliver? You can infer or create some of questions on your own like these for listing the limitations properly.
**Example fill:** `The authors claim cross-dataset TL is novel but only test within-subject 5-fold splits — the +2.3pp TL gain may not survive a subject-disjoint protocol, since cross-subject variability is often larger than the gain claimed. Their related-work section flags inter-subject variability as a key open problem but they never benchmark on it. Proposed fix: re-run the same TL setup under LOSO + cross-dataset (DB1→DB4 with the target subject held out), with linear-probe vs full fine-tune ablation to isolate where the gain comes from. Hyperparameter to probe: window length (250 vs 500 vs 1000 ms) — longer windows often help under cross-subject because they smooth out short-term variance, and this is unexplored. Game-changer experiment: add self-supervised masked-reconstruction pretraining on combined unlabeled DB1+DB4 before TL — based on calibration-free SSR results in nearby literature, this likely adds 5+pp on LOSO. Why it matters: would convert a within-subject result into a deployment-credible cross-subject result, which is what reviewers and downstream prosthetics work actually need.` This is just an example format, please use your own thinking and reasoning abilities and properly note down the limitations and the improved framework details to follow on. List any other details which are useful as you wish here in this column.

### Column 19. Reusable techniques for our research
Specific techniques, design choices, or tricks from this paper that are worth borrowing for our own experiments — and what to explicitly *not* borrow.
**Three categories the agent should address:**
1. **Architecture / methodology tricks** worth borrowing.
2. **Preprocessing / feature choices** that seem to work well.
3. **Training / evaluation protocols** worth adopting (or avoiding).
4. Any other things to note down
**Example fill:** `Borrow: per-channel z-score normalization keeps things subject-comparable without needing MVC (good for ADL where MVC trials are awkward); their causal sliding-window scheme translates cleanly to real-time. Adopt as baseline: cross-dataset TL setup (DB1↔DB4) for our experiments — gives us an apples-to-apples comparison number. Borrow with caution: their Transformer architecture is small enough to fine-tune cheaply, but we should add positional encoding (they used learned, sinusoidal may transfer better across subjects). Do NOT borrow: within-subject-only evaluation — we always include LOSO. Do NOT borrow: their lack of a statistical test on a 2pp gain across 25 subjects — likely not significant.`

### Column 20. Game-changer hyperparameters / experimental knobs
Specific values, design choices, or experimental setups in this paper that — if changed — could substantially improve results, OR knobs we should systematically vary in our own experiments. What type of changes we can do to improve the accuracy presented in the proposed baslines. Clearly state the obvious and game changing tricks here.
**Example fill:** `Window length: 250 ms is conservative for ADL; sweep 200/300/500/750/1000 ms — longer windows likely help on coarse FAABOS categories but hurt latency. Learning rate: 1e-3 with cosine decay is fine but no warmup — Transformer training often needs 1k-step linear warmup; could improve final accuracy by 1–2pp. Batch size: 64 is small for ~55k windows; try 256 with linear LR scaling (LR × 4). Augmentation magnitude: σ=0.05 jitter is mild — larger magnitude warping (0.1–0.2) plus time warp may improve cross-subject robustness. Encoder depth: only 2 Transformer layers — sweep 2/4/6/8 to find capacity sweet spot. Pretraining schedule on source dataset before TL is barely tuned (just "fine-tune for 50 epochs") — worth a proper LR/epochs sweep. Patch size for Transformer input: not stated; if they used the whole window as a single token, sub-windowing (10 ms patches → token sequence) is likely a major win.`


## Group 3 — Scientific questions tracking columns (Cols 20–32)
Each scientific question is one column. Inside each column, fill a structured Y/N/Partial for each sub-question of that theme, with a brief evidence note (number + Δ where possible).
**Fill format per cell:**
```
Q-ID: Y/N/Partial — one-line evidence with number + Δ
Q-ID: Y/N/Partial — ...
```
If the paper doesn't touch the theme at all, the agent writes `Question not addressed.`

### Column 21. Generalization & subject-independence
**Sub-questions:**
- A1. Single architecture across multiple datasets without per-dataset tuning?
- A2. LOSO performance reported? Δ vs intra-subject?
- A3. Cross-session / cross-day Δ?
- A4. ★ Is inter-day variability larger or smaller than inter-subject variability?
- A5. Cross-dataset transfer (e.g., DB1→DB4) — accuracy retained?
- A6. ★ Does cross-dataset transfer depend more on shared electrode layout, shared activity taxonomy, or shared population?
- A7. Calibration-free / zero-shot to new subject?
- A8. Few-shot adaptation curve (accuracy vs # calibration samples)?
- A9. ★ Subject-difficulty predictor from a short calibration sample?
**Example fill:**
`A1: Y — same Transformer applied to DB1 and DB4.`
`A2: N — only 5-fold WS, no LOSO.`
`A3: N — intra-day data only.`
`A4: N — not addressed.`
`A5: Y — DB1↔DB4 TL, +2.2pp.`
`A6: N — confounded (DB1 and DB4 share electrode layout and population).`
`A7: N`
`A8: N`
`A9: N`

### Column 22. ML vs DL trade-offs
**Sub-questions:**
- B1. Classical ML (LDA/SVM/RF) head-to-head with DL on identical splits?
- B2. Where does DL win — large data, cross-subject, multi-task, or just headline?
- B3. Parameter/accuracy Pareto curve?
- B4. ★ Is DL's gain statistically significant on LOSO, not just intra-subject?
- B5. Hybrid pipelines (handcrafted → DL classifier, or DL embedding → SVM)?
- B6. Compute/energy cost of DL gain quantified?
**Example fill:**
`B1: Y — SVM 60%, LDA 58%, RF 56%, Transformer 64.5%, +TL 66.8%.`
`B2: Partial — DL wins headline, but no LOSO comparison.`
`B3: N — only one Transformer size tested.`
`B4: N — no statistical test, no LOSO.`
`B5: N — DL vs ML kept separate.`
`B6: N — inference latency not measured.`

### Column 23. Features
**Sub-questions:**
- C1. Handcrafted vs learned features head-to-head?
- C2. Which handcrafted feature set is reported best (Hudgins, Phinyomark, TD-PSD, custom)?
- C3. Dimensionality reduction tested and impact?
- C4. Automated FS methods (mRMR / ReliefF / SFS / Boruta / tsfresh) used?
- C5. ★ Are top features physiologically interpretable?
- C6. ★ Muscle synergies (NMF) used as a stable, subject-invariant basis?
- C7. EMG envelope alone vs full bandwidth — does high-frequency content carry independent info?
- C8. Frequency-band importance analysis (which Hz bands matter)?

### Column 24. Architecture & training
**Sub-questions:**
- D1. CNN vs LSTM/GRU vs Transformer vs hybrid head-to-head after fair tuning?
- D2. Does attention give real gain over CNN+LSTM?
- D3. Multi-task learning (gesture + ADL + force jointly) tested?
- D4. ★ Adversarial subject-invariant training (DANN / gradient reversal)?
- D5. ★ Feature disentanglement (pattern-branch vs subject-branch)?
- D6. Test-time adaptation (entropy minimization, TENT-style) for EMG?
- D7. Subject-identity as auxiliary task?

### Column 25. Data, augmentation, synthetic generation
**Sub-questions:**
- E1. Data-scaling law plotted (accuracy vs # subjects, # trials/class)?
- E2. Synthetic data (GAN/VAE/diffusion) vs classical augmentation comparison?
- E3. ★ Synthetic EMG validated for physiological realism (expert review or reproducing EMG statistics)?
- E4. Mixing multiple public datasets in training — helps or hurts?
- E5. ★ Few-shot real labels vs synthesized samples — equivalence ratio?
- E6. Imbalanced-class handling (focal loss, weighted loss, SMOTE)?

### Column 26. Self-supervised & foundation models
**Sub-questions:**
- F1. SSL pretraining (masked reconstruction, contrastive, predictive) used?
- F2. ★ Which SSL pretext task is best for EMG?
- F3. Does SSL gain scale with pretraining-corpus size (scaling law)?
- F4. ★ Single pretrained encoder transfers across gesture, ADL, force, clinical?
- F5. Linear-probe vs full-fine-tune split of the gain?
- F6. ★ Cross-modal pretraining (EEG, ECG, IMU → EMG) tested?
- F7. Edge-deployable foundation-model size addressed?

### Column 27. Robustness
**Sub-questions:**
- G1. Electrode shift (±1–2 cm) tested?
- G2. Limb-position effect (different arm postures) tested?
- G3. Muscle fatigue (within-session drift) tested?
- G4. Sweat / impedance change effect?
- G5. ★ Sensor doff-don (reattachment) recovery tested?
- G6. Time-of-day effect?
- G7. Hardware/device transfer (different EMG device) tested?

### Column 28. Real-time, deployment, online control
**Sub-questions:**
- H1. End-to-end inference latency reported (≤300 ms prosthetics, ≤100 ms HCI)?
- H2. ★ Offline accuracy vs online closed-loop control gap quantified?
- H3. Edge / MCU deployment (size, quantization, pruning)?
- H4. Throughput on edge (FPS, battery)?
- H5. Continual learning (new class added online without forgetting)?

### Column 29. ADL-specific (your focus area)
**Sub-questions:**
- I1. Continuous ADL sequences with natural transitions, or only pre-segmented isolated activities?
- I2. ★ Hierarchical classifier (FAABOS coarse → ADL fine) vs flat N-class?
- I3. ★ Open-set recognition (reject untrained everyday actions)?
- I4. Within-class intensity / speed variation handled?
- I5. Object-conditioning analyzed (does EMG pattern depend on grasped object)?
- I6. ★ Cross-task transfer (pretrain on isolated gestures, fine-tune on ADL)?
- I7. Co-articulation accuracy at transitions vs steady-state?
- I8. Rest/null class included? Effect on confusion matrix?

### Column 30. Clinical validity
**Sub-questions:**
- J1. Tested on impaired subjects (amputee/stroke/SCI/MS), or healthy only?
- J2. ★ Healthy-trained model evaluated on amputees?
- J3. Performance vs disability severity quantified?
- J4. Performance vs time-post-amputation?
- J5. Sensor count reduction (fewer channels, equivalent accuracy)?

### Column 31. Channels, sampling, sensor design
**Sub-questions:**
- K1. ★ Minimum-channel subset for within-X% of full accuracy (channel selection)?
- K2. High sampling rate (≥1 kHz) necessary, or does 200–500 Hz suffice?
- K3. HD-EMG (64+ ch) vs sparse — accuracy gain worth cost?
- K4. ★ Information-theoretic accuracy ceiling for given (channels × duration × classes)?

### Column 32. Methodological honesty
**Sub-questions:**
- L1. Test set subject-disjoint and untouched during model selection?
- L2. Baselines re-implemented with comparable tuning, or copied?
- L3. Statistical significance + effect size + CI + multiple seeds?
- L4. ★ Is method ranking stable across datasets, or dataset-specific (SOTA-as-noise)?
- L5. ★ Do "novel" methods beat well-tuned baselines on LOSO, not just within-subject?
- L6. Confusion-matrix analysis — which classes drive errors?

### Column 33. Multimodal
**Sub-questions:**
- M1. EMG + IMU fusion tested?
- M2. EMG + force/kinematics tested?
- M3. ★ Fusion strategy compared (late vs early vs cross-attention)?

### Column 34. Any other scientific questions
Here, you have to see the entire paper at once and understand how it is structured and what kind of explicit sceintific question we could answer or has been answered by the author. Check if author has explicitely addresses any questions that we want and if not, you have to explicitly do the literature review in such a way that, each and every detail is captured and properly addressed according to decription of each column. If author doesn;t mention any, just go through the complete paper again, understand the paper as a whole and note down any other scientific questions if is trying to answer.



## 4. Some Rules
**Rule 1** The examples listed in each column is just for your understanding and you should not try to structure in the same format as is present. Approach and note down the information in your manner and make sure it addresses the column description well. 
**Rule 2 — Quote or admit ignorance.** For every non-trivial claim (the focus areas), the you must either quote a short phrase + page number, or write `Not stated`. No inference from typical pipelines.
**Rule 3 — Separate stated vs inferred.** mark explicitly what the authors say vs what the agent (you) concludes. Use prefixes like `Authors:` and `Inferred:` if needed.
**Rule 4 — Two-pass extraction.** For each paper, read the paper twice (different temperatures or different prompt orderings) and diff the outputs. Disagreements flag fields where the paper is genuinely ambiguous — exactly where a human should review.
For the three analysis columns (18, 19, 20), give the agent permission to be opinionated. The whole point of those columns is judgment, not extraction. A neutral cell ("the paper does not address X") is failure; a cell that says "the paper doesn't address X, and changing Y would likely help by Z" is success.

Act like a Machine Learning Researcher with expertise in EMG Signals with more than 25 years of research experience. Remember these are your instructions on how to properly read a EMG research paper, the examples given for each column are just for your reference, you should fill it with your own findings and wording. You can be brief about your findings as well. Enter the contents for each column only after you have performed a through reading of the paper. Go through the entire paper's contents twice for proper understanding and information extraction. Avoid jargon and try to answer as many questions as possible, be concise, explore the novelty, what we can learn from this paper, what type of scientific questions this paper answering or answering which will help our research further in EMG space. Remember here for the scientific questions description, the paper might not explicitly tell what question it addresses or what are its scientific objectives. It is your own duty to identify what scientific questions is the paper answering if there are any. **TAKE YOUR TIME**, **REASON**, **REITERATE**, **ANALYZE** and only then note down the observations.