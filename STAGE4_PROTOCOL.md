# Stage 4 protocol — post hoc robustness audit of the Stage 3 relational result

Status: written and checksummed **before** any Stage 4 result was computed (2026-10-20).
Scope: exploratory, post hoc, synthetic. Uses only the frozen Stage 2 corpus/features and the
frozen Stage 3 relational features. No new text generation, no API calls, no real-world data.
Seed: 20260604. The Stage 1–3 outputs are not modified.

## Motivation
Stage 3 relational features were computed transductively within held-out generator-family
batches in which each coordinated account had 9 same-campaign siblings (20 coordinated of 52
accounts, prevalence 38%). Each campaign also used a fixed talking-point repertoire that was
shared across generator families, and the two repertoires cover disjoint topic subsets.
Stage 4 asks how much of the Stage 3 result depends on these design properties.

## Analyses (all leave-one-generator-family-out; development-only scaling and fitting)

**R1 Campaign-size dilution.** For each held-out family h and campaign c, form batches of k
accounts sampled from c (k in {1, 2, 3, 5, 10}) plus all 32 controls of h; the other campaign
is removed. Reference: the frozen Stage 3 batch (both campaigns, k=20). Relational features are
recomputed within each batch with the frozen Stage 3 function. Relational-only and hybrid models
are trained on the frozen development-family features exactly as in Stage 3. CCSF-LR scores are
batch-invariant. A training-free score (max nearest-neighbour TF-IDF cosine) is also reported.
100 random draws per (h, c, k) for k < 10. Metrics per draw: ROC AUC vs all controls, ROC AUC vs
organic-style controls, and R-precision (precision among the top-k ranked accounts).
Summaries: mean and 2.5/97.5 percentiles over draws (sampling variability, not a CI).

**R2 Mixed-generator background.** As R1 for k in {1, 3, 10}, but the unlabeled batch also
contains all 64 controls from the two development families (96 controls in total). Scoring uses
only the held-out family's accounts (k positives vs 32 held-out controls). Labels of background
accounts are never used.

**R3 Baselines on the frozen full batches.**
(a) Training-free scores: max TF-IDF cosine and mean top-3 Jaccard, used directly as scores.
(b) Supervised content classifier: one document per account (8 posts concatenated); TF-IDF
word 1-2 grams, sublinear tf, min_df=2, fit on development documents only; L2 logistic
regression (C=1, class_weight=balanced, liblinear).
ROC AUC with 2000 stratified bootstrap 95% CIs; pooled and per family.

**R4 Cross-repertoire transfer.** Train on development families using coordinated accounts from
repertoire A only (plus all development controls); test on the held-out family using coordinated
accounts from repertoire B only (plus all held-out controls); both directions x 3 families = 6
splits. Because repertoires cover disjoint topics by design, this is a joint unseen-repertoire,
unseen-topic test. Models: CCSF-LR, content classifier (R3b), relational-only, hybrid. For
relational models, features are recomputed within each development and held-out batch that
contains only the retained campaign plus controls. Pooled ROC AUC with bootstrap CIs and paired
bootstrap differences (hybrid minus content classifier; relational-only minus content).

**R5 MiniLM sensitivity of R1.** Repeat R1 (k in {1, 3, 10}) with the six MiniLM semantic
relational features of the frozen Stage 3 sensitivity analysis, using the pinned model revision
and file checksum from stage3_minilm_sensitivity_config.json. Run only if the checksum matches.

## Expectations stated in advance (reported whatever the outcome)
E1: relational discrimination declines as k decreases; at k=1 it is no better than CCSF-LR.
E2: the supervised content classifier is near ceiling under leave-one-family-out (shared
repertoire) and degrades under cross-repertoire transfer.
E3: relational features degrade less than the content classifier under cross-repertoire
transfer because they do not require repertoire-specific vocabulary.

## Integrity
Inputs are verified against the SHA256 values recorded in stage3_minilm_sensitivity_config.json
and stage3_outputs/stage3_manifest.json. Outputs go to stage4_outputs/ with a manifest of input,
protocol, and code checksums and package versions. verify_stage4.py re-checks partitions,
label-free feature construction (invariance to label permutation), reproduction of the frozen
Stage 3 features for k=20, and determinism.
