# Stage 4, analysis R6 — stance- and prompt-family-matched sensitivity analysis (post hoc)

Status: written and checksummed **before** any R6 result was computed (2026-10-20).
Scope: exploratory, post hoc, synthetic. Part of the Stage 4 robustness audit; it does not add a stage.
Inputs: the frozen Stage 2 corpus and account features, the frozen Stage 3 relational features and
predictions, and the frozen Stage 4 content-classifier predictions. No new text generation, no API
calls, no real-world data. Seed 20260604. No existing output is modified.

## Motivation

Stage 4 (R3, R4) found that a supervised TF-IDF content classifier separates coordinated from control
accounts almost perfectly, and a descriptive inspection suggested prompt-induced register. But every
coordinated account argued one pro-access position, whereas organic controls were assigned one of three
stances. In `stage2_run.py`, organic stance is `ORGANIC_STANCES[k % 3]` and prompt family is
`families[k % 3]`, with the same index k used for topic (`k % 8`), temperature (`k % 2`) and
transformation (`k % 4`) in every role. Hence the supportive organic controls (k % 3 == 0) all used the
direct-talking-points prompt family, and coordinated account k and organic account k in the same
generator family share topic, prompt family, temperature and transformation.

R6 asks whether near-ceiling discrimination persists when organic controls share the campaign stance
and are index-matched to coordinated accounts on prompt family, topic, temperature and transformation.

## Matched subset (defined from generation metadata only, before any result)

Within each generator family, the matched subset is the 7 coordinated accounts and the 7 organic
accounts with index k in {0, 3, 6, 9, 12, 15, 18} (k % 3 == 0). Each coordinated account k is paired
with organic account k (identical topic, prompt family = direct talking points, temperature and
transformation). Organic stance = "broadly supportive of these products". Pooled: 21 vs 21 accounts.
Professional controls are excluded. The script verifies the pairing from the corpus metadata and from
the k-mod rules, and aborts if any pair differs on topic, prompt family, temperature or transformation.

Remaining known differences between the paired conditions: the campaign prompt (shared talking-point
repertoire, campaign voice) versus the organic prompt (named persona, personal voice); campaign
membership (two campaigns: k even → repertoire 0, k odd → repertoire 1).

## Analyses (leave-one-generator-family-out throughout)

**R6a — original trained models, matched evaluation subset (no retraining).** The frozen held-out
predictions of the models trained on full development families (Stage 3 CCSF-LR, relational-only,
hybrid; Stage 4 content classifier) are restricted to the matched subset of each held-out family and
scored. Relational features are those of the frozen full 52-account batches. This measures how the
original models rank stance- and prompt-matched pairs.

**R6b — retrained on the matched subset.** For each held-out family, every model is refitted using only
the matched subsets of the two development families (14 coordinated vs 14 supportive organic) and
evaluated on the matched subset of the held-out family (7 vs 7). Relational features are recomputed
with the frozen Stage 3 batch function within each family's matched 14-account unlabeled batch
(development and held-out batches separately). Standardization and the TF-IDF vocabulary are fitted on
development accounts only. Models: content classifier (TF-IDF word 1-2 grams, sublinear tf, min_df=2,
L2 logistic regression C=1, balanced, liblinear), CCSF-LR (4 account-level features), relational-only
(8), hybrid (12); same logistic specification as Stage 3.

**R6c — label-permutation null for R6b.** 200 permutations of development labels for the content
classifier and the hybrid, giving the distribution of pooled held-out AUC under no signal at this
sample size.

**R6d — descriptive register markers.** Share of matched accounts whose posts contain the tokens
"i", "my", "we", "our" (case-insensitive, word boundary), by condition.

## Reporting

Per-family ROC AUC (n = 7 vs 7) and pooled ROC AUC (21 vs 21) with stratified bootstrap 95% CIs
(2000 resamples), and the paired pooled differences hybrid − content and relational-only − content.
Per-family estimates on 7 vs 7 are imprecise and are reported as such. R6a and R6b are reported
separately and never combined.

## Interpretation rule (stated in advance)

Residual discrimination in R6a or R6b will be interpreted as evidence that the generation conditions
still differ in ways a classifier can detect after matching on stance, prompt family, topic,
temperature and transformation (for example campaign versus persona voice or repertoire language). It
will not be interpreted as proof that register alone explains the Stage 4 results. A drop to
chance-level discrimination would indicate that stance and prompt family accounted for most of the
separation in the unmatched comparison.

## Integrity

`verify_stage4_stance.py` checks the protocol checksum (`STAGE4_STANCE_PROTOCOL.sha256`), input
checksums, the pairing, subset sizes, family-disjoint development/test partitions, development-only
vocabulary and scaling, invariance of recomputed relational features to label permutation, exact
reproduction of the frozen predictions used in R6a, and deterministic reproduction of R6b.
Outputs: `stage4_outputs/stance_matched/`.
