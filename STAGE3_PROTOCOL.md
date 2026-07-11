# Stage 3 frozen protocol

## Status and question

Stage 3 is an exploratory, post hoc, entirely synthetic extension prompted by the
below-chance fixed-direction Stage 2 result. It asks whether label-free
cross-account relational features improve leave-one-generator-family-out
discrimination of coordinated synthetic accounts relative to fixed-direction CCSF
and CCSF-LR. It uses no new text, APIs, or real-world data.
Results will be reported whatever their direction or stability. It uses no external or
real-world dataset of any kind.

## Inputs, unit, and batches

The frozen inputs are `stage2_outputs/stage2_validation_corpus.csv`,
`stage2_outputs/stage2_account_features.csv`, the frozen generator split JSON,
and the frozen generator-out evaluation JSON. SHA256 checksums are recorded in
`stage3_feature_manifest.json`. The unit is one account. For each held-out
generator family, relational features are calculated transductively within the
unlabelled held-out-family batch: all its account texts may be used, but no class,
campaign, account, prompt, transformation, filename, row-order, or model-ID
variable is used as a feature. Generator family defines batches/splits only.
Topic is not a predictive categorical feature.

## Frozen relational features

Each account is represented by a deterministic TF-IDF centroid of its eight posts
(unigrams and bigrams; English token pattern; L2-normalised). Within its batch,
exclude itself and compute: (1) mean cosine similarity to the three nearest
centroids; (2) maximum cosine similarity; (3) mean token-set Jaccard overlap to
the three nearest accounts; (4) maximum Jaccard overlap; (5) semantic-without-
literal-reuse index, maximum cosine multiplied by one minus the paired Jaccard;
(6) weighted degree; (7) local clustering coefficient; and (8) normalized
connected-component size. The graph is a deterministic mutual 3-nearest-neighbour
cosine graph with nonnegative cosine weights. Features are unsupervised and must
be invariant to label permutation.

## Models and splits

The three frozen leave-one-generator-family-out splits train on two families and
test on the remaining family. A: fixed-direction CCSF is the archived score and
is not retrained. B: CCSF-LR uses the four frozen account features with a
development-only StandardScaler and L2 logistic regression (`C=1`,
`class_weight=balanced`, seed 20260604). C: relational-only uses the eight frozen
relational features with that specification. D: hybrid concatenates B and C.
No model or feature changes will be made after held-out results are inspected.

## Evaluation and checks

For all controls and organic-style controls separately, report ROC AUC and PR AUC
with 2,000 stratified bootstrap 95% CIs, Brier score, five-bin calibration,
sensitivity/specificity at 0.5 and a development-only out-of-fold Youden
threshold, PPV/FDR at 1% assumed prevalence, and transformation heterogeneity.
Paired held-out bootstrap AUC differences compare relational-only minus CCSF-LR,
hybrid minus CCSF-LR, and hybrid minus fixed CCSF.

Automated checks forbid prohibited columns from matrices, assert disjoint account
partitions and development-only scaling/training, make predictions before loading
test labels for scoring, check label-permutation feature invariance, and run a
permutation-label negative control. Unexpectedly above-chance permutation results
are a leakage stop condition.

## Success criterion

Stage 3 is not called successful unless hybrid improvement is reasonably
consistent across held-out families and visible for coordinated versus
organic-style controls. It is transductive/batch-relational, not evidence of
real-world detection, human-authorship detection, attribution, or readiness.
