# Frozen Stage 3 MiniLM sensitivity protocol

## Status and purpose

This is an additional post hoc sensitivity analysis, specified after the frozen
TF-IDF Stage 3 results were known and before any MiniLM cross-account performance
was calculated. It does not replace or alter the primary Stage 3 analysis. It
tests whether semantic cross-account structure contributes beyond literal lexical
overlap in the same frozen synthetic corpus.

## Frozen inputs and representation

The analysis uses only the frozen Stage 2 corpus, account features, generator-
family splits, and frozen Stage 3 outputs listed with SHA256 checksums in
`stage3_minilm_sensitivity_config.json`. No text is generated or changed.

Each post is encoded once with `sentence-transformers/all-MiniLM-L6-v2`, Hugging
Face revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, using normalized
embeddings. Each account centroid is the L2-normalized mean of its eight post
embeddings. Relational features are calculated transductively within each
unlabeled 52-account generator-family batch. Labels and identifiers never enter
the representation or graph construction.

## Frozen semantic and lexical features

The semantic-only set contains six MiniLM features:

1. mean cosine similarity to the three nearest account centroids;
2. maximum cosine similarity to another account centroid;
3. cosine similarity to the normalized centroid of the three nearest accounts;
4. weighted degree in a deterministic mutual 3-nearest-neighbor MiniLM graph;
5. local clustering coefficient in that graph; and
6. normalized connected-component size.

The semantic-plus-lexical set adds the two lexical-overlap measures already
frozen in Stage 3: mean top-three Jaccard overlap and maximum Jaccard overlap.
No TF-IDF cosine or TF-IDF graph feature enters this combined sensitivity set.

## Frozen models and evaluation

Three new L2 logistic-regression models use `C=1`, `class_weight=balanced`, the
`liblinear` solver, and seed 20260604:

- MiniLM semantic-only (6 features);
- MiniLM semantic plus lexical overlap (8 features); and
- CCSF plus MiniLM semantic plus lexical overlap (12 features).

All scaling, fitting, and out-of-fold development threshold selection use the two
development families only. The held-out family is evaluated unchanged. Endpoints,
2,000 stratified bootstrap intervals, calibration, thresholds, and prevalence
metrics follow the frozen Stage 3 procedure for all controls and organic-style
controls separately.

Paired bootstrap comparisons are fixed as:

- MiniLM semantic-only minus CCSF-LR;
- MiniLM semantic-plus-lexical minus frozen TF-IDF relational-only;
- CCSF plus MiniLM semantic-plus-lexical minus frozen TF-IDF hybrid;
- MiniLM semantic-plus-lexical minus MiniLM semantic-only; and
- CCSF plus MiniLM semantic-plus-lexical minus CCSF-LR.

## Verification and reporting rule

Checks must confirm input/model-file hashes, disjoint splits, absence of prohibited
predictors, label-permutation and row-order invariance of features, development-
only scaling, deterministic predictions, and chance-level label-permutation
performance. Results will be reported regardless of direction. This sensitivity
analysis cannot be described as prespecified or as real-world validation.
