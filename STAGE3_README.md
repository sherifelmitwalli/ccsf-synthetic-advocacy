# Stage 3: exploratory post hoc synthetic relational extension

> **Status.** Stage 3 is an **exploratory, post hoc, entirely synthetic** extension, designed
> *after* the Stage 2 results were known and motivated by them. It is **not prespecified**. It uses
> **no new text generation, no APIs, no real-world data, and no external corpus**, only the frozen
> Stage 2 outputs. Its protocol, configuration, and feature list were frozen and committed **before**
> any Stage 3 held-out performance was computed. All results are reported regardless of direction.

## Scientific question

> Do **label-free cross-account relational features** improve leave-one-generator-family-out
> discrimination of coordinated synthetic accounts compared with the fixed-direction CCSF composite
> and the account-level CCSF-LR model?

Stage 2 showed that the four intuitive *account-level* linguistic signals do not transfer reliably to
LLM text (the fixed-direction composite is significantly below chance). Coordination, however, is a
*relational* property across accounts, not only a property of one account's writing style. Stage 3
tests whether relational structure adds incremental synthetic discrimination.

## Design (batch-relational / transductive, stated openly)

- **Unit:** one account (156 accounts; 60 coordinated, 96 controls).
- **Representation:** deterministic, download-free TF-IDF (word 1-2 grams, L2-normalised) fit within
  each generator-family batch; each account centroid is the L2-normalised mean of its 8 post vectors.
  This was the frozen primary Stage 3 representation. A separately frozen post hoc sensitivity
  analysis subsequently evaluated MiniLM semantic relational features without changing the primary
  analysis.
- **Relational batch:** for each held-out generator family, relational features are computed
  **transductively within that unlabelled 52-account batch**. Generator family defines the batch and
  the splits only, it is never a predictor. No class, campaign, account, prompt, transformation,
  record, model-ID, revision, filename, or row-order variable is used as a feature. This is a
  batch-relational/transductive design in which a test account is related to other *unlabelled*
  accounts in the same held-out batch; it is not obscured.
- **Eight relational features:** `top3_cosine_mean`, `max_cosine`, `top3_jaccard_mean`,
  `max_jaccard`, `semantic_without_literal_reuse` (max cosine × (1 − paired Jaccard)),
  `mutual_knn_weighted_degree`, `mutual_knn_clustering`, `mutual_knn_component_size_normalized`, from a
  deterministic **mutual 3-nearest-neighbour** cosine graph.
- **Four models:** A fixed-direction CCSF (archived, not retrained); B CCSF-LR (4 account features);
  C relational-only (8 relational features); D hybrid (4 account + 8 relational). B-D are L2 logistic
  regression, `C=1`, `class_weight=balanced`, seed 20260604, standardisation fitted on development
  families only.
- **Evaluation:** leave-one-generator-family-out. Two endpoints with equal prominence, coordinated
  vs **all** controls and coordinated vs **organic-style** controls only. ROC/PR AUC with 2,000
  stratified bootstrap CIs, Brier, calibration, sens/spec at 0.5 and a development-only Youden
  threshold, PPV/FDR at 1% prevalence, transformation heterogeneity, and paired bootstrap AUC
  differences.

## Main findings

| Model | Pooled ROC AUC (all controls) | Pooled ROC AUC (organic-only) |
|---|---|---|
| Fixed-direction CCSF | 0.394 [0.308, 0.487] | 0.617 [0.511, 0.716] |
| CCSF-LR (account-level) | 0.799 [0.729, 0.865] | 0.721 [0.624, 0.809] |
| Relational-only | 0.974 [0.951, 0.991] | 0.982 [0.961, 0.997] |
| Hybrid | 0.993 [0.982, 1.000] | 0.996 [0.987, 1.000] |

- **Hybrid − CCSF-LR:** +0.193 [0.129, 0.268] (all controls); +0.275 [0.185, 0.365] (organic-only).
  Both paired-bootstrap CIs exclude zero, and improvement is consistent across all three held-out
  families (hybrid ROC AUC ≥ 0.97 in every family) **and** visible on the harder organic-style
  endpoint, the protocol's success criterion.
- **Calibration improves:** pooled mean Brier 0.192 (CCSF-LR) → 0.038 (hybrid).
- **Interpretation of coefficients:** the dominant relational predictors are nearest-neighbour
  **lexical Jaccard overlap** (`top3_jaccard_mean`, `max_jaccard`), i.e. shared talking-point wording,
  not the graph-topology features.
- **Negative controls preserved:** the fixed-direction CCSF remains below chance (0.394), and the
  **label-permutation control sits at chance** (hybrid 0.501, relational-only 0.502), no leakage.
- **Robustness:** hybrid separation is near-perfect across every transformation cell (0.98-1.00),
  including noise insertion.

## Additional MiniLM sensitivity analysis

The sensitivity protocol was frozen in commit `49a221e` before MiniLM cross-account performance was
calculated. MiniLM semantic-only achieved pooled AUC 0.842 (95% CI 0.776-0.898) against all controls
and 0.976 (95% CI 0.953-0.994) against organic-style controls. The CCSF-MiniLM-lexical hybrid reached
0.998 (95% CI 0.993-1.000) and 0.997 (95% CI 0.992-1.000), statistically equivalent to the primary
TF-IDF hybrid. This shows that semantic relational structure reproduces the harder coordinated-versus-
organic result; semantic relations alone are weaker against professional controls. See
`STAGE3_MINILM_SENSITIVITY_README.md` and `stage3_outputs/minilm_sensitivity/`.

## Essential caveat

The very high hybrid AUC reflects that the synthetic **coordinated accounts were generated from a
shared talking-point repertoire**, so within a batch their content shares recurring repertoire language and
relational similarity largely recovers the generative design. This is a **synthetic construct
demonstration under a transductive batch design**, not evidence of real-world coordination detection,
authentic-human detection, attribution, or operational readiness. The Stage 2 negative finding is not
replaced or hidden by Stage 3.

## Files

Protocol/config (frozen before results): `STAGE3_PROTOCOL.md`, `stage3_config.json`,
`stage3_feature_manifest.json`. Code: `stage3_relational_features.py`, `stage3_evaluate.py`,
`stage3_figures.py`, `verify_stage3.py`. Outputs (`stage3_outputs/`):
`stage3_account_relational_features.csv`, `stage3_predictions.csv`, `stage3_results.json`,
`stage3_model_comparisons.json`, `stage3_permutation_results.json`, `stage3_manifest.json`,
`figures/`.

## Reproduce

```bash
python stage3_relational_features.py   # label-free relational features from frozen Stage 2 text
python stage3_evaluate.py              # four-model leave-one-generator-family-out evaluation (~20 s)
python verify_stage3.py                # 10 leakage / partition / negative-control checks (all PASS)
python stage3_figures.py               # grayscale-readable figures
```

Environment: Python 3.10, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, networkx 3.4.2.
Frozen-input SHA256 checksums are recorded in `stage3_feature_manifest.json` and re-checked by
`verify_stage3.py` (check C1). Stage 1 and Stage 2 outputs are unchanged.
