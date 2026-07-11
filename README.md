# Coordinated Communication Signal Fingerprinting (CCSF) reproducibility and validation package

This repository contains Coordinated Communication Signal Fingerprinting (CCSF), an interpretable tobacco-control surveillance and triage framework for prioritising potentially coordinated synthetic or automatically transformed advocacy for later expert review. It is not an individual-attribution, enforcement, legal-compliance, or real-world-ready system.

## Evidence status

`baseline/` preserves the original seeded 524-post/78-account simulation and its derived artifacts. It is a **controlled construct-validity baseline**: the generator deliberately instantiates the regularities measured by the fingerprint, so its AUCs are not independent validation or operational-performance estimates.

The historical internal group value `mixed` is retained only for compatibility. It always means **rule-perturbed synthetic**: an automated condition created by programmatic transformations, never human editing or human rewriting.

A held-out multigenerator stress test was executed on 2026-07-11 at a preliminary scale (**Stage 2**, `stage2_outputs/`): three unseen LLM generator families with immutable dated revisions, same-model organic and vocabulary-matched professional controls, automated transformations, and leave-one-generator-family-out evaluation. Headline results: the Stage 1 fixed-direction composite does not transfer to LLM text and is significantly below chance (pooled AUC 0.394, 95% CI 0.310-0.481); a logistic reweighting of the same four features (CCSF-LR) reaches held-out-family ROC AUC 0.711-0.878 (mean 0.797), with coefficients that preserve only the burstiness direction; under equivalent-group comparison, untransformed coordinated-vs-organic separation pools near chance and transformation cells are heterogeneous. These are preliminary held-out-generalisation estimates with wide CIs — see `STAGE2_README.md`, `stage2_outputs/stage2_results.json` and the versioned derivative analyses in `stage2_outputs/derived_v1/`. Full provenance (raw responses, generation IDs, catalogue snapshot) is retained. Note on reproducibility: all statistical analyses are exactly reproducible from the archived outputs; the generation step itself queried remotely hosted model revisions and is not guaranteed re-executable — the archived raw responses are the authoritative record.

## Repository map

| File or folder | Purpose |
|---|---|
| `baseline/` | Immutable copy of the original baseline corpus, results, arrays and figures. |
| `corpus_gen.py` | Reproduces the baseline simulation. It uses legacy `mixed` internally; its external condition label is rule-perturbed synthetic. |
| `analysis.py` | Baseline feature extraction at the account level. Logistic-regression scaling is inside its cross-validation pipeline. Pin the declared model revisions before release. |
| `figures.py` | Figure generator with corrected account-level/data-minimising terminology; regenerates repository figures from verified outputs. |
| `validation_corpus_schema.csv` | Required provenance schema for independently generated validation text. |
| `validation_generation.py` | Builds provenance-complete pending generation requests and ingests only verified external model responses. It never fabricates text. |
| `validation_framework.py` | Validates provenance and produces disjoint leave-generator, leave-prompt-family, leave-topic and leave-transformation account splits. |
| `validation_evaluation.py` | Evaluates actual account-level feature outputs with ROC AUC, PR AUC, threshold metrics, bootstrap CIs, calibration and PPV/FDR scenarios. |
| `validation_config.example.json` | Non-executed configuration template for three independent generator families. |
| `dataset_registry.json` | Status and provenance of the baseline and independent-validation datasets. |
| `verify_outputs.py` | Verifies baseline invariants and the independent-validation framework state. |
| `stage2_derived_analysis.py` | Versioned derivative analyses over the frozen Stage 2 outputs (coefficients, development-only thresholds, calibration summary, equivalent-group transformation contrasts, Stage 2 coordinated-originality test). Writes `stage2_outputs/derived_v1/`; no generation, no API calls. |
| `stage2_figures_v2.py` | Revised manuscript and appendix figures from frozen + derived outputs (writes `stage2_outputs/derived_v1/figures/`). |
| `build_r3_docx.py` | Builds the R3 submission DOCX and Multimedia Appendix 7 from the clean manuscript markdown. |
| `verify_stage2_manuscript.py` | Traces every headline R3 manuscript value to frozen or versioned machine-readable outputs; asserts the derived refit matches the frozen evaluation exactly. |
| `MANUSCRIPT_REVISION_NOTES.md` | Deferred manuscript notes and claim-audit guidance. |

## Terminology

- **Commercial/regulatory policy-framing lexicon**: a fixed list of tobacco/nicotine policy-framing phrases. It is not a legal-compliance lexicon, and a match is never evidence of illegality or inauthenticity.
- **Account-level fingerprint**: CCSF scores account-level aggregates. It may support later cluster or campaign investigation, but it does not currently implement cluster discovery as its primary procedure.
- **Data-minimising/privacy-conscious**: derived outputs can omit usernames and text, but the pipeline does not provide a formal privacy guarantee.
- **Textual signals**: perplexity, sentence-length regularity, framing density and semantic convergence are linguistic/stylometric/semantic signals, not behavioural signals.

## Independent-validation workflow

1. Replace the placeholders in `validation_config.example.json`; then build pending requests with `python validation_generation.py build-requests validation_config.json --output generation_requests.jsonl`. Submit those through approved model adapters or retain exported model responses.
2. Ingest only actual responses whose model IDs/revisions match their requests: `python validation_generation.py ingest generation_requests.jsonl model_responses.jsonl --output validation_corpus.csv`. The responses must include matched comparison-control accounts in every evaluation batch.
3. Validate provenance and build only disjoint account-level condition holdouts:

```bash
python validation_framework.py validation_corpus.csv \
  --manifest validation_dataset_manifest.json --split-dir validation_splits
```

4. Extract the four account-level CCSF features into `validation_account_features.csv`. The table must include the provenance columns needed by the selected split plus `class_label` and one row per account.
5. Evaluate each held-out condition, for example:

```bash
python validation_evaluation.py validation_account_features.csv \
  validation_splits/leave_one_generator_id_out.json \
  --output validation_results_leave_generator_out.json
```

Do not use random folds as the principal generalisation evidence. The generated holdouts prevent a held-out generator, prompt family, topic, or automated transformation from appearing in both training and test accounts.

## Current checks

```bash
python verify_outputs.py
```

This validates the archived baseline and confirms that independent validation remains marked `not_run` until real, provenance-complete corpus outputs are supplied. No independent generator result is currently claimed.

## Legacy mappings

Historical code/data fields retain `mixed` and `compliance` only for compatibility. They map respectively to **rule-perturbed synthetic** and **commercial-policy framing**; neither implies human editing or legal compliance.
