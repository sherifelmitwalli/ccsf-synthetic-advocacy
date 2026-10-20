# Stage 4 — post hoc robustness audit of the Stage 3 relational result

**Status:** exploratory, post hoc, synthetic. Uses only the frozen Stage 2 corpus/features and the frozen
Stage 3 relational features. No new text generation, no API calls, no real-world data. Seed 20260604.

The protocol (`STAGE4_PROTOCOL.md`) was written and its SHA-256 recorded in `STAGE4_PROTOCOL.sha256`
before any Stage 4 result was computed. It specifies five analyses and three expectations, all of which are
reported whatever the outcome.

## Why

Stage 3 batches contained 20 coordinated accounts among 52 (two campaigns of ten), so every campaign account
had nine siblings in the batch, and the two campaign repertoires covered disjoint topic subsets. Stage 4 asks
how much of the near-ceiling Stage 3 result depends on these design properties.

## Analyses

| ID | Analysis | Output |
|---|---|---|
| R1 | Campaign-size dilution: k ∈ {1,2,3,5,10} accounts from one campaign + 32 controls; relational features recomputed within each batch; 100 draws per cell | `stage4_dilution_draws.csv` (background `heldout_32`) |
| R2 | As R1 (k ∈ {1,3,10}) with the 64 development-family controls added as unlabeled background | `stage4_dilution_draws.csv` (background `mixed_96`) |
| R3 | Baselines on frozen full batches: training-free max cosine / top-3 Jaccard; supervised TF-IDF content classifier (dev-only vocabulary) | `stage4_results.json` → `R3_baselines`; `stage4_content_predictions.csv` |
| R4 | Cross-repertoire transfer: train on repertoire A, test on repertoire B (unseen repertoire and topics), both directions × 3 families | `stage4_results.json` → `R4_cross_repertoire` |
| R5 | MiniLM sensitivity of R1 (k ∈ {1,3,10}) with the pinned all-MiniLM-L6-v2 revision; model checksum verified | `stage4_minilm_dilution_draws.csv` |

| R6 | Stance- and prompt-matched sensitivity (protocol `STAGE4_STANCE_PROTOCOL.md`, committed before results): 7 coordinated vs 7 index-matched supportive organic accounts per family; R6a original models scored without retraining, R6b retrained on matched subsets, R6c permutation null, R6d pronoun rates | `stance_matched/` |

## Headline results

* **Campaign size matters (E1 supported).** Mean held-out hybrid AUC vs all controls: 0.795 (k=1), 0.888 (2),
  0.933 (3), 0.976 (5), 0.992 (10); CCSF-LR is batch-invariant at ≈0.80. With one campaign account the hybrid
  ranks it first in only 9% of draws (CCSF-LR 32%). MiniLM semantic-only ranks a lone campaign account below
  chance (0.434).
* **The corpus is content-separable (E2 partly supported).** A supervised TF-IDF content classifier reaches
  pooled AUC 1.000 under leave-one-generator-family-out.
* **…even for an unseen repertoire (E2/E3 not supported).** Under cross-repertoire transfer the content
  classifier keeps AUC 0.981 / 0.980; relational-only 0.831 / 0.978; hybrid 0.933 / 0.999; CCSF-LR 0.668 / 0.703.
  A descriptive inspection shows the content classifier uses prompt-condition register (first-person organic
  personas; framing vocabulary in professional prompts; collective, prescriptive campaign voice), which is a
  benchmark-design confound.

* **Stance and prompt matching do not remove separation (R6, post hoc).** Against supportive organic controls
  index-matched on prompt family, topic, temperature and transformation, the content classifier reaches AUC 1.000
  both with the original models (R6a) and when retrained on matched data (R6b; permutation null mean 0.505,
  P = 0.005). This indicates remaining differences between generation conditions (e.g. campaign versus persona
  voice, repertoire language), not proof that register alone explains the results.

## Reproduce and verify

```bash
python stage4_robustness.py --draws 100      # ~18 min on CPU; writes stage4_outputs/
python verify_stage4.py                      # 10 checks (all PASS)
python stage4_stance_matched.py              # R6, ~30 s
python verify_stage4_stance.py               # 10 checks (all PASS)
python stage4_figures.py                     # stage4_outputs/figures/
python make_figure1_frontiers.py             # frontiers_figures/figure1_design.{png,tif}
```

`stage4_robustness.py --skip-minilm` omits R5 (no model download). R5 downloads the pinned MiniLM revision
via `huggingface_hub` and refuses to run if the model-file SHA-256 differs from
`stage3_minilm_sensitivity_config.json`.

Checks in `verify_stage4.py`: (C1) protocol unchanged since the pre-results checksum; (C2) input and (C3) output
checksums; (C4) the batch function reproduces the frozen Stage 3 features exactly; (C5) diluted-batch features
are invariant to label permutation and row shuffling; (C6) cross-repertoire splits are disjoint by family and
repertoire; (C7) dilution draws reproduce deterministically; (C8) content-classifier predictions reproduce;
(C9) the content vocabulary is fitted on development documents only; (C10) the k=20 reference equals the frozen
Stage 3 per-family results.

The full run was executed twice (Python 3.13, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5, NetworkX 3.7,
PyTorch 2.14.1 CPU, sentence-transformers 6.1.0); both runs produced byte-identical outputs. Package versions are
recorded in `stage4_outputs/stage4_manifest.json`.

## Interpretation limits

Stage 4 is post hoc, designed with knowledge of Stages 2 and 3. The descriptive content-classifier inspection
was not pre-specified. Dilution varies campaign size but not the volume or diversity of authentic background
traffic. Nothing here is evidence of real-world detection or attribution.
