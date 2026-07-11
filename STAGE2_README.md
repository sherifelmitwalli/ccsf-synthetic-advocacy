# Stage 2: held-out multi-generator validation

Stage 1 (`baseline/`) is a rule-based construct verification whose separation statistics are
manipulation checks, not performance estimates. Stage 2 evaluates CCSF against three independent
LLM generator families it was never tuned on, with same-model controls, a vocabulary-matched
professional control, and automated adversarial transformations. No human editing is involved;
human-in-the-loop editing remains an optional future extension.

## Design

| Component | Value |
|---|---|
| Generator families | Google Gemini (`google/gemini-3.1-flash-lite`), Alibaba Qwen (`qwen/qwen3.6-35b-a3b`), ByteDance Seed (`bytedance-seed/seed-2.0-mini`) |
| Transform-only family | Tencent (`tencent/hy3`, paid endpoint) — used only for paraphrase so transforms never leak a generator family |
| Per generator | 20 coordinated + 20 organic-control + 12 vocabulary-matched professional-control accounts, 8 posts each |
| Totals | 156 accounts, 1,248 posts |
| Prompt families | direct_talking_points, grassroots_persona, indirect_policy_discussion |
| Temperatures | 0.35 (controlled), 0.95 (diverse) |
| Transformations | none, llm_paraphrase, noise_insertion, sentence_restructure — applied at account level to coordinated AND organic accounts so every transformation value contains both classes |
| Decisive analysis | leave-one-generator-family-out (train on 2 families, test on the unseen third, rotate), standardisation fitted on development families only |

Model IDs were verified against the live OpenRouter catalogue on 2026-07-11; `snapshot` re-verifies
and archives the catalogue on the execution date. Never use `:free` variants, `latest` aliases, or
auto-routers.

## Run

```
python stage2_run.py selftest                       # offline checks, no API calls
python stage2_run.py all --config stage2_config.json
```

Or phase by phase: `build` -> `snapshot` -> `generate` -> `transform` -> `ingest` -> `features` -> `evaluate`.
`generate` and `transform` call OpenRouter (expected cost: low single-digit dollars; ~192 requests
total) and are resume-safe: re-running retries only failed requests. `OPENROUTER_API_KEY` is read
from the environment or `.env` (gitignored — never commit it).

## Outputs (in `stage2_outputs/`)

- `stage2_requests.jsonl` — full prompts + provenance per account request
- `openrouter_models_snapshot_<date>.json` — catalogue archive
- `stage2_raw_responses.jsonl`, `stage2_paraphrase_responses.jsonl` — raw responses with model ID,
  provider, OpenRouter generation ID, token usage, timestamps (retain for audit)
- `stage2_validation_corpus.csv` — schema-validated corpus (`validation_corpus_schema.csv`)
- `stage2_dataset_manifest.json`, `stage2_splits/` — via `validation_framework.py`
- `stage2_account_features.csv` — GPT-2 perplexity, burstiness, framing density, MiniLM convergence
  (identical instruments and formulas to Stage 1 `analysis.py`)
- `stage2_eval_leave_one_*_out.json` — via `validation_evaluation.py` (ROC/PR AUC, bootstrap CIs,
  calibration, prevalence scenarios)
- `stage2_results.json` — plus per-control-class and per-transformation breakdowns and token usage

## Execution note (2026-07-11)

Stage 2 was executed on 2026-07-11. Returned dated revisions: `gemini-3.1-flash-lite-20260507`,
`qwen3.6-35b-a3b-20260415`, `seed-2.0-mini-20260224`, `hy3-20260706` (paraphrase). 186 requests,
~141k tokens. Parsing rule applied during the run: one account returned more than the requested
8 posts and was truncated to the first 8 (`extract_posts` errors only on insufficient posts);
no text was manually written or edited.

## Interpretation rules

Report Stage 2 as a preliminary external-generalisation study: 3 families, English-only, ANDS-only,
automated transformations only. Controls are LLM-generated, so the contrast is coordinated versus
uncoordinated generation within the same model — not synthetic versus authentic human text. Wide
CIs are expected at 20 coordinated accounts per held-out family.

## Derivative analyses (derived_v1, 2026-07-11)

`python stage2_derived_analysis.py` (optionally `--sections core,coefficients,originality`) reads only
the frozen outputs above and writes versioned derivatives to `stage2_outputs/derived_v1/`:
fixed-composite AUCs with CIs (pooled 0.394, 95% CI 0.310-0.481), per-feature transfer AUCs,
standardized CCSF-LR coefficients with development-bootstrap CIs, development-only Youden thresholds
applied unchanged to held-out families, calibration summaries, equivalent-group transformation
contrasts (professional controls excluded from every cell), and a Stage 2 test of the Stage 1
"coordinated originality" signature. A consistency check refits the frozen evaluation and asserts
identical held-out AUCs. Original Stage 2 evidence is never modified.

## Stage 3 (exploratory, post hoc, synthetic — see `STAGE3_README.md`)

Prompted by the below-chance fixed-direction Stage 2 result, an **exploratory post hoc synthetic**
extension (`stage3_outputs/`) tests whether **label-free cross-account relational features** improve
leave-one-generator-family-out discrimination beyond the account-level features. Stage 3 is **not
prespecified**, performs **no new generation and uses no APIs or real-world data**, and reads only the
frozen Stage 2 outputs above (which it leaves unchanged); its protocol and feature list were frozen
before any Stage 3 performance was computed. Adding relational features raises pooled held-out ROC AUC
from 0.799 (CCSF-LR) to 0.993 (hybrid; 0.996 vs organic-style controls), the fixed-direction composite
stays below chance, and a label-permutation control sits at chance. Because coordinated accounts share
a fixed talking-point repertoire, this is a synthetic construct demonstration under a transductive
batch design — not real-world detection — and it does not replace the Stage 2 negative finding. Full
detail, protocol, and reproduction commands are in `STAGE3_README.md` and `STAGE3_PROTOCOL.md`.
