# Stage 3 MiniLM sensitivity analysis

This additional post hoc sensitivity analysis was frozen in commit `49a221e`
before MiniLM cross-account performance was calculated. It does not replace the
primary frozen TF-IDF Stage 3 analysis.

The analysis uses `all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` to create normalized account
centroids and six semantic relational features within each unlabeled held-out
generator-family batch. Two frozen Jaccard features provide the lexical-overlap
channel. Models, splits, scaling, thresholds, bootstrap intervals, and
permutation controls follow the frozen Stage 3 procedure.

## Pooled held-out results

| Model | ROC AUC, all controls (95% CI) | ROC AUC, organic-style controls (95% CI) |
|---|---:|---:|
| MiniLM semantic-only | 0.842 (0.776-0.898) | 0.976 (0.953-0.994) |
| MiniLM semantic + lexical overlap | 0.847 (0.784-0.902) | 0.982 (0.961-0.997) |
| CCSF + MiniLM semantic + lexical overlap | 0.998 (0.993-1.000) | 0.997 (0.992-1.000) |

The MiniLM hybrid differed from the frozen TF-IDF hybrid by +0.005 (95% CI
-0.001 to +0.014) against all controls and +0.002 (95% CI -0.005 to +0.010)
against organic-style controls. MiniLM semantic-only improved over CCSF-LR by
+0.256 (95% CI +0.159 to +0.353) for the harder organic-style endpoint. Adding
lexical overlap to MiniLM semantic features changed organic-style AUC by only
+0.006 (95% CI -0.001 to +0.015).

Interpretation: semantic cross-account structure reproduces the relational
separation for coordinated versus organic-style synthetic accounts, so the
finding is not solely an artifact of literal overlap. MiniLM-only models are
weaker against the professional controls; combining semantic/lexical relational
features with the account-level CCSF features restores near-perfect synthetic
separation. This remains a transductive, entirely synthetic, post hoc sensitivity
analysis and is not real-world validation.

## Run and verify

```bash
python stage3_minilm_sensitivity.py
python verify_stage3_minilm_sensitivity.py
```

Outputs are under `stage3_outputs/minilm_sensitivity/`.
