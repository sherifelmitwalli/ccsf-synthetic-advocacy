"""Verify CCSF preserved baseline and independent-validation package state.

The baseline checks target baseline/ so later experimentation cannot overwrite the
construct-validity artifacts. Independent-validation claims are checked only when
an executed provenance-complete corpus is present.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import corpus_gen

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "baseline"
fails = []


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        fails.append(name)


# ---- preserved baseline ----
check("preserved baseline directory exists", BASELINE.is_dir())
required = ["synthetic_corpus.csv", "account_features.csv", "results.json", "arrays.npz",
            "fingerprint_profiles.csv", "results_provenance.json", "dataset_manifest.json"]
for name in required:
    check(f"baseline/{name} exists", (BASELINE / name).is_file())

posts = corpus_gen.build_corpus()
groups = [p["group"] for p in posts]
accounts = {p["account"]: p["group"] for p in posts}
check("baseline generator has 524 posts", len(posts) == 524)
check("baseline generator has 78 accounts", len(accounts) == 78)
check("baseline generator keeps legacy mixed group for compatibility", groups.count("mixed") == 80)
check("legacy mixed is documented as rule-perturbed synthetic",
      "rule-perturbed synthetic" in (BASELINE / "BASELINE_README.md").read_text(encoding="utf-8").lower())

if (BASELINE / "results.json").is_file():
    results = json.loads((BASELINE / "results.json").read_text(encoding="utf-8"))
    auc = results["auc"]
    def close(value, target, tolerance=0.02):
        return abs(value - target) <= tolerance
    check("baseline composite AUC vs simulated human ~0.995", close(auc["composite_vs_both"], 0.995))
    check("baseline rule-perturbed contrast retained (~0.955)", close(auc["composite_mixed_vs_organic"], 0.955))
    check("baseline random-fold result is retained only as interpolation (~0.980)",
          close(auc["logreg_cv_mean"], 0.980, tolerance=0.03))
    check("baseline provenance maps legacy output terminology", (BASELINE / "results_provenance.json").is_file())
    check("baseline dataset manifest identifies construct-validity scope",
          json.loads((BASELINE / "dataset_manifest.json").read_text(encoding="utf-8"))["kind"] == "controlled synthetic construct-validity baseline")

# ---- reproducible account-level analysis safeguards ----
analysis_source = (ROOT / "analysis.py").read_text(encoding="utf-8")
check("scaling is inside supervised cross-validation pipeline",
      "make_pipeline(StandardScaler(), LogisticRegression" in analysis_source and "Xcv = M" in analysis_source)
check("analysis records model revision status", "MODEL_SPECS" in analysis_source)
check("baseline exports rule-perturbed external label", "condition_label" in analysis_source)

# ---- independent-validation framework ----
for name in ["validation_generation.py", "validation_framework.py", "validation_evaluation.py", "validation_corpus_schema.csv",
             "validation_config.example.json", "dataset_registry.json"]:
    check(f"{name} exists", (ROOT / name).is_file())

from validation_framework import REQUIRED_COLUMNS, read_records, validate_records, build_all_holdouts
schema = next(csv.reader((ROOT / "validation_corpus_schema.csv").open(encoding="utf-8")))
check("validation schema matches provenance contract", tuple(schema) == REQUIRED_COLUMNS)
config = json.loads((ROOT / "validation_config.example.json").read_text(encoding="utf-8"))
check("independent validation is not falsely marked as executed", config.get("status") == "not_run")
check("configuration specifies multiple independent generator slots", len(config.get("generators", [])) >= 2)
check("generation template records temperature, diversity, repertoire and transformations", all(config.get(k) for k in ("temperatures", "diversity_settings", "talking_point_repertoires", "automated_transformations")))
check("configuration specifies all required holdouts",
      set(config.get("required_holdouts", [])) == {"generator_id", "prompt_family", "topic", "transformation_id"})

corpus = ROOT / "validation_corpus.csv"
if corpus.exists():
    try:
        records = read_records(corpus)
        validate_records(records)
        output = ROOT / "validation_splits"
        build_all_holdouts(records, output)
        check("executed independent validation corpus has complete provenance", True)
        check("independent validation split manifests were built", all((output / f"leave_one_{field}_out.json").exists()
              for field in ("generator_id", "prompt_family", "topic", "transformation_id")))
    except Exception as exc:
        check(f"executed independent validation corpus is valid ({exc})", False)
else:
    print("[SKIP] Independent generator validation has not run; no validation_corpus.csv is present.")

if fails:
    print(f"{len(fails)} check(s) FAILED: {'; '.join(fails)}")
    sys.exit(1)
print("All applicable checks PASSED.")
