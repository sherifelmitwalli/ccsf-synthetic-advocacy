#!/usr/bin/env python3
"""Verify frozen MiniLM sensitivity outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CFG = json.loads(Path("stage3_minilm_sensitivity_config.json").read_text(encoding="utf-8"))
OUT = Path("stage3_outputs/minilm_sensitivity")
SEMANTIC = [
    "minilm_top3_cosine_mean",
    "minilm_max_cosine",
    "minilm_local_neighborhood_centroid_similarity",
    "minilm_mutual_knn_weighted_degree",
    "minilm_mutual_knn_clustering",
    "minilm_mutual_knn_component_size_normalized",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"[PASS] {name}")


def main():
    for path, expected in CFG["input_sha256"].items():
        check(f"input checksum: {path}", sha(path) == expected)
    features = pd.read_csv(OUT / "stage3_minilm_account_features.csv")
    check("156 unique accounts", len(features) == features.account_id.nunique() == 156)
    check("six semantic features present", set(SEMANTIC).issubset(features.columns))
    check("no labels or campaign IDs in semantic feature file", not {"class_label", "campaign_id"}.intersection(features.columns))
    check("finite semantic features", np.isfinite(features[SEMANTIC].to_numpy()).all())
    check("three 52-account batches", features.groupby("generator_family").size().eq(52).all())
    predictions = pd.read_csv(OUT / "stage3_minilm_predictions.csv")
    model_cols = ["minilm_semantic_only", "minilm_semantic_plus_lexical", "ccsf_plus_minilm_semantic_plus_lexical"]
    check("prediction probabilities bounded", predictions[model_cols].apply(lambda x: x.between(0, 1).all()).all())
    permutation = json.loads((OUT / "stage3_minilm_permutation_results.json").read_text(encoding="utf-8"))
    check("permutation controls at chance", all(0.40 <= value["mean_auc"] <= 0.60 for value in permutation.values()))
    manifest = json.loads((OUT / "stage3_minilm_manifest.json").read_text(encoding="utf-8"))
    model_file = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / CFG["model"]["revision"] / "model.safetensors"
    check("frozen MiniLM model file checksum", sha(model_file) == manifest["model_file_sha256"])
    check("model checksum matches frozen config", manifest["model_file_sha256"] == CFG["model"]["model_file_sha256"])
    check("feature invariance checks passed", manifest["feature_invariance"] == {"label_permutation": "passed", "row_shuffle": "passed"})
    print("ALL MINILM SENSITIVITY CHECKS PASSED")


if __name__ == "__main__":
    main()
