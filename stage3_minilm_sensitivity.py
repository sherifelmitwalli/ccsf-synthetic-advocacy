#!/usr/bin/env python3
"""Frozen post hoc MiniLM sensitivity analysis for Stage 3."""

import hashlib
import json
import os
import platform
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import stage3_evaluate as ev


CFG_PATH = Path("stage3_minilm_sensitivity_config.json")
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
SEED = CFG["seed"]
OUTDIR = Path("stage3_outputs/minilm_sensitivity")
CORPUS_PATH = Path("stage2_outputs/stage2_validation_corpus.csv")
ACCOUNT_PATH = Path("stage2_outputs/stage2_account_features.csv")
PRIMARY_REL_PATH = Path("stage3_outputs/stage3_account_relational_features.csv")
PRIMARY_PRED_PATH = Path("stage3_outputs/stage3_predictions.csv")
K = CFG["graph"]["k"]

SEMANTIC_FEATURES = [
    "minilm_top3_cosine_mean",
    "minilm_max_cosine",
    "minilm_local_neighborhood_centroid_similarity",
    "minilm_mutual_knn_weighted_degree",
    "minilm_mutual_knn_clustering",
    "minilm_mutual_knn_component_size_normalized",
]
LEXICAL_OVERLAP_FEATURES = ["top3_jaccard_mean", "max_jaccard"]
COMBINED_FEATURES = SEMANTIC_FEATURES + LEXICAL_OVERLAP_FEATURES
CCSF_FEATURES = ev.CCSF_FEATURES


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_inputs():
    for path, expected in CFG["input_sha256"].items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Input checksum mismatch: {path}")


def model_path():
    root = Path.home() / ".cache" / "huggingface" / "hub"
    path = root / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / CFG["model"]["revision"]
    if not (path / "model.safetensors").exists():
        raise RuntimeError(f"Frozen MiniLM snapshot not found: {path}")
    return path


def mutual_knn_features(centroids):
    n = len(centroids)
    similarities = centroids @ centroids.T
    np.fill_diagonal(similarities, -np.inf)
    nearest = np.argsort(-similarities, axis=1, kind="mergesort")[:, :K]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in nearest[i]:
            if i in nearest[j]:
                graph.add_edge(i, int(j), weight=max(float(similarities[i, j]), 0.0))
    weighted_degree = dict(graph.degree(weight="weight"))
    clustering = nx.clustering(graph, weight="weight")
    component_size = {}
    for component in nx.connected_components(graph):
        for node in component:
            component_size[node] = len(component) / n
    return similarities, nearest, weighted_degree, clustering, component_size


def features_from_vectors(corpus, post_vectors):
    corpus = corpus.copy()
    corpus["_row"] = np.arange(len(corpus))
    records = []
    for family in sorted(corpus["generator_family"].unique()):
        batch = corpus[corpus.generator_family == family]
        accounts = sorted(batch["account_id"].unique())
        centroids = []
        for account in accounts:
            rows = batch.loc[batch.account_id == account, "_row"].to_numpy()
            centroid = post_vectors[rows].mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids.append(centroid / norm if norm else centroid)
        centroids = np.asarray(centroids)
        sim, nearest, degree, clustering, component = mutual_knn_features(centroids)
        for i, account in enumerate(accounts):
            top = nearest[i]
            neighborhood = centroids[top].mean(axis=0)
            norm = np.linalg.norm(neighborhood)
            neighborhood = neighborhood / norm if norm else neighborhood
            records.append({
                "account_id": account,
                "generator_family": family,
                "minilm_top3_cosine_mean": float(sim[i, top].mean()),
                "minilm_max_cosine": float(sim[i, top[0]]),
                "minilm_local_neighborhood_centroid_similarity": float(centroids[i] @ neighborhood),
                "minilm_mutual_knn_weighted_degree": float(degree.get(i, 0.0)),
                "minilm_mutual_knn_clustering": float(clustering.get(i, 0.0)),
                "minilm_mutual_knn_component_size_normalized": float(component.get(i, 1 / len(accounts))),
            })
    return pd.DataFrame(records).sort_values("account_id").reset_index(drop=True)


def build_features(corpus):
    model = SentenceTransformer(str(model_path()), local_files_only=True)
    post_vectors = model.encode(
        corpus["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    return features_from_vectors(corpus, post_vectors), post_vectors


def load_analysis_frame(features):
    accounts = pd.read_csv(ACCOUNT_PATH)
    lexical = pd.read_csv(PRIMARY_REL_PATH)[["account_id"] + LEXICAL_OVERLAP_FEATURES]
    frame = accounts.merge(features.drop(columns=["generator_family"]), on="account_id", validate="1:1")
    frame = frame.merge(lexical, on="account_id", validate="1:1")
    frame["y"] = (frame["class_label"] == "coordinated_synthetic").astype(int)
    frame["role"] = frame["account_id"].str.extract(r"_(coordinated|organic|professional)")[0]
    return frame


def primary_predictions(frame):
    primary = pd.read_csv(PRIMARY_PRED_PATH)
    columns = {
        "proba_ccsf_lr": "ccsf_lr",
        "proba_relational_only": "relational_only",
        "proba_hybrid": "hybrid",
    }
    if not set(columns).issubset(primary.columns):
        raise RuntimeError(f"Unexpected primary prediction columns: {primary.columns.tolist()}")
    return frame[["account_id"]].merge(
        primary[["account_id"] + list(columns)].rename(columns=columns),
        on="account_id",
        validate="1:1",
    )


def paired_bootstrap(frame, a, b, endpoint, nboot=2000):
    mask = np.ones(len(frame), dtype=bool) if endpoint == "all" else frame.role.isin(["coordinated", "organic"]).to_numpy()
    y = frame.loc[mask, "y"].to_numpy()
    sa = frame.loc[mask, a].to_numpy()
    sb = frame.loc[mask, b].to_numpy()
    point = ev.fast_auc(y, sa) - ev.fast_auc(y, sb)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(nboot):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        values.append(ev.fast_auc(y[idx], sa[idx]) - ev.fast_auc(y[idx], sb[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"delta_auc": float(point), "ci": [float(lo), float(hi)]}


def main():
    np.random.seed(SEED)
    verify_inputs()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(CORPUS_PATH)
    features, post_vectors = build_features(corpus)
    permuted = corpus.copy()
    permuted["class_label"] = np.random.default_rng(SEED).permutation(permuted["class_label"].to_numpy())
    permuted_features = features_from_vectors(permuted, post_vectors)
    if not np.allclose(features[SEMANTIC_FEATURES], permuted_features[SEMANTIC_FEATURES], atol=1e-12):
        raise RuntimeError("MiniLM features changed after label permutation")
    shuffled = corpus.sample(frac=1, random_state=SEED)
    shuffled_vectors = post_vectors[shuffled.index.to_numpy()]
    shuffled = shuffled.reset_index(drop=True)
    shuffled_features = features_from_vectors(shuffled, shuffled_vectors)
    if not np.allclose(features[SEMANTIC_FEATURES], shuffled_features[SEMANTIC_FEATURES], atol=1e-12):
        raise RuntimeError("MiniLM features changed after row shuffle")
    features.to_csv(OUTDIR / "stage3_minilm_account_features.csv", index=False)
    frame = load_analysis_frame(features)

    model_features = {
        "minilm_semantic_only": SEMANTIC_FEATURES,
        "minilm_semantic_plus_lexical": COMBINED_FEATURES,
        "ccsf_plus_minilm_semantic_plus_lexical": CCSF_FEATURES + COMBINED_FEATURES,
    }
    results = {}
    predictions = frame[["account_id", "generator_family", "class_label", "role", "transformation_id"]].copy()
    for name, columns in model_features.items():
        ev.rng = np.random.default_rng(SEED)
        results[name] = ev.evaluate_model(frame, name, columns)
        predictions[name] = ev.logo_predict(frame, columns)

    primary = primary_predictions(frame)
    predictions = predictions.merge(primary, on="account_id", validate="1:1")
    comparisons_frame = frame[["account_id", "role", "y"]].merge(
        predictions.drop(columns=["generator_family", "class_label", "role", "transformation_id"]),
        on="account_id",
        validate="1:1",
    )
    comparisons = {endpoint: {} for endpoint in ["all", "organic_only"]}
    pairs = [
        ("minilm_semantic_only", "ccsf_lr"),
        ("minilm_semantic_plus_lexical", "relational_only"),
        ("ccsf_plus_minilm_semantic_plus_lexical", "hybrid"),
        ("minilm_semantic_plus_lexical", "minilm_semantic_only"),
        ("ccsf_plus_minilm_semantic_plus_lexical", "ccsf_lr"),
    ]
    for endpoint in comparisons:
        for a, b in pairs:
            comparisons[endpoint][f"{a}_minus_{b}"] = paired_bootstrap(comparisons_frame, a, b, endpoint)

    permutations = {}
    for name, columns in model_features.items():
        permutations[name] = ev.permutation_control(frame, columns, CFG["permutation_resamples"])

    predictions.to_csv(OUTDIR / "stage3_minilm_predictions.csv", index=False)
    (OUTDIR / "stage3_minilm_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUTDIR / "stage3_minilm_comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (OUTDIR / "stage3_minilm_permutation_results.json").write_text(json.dumps(permutations, indent=2), encoding="utf-8")

    import sentence_transformers
    import sklearn
    import torch
    manifest = {
        "status": "complete_pending_independent_verification",
        "analysis": "post_hoc_stage3_minilm_sensitivity",
        "seed": SEED,
        "model": CFG["model"],
        "input_sha256": CFG["input_sha256"],
        "protocol_sha256": sha256("STAGE3_MINILM_SENSITIVITY_PROTOCOL.md"),
        "config_sha256": sha256(CFG_PATH),
        "code_sha256": sha256(__file__),
        "model_file_sha256": sha256(model_path() / "model.safetensors"),
        "feature_invariance": {"label_permutation": "passed", "row_shuffle": "passed"},
        "features": model_features,
        "packages": {
            "python": platform.python_version(),
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "networkx": nx.__version__,
        },
    }
    (OUTDIR / "stage3_minilm_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in results:
        all_auc = results[name]["pooled"]["all"]["roc_auc"]
        organic_auc = results[name]["pooled"]["organic_only"]["roc_auc"]
        print(f"{name}: all={all_auc:.3f}, organic={organic_auc:.3f}")


if __name__ == "__main__":
    main()
