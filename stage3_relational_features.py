#!/usr/bin/env python3
"""
Stage 3 (exploratory, post hoc, synthetic) label-free relational features.

Builds one row per Stage 2 account from FROZEN Stage 2 post text only.
No new text, no APIs, no real-world data. Deterministic; seed 20260604.

Relational batch = the set of accounts sharing one generator family (52 accounts).
Generator family is used ONLY to define this unlabelled batch (and, downstream, the
leave-one-generator-family-out splits). It is never a predictive feature. No class,
campaign, account, prompt, transformation, record, model-id, revision, filename, or
row-order variable is used to construct any feature.

Text representation (frozen in STAGE3_PROTOCOL.md / stage3_config.json):
  Deterministic TF-IDF (word 1-2 grams, min_df=1, L2 norm), fit per family batch on
  that batch's posts. Each account centroid = L2-normalized mean of its 8 post vectors.
  (The project's neural MiniLM instrument is network-unavailable offline; this
  download-free TF-IDF representation was frozen before any Stage 3 performance was
  seen. Documented as a Stage 3 limitation.)
"""
import json, hashlib
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 20260604
CORPUS = "stage2_outputs/stage2_validation_corpus.csv"
OUT = "stage3_outputs/stage3_account_relational_features.csv"
CFG = json.load(open("stage3_config.json"))
NGRAM = tuple(CFG["tfidf"]["ngram_range"])
KNN = CFG["graph"]["k"]

REL_FEATURES = [
    "top3_cosine_mean", "max_cosine", "top3_jaccard_mean", "max_jaccard",
    "semantic_without_literal_reuse", "mutual_knn_weighted_degree",
    "mutual_knn_clustering", "mutual_knn_component_size_normalized",
]

TOKEN_PATTERN = r"(?u)\b\w\w+\b"  # sklearn default; used for both TF-IDF and Jaccard


def token_set(text, vectorizer):
    analyzer = vectorizer.build_analyzer()
    # unigram token set for lexical Jaccard (from a unigram analyzer)
    return set(t for t in analyzer(text) if " " not in t)


def batch_features(df_batch):
    """Compute relational features for one generator-family batch (label-free)."""
    accounts = sorted(df_batch["account_id"].unique())  # deterministic order
    # aggregate 8 posts per account (join with space; order irrelevant to bag features)
    posts_by_acc = {a: list(df_batch.loc[df_batch.account_id == a, "text"]) for a in accounts}

    # ---- TF-IDF fit on this batch's posts only (unsupervised, within-batch) ----
    all_posts = [p for a in accounts for p in posts_by_acc[a]]
    vec = TfidfVectorizer(ngram_range=NGRAM, min_df=1, norm="l2",
                          lowercase=True, token_pattern=TOKEN_PATTERN)
    vec.fit(all_posts)
    # account centroid = L2-normalized mean of its post vectors
    centroids = np.zeros((len(accounts), len(vec.vocabulary_)))
    for i, a in enumerate(accounts):
        M = vec.transform(posts_by_acc[a]).toarray()
        c = M.mean(axis=0)
        n = np.linalg.norm(c)
        centroids[i] = c / n if n > 0 else c
    S = centroids @ centroids.T  # cosine similarity (centroids are L2-normed)
    np.fill_diagonal(S, -np.inf)  # exclude self

    # lexical token sets (unigram) for Jaccard
    uni = TfidfVectorizer(ngram_range=(1, 1), min_df=1, lowercase=True,
                          token_pattern=TOKEN_PATTERN)
    uni.fit(all_posts)
    tsets = [token_set(" ".join(posts_by_acc[a]), uni) for a in accounts]
    n = len(accounts)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            u = len(tsets[i] | tsets[j])
            jac = (len(tsets[i] & tsets[j]) / u) if u else 0.0
            J[i, j] = J[j, i] = jac
    np.fill_diagonal(J, -np.inf)

    # ---- mutual kNN cosine graph (deterministic) ----
    Scos = S.copy()
    Scos[np.isneginf(Scos)] = -1.0
    knn = {i: set(np.argsort(-Scos[i])[:KNN]) for i in range(n)}
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in knn[i]:
            if i in knn[j] and i != j:
                w = max(0.0, float(Scos[i, j]))
                G.add_edge(i, j, weight=w)
    wdeg = dict(G.degree(weight="weight"))
    clust = nx.clustering(G)  # unweighted local clustering coefficient
    comp_of = {}
    for comp in nx.connected_components(G):
        for node in comp:
            comp_of[node] = len(comp)

    rows = []
    for i, a in enumerate(accounts):
        cos_sorted = np.sort(S[i][~np.isneginf(S[i])])[::-1]
        jac_sorted = np.sort(J[i][~np.isneginf(J[i])])[::-1]
        max_cos = float(cos_sorted[0])
        jstar = int(np.argmax(np.where(np.isneginf(S[i]), -np.inf, S[i])))
        paired_jac = float(J[i, jstar]) if not np.isneginf(J[i, jstar]) else 0.0
        rows.append({
            "account_id": a,
            "top3_cosine_mean": float(np.mean(cos_sorted[:3])),
            "max_cosine": max_cos,
            "top3_jaccard_mean": float(np.mean(jac_sorted[:3])),
            "max_jaccard": float(jac_sorted[0]),
            "semantic_without_literal_reuse": max_cos * (1.0 - paired_jac),
            "mutual_knn_weighted_degree": float(wdeg.get(i, 0.0)),
            "mutual_knn_clustering": float(clust.get(i, 0.0)),
            "mutual_knn_component_size_normalized": comp_of.get(i, 1) / float(n),
        })
    return rows


def main():
    np.random.seed(SEED)
    df = pd.read_csv(CORPUS)
    assert df.groupby("account_id").size().eq(8).all(), "expected 8 posts/account"
    out = []
    for fam in sorted(df["generator_family"].unique()):
        sub = df[df["generator_family"] == fam].copy()
        recs = batch_features(sub)
        for r in recs:
            r["generator_family"] = fam
        out.extend(recs)
    res = pd.DataFrame(out)[["account_id", "generator_family"] + REL_FEATURES]
    res = res.sort_values("account_id").reset_index(drop=True)
    import os
    os.makedirs("stage3_outputs", exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {res.shape[0]} accounts x {len(REL_FEATURES)} relational features")
    print(res.groupby("generator_family").size().to_dict())
    print(res[REL_FEATURES].describe().round(3).to_string())


if __name__ == "__main__":
    main()
