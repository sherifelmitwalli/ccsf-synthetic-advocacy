#!/usr/bin/env python3
"""
Stage 4, analysis R6: stance- and prompt-family-matched sensitivity analysis (post hoc).

Implements STAGE4_STANCE_PROTOCOL.md (checksummed and committed before any R6 result).
Uses only frozen Stage 2/3/4 inputs; no generation, no APIs, no real-world data. Seed 20260604.

  R6a  original trained models (full development families), scored on the matched subset
  R6b  all models retrained on matched development subsets, evaluated on the held-out matched subset
  R6c  label-permutation null for R6b (content classifier, hybrid)
  R6d  descriptive register markers in the matched subset

Matched subset per generator family: coordinated and organic accounts with index k % 3 == 0
(supportive organic stance; direct-talking-points prompt family), paired by k on topic,
temperature and transformation. 7 vs 7 per family; 21 vs 21 pooled.
"""
import hashlib
import json
import os
import platform
import re

import numpy as np
import pandas as pd

import stage4_robustness as s4

SEED = 20260604
OUT = os.path.join(s4.OUTDIR, "stance_matched")
FAMILIES = s4.FAMILIES
CCSF, REL = s4.CCSF, s4.REL
NBOOT = 2000
NPERM = 200
MARKERS = ["i", "my", "we", "our"]


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ----------------------------------------------------------------------------- matched subset
def matched_subset(corpus, F):
    meta = corpus.groupby("account_id").first()
    F = F.copy()
    F["k"] = F["account_id"].str.extract(r"_(\d+)$")[0].astype(int)
    keep = F[(F.role.isin(["coordinated", "organic"])) & (F.k % 3 == 0)].copy()
    pairs = []
    for fam in FAMILIES:
        sub = keep[keep.generator_family == fam]
        co = sub[sub.role == "coordinated"].set_index("k")
        org = sub[sub.role == "organic"].set_index("k")
        if sorted(co.index) != [0, 3, 6, 9, 12, 15, 18] or sorted(org.index) != sorted(co.index):
            raise RuntimeError(f"unexpected matched indices in {fam}")
        for k in co.index:
            a, b = meta.loc[co.loc[k, "account_id"]], meta.loc[org.loc[k, "account_id"]]
            for col in ["topic", "prompt_family", "temperature", "transformation_id"]:
                if a[col] != b[col]:
                    raise RuntimeError(f"pair {fam} k={k} differs on {col}")
            if a["prompt_family"] != "direct_talking_points":
                raise RuntimeError("matched pair not in direct-talking-points family")
            if b["style_instruction"] != f"organic_persona_{k % 10}":
                raise RuntimeError("organic persona index does not match k")
            pairs.append({"generator_family": fam, "k": int(k),
                          "coordinated": co.loc[k, "account_id"], "organic": org.loc[k, "account_id"],
                          "campaign": a["campaign_id"], "topic": a["topic"],
                          "temperature": float(a["temperature"]),
                          "transformation_id": a["transformation_id"],
                          "organic_stance_rule": "ORGANIC_STANCES[k % 3] = supportive"})
    return keep.set_index("account_id", drop=False), pairs


# ----------------------------------------------------------------------------- metrics
def boot(y, s, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(NBOOT):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        v.append(s4.ev.fast_auc(y[i], s[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def summarise(P, models):
    rng = np.random.default_rng(SEED)
    out = {"per_family": {}, "pooled": {}}
    for fam in FAMILIES:
        T = P[P.generator_family == fam]
        out["per_family"][fam] = {m: {"auc": s4.ev.fast_auc(T.y.values, T[m].values),
                                      "ci": boot(T.y.values, T[m].values, rng),
                                      "n_coordinated": int(T.y.sum()), "n_organic": int((T.y == 0).sum())}
                                  for m in models}
    for m in models:
        out["pooled"][m] = {"auc": s4.ev.fast_auc(P.y.values, P[m].values),
                            "ci": boot(P.y.values, P[m].values, rng),
                            "n_coordinated": int(P.y.sum()), "n_organic": int((P.y == 0).sum())}
    prng = np.random.default_rng(SEED)
    out["paired_pooled"] = {}
    for a in [m for m in ["hybrid", "relational_only", "ccsf_lr"] if m in models]:
        y, sa, sb = P.y.values, P[a].values, P["content_tfidf_lr"].values
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        d = []
        for _ in range(NBOOT):
            i = np.concatenate([prng.choice(pos, len(pos), True), prng.choice(neg, len(neg), True)])
            d.append(s4.ev.fast_auc(y[i], sa[i]) - s4.ev.fast_auc(y[i], sb[i]))
        out["paired_pooled"][f"{a}_minus_content"] = {
            "delta_auc": s4.ev.fast_auc(y, sa) - s4.ev.fast_auc(y, sb),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out


# ----------------------------------------------------------------------------- R6a
def r6a(F, M):
    content = pd.read_csv(os.path.join(s4.OUTDIR, "stage4_content_predictions.csv")).set_index("account_id")
    P = M[["account_id", "generator_family", "role", "y"]].copy()
    P["ccsf_lr"] = F.loc[M.index, "proba_ccsf_lr"].values
    P["relational_only"] = F.loc[M.index, "proba_relational_only"].values
    P["hybrid"] = F.loc[M.index, "proba_hybrid"].values
    P["content_tfidf_lr"] = content.loc[M.index, "proba_content_tfidf_lr"].values
    return P


# ----------------------------------------------------------------------------- R6b
def matched_features(corpus, M):
    """Relational features recomputed within each family's matched 14-account unlabeled batch."""
    G = M.copy()
    for fam in FAMILIES:
        ids = G[G.generator_family == fam].index.tolist()
        G.loc[ids, REL] = s4.batch_rel(corpus, ids).loc[ids, REL].values
    return G


def r6b(corpus, G, y_override=None, models=("ccsf_lr", "relational_only", "hybrid", "content_tfidf_lr"),
        rng=None):
    docs = s4.account_docs(corpus)
    P = G[["account_id", "generator_family", "role", "y"]].copy()
    for fam in FAMILIES:
        dev, test = G[G.generator_family != fam], G[G.generator_family == fam]
        ydev = dev.y.values if y_override is None else rng.permutation(dev.y.values)
        for name, feats in [("ccsf_lr", CCSF), ("relational_only", REL), ("hybrid", CCSF + REL)]:
            if name in models:
                P.loc[test.index, name] = s4.predict(s4.fit(dev[feats].values, ydev), test[feats].values)
        if "content_tfidf_lr" in models:
            P.loc[test.index, "content_tfidf_lr"] = s4.content_scores(docs, dev.index, ydev, test.index)
    return P


def r6c(corpus, G):
    rng = np.random.default_rng(SEED)
    res = {"content_tfidf_lr": [], "hybrid": []}
    for _ in range(NPERM):
        P = r6b(corpus, G, y_override=True, models=("hybrid", "content_tfidf_lr"), rng=rng)
        for m in res:
            res[m].append(s4.ev.fast_auc(P.y.values, P[m].values))
    return {m: {"n_perm": NPERM, "mean_auc": float(np.mean(v)),
                "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5)),
                "p_value_vs_observed": None} for m, v in res.items()}, res


# ----------------------------------------------------------------------------- R6d
def r6d(corpus, M):
    docs = s4.account_docs(corpus).loc[M.index].str.lower()
    out = {}
    for role in ["coordinated", "organic"]:
        d = docs[M.role == role]
        out[role] = {w: float(d.apply(lambda t: bool(re.search(rf"\b{w}\b", t))).mean()) for w in MARKERS}
        out[role]["n_accounts"] = int(len(d))
    return out


def main():
    np.random.seed(SEED)
    s4.verify_inputs()
    rec = open("STAGE4_STANCE_PROTOCOL.sha256").read().split()[0]
    if sha256("STAGE4_STANCE_PROTOCOL.md") != rec:
        raise RuntimeError("R6 protocol changed after its pre-results checksum")
    os.makedirs(OUT, exist_ok=True)
    corpus, F = s4.load()
    M, pairs = matched_subset(corpus, F)

    models = ["ccsf_lr", "relational_only", "hybrid", "content_tfidf_lr"]
    A = r6a(F, M)
    G = matched_features(corpus, M)
    B = r6b(corpus, G)
    perm, perm_raw = r6c(corpus, G)
    for m in perm:
        obs = s4.ev.fast_auc(B.y.values, B[m].values)
        perm[m]["observed_auc"] = obs
        perm[m]["p_value_vs_observed"] = float((1 + sum(v >= obs for v in perm_raw[m])) / (1 + NPERM))

    results = {
        "seed": SEED, "analysis": "stage4_R6_stance_prompt_matched_post_hoc",
        "matched_subset": {"per_family": "7 coordinated vs 7 supportive organic (k % 3 == 0)",
                           "pooled": "21 vs 21", "pairs": pairs},
        "R6a_original_models_matched_subset": summarise(A, models),
        "R6b_retrained_on_matched_subset": summarise(B, models),
        "R6c_permutation_null_R6b": perm,
        "R6d_register_markers": r6d(corpus, M),
        "reference_unmatched_full_batches": {
            "content_tfidf_lr_pooled_auc_all": json.load(open(os.path.join(s4.OUTDIR, "stage4_results.json")))
            ["R3_baselines"]["pooled"]["content_tfidf_lr"]["auc_all"]},
    }
    A.to_csv(os.path.join(OUT, "r6a_predictions.csv"), index=False)
    B.to_csv(os.path.join(OUT, "r6b_predictions.csv"), index=False)
    G[["account_id", "generator_family"] + REL].to_csv(os.path.join(OUT, "r6b_matched_relational_features.csv"),
                                                       index=False)
    json.dump(results, open(os.path.join(OUT, "r6_results.json"), "w"), indent=1)
    import networkx, scipy, sklearn
    manifest = {
        "analysis": "stage4_R6_stance_prompt_matched_post_hoc", "seed": SEED,
        "protocol_sha256": sha256("STAGE4_STANCE_PROTOCOL.md"),
        "protocol_commit": "1a9485f (pushed before any R6 result was computed)",
        "code_sha256": {f: sha256(f) for f in ["stage4_stance_matched.py", "stage4_robustness.py",
                                               "stage3_relational_features.py"]},
        "input_sha256": {f: sha256(f) for f in [s4.CORPUS, s4.ACCOUNTS, s4.REL_FROZEN, s4.PRED_FROZEN,
                                                os.path.join(s4.OUTDIR, "stage4_content_predictions.csv")]},
        "output_sha256": {f: sha256(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))
                          if f.endswith((".csv", ".json")) and f != "r6_manifest.json"},
        "packages": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "scipy": scipy.__version__, "networkx": networkx.__version__},
        "generation_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    json.dump(manifest, open(os.path.join(OUT, "r6_manifest.json"), "w"), indent=1)
    for key in ["R6a_original_models_matched_subset", "R6b_retrained_on_matched_subset"]:
        print(key)
        for m, v in results[key]["pooled"].items():
            fam = " ".join(f"{f.split('_')[1]}={results[key]['per_family'][f][m]['auc']:.3f}" for f in FAMILIES)
            print(f"  {m:18s} pooled {v['auc']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]  {fam}")
        print("  paired", {k: round(v["delta_auc"], 3) for k, v in results[key]["paired_pooled"].items()})
    print("R6c", {m: (round(v["mean_auc"], 3), round(v["p2.5"], 3), round(v["p97.5"], 3), round(v["p_value_vs_observed"], 4))
                  for m, v in perm.items()})
    print("R6d", results["R6d_register_markers"])


if __name__ == "__main__":
    main()
