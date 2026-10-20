#!/usr/bin/env python3
"""
Stage 4: post hoc robustness audit of the Stage 3 relational result.

Exploratory, post hoc, synthetic. Implements STAGE4_PROTOCOL.md (checksummed before any
Stage 4 result was computed). Uses only frozen Stage 2/3 inputs; no generation, no APIs,
no real-world data. Seed 20260604.

  R1  campaign-size dilution of the unlabeled held-out batch
  R2  mixed-generator background (held-out batch + 64 development-family controls)
  R3  baselines on frozen full batches (training-free scores; supervised content classifier)
  R4  cross-repertoire transfer (unseen talking-point repertoire and topic subset)
  R5  MiniLM sensitivity of R1 (pinned revision; runs only if the checksum matches)

Usage:  python stage4_robustness.py [--draws 100] [--skip-minilm]
"""
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import stage3_evaluate as ev
import stage3_relational_features as s3

SEED = 20260604
OUTDIR = "stage4_outputs"
CORPUS = "stage2_outputs/stage2_validation_corpus.csv"
ACCOUNTS = "stage2_outputs/stage2_account_features.csv"
REL_FROZEN = "stage3_outputs/stage3_account_relational_features.csv"
PRED_FROZEN = "stage3_outputs/stage3_predictions.csv"
MINILM_FROZEN = "stage3_outputs/minilm_sensitivity/stage3_minilm_account_features.csv"
MINILM_CFG = "stage3_minilm_sensitivity_config.json"

FAMILIES = ev.FAMILIES
CCSF = ev.CCSF_FEATURES
REL = ev.REL_FEATURES
K_GRID = [1, 2, 3, 5, 10]
K_GRID_R2 = [1, 3, 10]
NBOOT = 2000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_inputs():
    cfg = json.load(open(MINILM_CFG))
    for path, expected in cfg["input_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"input checksum mismatch: {path}")


# ----------------------------------------------------------------------------- data
def load():
    corpus = pd.read_csv(CORPUS)
    af = pd.read_csv(ACCOUNTS)
    rel = pd.read_csv(REL_FROZEN).drop(columns=["generator_family"])
    pred = pd.read_csv(PRED_FROZEN)[["account_id", "proba_ccsf_lr", "proba_relational_only",
                                     "proba_hybrid"]]
    rep = corpus.groupby("account_id")["talking_point_repertoire"].first().rename("repertoire")
    F = af.merge(rel, on="account_id", validate="1:1").merge(pred, on="account_id", validate="1:1")
    F = F.merge(rep, left_on="account_id", right_index=True, validate="1:1")
    F["y"] = (F["class_label"] == "coordinated_synthetic").astype(int)
    F["role"] = F["account_id"].str.extract(r"_(coordinated|organic|professional)")[0]
    return corpus, F.set_index("account_id", drop=False)


def lr():
    return LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                              random_state=SEED)


def fit(X, y):
    sc = StandardScaler().fit(X)
    return sc, lr().fit(sc.transform(X), y)


def predict(model, X):
    sc, clf = model
    return clf.predict_proba(sc.transform(X))[:, 1]


def rprec(y, s):
    k = int(y.sum())
    if k == 0:
        return float("nan")
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].mean())


def batch_rel(corpus, accounts):
    """Recompute frozen-protocol TF-IDF relational features within an unlabeled batch."""
    sub = corpus[corpus.account_id.isin(accounts)].copy()
    return pd.DataFrame(s3.batch_features(sub)).set_index("account_id")


# ----------------------------------------------------------------------------- R1/R2
def dilution(corpus, F, draws, k_grid, extra_background, first_cell_only=False):
    """first_cell_only=True reproduces only the first (family, campaign) cell; used by
    verify_stage4.py to check determinism without re-running every draw."""
    rng = np.random.default_rng(SEED)
    rows = []
    for h in FAMILIES[:1] if first_cell_only else FAMILIES:
        dev = F[F.generator_family != h]
        m_rel = fit(dev[REL].values, dev.y.values)
        m_hyb = fit(dev[CCSF + REL].values, dev.y.values)
        controls = F[(F.generator_family == h) & (F.y == 0)].account_id.tolist()
        background = (F[(F.generator_family != h) & (F.y == 0)].account_id.tolist()
                      if extra_background else [])
        camps = sorted(F[(F.generator_family == h) & (F.y == 1)].campaign_id.unique())
        for camp in camps[:1] if first_cell_only else camps:
            members = sorted(F[F.campaign_id == camp].account_id)
            for k in k_grid:
                n_draw = 1 if k >= len(members) else draws
                for d in range(n_draw):
                    pos = sorted(rng.choice(members, k, replace=False).tolist())
                    batch = pos + controls + background
                    R = batch_rel(corpus, batch)
                    scored = pos + controls  # only held-out-family accounts are scored
                    T = F.loc[scored].copy()
                    T[REL] = R.loc[scored, REL].values
                    s = {
                        "ccsf_lr": T["proba_ccsf_lr"].values,
                        "max_cosine_untrained": T["max_cosine"].values,
                        "relational_only": predict(m_rel, T[REL].values),
                        "hybrid": predict(m_hyb, T[CCSF + REL].values),
                    }
                    y = T.y.values
                    org = (T.role != "professional").values
                    for name, sc in s.items():
                        rows.append({
                            "family": h, "campaign": camp, "k": k, "draw": d, "model": name,
                            "auc_all": ev.fast_auc(y, sc),
                            "auc_organic": ev.fast_auc(y[org], sc[org]),
                            "r_precision": rprec(y, sc),
                            "background": "mixed_96" if extra_background else "heldout_32",
                        })
    return pd.DataFrame(rows)


def reference_full_batch(F):
    """Frozen Stage 3 batches (both campaigns, k=20) as the dilution reference."""
    rows = []
    for h in FAMILIES:
        T = F[F.generator_family == h]
        y = T.y.values
        org = (T.role != "professional").values
        for name, col in [("ccsf_lr", "proba_ccsf_lr"), ("relational_only", "proba_relational_only"),
                          ("hybrid", "proba_hybrid"), ("max_cosine_untrained", "max_cosine")]:
            sc = T[col].values
            rows.append({"family": h, "campaign": "both", "k": 20, "draw": 0, "model": name,
                         "auc_all": ev.fast_auc(y, sc), "auc_organic": ev.fast_auc(y[org], sc[org]),
                         "r_precision": rprec(y, sc), "background": "heldout_32"})
    return pd.DataFrame(rows)


def summarise_draws(D):
    out = {}
    for (bg, model, k), g in D.groupby(["background", "model", "k"]):
        e = {}
        for m in ["auc_all", "auc_organic", "r_precision"]:
            v = g[m].dropna().values
            e[m] = {"mean": float(v.mean()), "p2.5": float(np.percentile(v, 2.5)),
                    "p97.5": float(np.percentile(v, 97.5)), "n_batches": int(len(v))}
        e["per_family_mean_auc_all"] = g.groupby("family")["auc_all"].mean().round(6).to_dict()
        out.setdefault(bg, {}).setdefault(model, {})[str(k)] = e
    return out


# ----------------------------------------------------------------------------- R3
def account_docs(corpus):
    return corpus.groupby("account_id")["text"].apply(lambda t: " ".join(t))


def content_scores(docs, train_ids, y_train, test_ids):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          token_pattern=s3.TOKEN_PATTERN)
    Xtr = vec.fit_transform(docs.loc[train_ids].values)
    clf = lr().fit(Xtr, y_train)
    return clf.predict_proba(vec.transform(docs.loc[test_ids].values))[:, 1]


def pooled_eval(y, s, org_mask):
    ev.rng = np.random.default_rng(SEED)
    a = ev.boot_ci(y, s, ev.fast_auc, NBOOT)
    o = ev.boot_ci(y[org_mask], s[org_mask], ev.fast_auc, NBOOT)
    return {"auc_all": a[0], "auc_all_ci": [a[1], a[2]],
            "auc_organic": o[0], "auc_organic_ci": [o[1], o[2]], "n": int(len(y)),
            "n_pos": int(y.sum())}


def paired(y, sa, sb):
    rng = np.random.default_rng(SEED)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    point = ev.fast_auc(y, sa) - ev.fast_auc(y, sb)
    d = []
    for _ in range(NBOOT):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        d.append(ev.fast_auc(y[i], sa[i]) - ev.fast_auc(y[i], sb[i]))
    return {"delta_auc": float(point), "ci": [float(np.percentile(d, 2.5)),
                                              float(np.percentile(d, 97.5))]}


def baselines(corpus, F):
    docs = account_docs(corpus)
    content = pd.Series(index=F.index, dtype=float)
    for h in FAMILIES:
        dev = F[F.generator_family != h]; test = F[F.generator_family == h]
        content.loc[test.index] = content_scores(docs, dev.index, dev.y.values, test.index)
    y = F.y.values; org = (F.role != "professional").values
    scores = {"max_cosine_untrained": F["max_cosine"].values,
              "top3_jaccard_untrained": F["top3_jaccard_mean"].values,
              "content_tfidf_lr": content.values,
              "ccsf_lr": F["proba_ccsf_lr"].values,
              "hybrid": F["proba_hybrid"].values}
    res = {"pooled": {n: pooled_eval(y, s, org) for n, s in scores.items()}, "per_family": {}}
    for h in FAMILIES:
        m = (F.generator_family == h).values
        res["per_family"][h] = {n: {"auc_all": ev.fast_auc(y[m], s[m]),
                                    "auc_organic": ev.fast_auc(y[m & org], s[m & org])}
                                for n, s in scores.items()}
    res["paired_all"] = {
        "content_minus_hybrid": paired(y, scores["content_tfidf_lr"], scores["hybrid"]),
        "max_cosine_untrained_minus_hybrid": paired(y, scores["max_cosine_untrained"],
                                                    scores["hybrid"]),
    }
    return res, content


# ----------------------------------------------------------------------------- R4
def cross_repertoire(corpus, F):
    docs = account_docs(corpus)
    reps = sorted(F[F.y == 1].repertoire.unique())
    res, split_log = {}, []
    for a, b in [(reps[0], reps[1]), (reps[1], reps[0])]:
        direction = f"train_{a}__test_{b}"
        pieces = []
        for h in FAMILIES:
            keep_dev = F[(F.generator_family != h) & ((F.y == 0) | (F.repertoire == a))].copy()
            keep_test = F[(F.generator_family == h) & ((F.y == 0) | (F.repertoire == b))].copy()
            assert set(keep_dev.index).isdisjoint(keep_test.index)
            # recompute relational features within each retained batch (label-free)
            for fam in keep_dev.generator_family.unique():
                ids = keep_dev[keep_dev.generator_family == fam].index.tolist()
                keep_dev.loc[ids, REL] = batch_rel(corpus, ids).loc[ids, REL].values
            ids = keep_test.index.tolist()
            keep_test.loc[ids, REL] = batch_rel(corpus, ids).loc[ids, REL].values
            out = keep_test[["account_id", "generator_family", "role", "y"]].copy()
            for name, feats in [("ccsf_lr", CCSF), ("relational_only", REL),
                                ("hybrid", CCSF + REL)]:
                out[name] = predict(fit(keep_dev[feats].values, keep_dev.y.values),
                                    keep_test[feats].values)
            out["content_tfidf_lr"] = content_scores(docs, keep_dev.index, keep_dev.y.values,
                                                     keep_test.index)
            out["max_cosine_untrained"] = keep_test["max_cosine"].values
            pieces.append(out)
            split_log.append({"direction": direction, "heldout": h, "n_dev": int(len(keep_dev)),
                              "n_test": int(len(keep_test)),
                              "dev_pos": int(keep_dev.y.sum()), "test_pos": int(keep_test.y.sum())})
        P = pd.concat(pieces)
        y = P.y.values; org = (P.role != "professional").values
        models = ["ccsf_lr", "content_tfidf_lr", "relational_only", "hybrid", "max_cosine_untrained"]
        res[direction] = {
            "pooled": {m: pooled_eval(y, P[m].values, org) for m in models},
            "per_family_auc_all": {h: {m: ev.fast_auc(P[P.generator_family == h].y.values,
                                                      P[P.generator_family == h][m].values)
                                       for m in models} for h in FAMILIES},
            "paired_all": {
                "hybrid_minus_content": paired(y, P.hybrid.values, P.content_tfidf_lr.values),
                "relational_only_minus_content": paired(y, P.relational_only.values,
                                                        P.content_tfidf_lr.values),
            },
        }
    return res, split_log


# ----------------------------------------------------------------------------- R5
def minilm_vectors(corpus):
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    cfg = json.load(open(MINILM_CFG))["model"]
    path = snapshot_download(cfg["model_id"], revision=cfg["revision"])
    got = sha256(os.path.join(path, "model.safetensors"))
    if got != cfg["model_file_sha256"]:
        raise RuntimeError("MiniLM model checksum mismatch; R5 not run")
    model = SentenceTransformer(path, device="cpu")
    vec = model.encode(corpus["text"].tolist(), normalize_embeddings=True, batch_size=64,
                       show_progress_bar=False)
    return np.asarray(vec), got


def minilm_batch(corpus, vectors, accounts):
    import stage3_minilm_sensitivity as mm
    sub = corpus[corpus.account_id.isin(accounts)].copy()
    sub["generator_family"] = "batch"  # one unlabeled batch regardless of source family
    return mm.features_from_vectors(sub.reset_index(drop=True),
                                    vectors[sub.index.to_numpy()]).set_index("account_id")


def minilm_dilution(corpus, F, vectors, draws):
    import stage3_minilm_sensitivity as mm
    SEM = mm.SEMANTIC_FEATURES
    frozen = pd.read_csv(MINILM_FROZEN).set_index("account_id")
    G = F.join(frozen[SEM])
    # reproduction check: full-family batches must match the frozen MiniLM features
    max_dev = 0.0
    for h in FAMILIES:
        ids = G[G.generator_family == h].index.tolist()
        R = minilm_batch(corpus, vectors, ids)
        max_dev = max(max_dev, float(np.abs(R.loc[ids, SEM].values - G.loc[ids, SEM].values).max()))
    rng = np.random.default_rng(SEED)
    rows = []
    for h in FAMILIES:
        dev = G[G.generator_family != h]
        m_sem = fit(dev[SEM].values, dev.y.values)
        m_hyb = fit(dev[CCSF + SEM + ["top3_jaccard_mean", "max_jaccard"]].values, dev.y.values)
        controls = G[(G.generator_family == h) & (G.y == 0)].index.tolist()
        for camp in sorted(G[(G.generator_family == h) & (G.y == 1)].campaign_id.unique()):
            members = sorted(G[G.campaign_id == camp].index)
            for k in K_GRID_R2:
                for d in range(1 if k >= len(members) else draws):
                    pos = sorted(rng.choice(members, k, replace=False).tolist())
                    ids = pos + controls
                    T = G.loc[ids].copy()
                    T[SEM] = minilm_batch(corpus, vectors, ids).loc[ids, SEM].values
                    T[["top3_jaccard_mean", "max_jaccard"]] = batch_rel(corpus, ids).loc[
                        ids, ["top3_jaccard_mean", "max_jaccard"]].values
                    y = T.y.values; org = (T.role != "professional").values
                    for name, sc in [("minilm_semantic_only", predict(m_sem, T[SEM].values)),
                                     ("ccsf_minilm_lexical_hybrid",
                                      predict(m_hyb, T[CCSF + SEM + ["top3_jaccard_mean",
                                                                     "max_jaccard"]].values))]:
                        rows.append({"family": h, "campaign": camp, "k": k, "draw": d,
                                     "model": name, "auc_all": ev.fast_auc(y, sc),
                                     "auc_organic": ev.fast_auc(y[org], sc[org]),
                                     "r_precision": rprec(y, sc), "background": "heldout_32"})
    return pd.DataFrame(rows), max_dev


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--skip-minilm", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    np.random.seed(SEED)
    verify_inputs()
    os.makedirs(OUTDIR, exist_ok=True)
    corpus, F = load()

    # reproduction check: recomputing a full family batch must equal the frozen Stage 3 features
    max_dev = 0.0
    for h in FAMILIES:
        ids = F[F.generator_family == h].index.tolist()
        R = batch_rel(corpus, ids)
        max_dev = max(max_dev, float(np.abs(R.loc[ids, REL].values - F.loc[ids, REL].values).max()))
    if max_dev > 1e-9:
        raise RuntimeError(f"relational features do not reproduce frozen Stage 3 ({max_dev})")

    D1 = dilution(corpus, F, args.draws, K_GRID, extra_background=False)
    D2 = dilution(corpus, F, args.draws, K_GRID_R2, extra_background=True)
    REF = reference_full_batch(F)
    D = pd.concat([D1, D2, REF], ignore_index=True)
    D.to_csv(f"{OUTDIR}/stage4_dilution_draws.csv", index=False)
    print(f"R1/R2 done ({time.time() - t0:.0f}s)")

    R3, content = baselines(corpus, F)
    pd.DataFrame({"account_id": F.index, "generator_family": F.generator_family.values,
                  "role": F.role.values, "y": F.y.values,
                  "proba_content_tfidf_lr": content.values}).to_csv(
        f"{OUTDIR}/stage4_content_predictions.csv", index=False)
    R4, split_log = cross_repertoire(corpus, F)
    print(f"R3/R4 done ({time.time() - t0:.0f}s)")

    results = {
        "seed": SEED, "draws_per_cell": args.draws,
        "reproduction_max_abs_deviation_stage3_features": max_dev,
        "R1_R2_dilution": summarise_draws(D),
        "R3_baselines": R3,
        "R4_cross_repertoire": R4, "R4_splits": split_log,
    }

    model_sha = None
    if not args.skip_minilm:
        try:
            vectors, model_sha = minilm_vectors(corpus)
            D5, mdev = minilm_dilution(corpus, F, vectors, args.draws)
            D5.to_csv(f"{OUTDIR}/stage4_minilm_dilution_draws.csv", index=False)
            results["R5_minilm_dilution"] = summarise_draws(D5)
            results["R5_minilm_reproduction_max_abs_deviation"] = mdev
        except Exception as exc:  # documented, never silently fabricated
            results["R5_minilm_dilution"] = {"status": "not_run", "reason": repr(exc)}
    json.dump(results, open(f"{OUTDIR}/stage4_results.json", "w"), indent=1)

    import networkx, scipy, sklearn
    manifest = {
        "stage": "stage4_post_hoc_robustness_audit", "seed": SEED,
        "protocol_sha256": sha256("STAGE4_PROTOCOL.md"),
        "code_sha256": {f: sha256(f) for f in ["stage4_robustness.py",
                                               "stage3_relational_features.py",
                                               "stage3_evaluate.py"]},
        "input_sha256": {f: sha256(f) for f in [CORPUS, ACCOUNTS, REL_FROZEN, PRED_FROZEN,
                                                MINILM_FROZEN]},
        "minilm_model_file_sha256": model_sha,
        "output_sha256": {f: sha256(os.path.join(OUTDIR, f)) for f in sorted(os.listdir(OUTDIR))
                          if f.endswith((".csv", ".json")) and f != "stage4_manifest.json"},
        "packages": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "scipy": scipy.__version__, "networkx": networkx.__version__},
        "runtime_seconds": round(time.time() - t0, 1),
        "generation_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    json.dump(manifest, open(f"{OUTDIR}/stage4_manifest.json", "w"), indent=1)
    print(f"wrote {OUTDIR}/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
