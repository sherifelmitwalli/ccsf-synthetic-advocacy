"""Versioned derivative analyses over the frozen Stage 2 outputs (derived_v1).

This script performs NO text generation and NO model API calls. It reads the
immutable Stage 2 evidence (stage2_outputs/stage2_account_features.csv,
stage2_outputs/stage2_validation_corpus.csv, stage2_outputs/stage2_splits/,
stage2_outputs/stage2_eval_*.json) and writes derivative outputs to
stage2_outputs/derived_v1/. Original Stage 2 files are never modified.

Analyses (all deterministic; seed 20260604):
 1. fixed_composite   - Stage 1 fixed-direction equal-weight composite applied
                        without retraining: pooled and per-family ROC AUC with
                        stratified bootstrap 95% CIs.
 2. signal_transfer   - per-feature group means/SDs by account role and
                        oriented single-feature AUCs (vs organic-style controls
                        and vs all controls), pooled with bootstrap CIs.
 3. logo_predictions  - exact re-fit of the leave-one-generator-family-out
                        logistic model (identical pipeline to
                        validation_evaluation.py); recomputed AUCs are asserted
                        to match the frozen eval JSON to 1e-9 (consistency
                        anchor). Out-of-family probabilities are saved for
                        descriptive use only.
 4. coefficients      - standardized logistic coefficients per LOGO split with
                        development-account bootstrap percentile CIs.
 5. dev_threshold     - development-only threshold selection (Youden J on
                        5-fold out-of-fold development predictions), applied
                        unchanged to the held-out family. No held-out tuning.
 6. calibration       - Brier scores and quantile calibration bins copied from
                        the frozen eval JSON (no recomputation), summarized.
 7. transformation_equivalent - coordinated-vs-organic ROC AUC within each
                        transformation cell using EQUIVALENT groups (the
                        untransformed cell is restricted to untransformed
                        coordinated and organic accounts; professional controls
                        excluded from all cells), per held-out family and
                        pooled across families (descriptive pooling of
                        out-of-family predictions; no refitting).
 8. coordinated_originality_stage2 - campaign-level mean pairwise lexical
                        Jaccard (Stage 1 formula: [A-Za-z']+ tokens, pair cap
                        400) and, when the embedding model is available, mean
                        pairwise semantic cosine, per class within each
                        generator family. Tests whether the Stage 1
                        'coordinated originality' signature recurs in LLM text.

Usage: python stage2_derived_analysis.py [--skip-embeddings]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260604
FEATURES = ("ppl_mean", "burstiness", "commercial_policy_framing", "convergence")
ORIENTATION = {"ppl_mean": -1.0, "burstiness": -1.0,
               "commercial_policy_framing": 1.0, "convergence": 1.0}
WORD = re.compile(r"[A-Za-z']+")
BASE = Path(__file__).resolve().parent
S2 = BASE / "stage2_outputs"
OUT = S2 / "derived_v1"


def account_role(account_id: str) -> str:
    if "professional" in account_id:
        return "professional"
    if "organic" in account_id:
        return "organic"
    return "coordinated"


def bootstrap_ci(y, s, metric=roc_auc_score, n_boot=2000, seed=SEED):
    y = np.asarray(y, int); s = np.asarray(s, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate((rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)))
        vals.append(metric(y[idx], s[idx]))
    lo, hi = np.quantile(vals, (0.025, 0.975))
    return [float(lo), float(hi)]


def load_features():
    with (S2 / "stage2_account_features.csv").open(encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    y = np.array([r["class_label"] == "coordinated_synthetic" for r in rows], int)
    return rows, X, y


def fixed_composite(rows, X, y):
    Z = (X - X.mean(0)) / X.std(0)
    w = np.array([ORIENTATION[f] for f in FEATURES])
    comp = (w * Z).mean(1)
    fams = sorted({r["generator_family"] for r in rows})
    roles = np.array([account_role(r["account_id"]) for r in rows])
    res = {"description": "Stage 1 fixed-direction equal-weight composite applied to Stage 2 "
                          "accounts without retraining (z-scored across all 156 Stage 2 accounts).",
           "pooled": {"n": int(len(y)), "roc_auc": float(roc_auc_score(y, comp)),
                      "roc_auc_ci": bootstrap_ci(y, comp)},
           "pooled_vs_organic_only": {}, "per_family": {}}
    m = roles != "professional"
    res["pooled_vs_organic_only"] = {"n": int(m.sum()), "roc_auc": float(roc_auc_score(y[m], comp[m])),
                                     "roc_auc_ci": bootstrap_ci(y[m], comp[m])}
    for fam in fams:
        fm = np.array([r["generator_family"] == fam for r in rows])
        res["per_family"][fam] = {"n": int(fm.sum()), "roc_auc": float(roc_auc_score(y[fm], comp[fm])),
                                  "roc_auc_ci": bootstrap_ci(y[fm], comp[fm])}
    return res, comp


def signal_transfer(rows, X, y):
    roles = np.array([account_role(r["account_id"]) for r in rows])
    res = {"description": "Per-feature Stage 2 transfer: group means (SD) by account role and "
                          "oriented single-feature ROC AUCs (orientation fixed by the Stage 1 "
                          "hypothesis).", "features": {}}
    m_org = roles != "professional"
    for i, f in enumerate(FEATURES):
        s = ORIENTATION[f] * X[:, i]
        res["features"][f] = {
            "orientation": "higher-composite when " + ("lower" if ORIENTATION[f] < 0 else "higher"),
            "means": {role: {"mean": float(X[roles == role, i].mean()),
                             "sd": float(X[roles == role, i].std(ddof=1))}
                      for role in ("coordinated", "organic", "professional")},
            "oriented_auc_vs_organic": {"value": float(roc_auc_score(y[m_org], s[m_org])),
                                        "ci": bootstrap_ci(y[m_org], s[m_org])},
            "oriented_auc_vs_all_controls": {"value": float(roc_auc_score(y, s)),
                                             "ci": bootstrap_ci(y, s)}}
        m_prof = roles != "organic"
        res["features"][f]["oriented_auc_vs_professional"] = {
            "value": float(roc_auc_score(y[m_prof], s[m_prof])),
            "ci": bootstrap_ci(y[m_prof], s[m_prof])}
    return res


def logo_refit(rows, X, y):
    manifest = json.loads((S2 / "stage2_splits" / "leave_one_generator_id_out.json").read_text())
    frozen = json.loads((S2 / "stage2_eval_leave_one_generator_id_out.json").read_text())
    frozen_auc = {s["held_out_value"]: s["roc_auc"] for s in frozen["splits"]}
    idx_by_acc = {r["account_id"]: i for i, r in enumerate(rows)}
    prob = np.full(len(rows), np.nan)
    fits = {}
    checks = {}
    for split in manifest["splits"]:
        tr = np.array([idx_by_acc[a] for a in split["train_account_ids"]])
        te = np.array([idx_by_acc[a] for a in split["test_account_ids"]])
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        prob[te] = p
        auc = float(roc_auc_score(y[te], p))
        held = split["held_out_value"]
        checks[held] = {"recomputed_roc_auc": auc, "frozen_roc_auc": frozen_auc[held],
                        "abs_diff": abs(auc - frozen_auc[held])}
        assert abs(auc - frozen_auc[held]) < 1e-9, f"Mismatch vs frozen eval for {held}"
        fits[held] = (model, tr, te)
    assert not np.isnan(prob).any()
    return fits, prob, checks


def coefficients(rows, X, y, fits, n_boot=1000):
    res = {"description": "Standardized logistic coefficients (per SD of each development-fitted "
                          "feature) for each leave-one-generator-family-out split, with "
                          "development-account stratified bootstrap 95% percentile CIs. "
                          "Positive = higher value increases predicted coordination probability.",
           "splits": {}}
    rng = np.random.default_rng(SEED)
    for held, (model, tr, _te) in fits.items():
        lr = model.named_steps["logisticregression"]
        entry = {"n_dev_accounts": int(len(tr)),
                 "intercept": float(lr.intercept_[0]),
                 "coefficients": {f: float(c) for f, c in zip(FEATURES, lr.coef_[0])},
                 "bootstrap_ci": {}}
        ytr = y[tr]
        pos, neg = tr[ytr == 1], tr[ytr == 0]
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate((rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)))
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
            m.fit(X[idx], y[idx])
            boots.append(m.named_steps["logisticregression"].coef_[0])
        boots = np.array(boots)
        for j, f in enumerate(FEATURES):
            lo, hi = np.quantile(boots[:, j], (0.025, 0.975))
            entry["bootstrap_ci"][f] = [float(lo), float(hi)]
        res["splits"][held] = entry
    return res


def dev_threshold(rows, X, y, fits):
    res = {"description": "Development-only operating threshold: Youden J maximized on 5-fold "
                          "out-of-fold development predictions, then applied unchanged to the "
                          "held-out family. The held-out family contributes nothing to threshold "
                          "choice. Reported alongside the default 0.5 threshold.", "splits": {}}
    for held, (model, tr, te) in fits.items():
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        oof = np.full(len(tr), np.nan)
        for a, b in skf.split(X[tr], y[tr]):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
            m.fit(X[tr][a], y[tr][a])
            oof[b] = m.predict_proba(X[tr][b])[:, 1]
        cand = np.unique(oof)
        best_t, best_j = 0.5, -1.0
        for t in cand:
            pred = (oof >= t).astype(int)
            tp = int(((pred == 1) & (y[tr] == 1)).sum()); fn = int(((pred == 0) & (y[tr] == 1)).sum())
            tn = int(((pred == 0) & (y[tr] == 0)).sum()); fp = int(((pred == 1) & (y[tr] == 0)).sum())
            j = tp / (tp + fn) + tn / (tn + fp) - 1
            if j > best_j:
                best_j, best_t = j, float(t)
        p_te = model.predict_proba(X[te])[:, 1]
        entry = {"dev_youden_threshold": best_t, "dev_oof_youden_j": float(best_j), "held_out": {}}
        for label, t in (("dev_selected", best_t), ("default_0.5", 0.5)):
            pred = (p_te >= t).astype(int)
            tp = int(((pred == 1) & (y[te] == 1)).sum()); fn = int(((pred == 0) & (y[te] == 1)).sum())
            tn = int(((pred == 0) & (y[te] == 0)).sum()); fp = int(((pred == 1) & (y[te] == 0)).sum())
            sens = tp / (tp + fn) if tp + fn else None
            spec = tn / (tn + fp) if tn + fp else None
            prec = tp / (tp + fp) if tp + fp else None
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
            entry["held_out"][label] = {"threshold": t, "sensitivity": sens, "specificity": spec,
                                        "precision": prec, "f1": f1,
                                        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
        res["splits"][held] = entry
    return res


def calibration_summary():
    frozen = json.loads((S2 / "stage2_eval_leave_one_generator_id_out.json").read_text())
    return {"description": "Brier scores and quantile calibration bins copied verbatim from the "
                           "frozen leave-one-generator-family-out evaluation output "
                           "(stage2_eval_leave_one_generator_id_out.json).",
            "splits": {s["held_out_value"]: {"brier_score": s["brier_score"],
                                             "calibration_bins": s["calibration_bins"]}
                       for s in frozen["splits"]}}


def transformation_equivalent(rows, y, prob):
    fams = sorted({r["generator_family"] for r in rows})
    trans = ("none", "llm_paraphrase", "noise_insertion", "sentence_restructure")
    roles = np.array([account_role(r["account_id"]) for r in rows])
    tvals = np.array([r["transformation_id"] for r in rows])
    fam_arr = np.array([r["generator_family"] for r in rows])
    res = {"description": "Coordinated-vs-organic ROC AUC of the held-out-family predictions "
                          "within each transformation cell, using EQUIVALENT groups: professional "
                          "controls are excluded everywhere, so every cell (including the "
                          "untransformed cell) contains 5 coordinated and 5 organic accounts per "
                          "family (15 vs 15 pooled). Per-family values are exploratory point "
                          "estimates; pooled values are descriptive summaries of out-of-family "
                          "predictions with stratified bootstrap CIs (no refitting).",
           "per_family": {}, "pooled": {}}
    for fam in fams:
        res["per_family"][fam] = {}
        for t in trans:
            m = (fam_arr == fam) & (tvals == t) & (roles != "professional")
            res["per_family"][fam][t] = {"n": int(m.sum()),
                                         "n_coordinated": int((y[m] == 1).sum()),
                                         "n_organic": int((y[m] == 0).sum()),
                                         "roc_auc": float(roc_auc_score(y[m], prob[m]))}
    for t in trans:
        m = (tvals == t) & (roles != "professional")
        res["pooled"][t] = {"n": int(m.sum()), "roc_auc": float(roc_auc_score(y[m], prob[m])),
                            "roc_auc_ci": bootstrap_ci(y[m], prob[m])}
    return res


def mean_pairwise_jaccard(texts, cap=400, rng=None):
    sets = [set(WORD.findall(t.lower())) for t in texts]
    pairs = list(itertools.combinations(range(len(sets)), 2))
    if len(pairs) > cap:
        idx = rng.choice(len(pairs), cap, replace=False)
        pairs = [pairs[i] for i in idx]
    js = []
    for i, j in pairs:
        u = sets[i] | sets[j]
        if u:
            js.append(len(sets[i] & sets[j]) / len(u))
    return float(np.mean(js)) if js else 0.0, float(np.max(js)) if js else 0.0


def coordinated_originality(skip_embeddings=False):
    with (S2 / "stage2_validation_corpus.csv").open(encoding="utf-8", newline="") as h:
        posts = list(csv.DictReader(h))
    emb_by_group = None
    embed_note = "semantic cosine computed with all-MiniLM-L6-v2 (same instrument as Stage 1)"
    if not skip_embeddings:
        try:
            import os
            cache = os.environ.get("CCSF_EMB_CACHE", "")
            if cache and Path(cache).exists():
                emb_by_group = np.load(cache)["emb"]
                if len(emb_by_group) != len(posts):
                    raise ValueError("embedding cache size mismatch")
            else:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                texts = [p["text"] for p in posts]
                emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                emb_by_group = np.asarray(emb)
                if cache:
                    np.savez(cache, emb=emb_by_group)
        except Exception as exc:  # noqa: BLE001
            embed_note = f"embedding model unavailable in this environment ({exc}); cosine omitted"
    else:
        embed_note = "embeddings skipped by flag; cosine omitted"
    rng = np.random.default_rng(SEED)
    res = {"description": "Stage 2 test of the Stage 1 'coordinated originality' signature "
                          "(campaign-level semantic alignment without lexical duplication). "
                          "Groups: each coordinated campaign, and the organic-style and "
                          "professional controls, within each generator family. Same lexical "
                          "tokenizer and pair cap as Stage 1 analysis.py.",
           "embedding_note": embed_note, "groups": {}}
    def group_key(p):
        role = account_role(p["account_id"])
        if role == "coordinated":
            return f'{p["generator_family"]}::{p["campaign_id"]}'
        return f'{p["generator_family"]}::{role}'
    keys = sorted({group_key(p) for p in posts})
    for k in keys:
        idx = [i for i, p in enumerate(posts) if group_key(p) == k]
        texts = [posts[i]["text"] for i in idx]
        mj, xj = mean_pairwise_jaccard(texts, rng=rng)
        entry = {"n_posts": len(texts), "mean_jaccard": mj, "max_jaccard": xj}
        if emb_by_group is not None:
            e = emb_by_group[idx]
            sim = e @ e.T
            iu = np.triu_indices(len(e), k=1)
            entry["mean_embed_cosine"] = float(sim[iu].mean())
        res["groups"][k] = entry
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--sections", default="all",
                    help="comma list of: core,coefficients,originality (default all)")
    args = ap.parse_args()
    sections = set(args.sections.split(",")) if args.sections != "all" else {
        "core", "coefficients", "originality"}
    OUT.mkdir(exist_ok=True)
    out_path = OUT / "stage2_derived_results_v1.json"
    bundle = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    bundle.update({"version": "derived_v1", "seed": SEED,
                   "inputs": ["stage2_outputs/stage2_account_features.csv",
                              "stage2_outputs/stage2_validation_corpus.csv",
                              "stage2_outputs/stage2_splits/leave_one_generator_id_out.json",
                              "stage2_outputs/stage2_eval_leave_one_generator_id_out.json"]})
    rows, X, y = load_features()
    fits, prob, checks = logo_refit(rows, X, y)
    bundle["frozen_consistency_checks"] = checks
    if "core" in sections:
        fixed, comp = fixed_composite(rows, X, y)
        bundle["fixed_composite"] = fixed
        bundle["signal_transfer"] = signal_transfer(rows, X, y)
        bundle["dev_only_threshold"] = dev_threshold(rows, X, y, fits)
        bundle["calibration"] = calibration_summary()
        bundle["transformation_equivalent_groups"] = transformation_equivalent(rows, y, prob)
        np.savez(OUT / "stage2_out_of_family_probabilities_v1.npz",
                 account_id=np.array([r["account_id"] for r in rows]),
                 y=y, probability=prob, fixed_composite=comp)
    if "coefficients" in sections:
        bundle["logo_coefficients"] = coefficients(rows, X, y, fits)
    if "originality" in sections:
        bundle["coordinated_originality_stage2"] = coordinated_originality(
            skip_embeddings=args.skip_embeddings)
    out_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path, "sections:", sorted(sections))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
