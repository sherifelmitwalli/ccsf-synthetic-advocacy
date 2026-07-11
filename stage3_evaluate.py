#!/usr/bin/env python3
"""
Stage 3 evaluation (exploratory, post hoc, synthetic).

Compares four interpretable models under the FROZEN leave-one-generator-family-out
design from Stage 2:
  A. Fixed-direction CCSF   (archived Stage-1-oriented composite; no retraining)
  B. CCSF-LR                (4 frozen account features)
  C. Relational-only        (8 frozen Stage 3 relational features)
  D. Hybrid                 (4 account + 8 relational features)

All standardization/fitting/threshold-selection use development families only.
Held-out labels are used only after per-account predictions are frozen, for scoring.
Two endpoints reported with equal prominence: coordinated vs all controls, and
coordinated vs organic-style controls only. Seed 20260604.
"""
import json, os, hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SEED = 20260604
rng = np.random.default_rng(SEED)
CFG = json.load(open("stage3_config.json"))
NBOOT = CFG["bootstrap"]["n_resamples"]
PREV = CFG["assumed_prevalence"]
NBINS = CFG["calibration_bins"]
OUTDIR = "stage3_outputs"
os.makedirs(OUTDIR, exist_ok=True)

CCSF_FEATURES = ["ppl_mean", "burstiness", "commercial_policy_framing", "convergence"]
REL_FEATURES = ["top3_cosine_mean", "max_cosine", "top3_jaccard_mean", "max_jaccard",
                "semantic_without_literal_reuse", "mutual_knn_weighted_degree",
                "mutual_knn_clustering", "mutual_knn_component_size_normalized"]
FAMILIES = ["alibaba_qwen", "google_gemini", "bytedance_seed"]
FAM_LABEL = {"alibaba_qwen": "Alibaba Qwen", "google_gemini": "Google Gemini",
             "bytedance_seed": "ByteDance Seed"}


def load():
    af = pd.read_csv("stage2_outputs/stage2_account_features.csv")
    rel = pd.read_csv(f"{OUTDIR}/stage3_account_relational_features.csv")
    df = af.merge(rel.drop(columns=["generator_family"]), on="account_id", validate="1:1")
    df["y"] = (df["class_label"] == "coordinated_synthetic").astype(int)
    df["role"] = df["account_id"].str.extract(r"_(coordinated|organic|professional)")[0]
    # archived fixed-direction composite (no retraining)
    z = np.load("stage2_outputs/derived_v1/stage2_out_of_family_probabilities_v1.npz",
                allow_pickle=True)
    fc = pd.DataFrame({"account_id": z["account_id"], "fixed_composite": z["fixed_composite"]})
    df = df.merge(fc, on="account_id", validate="1:1")
    return df


def logo_predict(df, feats):
    """Leave-one-generator-family-out probabilities; dev-only scaler+fit. One row/account."""
    proba = pd.Series(index=df.index, dtype=float)
    for held in FAMILIES:
        dev = df[df.generator_family != held]
        test = df[df.generator_family == held]
        sc = StandardScaler().fit(dev[feats].values)
        clf = LogisticRegression(penalty="l2", C=CFG["model"]["C"],
                                 class_weight="balanced", solver="liblinear",
                                 random_state=SEED)
        clf.fit(sc.transform(dev[feats].values), dev["y"].values)
        proba.loc[test.index] = clf.predict_proba(sc.transform(test[feats].values))[:, 1]
    return proba.values


def oof_youden_threshold(df, feats):
    """Development-only Youden threshold per held-out family via 5-fold OOF on dev."""
    thr = {}
    for held in FAMILIES:
        dev = df[df.generator_family != held].reset_index(drop=True)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(dev))
        for tr, va in skf.split(dev[feats].values, dev["y"].values):
            sc = StandardScaler().fit(dev.loc[tr, feats].values)
            clf = LogisticRegression(penalty="l2", C=CFG["model"]["C"],
                                     class_weight="balanced", solver="liblinear",
                                     random_state=SEED)
            clf.fit(sc.transform(dev.loc[tr, feats].values), dev.loc[tr, "y"].values)
            oof[va] = clf.predict_proba(sc.transform(dev.loc[va, feats].values))[:, 1]
        thr[held] = youden(dev["y"].values, oof)
    return thr


def youden(y, s):
    order = np.argsort(-s)
    ys = y[order]; ss = s[order]
    P = ys.sum(); N = len(ys) - P
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    tpr = tp / P if P else np.zeros_like(tp, float)
    fpr = fp / N if N else np.zeros_like(fp, float)
    j = tpr - fpr
    return float(ss[np.argmax(j)])


def fast_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    sv = s[order]
    ranks = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r[order] = ranks
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def fast_ap(y, s):
    y = np.asarray(y); s = np.asarray(s)
    P = y.sum()
    if P == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float(precision[ys == 1].sum() / P)


def boot_ci(y, s, metric, nboot=NBOOT):
    y = np.asarray(y); s = np.asarray(s)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = metric(y, s)
    vals = []
    for _ in range(nboot):
        bi = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[bi]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        vals.append(metric(yy, s[bi]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(point), float(lo), float(hi))


def sens_spec(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def ppv_fdr(sens, spec, prev=PREV):
    denom = sens * prev + (1 - spec) * (1 - prev)
    ppv = (sens * prev) / denom if denom else float("nan")
    return ppv, (1 - ppv if ppv == ppv else float("nan"))


def calib_bins(y, s, nbins=NBINS):
    q = np.quantile(s, np.linspace(0, 1, nbins + 1))
    q[0] -= 1e-9; q[-1] += 1e-9
    out = []
    for b in range(nbins):
        m = (s > q[b]) & (s <= q[b + 1])
        if m.sum() == 0:
            continue
        out.append({"mean_predicted": float(s[m].mean()),
                    "fraction_positive": float(y[m].mean()), "n": int(m.sum())})
    return out


def endpoint_mask(df_test, endpoint):
    if endpoint == "all":
        return np.ones(len(df_test), dtype=bool)
    return (df_test["role"].isin(["coordinated", "organic"])).values  # vs organic-only


def evaluate_model(df, name, feats, is_fixed=False):
    if is_fixed:
        proba = df["fixed_composite"].values
        thr_dev = {h: youden(df[df.generator_family != h]["y"].values,
                             df[df.generator_family != h]["fixed_composite"].values)
                   for h in FAMILIES}
    else:
        proba = logo_predict(df, feats)
        thr_dev = oof_youden_threshold(df, feats)
    df = df.copy(); df["proba"] = proba
    res = {"model": name, "is_probability": (not is_fixed), "per_family": {}, "pooled": {}}
    for endpoint in ["all", "organic_only"]:
        res["per_family"][endpoint] = {}
        for held in FAMILIES:
            t = df[df.generator_family == held]
            m = endpoint_mask(t, endpoint)
            y = t["y"].values[m]; s = t["proba"].values[m]
            roc = boot_ci(y, s, fast_auc)
            pr = boot_ci(y, s, fast_ap)
            e = {"n": int(m.sum()), "n_pos": int(y.sum()),
                 "roc_auc": roc[0], "roc_auc_ci": [roc[1], roc[2]],
                 "pr_auc": pr[0], "pr_auc_ci": [pr[1], pr[2]]}
            if res["is_probability"]:
                e["brier"] = float(brier_score_loss(y, s))
                e["calibration_bins"] = calib_bins(y, s)
                s05, p05 = sens_spec(y, s, 0.5)
                sd, pd_ = sens_spec(y, s, thr_dev[held])
                e["sens_spec_0.5"] = [s05, p05]
                e["dev_threshold"] = thr_dev[held]
                e["sens_spec_dev"] = [sd, pd_]
                ppv, fdr = ppv_fdr(s05, p05)
                e["ppv_at_1pct_default"] = ppv
                e["fdr_at_1pct_default"] = fdr
            res["per_family"][endpoint][held] = e
        # pooled across held-out families (each account once)
        mall = endpoint_mask(df, endpoint)
        y = df["y"].values[mall]; s = df["proba"].values[mall]
        roc = boot_ci(y, s, fast_auc); pr = boot_ci(y, s, fast_ap)
        res["pooled"][endpoint] = {"n": int(mall.sum()), "n_pos": int(y.sum()),
                                   "roc_auc": roc[0], "roc_auc_ci": [roc[1], roc[2]],
                                   "pr_auc": pr[0], "pr_auc_ci": [pr[1], pr[2]]}
    # transformation heterogeneity (equivalent groups: coordinated vs organic only, per cell)
    res["transformation_heterogeneity"] = transformation_cells(df)
    return res, df["proba"].values


def transformation_cells(df):
    out = {}
    for tx in ["none", "llm_paraphrase", "noise_insertion", "sentence_restructure"]:
        sub = df[(df["role"].isin(["coordinated", "organic"])) & (df["transformation_id"] == tx)]
        if sub["y"].nunique() < 2:
            out[tx] = {"n": int(len(sub)), "roc_auc": None}
            continue
        roc = boot_ci(sub["y"].values, sub["proba"].values, fast_auc)
        out[tx] = {"n": int(len(sub)), "n_coord": int(sub["y"].sum()),
                   "roc_auc": roc[0], "roc_auc_ci": [roc[1], roc[2]]}
    return out


def paired_diff(df, probs, a, b, endpoint):
    """Paired bootstrap AUC difference (model a - model b) on pooled held-out accounts."""
    m = endpoint_mask(df, endpoint)
    y = df["y"].values[m]; sa = probs[a][m]; sb = probs[b][m]
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    point = fast_auc(y, sa) - fast_auc(y, sb)
    diffs = []
    for _ in range(NBOOT):
        bi = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[bi]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        diffs.append(fast_auc(yy, sa[bi]) - fast_auc(yy, sb[bi]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_auc": float(point), "ci": [float(lo), float(hi)],
            "auc_a": float(fast_auc(y, sa)), "auc_b": float(fast_auc(y, sb))}


def permutation_control(df, feats, n_perm=150):
    """Permute training labels within dev; expect held-out pooled AUC ~ 0.5."""
    aucs = []
    prng = np.random.default_rng(SEED)
    for _ in range(n_perm):
        proba = np.zeros(len(df))
        dfr = df.reset_index(drop=True)
        for held in FAMILIES:
            dev = dfr[dfr.generator_family != held]
            test = dfr[dfr.generator_family == held]
            yperm = prng.permutation(dev["y"].values)
            sc = StandardScaler().fit(dev[feats].values)
            clf = LogisticRegression(penalty="l2", C=CFG["model"]["C"],
                                     class_weight="balanced", solver="liblinear",
                                     random_state=SEED)
            clf.fit(sc.transform(dev[feats].values), yperm)
            proba[test.index] = clf.predict_proba(sc.transform(test[feats].values))[:, 1]
        aucs.append(fast_auc(dfr["y"].values, proba))
    aucs = np.array(aucs)
    return {"n_perm": n_perm, "mean_auc": float(aucs.mean()),
            "ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "max_auc": float(aucs.max()), "feats": feats}


def sha(f):
    return hashlib.sha256(open(f, "rb").read()).hexdigest()


def main():
    df = load()
    models = {}
    probs = {}
    specs = [("fixed_ccsf", None, True), ("ccsf_lr", CCSF_FEATURES, False),
             ("relational_only", REL_FEATURES, False),
             ("hybrid", CCSF_FEATURES + REL_FEATURES, False)]
    for name, feats, isf in specs:
        r, p = evaluate_model(df, name, feats, is_fixed=isf)
        models[name] = r; probs[name] = p

    # model comparisons (paired) on pooled held-out predictions
    comparisons = {}
    for endpoint in ["all", "organic_only"]:
        comparisons[endpoint] = {
            "relational_only_minus_ccsf_lr": paired_diff(df, probs, "relational_only", "ccsf_lr", endpoint),
            "hybrid_minus_ccsf_lr": paired_diff(df, probs, "hybrid", "ccsf_lr", endpoint),
            "hybrid_minus_fixed_ccsf": paired_diff(df, probs, "hybrid", "fixed_ccsf", endpoint),
        }

    # coefficients for relational-only and hybrid (dev-only fits per split, standardized)
    coefs = {}
    for name, feats in [("relational_only", REL_FEATURES), ("hybrid", CCSF_FEATURES + REL_FEATURES)]:
        coefs[name] = {}
        for held in FAMILIES:
            dev = df[df.generator_family != held]
            sc = StandardScaler().fit(dev[feats].values)
            clf = LogisticRegression(penalty="l2", C=CFG["model"]["C"], class_weight="balanced",
                                     solver="liblinear", random_state=SEED)
            clf.fit(sc.transform(dev[feats].values), dev["y"].values)
            # bootstrap dev CIs
            bco = {f: [] for f in feats}
            devr = dev.reset_index(drop=True)
            for _ in range(400):
                bi = rng.choice(len(devr), len(devr), True)
                if devr.loc[bi, "y"].nunique() < 2:
                    continue
                sc2 = StandardScaler().fit(devr.loc[bi, feats].values)
                c2 = LogisticRegression(penalty="l2", C=CFG["model"]["C"], class_weight="balanced",
                                        solver="liblinear", random_state=SEED)
                c2.fit(sc2.transform(devr.loc[bi, feats].values), devr.loc[bi, "y"].values)
                for k, f in enumerate(feats):
                    bco[f].append(c2.coef_[0][k])
            coefs[name][held] = {f: {"coef": float(clf.coef_[0][k]),
                                     "ci": [float(np.percentile(bco[f], 2.5)),
                                            float(np.percentile(bco[f], 97.5))]}
                                 for k, f in enumerate(feats)}

    perm = {"hybrid": permutation_control(df, CCSF_FEATURES + REL_FEATURES),
            "relational_only": permutation_control(df, REL_FEATURES)}

    # predictions table
    pred = df[["account_id", "generator_family", "role", "y", "transformation_id"]].copy()
    for name in probs:
        pred[f"proba_{name}"] = probs[name]
    pred.to_csv(f"{OUTDIR}/stage3_predictions.csv", index=False)

    json.dump({"seed": SEED, "config": CFG, "models": models, "coefficients": coefs},
              open(f"{OUTDIR}/stage3_results.json", "w"), indent=1)
    json.dump(comparisons, open(f"{OUTDIR}/stage3_model_comparisons.json", "w"), indent=1)
    json.dump(perm, open(f"{OUTDIR}/stage3_permutation_results.json", "w"), indent=1)

    inputs = ["stage2_outputs/stage2_validation_corpus.csv",
              "stage2_outputs/stage2_account_features.csv",
              "stage2_outputs/stage2_splits/leave_one_generator_id_out.json",
              "stage2_outputs/stage2_eval_leave_one_generator_id_out.json",
              f"{OUTDIR}/stage3_account_relational_features.csv"]
    import sklearn, scipy, networkx, platform
    manifest = {
        "stage": "exploratory_post_hoc_synthetic_stage3", "seed": SEED,
        "input_checksums": {f: sha(f) for f in inputs},
        "protocol_checksums": {f: sha(f) for f in ["STAGE3_PROTOCOL.md", "stage3_config.json",
                                                   "stage3_feature_manifest.json"]},
        "code_checksums": {f: sha(f) for f in ["stage3_relational_features.py", "stage3_evaluate.py"]},
        "feature_names": {"ccsf": CCSF_FEATURES, "relational": REL_FEATURES},
        "sample_counts": {"accounts": int(len(df)), "coordinated": int(df.y.sum()),
                          "controls": int((df.y == 0).sum())},
        "package_versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__,
                             "scipy": scipy.__version__, "networkx": networkx.__version__},
        "generation_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "verification_status": "pending_verify_stage3",
    }
    json.dump(manifest, open(f"{OUTDIR}/stage3_manifest.json", "w"), indent=1)

    # console summary
    print("=== pooled held-out ROC AUC (vs ALL controls / vs ORGANIC-style controls) ===")
    for name in probs:
        pa = models[name]["pooled"]["all"]
        po = models[name]["pooled"]["organic_only"]
        print(f"  {name:16s}  all={pa['roc_auc']:.3f} "
              f"[{pa['roc_auc_ci'][0]:.3f},{pa['roc_auc_ci'][1]:.3f}]   "
              f"organic={po['roc_auc']:.3f} "
              f"[{po['roc_auc_ci'][0]:.3f},{po['roc_auc_ci'][1]:.3f}]")
    print("=== paired AUC differences (pooled) ===")
    for endpoint in ["all", "organic_only"]:
        for k, v in comparisons[endpoint].items():
            print(f"  [{endpoint}] {k:32s} delta={v['delta_auc']:+.3f} "
                  f"CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
    print("=== label-permutation negative control (expect ~0.5) ===")
    for k, v in perm.items():
        print(f"  {k:16s} mean_auc={v['mean_auc']:.3f} "
              f"CI[{v['ci'][0]:.3f},{v['ci'][1]:.3f}] max={v['max_auc']:.3f}")
    print(f"wrote outputs under {OUTDIR}/")


if __name__ == "__main__":
    main()
