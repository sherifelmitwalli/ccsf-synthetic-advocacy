"""
verify_outputs.py -- sanity checks for the CCSF reproducibility package.

Run AFTER analysis.py (and optionally figures.py). It verifies data-generation
invariants and that the headline numbers in results.json match the values reported
in the manuscript, within tolerances. Prints PASS/FAIL for each check and exits
non-zero if any check fails.

    python verify_outputs.py
"""
import json, sys
from pathlib import Path
import numpy as np
import corpus_gen

BASE = Path(__file__).resolve().parent
fails = []

def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)

# ---- data-generation invariants ----
posts = corpus_gen.build_corpus()
groups = [p["group"] for p in posts]
accounts = {p["account"]: p["group"] for p in posts}
n_by_group = {g: sum(1 for q in posts if q["group"] == g) for g in set(groups)}
acc_by_group = {g: sum(1 for a, gg in accounts.items() if gg == g) for g in set(accounts.values())}

check("corpus has 524 posts", len(posts) == 524)
check("78 unique accounts", len(accounts) == 78)
check("account counts 40/16/12/10",
      acc_by_group.get("organic") == 40 and acc_by_group.get("synthetic") == 16
      and acc_by_group.get("professional") == 12 and acc_by_group.get("mixed") == 10)
check("post counts 200/160/84/80",
      n_by_group.get("organic") == 200 and n_by_group.get("synthetic") == 160
      and n_by_group.get("professional") == 84 and n_by_group.get("mixed") == 80)
check("deterministic post IDs (p0000..)", posts[0]["id"] == "p0000" and len(posts[0]["id"]) == 5)
check("compliance lexicon non-empty", len(corpus_gen.COMPLIANCE_LEXICON) > 0)
check("no real usernames (all synthetic org_/syn_/pro_/mix_ handles)",
      all(any(a.startswith(pre) for pre in ("org", "syn", "pro", "mix")) for a in accounts))

# ---- results.json headline numbers (four-signal primary composite) ----
R = json.load(open(BASE / "results.json"))
auc = R["auc"]

def close(a, b, tol=0.02):
    return abs(a - b) <= tol

check("composite (4-signal) AUC vs both human ~0.995", close(auc["composite_vs_both"], 0.995))
check("composite AUC syn vs organic ~0.994", close(auc["composite_vs_organic"], 0.994))
check("composite AUC syn vs professional ~1.00", close(auc["composite_vs_professional"], 1.00))
check("composite AUC mixed (human-edited) vs organic ~0.955", close(auc["composite_mixed_vs_organic"], 0.955))
check("machine-origin vs human AUC ~0.984", close(auc["composite_machineorigin_vs_human"], 0.984))
check("logreg 5-fold CV AUC ~0.980", close(auc["logreg_cv_mean"], 0.980, tol=0.03))
check("stance single-feature AUC ~0.50 (at chance)", close(auc["feature_stance_std_vs_both"], 0.50, tol=0.05))
check("perplexity single-feature AUC ~0.989", close(auc["feature_ppl_mean_vs_both"], 0.989))
check("convergence single-feature AUC ~0.948", close(auc["feature_convergence_vs_both"], 0.948))
check("composite exceeds the strongest single signal in this corpus",
      auc["composite_vs_both"] > auc["feature_ppl_mean_vs_both"])

abl = R["ablation"]
check("ablation covers exactly the 4 primary signals",
      sorted(k for k in abl if k.startswith("drop_")) ==
      ["drop_burstiness", "drop_compliance", "drop_convergence", "drop_ppl_mean"])
check("every primary signal contributes positively (all drop deltas < 0)",
      all(abl[k]["delta"] < 0 for k in abl if k.startswith("drop_")))
check("dropping burstiness is the largest AUC decrease",
      abl["drop_burstiness"]["auc"] == min(abl[k]["auc"] for k in abl if k.startswith("drop_")))
check("adding stance (sensitivity) reduces AUC", abl["with_stance_sensitivity"]["delta"] < 0)
check("with-stance sensitivity AUC ~0.989", close(abl["with_stance_sensitivity"]["auc"], 0.989))

clu = R["clustering"]
check("KMeans ARI ~0.35", close(clu["adjusted_rand"], 0.35, tol=0.05))

ppv = auc.get("ppv_scenarios", {})
check("ppv scenarios present (sens 1.00, spec ~0.94)",
      close(ppv.get("sensitivity", 0), 1.00) and close(ppv.get("specificity", 0), 0.942, tol=0.02))
check("ppv scenario at 1% prevalence is cautionary (PPV < 0.5)",
      ppv.get("scenarios", {}).get("prev_0.01", {}).get("ppv", 1.0) < 0.5)

nd = R["nonduplication"]
check("synthetic: low Jaccard, high cosine (coordinated originality)",
      nd["synthetic"]["mean_jaccard"] < 0.20 and nd["synthetic"]["mean_embed_cosine"] > 0.50)

# ---- figures present ----
for fp in ["fig1_pipeline.png", "fig1_pipeline.svg", "fig2_features.png",
           "fig3_fingerprint.png", "fig4_tsne.png", "fig5_roc_ablation.png",
           "fig6_nonduplication.png"]:
    check(f"{fp} exists", (BASE / fp).exists())

# ---- fingerprint profile (figure 3 / fingerprint_profiles.csv) ----
# The synthetic group's mean z-scores must show the hypothesised joint pattern,
# and professional advocacy must diverge on compliance density and convergence.
import csv as _csv
fp_rows = list(_csv.DictReader(open(BASE / "fingerprint_profiles.csv")))
prof = {(r["group"], r["feature"]): float(r["mean_z"]) for r in fp_rows}
check("fingerprint_profiles.csv has 20 rows (4 groups x 5 signals)", len(fp_rows) == 20)
check("synthetic fingerprint: low perplexity (z<0)", prof[("synthetic", "ppl_mean")] < 0)
check("synthetic fingerprint: low burstiness (z<0)", prof[("synthetic", "burst")] < 0)
check("synthetic fingerprint: high compliance (z>0)", prof[("synthetic", "compliance")] > 0)
check("synthetic fingerprint: high convergence (z>0)", prof[("synthetic", "conv")] > 0)
check("synthetic fingerprint: stance uninformative (|z|<0.3)",
      abs(prof[("synthetic", "stance_std")]) < 0.3)
check("professional diverges: compliance z<0 despite low burstiness",
      prof[("professional", "compliance")] < 0 and prof[("professional", "burst")] < 0)
check("professional diverges: convergence z<0", prof[("professional", "conv")] < 0)

# ---- account_features.csv present and complete ----
import csv
af = list(csv.DictReader(open(BASE / "account_features.csv")))
check("account_features.csv has 78 rows", len(af) == 78)
check("no missing feature values",
      all(all(row[c] not in ("", None) for c in
              ["ppl_mean", "ppl_var", "burstiness", "compliance", "stance_std", "convergence"])
          for row in af))
# stored anomaly column must equal the 4-signal composite recomputed from the table
M4 = np.array([[float(r[c]) for c in ["ppl_mean", "burstiness", "compliance", "convergence"]] for r in af])
Z4 = (M4 - M4.mean(axis=0)) / M4.std(axis=0)
an = (Z4 * np.array([-1, -1, 1, 1])).mean(axis=1)
stored = np.array([float(r["anomaly"]) for r in af])
check("stored anomaly equals 4-signal composite", float(np.abs(an - stored).max()) < 1e-4)

print()
if fails:
    print(f"{len(fails)} check(s) FAILED:", "; ".join(fails))
    sys.exit(1)
print("All checks PASSED.")
