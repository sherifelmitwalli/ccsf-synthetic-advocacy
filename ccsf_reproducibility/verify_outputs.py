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

# ---- results.json headline numbers ----
R = json.load(open(BASE / "results.json"))
auc = R["auc"]

def close(a, b, tol=0.02):
    return abs(a - b) <= tol

check("composite AUC vs both human ~0.989", close(auc["composite_vs_both"], 0.989))
check("composite AUC syn vs organic ~0.991", close(auc["composite_vs_organic"], 0.991))
check("composite AUC syn vs professional ~0.984", close(auc["composite_vs_professional"], 0.984))
check("composite AUC mixed (human-edited) vs organic ~0.935", close(auc["composite_mixed_vs_organic"], 0.935))
check("machine-origin vs human AUC ~0.957", close(auc["composite_machineorigin_vs_human"], 0.957))
check("logreg 5-fold CV AUC ~0.975", close(auc["logreg_cv_mean"], 0.975, tol=0.03))
check("stance single-feature AUC ~0.50 (at chance)", close(auc["feature_stance_std_vs_both"], 0.50, tol=0.05))
check("perplexity single-feature AUC ~0.989", close(auc["feature_ppl_mean_vs_both"], 0.989))
check("convergence single-feature AUC ~0.948", close(auc["feature_convergence_vs_both"], 0.948))

abl = R["ablation"]
check("dropping compliance is the largest AUC decrease",
      abl["drop_compliance"]["auc"] == min(abl[k]["auc"] for k in abl if k.startswith("drop_")))
check("dropping stance does not reduce AUC", abl["drop_stance_std"]["delta"] >= 0)

clu = R["clustering"]
check("KMeans ARI ~0.26", close(clu["adjusted_rand"], 0.26, tol=0.05))

nd = R["nonduplication"]
check("synthetic: low Jaccard, high cosine (coordinated originality)",
      nd["synthetic"]["mean_jaccard"] < 0.20 and nd["synthetic"]["mean_embed_cosine"] > 0.50)

# ---- account_features.csv present and complete ----
import csv
af = list(csv.DictReader(open(BASE / "account_features.csv")))
check("account_features.csv has 78 rows", len(af) == 78)
check("no missing feature values",
      all(all(row[c] not in ("", None) for c in
              ["ppl_mean", "ppl_var", "burstiness", "compliance", "stance_std", "convergence"])
          for row in af))

print()
if fails:
    print(f"{len(fails)} check(s) FAILED:", "; ".join(fails))
    sys.exit(1)
print("All checks PASSED.")
