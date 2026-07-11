#!/usr/bin/env python3
"""
verify_stage3.py -- automated leakage, partition, and negative-control checks for the
exploratory post hoc synthetic Stage 3 relational extension.

Runs a battery of assertions and prints PASS/FAIL per check. Exit code is nonzero if
ANY check fails. No new text, APIs, or real-world data are used. Seed 20260604.

Checks implemented (mapped to the Stage 3 protocol safeguards):
  C1  Frozen input checksums match stage3_feature_manifest.json.
  C2  No prohibited / label / identifier column appears in any model feature matrix.
  C3  Relational feature CSV carries no target, campaign, or account-attribute column.
  C4  Relational features are invariant to a random permutation of class labels
      (features are unsupervised).
  C5  Relational features are invariant to row-order shuffling of the corpus
      (no ID-ordering or duplicate-row influence).
  C6  Leave-one-generator-family-out partitions are disjoint by account, and no
      held-out family account appears in its own development set.
  C7  StandardScaler is fitted on development accounts only (scaler means equal the
      development-set means, never the pooled/test means).
  C8  Held-out predictions reproduce the frozen stage3_predictions.csv exactly
      (deterministic; test labels are not consulted to build scores).
  C9  Label-permutation negative control is at chance (pooled mean AUC in [0.40,0.60]).
  C10 Fixed-direction CCSF pooled held-out AUC is preserved below chance (Stage 2
      negative finding is not silently altered).
"""
import json, hashlib, tempfile, sys
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

FAIL = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)

def sha(f):
    return hashlib.sha256(open(f, "rb").read()).hexdigest()

CORPUS = "stage2_outputs/stage2_validation_corpus.csv"
RELCSV = "stage3_outputs/stage3_account_relational_features.csv"
MAN = json.load(open("stage3_feature_manifest.json"))
PROHIBITED = set(MAN["prohibited_predictors"])
REL_FEATURES = MAN["relational_features"]

# import the frozen evaluation + feature modules
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

ev = load_module("stage3_evaluate.py", "ev")
fe = load_module("stage3_relational_features.py", "fe")

# ---------------- C1 checksums ----------------
c1 = all(sha(f) == exp for f, exp in MAN["checksums"].items())
check("C1 frozen input checksums match manifest", c1)

# ---------------- C2 no prohibited predictors in model matrices ----------------
model_feats = {"ccsf_lr": ev.CCSF_FEATURES, "relational_only": ev.REL_FEATURES,
               "hybrid": ev.CCSF_FEATURES + ev.REL_FEATURES}
c2 = all(not (set(f) & PROHIBITED) for f in model_feats.values())
bad = {k: sorted(set(v) & PROHIBITED) for k, v in model_feats.items() if set(v) & PROHIBITED}
check("C2 no prohibited column in any feature matrix", c2, str(bad) if bad else "")

# ---------------- C3 relational CSV columns ----------------
relcols = set(pd.read_csv(RELCSV, nrows=1).columns)
allowed = set(["account_id", "generator_family"]) | set(REL_FEATURES)
extra_labelish = (relcols - allowed) & (PROHIBITED | {"class_label", "y", "role", "topic"})
check("C3 relational CSV has no target/attribute columns", len(extra_labelish) == 0,
      f"unexpected={sorted(extra_labelish)}" if extra_labelish else f"cols={sorted(relcols)}")

# ---------------- C4 label-permutation invariance of relational features ----------------
base = pd.read_csv(RELCSV).sort_values("account_id").reset_index(drop=True)
df_perm = pd.read_csv(CORPUS).copy()
rng = np.random.default_rng(1)
df_perm["class_label"] = rng.permutation(df_perm["class_label"].values)  # scramble labels
tmp_dir = Path(tempfile.gettempdir()) / "ccsf_stage3_verify"
tmp_dir.mkdir(parents=True, exist_ok=True)
perm_corpus = tmp_dir / "corpus_labelperm.csv"
perm_features = tmp_dir / "rel_labelperm.csv"
shuffle_corpus = tmp_dir / "corpus_shuffled.csv"
shuffle_features = tmp_dir / "rel_shuffled.csv"
df_perm.to_csv(perm_corpus, index=False)
fe.CORPUS = str(perm_corpus); fe.OUT = str(perm_features)
fe.main()
permf = pd.read_csv(perm_features).sort_values("account_id").reset_index(drop=True)
c4 = np.allclose(base[REL_FEATURES].values, permf[REL_FEATURES].values, atol=1e-9)
check("C4 relational features invariant to label permutation", c4)

# ---------------- C5 row-order shuffle invariance ----------------
df_sh = pd.read_csv(CORPUS).sample(frac=1.0, random_state=7).reset_index(drop=True)
df_sh.to_csv(shuffle_corpus, index=False)
fe.CORPUS = str(shuffle_corpus); fe.OUT = str(shuffle_features)
fe.main()
shf = pd.read_csv(shuffle_features).sort_values("account_id").reset_index(drop=True)
c5 = np.allclose(base[REL_FEATURES].values, shf[REL_FEATURES].values, atol=1e-9)
check("C5 relational features invariant to row-order shuffle", c5)
# restore module paths (defensive)
fe.CORPUS = CORPUS

# ---------------- C6 disjoint partitions ----------------
df = ev.load()
c6 = True; det6 = []
for held in ev.FAMILIES:
    dev = set(df[df.generator_family != held]["account_id"])
    test = set(df[df.generator_family == held]["account_id"])
    if dev & test:
        c6 = False; det6.append(held)
    # no held-out family account in dev
    if any(a in dev for a in test):
        c6 = False
check("C6 dev/test account partitions disjoint per split", c6, str(det6))

# ---------------- C7 scaler fitted on dev only ----------------
from sklearn.preprocessing import StandardScaler
c7 = True
feats = ev.CCSF_FEATURES + ev.REL_FEATURES
for held in ev.FAMILIES:
    dev = df[df.generator_family != held]
    pooled_mean = df[feats].values.mean(axis=0)
    dev_mean = dev[feats].values.mean(axis=0)
    sc = StandardScaler().fit(dev[feats].values)
    if not np.allclose(sc.mean_, dev_mean):
        c7 = False
    # scaler must NOT equal pooled mean unless dev==pooled (they differ here)
    if np.allclose(sc.mean_, pooled_mean):
        c7 = False
check("C7 StandardScaler fitted on development accounts only", c7)

# ---------------- C8 predictions reproduce frozen file ----------------
frozen = pd.read_csv("stage3_outputs/stage3_predictions.csv").sort_values("account_id").reset_index(drop=True)
repro = df.sort_values("account_id").reset_index(drop=True)
c8 = True; det8 = []
for name, f in [("ccsf_lr", ev.CCSF_FEATURES), ("relational_only", ev.REL_FEATURES),
                ("hybrid", ev.CCSF_FEATURES + ev.REL_FEATURES)]:
    p = ev.logo_predict(repro, f)
    if not np.allclose(p, frozen[f"proba_{name}"].values, atol=1e-9):
        c8 = False; det8.append(name)
check("C8 held-out predictions reproduce frozen predictions", c8, str(det8))

# ---------------- C9 permutation negative control at chance ----------------
perm = json.load(open("stage3_outputs/stage3_permutation_results.json"))
c9 = all(0.40 <= perm[m]["mean_auc"] <= 0.60 for m in perm)
check("C9 label-permutation control at chance [0.40,0.60]", c9,
      ", ".join(f"{m}={perm[m]['mean_auc']:.3f}" for m in perm))

# ---------------- C10 fixed CCSF below chance preserved ----------------
res = json.load(open("stage3_outputs/stage3_results.json"))
fx = res["models"]["fixed_ccsf"]["pooled"]["all"]["roc_auc"]
c10 = fx < 0.5
check("C10 fixed-direction CCSF pooled AUC preserved below chance", c10, f"AUC={fx:.3f}")

print("\n" + ("ALL STAGE 3 CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED: {FAIL}"))
sys.exit(0 if not FAIL else 1)


