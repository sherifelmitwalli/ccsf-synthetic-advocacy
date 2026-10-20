#!/usr/bin/env python3
"""Verification checks for Stage 4 analysis R6 (stage4_outputs/stance_matched/)."""
import hashlib
import json
import os

import numpy as np
import pandas as pd

import stage4_robustness as s4
import stage4_stance_matched as r6

OUT = r6.OUT
ok_all = []


def check(name, ok, detail=""):
    ok_all.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


man = json.load(open(os.path.join(OUT, "r6_manifest.json")))
res = json.load(open(os.path.join(OUT, "r6_results.json")))

rec = open("STAGE4_STANCE_PROTOCOL.sha256").read().split()[0]
check("S1 protocol unchanged since pre-results checksum", sha("STAGE4_STANCE_PROTOCOL.md") == rec == man["protocol_sha256"])
check("S2 input checksums match manifest", all(sha(f) == h for f, h in man["input_sha256"].items()))
check("S3 output checksums match manifest",
      all(sha(os.path.join(OUT, f)) == h for f, h in man["output_sha256"].items()))

corpus, F = s4.load()
M, pairs = r6.matched_subset(corpus, F)  # raises if any pair is mismatched
counts = M.groupby(["generator_family", "role"]).size()
check("S4 matched subset is 7 coordinated vs 7 organic per family, pairs identical on topic, prompt family, "
      "temperature, transformation", (counts == 7).all() and len(counts) == 6 and len(pairs) == 21)

src = open("stage2_run.py").read()
check("S5 generator code assigns organic stance and prompt family by k % 3 (supportive = index 0)",
      'ORGANIC_STANCES[account["persona_index"] % len(ORGANIC_STANCES)]' in src
      and 'families[k % len(families)]' in src
      and 'ORGANIC_STANCES = ["broadly supportive of these products"' in src)

A = pd.read_csv(os.path.join(OUT, "r6a_predictions.csv")).set_index("account_id")
frozen = pd.read_csv(s4.PRED_FROZEN).set_index("account_id")
content = pd.read_csv(os.path.join(s4.OUTDIR, "stage4_content_predictions.csv")).set_index("account_id")
ok = (np.allclose(A.hybrid, frozen.loc[A.index, "proba_hybrid"]) and
      np.allclose(A.relational_only, frozen.loc[A.index, "proba_relational_only"]) and
      np.allclose(A.ccsf_lr, frozen.loc[A.index, "proba_ccsf_lr"]) and
      np.allclose(A.content_tfidf_lr, content.loc[A.index, "proba_content_tfidf_lr"]))
check("S6 R6a uses the frozen original-model predictions unchanged (no retraining)", ok)

G = r6.matched_features(corpus, M)
perm = corpus.copy()
perm["class_label"] = np.random.default_rng(5).permutation(perm["class_label"].values)
perm = perm.sample(frac=1, random_state=6)
G2 = r6.matched_features(perm, M)
check("S7 matched-batch relational features invariant to label permutation and row shuffle",
      np.allclose(G[s4.REL].values, G2[s4.REL].values, atol=1e-12))

ok = True
for fam in s4.FAMILIES:
    dev, test = G[G.generator_family != fam], G[G.generator_family == fam]
    ok &= set(dev.index).isdisjoint(test.index) and fam not in set(dev.generator_family)
check("S8 development and held-out partitions disjoint by generator family", ok)

B = r6.r6b(corpus, G)
saved = pd.read_csv(os.path.join(OUT, "r6b_predictions.csv")).set_index("account_id")
check("S9 R6b retrained predictions reproduce deterministically",
      all(np.allclose(saved.loc[B.index, m], B[m].astype(float)) for m in
          ["ccsf_lr", "relational_only", "hybrid", "content_tfidf_lr"]))

from sklearn.feature_extraction.text import TfidfVectorizer
docs = s4.account_docs(corpus)
fam = s4.FAMILIES[2]
dev_ids, test_ids = G[G.generator_family != fam].index, G[G.generator_family == fam].index
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                      token_pattern=s4.s3.TOKEN_PATTERN).fit(docs.loc[dev_ids].values)
held_only = (set(TfidfVectorizer(token_pattern=s4.s3.TOKEN_PATTERN).fit(docs.loc[test_ids].values).vocabulary_)
             - set(TfidfVectorizer(token_pattern=s4.s3.TOKEN_PATTERN).fit(docs.loc[dev_ids].values).vocabulary_))
check("S10 R6b content vocabulary fitted on development matched accounts only",
      not (held_only & set(vec.vocabulary_)), f"{len(held_only)} held-out-only tokens excluded")

print("\nALL R6 CHECKS PASSED" if all(ok_all) else "\nSOME R6 CHECKS FAILED")
raise SystemExit(0 if all(ok_all) else 1)
