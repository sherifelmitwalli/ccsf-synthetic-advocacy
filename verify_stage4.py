#!/usr/bin/env python3
"""Verification checks for the Stage 4 post hoc robustness audit (stage4_outputs/)."""
import hashlib
import json

import numpy as np
import pandas as pd

import stage4_robustness as s4

OUT = s4.OUTDIR
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


man = json.load(open(f"{OUT}/stage4_manifest.json"))
res = json.load(open(f"{OUT}/stage4_results.json"))

recorded = open("STAGE4_PROTOCOL.sha256").read().split()[0]
check("C1 protocol unchanged since pre-results checksum",
      sha("STAGE4_PROTOCOL.md") == recorded == man["protocol_sha256"])

check("C2 input checksums match manifest",
      all(sha(f) == h for f, h in man["input_sha256"].items()))

check("C3 output checksums match manifest",
      all(sha(f"{OUT}/{f}") == h for f, h in man["output_sha256"].items()))

corpus, F = s4.load()
dev = 0.0
for h in s4.FAMILIES:
    ids = F[F.generator_family == h].index.tolist()
    R = s4.batch_rel(corpus, ids)
    dev = max(dev, float(np.abs(R.loc[ids, s4.REL].values - F.loc[ids, s4.REL].values).max()))
check("C4 full-batch recomputation reproduces frozen Stage 3 features", dev < 1e-9, f"max dev={dev:.1e}")

# label-free construction: permuting labels and shuffling rows leaves diluted-batch features unchanged
h = s4.FAMILIES[1]
ids = (F[(F.generator_family == h) & (F.y == 1)].index[:3].tolist()
       + F[(F.generator_family == h) & (F.y == 0)].index.tolist())
base = s4.batch_rel(corpus, ids).loc[ids, s4.REL]
perm = corpus.copy()
perm["class_label"] = np.random.default_rng(1).permutation(perm["class_label"].values)
perm["campaign_id"] = np.random.default_rng(2).permutation(perm["campaign_id"].values)
perm = perm.sample(frac=1, random_state=3)
alt = s4.batch_rel(perm, ids).loc[ids, s4.REL]
check("C5 diluted-batch features invariant to label permutation and row shuffle",
      np.allclose(base.values, alt.values, atol=1e-12))

ok = True
for s in res["R4_splits"]:
    ok &= (s["n_dev"] == 84 and s["n_test"] == 42 and s["dev_pos"] == 20 and s["test_pos"] == 10)
reps = sorted(F[F.y == 1].repertoire.unique())
for a, b in [(reps[0], reps[1]), (reps[1], reps[0])]:
    for h in s4.FAMILIES:
        d = F[(F.generator_family != h) & ((F.y == 0) | (F.repertoire == a))]
        t = F[(F.generator_family == h) & ((F.y == 0) | (F.repertoire == b))]
        ok &= set(d.index).isdisjoint(t.index)
        ok &= not (d[d.y == 1].repertoire == b).any() and not (t[t.y == 1].repertoire == a).any()
        ok &= not (set(d.generator_family) & set(t.generator_family))
check("C6 cross-repertoire splits disjoint by family and repertoire", bool(ok))

D = pd.read_csv(f"{OUT}/stage4_dilution_draws.csv")
D1 = D[(D.background == "heldout_32") & (D.k < 20)]
first = D1[(D.family == s4.FAMILIES[0])]
first = first[first.campaign == sorted(first.campaign.unique())[0]]
rep = s4.dilution(corpus, F, res["draws_per_cell"], s4.K_GRID, False, first_cell_only=True)
m = first.merge(rep, on=["family", "campaign", "k", "draw", "model"], suffixes=("", "_re"))
check("C7 R1 draws reproduce deterministically (first cell)",
      len(m) == len(rep) and np.allclose(m.auc_all, m.auc_all_re, equal_nan=True),
      f"rows={len(m)}")

R3, content = s4.baselines(corpus, F)
saved = pd.read_csv(f"{OUT}/stage4_content_predictions.csv").set_index("account_id")
check("C8 content-classifier predictions reproduce",
      np.allclose(saved.loc[F.index, "proba_content_tfidf_lr"].values, content.values, atol=1e-12))

from sklearn.feature_extraction.text import TfidfVectorizer
docs = s4.account_docs(corpus)
held = s4.FAMILIES[0]
dev_ids = F[F.generator_family != held].index
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                      token_pattern=s4.s3.TOKEN_PATTERN).fit(docs.loc[dev_ids].values)
held_only = (set(TfidfVectorizer(token_pattern=s4.s3.TOKEN_PATTERN).fit(
    docs.loc[F[F.generator_family == held].index].values).vocabulary_)
    - set(TfidfVectorizer(token_pattern=s4.s3.TOKEN_PATTERN).fit(docs.loc[dev_ids].values).vocabulary_))
check("C9 content vocabulary fitted on development documents only",
      not (held_only & set(vec.vocabulary_)), f"{len(held_only)} held-out-only tokens excluded")

k1 = res["R1_R2_dilution"]["heldout_32"]
check("C10 fixed reference (k=20) equals frozen Stage 3 per-family means",
      abs(k1["hybrid"]["20"]["auc_all"]["mean"]
          - np.mean([json.load(open("stage3_outputs/stage3_results.json"))["models"]["hybrid"]
                     ["per_family"]["all"][f]["roc_auc"] for f in s4.FAMILIES])) < 1e-9)

print("\nALL STAGE 4 CHECKS PASSED" if all(results) else "\nSOME STAGE 4 CHECKS FAILED")
raise SystemExit(0 if all(results) else 1)
