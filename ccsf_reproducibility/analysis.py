"""
analysis.py  -- CCSF proof-of-concept analysis pipeline.
Computes the five fingerprint features with real models, runs inferential statistics,
account-level clustering, detection ROC/AUC, an ablation, and a non-duplication check,
then writes results.json and publication figures.
"""
import json, math, re, itertools, warnings, collections, csv
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent  # write/read all artifacts relative to this script
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import roc_auc_score, silhouette_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.manifold import TSNE
from scipy import stats
import corpus_gen

RNG = np.random.RandomState(corpus_gen.SEED)
torch.manual_seed(corpus_gen.SEED)

# ---------------------------------------------------------------- load models
print("Loading models ...")
TOK = GPT2TokenizerFast.from_pretrained("gpt2")
LM = GPT2LMHeadModel.from_pretrained("gpt2").eval()
EMB = SentenceTransformer("all-MiniLM-L6-v2")

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
WORD = re.compile(r"[A-Za-z']+")

def sentences(t):
    parts = [s.strip() for s in SENT_SPLIT.split(t.strip()) if s.strip()]
    return parts if parts else [t.strip()]

def tokens(t):
    return WORD.findall(t.lower())

def perplexity(text):
    ids = TOK(text, return_tensors="pt", truncation=True, max_length=256).input_ids
    if ids.shape[1] < 2:
        ids = TOK(text + " .", return_tensors="pt").input_ids
    with torch.no_grad():
        loss = LM(ids, labels=ids).loss.item()
    return math.exp(loss)

def compliance_density(text):
    # Density of pre-specified industry/commercial framing terms per word.
    # Each lexicon phrase is matched independently on word boundaries (case-insensitive).
    # Hyphens/apostrophes inside a phrase are treated as part of the phrase; a longer
    # phrase that contains a shorter listed phrase will count both, by design, because
    # each phrase is an independent framing cue. This single, documented rule replaces
    # the earlier ambiguous substring count.
    low = text.lower()
    toks = max(len(tokens(text)), 1)
    hits = 0
    for term in corpus_gen.COMPLIANCE_LEXICON:
        pattern = r"(?<![a-z])" + re.escape(term.lower()) + r"(?![a-z])"
        hits += len(re.findall(pattern, low))
    return hits / toks

# stance anchors (embedding-based directional score)
ANCHOR_LIGHT = "Adults should keep access to these products; regulation should be light-touch and proportionate."
ANCHOR_STRICT = "These products should be strictly restricted or banned to protect public health and young people."

# ---------------------------------------------------------------- build corpus + post features
rows = corpus_gen.build_corpus()
texts = [r["text"] for r in rows]
print(f"Corpus: {len(rows)} posts. Scoring perplexity + embeddings ...")

emb = EMB.encode(texts, normalize_embeddings=True, show_progress_bar=False)
anchors = EMB.encode([ANCHOR_LIGHT, ANCHOR_STRICT], normalize_embeddings=True)
for i, r in enumerate(rows):
    r["ppl"] = perplexity(r["text"])
    sl = [len(tokens(s)) for s in sentences(r["text"])]
    r["sent_lengths"] = sl
    r["compliance"] = compliance_density(r["text"])
    r["stance"] = float(emb[i] @ anchors[0] - emb[i] @ anchors[1])
    r["emb"] = emb[i]

groups = ["organic", "synthetic", "professional", "mixed"]
G = {g: [r for r in rows if r["group"] == g] for g in groups}

# ---------------------------------------------------------------- account-level features
def cv(xs):
    xs = np.asarray(xs, float)
    return float(xs.std(ddof=0) / xs.mean()) if xs.mean() > 0 and len(xs) > 1 else 0.0

accounts = {}
for r in rows:
    accounts.setdefault(r["account"], []).append(r)

acc_rows = []
for acc, posts in accounts.items():
    g = posts[0]["group"]
    all_sent_lengths = list(itertools.chain.from_iterable(p["sent_lengths"] for p in posts))
    ppls = np.array([p["ppl"] for p in posts])
    embs = np.vstack([p["emb"] for p in posts])
    # intra-account mean pairwise cosine (convergence)
    if len(embs) > 1:
        sim = embs @ embs.T
        iu = np.triu_indices(len(embs), k=1)
        conv = float(sim[iu].mean())
    else:
        conv = np.nan
    acc_rows.append(dict(
        account=acc, group=g, n_posts=len(posts),
        ppl_mean=float(ppls.mean()),
        ppl_var=float(ppls.var(ddof=0)),
        burstiness=cv(all_sent_lengths),
        compliance=float(np.mean([p["compliance"] for p in posts])),
        stance_std=float(np.std([p["stance"] for p in posts], ddof=0)),  # low = stable
        convergence=conv,
    ))

A = {g: [a for a in acc_rows if a["group"] == g] for g in groups}

def col(rowset, key):
    return np.array([r[key] for r in rowset], float)

# ---------------------------------------------------------------- effect sizes / tests
def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))

def compare(a, b, key, level="account"):
    src = A if level == "account" else G
    xa = col(src[a], key); xb = col(src[b], key)
    u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    return dict(a=a, b=b, feature=key, level=level,
                a_median=float(np.median(xa)), b_median=float(np.median(xb)),
                a_mean=float(np.mean(xa)), b_mean=float(np.mean(xb)),
                U=float(u), p=float(p), cliffs_delta=float(cliffs_delta(xa, xb)),
                n_a=len(xa), n_b=len(xb))

results = {"meta": {}, "tests": [], "auc": {}, "clustering": {}, "ablation": {},
           "nonduplication": {}, "descriptives": {}}

results["meta"] = dict(
    seed=corpus_gen.SEED, n_posts=len(rows),
    n_accounts={g: len(A[g]) for g in groups},
    n_posts_by_group={g: len(G[g]) for g in groups},
    lm="gpt2", embed_model="all-MiniLM-L6-v2",
    compliance_lexicon=corpus_gen.COMPLIANCE_LEXICON,
)

# account-level comparisons for the 5 fingerprint features
acc_feats = ["ppl_mean", "ppl_var", "burstiness", "compliance", "stance_std", "convergence"]
for f in acc_feats:
    results["tests"].append(compare("synthetic", "organic", f, "account"))
    results["tests"].append(compare("synthetic", "professional", f, "account"))
    results["tests"].append(compare("mixed", "organic", f, "account"))

# post-level distributional comparisons for perplexity, compliance, stance
for f in ["ppl", "compliance", "stance"]:
    results["tests"].append(compare("synthetic", "organic", f, "post"))
    results["tests"].append(compare("synthetic", "professional", f, "post"))

# descriptives (account-level mean/sd per group)
for g in groups:
    results["descriptives"][g] = {f: dict(mean=float(col(A[g], f).mean()),
                                          sd=float(col(A[g], f).std(ddof=1)))
                                  for f in acc_feats}

# ---------------------------------------------------------------- composite anomaly score (equal weights)
# A_i = mean[ Z(-P), Z(-B), Z(L), Z(-S_stance), Z(E) ]  (per-account, standardised across all accounts).
# Perplexity is treated as a feature FAMILY: mean perplexity enters the composite, while
# within-account perplexity variance (ppl_var) is reported descriptively only and is
# deliberately excluded from the composite to avoid double-weighting the perplexity family.
# The composite is the MEAN (not sum) of the five oriented z-scores; this scaling choice
# does not affect any AUC (a monotonic transform) but keeps the score on an interpretable scale.
M = np.array([[a["ppl_mean"], a["burstiness"], a["compliance"], a["stance_std"], a["convergence"]]
              for a in acc_rows], float)
labels_syn = np.array([1 if a["group"] == "synthetic" else 0 for a in acc_rows])
labels_3 = np.array([a["group"] for a in acc_rows])
Z = StandardScaler().fit_transform(M)
signs = np.array([-1, -1, +1, -1, +1])  # P-, B-, L+, S(stance var)-, E+
anomaly = (Z * signs).mean(axis=1)
for a, s in zip(acc_rows, anomaly):
    a["anomaly"] = float(s)

# AUCs: synthetic vs organic, synthetic vs professional, synthetic vs (organic+professional)
def auc_for(score_name_or_vec, positive="synthetic", negatives=("organic",), invert=False):
    mask = np.array([a["group"] in (positive,) + tuple(negatives) for a in acc_rows])
    y = np.array([1 if a["group"] == positive else 0 for a in acc_rows])[mask]
    if isinstance(score_name_or_vec, str):
        s = col([a for a in acc_rows], score_name_or_vec)[mask]
    else:
        s = np.asarray(score_name_or_vec)[mask]
    if invert:
        s = -s
    return float(roc_auc_score(y, s))

results["auc"]["composite_vs_organic"] = auc_for(anomaly, negatives=("organic",))
results["auc"]["composite_vs_professional"] = auc_for(anomaly, negatives=("professional",))
results["auc"]["composite_vs_both"] = auc_for(anomaly, negatives=("organic", "professional"))
results["auc"]["composite_mixed_vs_organic"] = auc_for(anomaly, positive="mixed", negatives=("organic",))

# Realistic operational target: any machine-origin coordination (synthetic OR mixed)
# treated as positive, vs genuine human discourse (organic + professional) as negative.
mask_op = np.array([a["group"] in ("synthetic", "mixed", "organic", "professional") for a in acc_rows])
y_op = np.array([1 if a["group"] in ("synthetic", "mixed") else 0 for a in acc_rows])
results["auc"]["composite_machineorigin_vs_human"] = float(roc_auc_score(y_op, anomaly))

# bootstrap 95% CIs for the two headline composite AUCs (stratified resampling of accounts)
def bootstrap_auc_ci(y, s, n_boot=2000, seed=corpus_gen.SEED):
    y = np.asarray(y); s = np.asarray(s)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)])
        if len(np.unique(y[bi])) < 2:
            continue
        boots.append(roc_auc_score(y[bi], s[bi]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)

mask_both = np.array([a["group"] in ("synthetic", "organic", "professional") for a in acc_rows])
y_both = np.array([1 if a["group"] == "synthetic" else 0 for a in acc_rows])[mask_both]
lo, hi = bootstrap_auc_ci(y_both, anomaly[mask_both])
results["auc"]["composite_vs_both_ci"] = [lo, hi]
lo2, hi2 = bootstrap_auc_ci(y_op, anomaly)
results["auc"]["composite_machineorigin_vs_human_ci"] = [lo2, hi2]

# single-feature account-level AUCs (synthetic vs organic+professional), orient each feature
feat_orient = dict(ppl_mean=True, burstiness=True, compliance=False, stance_std=True, convergence=False)
for f, inv in feat_orient.items():
    results["auc"][f"feature_{f}_vs_both"] = auc_for(f, negatives=("organic", "professional"), invert=inv)

# ---------------------------------------------------------------- supervised cross-validated detector (rigor check)
Xcv = Z  # standardized 5 features
ycv = labels_syn
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=corpus_gen.SEED)
clf = LogisticRegression(max_iter=1000)
cv_auc = cross_val_score(clf, Xcv, ycv, cv=skf, scoring="roc_auc")
results["auc"]["logreg_cv_mean"] = float(cv_auc.mean())
results["auc"]["logreg_cv_sd"] = float(cv_auc.std())

# ---------------------------------------------------------------- ablation (leave-one-feature-out, composite AUC vs both)
feat_names = ["ppl_mean", "burstiness", "compliance", "stance_std", "convergence"]
full_auc = results["auc"]["composite_vs_both"]
results["ablation"]["full"] = full_auc
for j, fn in enumerate(feat_names):
    keep = [k for k in range(5) if k != j]
    Zsub = Z[:, keep]
    sub_anom = (Zsub * signs[keep]).sum(axis=1)
    results["ablation"][f"drop_{fn}"] = dict(
        auc=auc_for(sub_anom, negatives=("organic", "professional")),
        delta=auc_for(sub_anom, negatives=("organic", "professional")) - full_auc)

# ---------------------------------------------------------------- unsupervised clustering (account level)
km = KMeans(n_clusters=4, n_init=10, random_state=corpus_gen.SEED).fit(Z)
sil = silhouette_score(Z, km.labels_)
# map predicted clusters to dominant group for a purity measure
purity = 0
comp = {}
for c in set(km.labels_):
    members = [labels_3[i] for i in range(len(labels_3)) if km.labels_[i] == c]
    dom = collections.Counter(members).most_common(1)[0]
    purity += dom[1]
    comp[int(c)] = dict(collections.Counter(members))
purity /= len(labels_3)
gmap = {"organic": 0, "synthetic": 1, "professional": 2, "mixed": 3}
ari = adjusted_rand_score([gmap[g] for g in labels_3], km.labels_)
results["clustering"] = dict(method="KMeans(k=4) on 5 standardized account features",
                             silhouette=float(sil), purity=float(purity),
                             adjusted_rand=float(ari), cluster_composition=comp)

# DBSCAN as anomaly-style alternative
db = DBSCAN(eps=1.6, min_samples=4).fit(Z)
results["clustering"]["dbscan_labels_by_group"] = {
    g: dict(collections.Counter(int(db.labels_[i]) for i in range(len(acc_rows)) if acc_rows[i]["group"] == g))
    for g in groups}

# ---------------------------------------------------------------- non-duplication check (coordinated ORIGINALITY)
def mean_pairwise_jaccard(posts, cap=400):
    sets = [set(tokens(p["text"])) for p in posts]
    pairs = list(itertools.combinations(range(len(sets)), 2))
    if len(pairs) > cap:
        idx = RNG.choice(len(pairs), cap, replace=False)
        pairs = [pairs[i] for i in idx]
    js = []
    for i, j in pairs:
        u = sets[i] | sets[j]; inter = sets[i] & sets[j]
        if u:
            js.append(len(inter) / len(u))
    if not js:
        return 0.0, 0.0
    return float(np.mean(js)), float(np.max(js))

for g in groups:
    mj, xj = mean_pairwise_jaccard(G[g])
    # mean intra-group embedding cosine (semantic convergence at post level)
    e = np.vstack([r["emb"] for r in G[g]])
    sim = e @ e.T; iu = np.triu_indices(len(e), k=1)
    results["nonduplication"][g] = dict(mean_jaccard=mj, max_jaccard=xj,
                                        mean_embed_cosine=float(sim[iu].mean()))

# ---------------------------------------------------------------- save results + arrays for plotting
with open(BASE / "results.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez(BASE / "arrays.npz",
         emb=np.vstack([r["emb"] for r in rows]),
         group=np.array([r["group"] for r in rows]),
         ppl=np.array([r["ppl"] for r in rows]),
         compliance=np.array([r["compliance"] for r in rows]),
         stance=np.array([r["stance"] for r in rows]),
         acc_group=labels_3,
         acc_burst=col(acc_rows, "burstiness"),
         acc_conv=col(acc_rows, "convergence"),
         acc_stance_std=col(acc_rows, "stance_std"),
         acc_compliance=col(acc_rows, "compliance"),
         acc_ppl_mean=col(acc_rows, "ppl_mean"),
         anomaly=anomaly)

# human-readable CSV exports (regenerated by this script for full reproducibility)
with open(BASE / "synthetic_corpus.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "account", "group", "topic", "text"])
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in ["id", "account", "group", "topic", "text"]})

with open(BASE / "account_features.csv", "w", newline="", encoding="utf-8") as f:
    fields = ["account", "group", "n_posts", "ppl_mean", "ppl_var", "burstiness",
              "compliance", "stance_std", "convergence", "anomaly"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for a in acc_rows:
        w.writerow({k: (round(a[k], 6) if isinstance(a[k], float) else a[k]) for k in fields})

print("\n=== KEY RESULTS ===")
print("Accounts:", results["meta"]["n_accounts"])
print("Composite AUC syn vs organic:", round(results["auc"]["composite_vs_organic"],3))
print("Composite AUC syn vs professional:", round(results["auc"]["composite_vs_professional"],3))
print("Composite AUC syn vs both human:", round(results["auc"]["composite_vs_both"],3))
print("Composite AUC mixed vs organic:", round(results["auc"]["composite_mixed_vs_organic"],3))
print("Composite AUC machine-origin vs human:", round(results["auc"]["composite_machineorigin_vs_human"],3))
print("LogReg 5-fold CV AUC (syn vs rest):", round(results["auc"]["logreg_cv_mean"],3), "+/-", round(results["auc"]["logreg_cv_sd"],3))
print("KMeans(k=4) silhouette:", round(sil,3), "purity:", round(purity,3), "ARI:", round(ari,3))
print("Non-dup mean Jaccard (syn):", round(results["nonduplication"]["synthetic"]["mean_jaccard"],3),
      "| embed cosine (syn):", round(results["nonduplication"]["synthetic"]["mean_embed_cosine"],3))
print("Done.")
