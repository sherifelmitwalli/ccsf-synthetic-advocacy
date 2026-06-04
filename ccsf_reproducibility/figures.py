"""figures.py -- publication figures for the CCSF simulation."""
import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score
import corpus_gen

BASE = Path(__file__).resolve().parent
RNG = np.random.default_rng(corpus_gen.SEED)  # seed all plotting jitter for reproducibility

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
R = json.load(open(BASE / "results.json"))
A = np.load(BASE / "arrays.npz", allow_pickle=True)

GROUPS = ["organic", "professional", "mixed", "synthetic"]
LABELS = {"organic": "Organic human", "professional": "Professional advocacy",
          "mixed": "Human-edited synthetic", "synthetic": "Synthetic (coordinated)"}
CMAP = {"organic": "#4C72B0", "professional": "#55A868",
        "mixed": "#C44E52", "synthetic": "#8172B3"}

# ============================================================== FIG 1: pipeline schematic
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    steps = ["De-identified\ndiscourse corpus", "Data minimisation\n& abstraction",
             "Feature extraction\n(5 signals)", "Account-level\nclustering",
             "Composite anomaly\nscoring", "Human expert review\n(future deployment)"]
    x = 2
    for i, s in enumerate(steps):
        c = "#EAEAF2" if i not in (2, 4) else "#DCD6EC"
        box = FancyBboxPatch((x, 9), 13.5, 12, boxstyle="round,pad=0.3",
                             linewidth=1, edgecolor="#555", facecolor=c)
        ax.add_patch(box)
        ax.text(x + 6.75, 15, s, ha="center", va="center", fontsize=7.6)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 13.7, 15), (x + 16, 15),
                         arrowstyle="-|>", mutation_scale=11, color="#555"))
        x += 16.3
    feats = ["Language-model perplexity", "Sentence-length burstiness", "Compliance-lexicon density",
             "Stance variability", "Embedding convergence"]
    ax.text(34.7, 6.5, "  •  ".join(feats), ha="center", va="center",
            fontsize=6.6, style="italic", color="#444")
    ax.add_patch(FancyArrowPatch((34.7, 8.7), (34.7, 7.6), arrowstyle="-",
                 color="#999", linestyle=":"))
    plt.tight_layout()
    plt.savefig(BASE / "fig1_pipeline.png", bbox_inches="tight"); plt.close()

# ============================================================== FIG 2: feature distributions
def fig_features():
    feats = [("acc_ppl_mean", "Language-model perplexity\n(account mean)", False),
             ("acc_burst", "Sentence-length burstiness\n(coefficient of variation)", False),
             ("acc_compliance", "Compliance-lexicon density", False),
             ("acc_conv", "Embedding-space convergence\n(mean intra-account cosine)", False)]
    acc_group = A["acc_group"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, (key, title, logy) in zip(axes.ravel(), feats):
        data = [A[key][acc_group == g] for g in GROUPS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(CMAP[g]); patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(1.2)
        for i, g in enumerate(GROUPS):
            y = A[key][acc_group == g]
            x = RNG.normal(i + 1, 0.06, len(y))
            ax.scatter(x, y, s=7, color=CMAP[g], edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(["Organic", "Prof.", "Mixed", "Synth."], fontsize=8)
        ax.set_title(title, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(BASE / "fig2_features.png", bbox_inches="tight"); plt.close()

# ============================================================== FIG 3: t-SNE of embeddings
def fig_tsne():
    emb = A["emb"]; grp = A["group"]
    ts = TSNE(n_components=2, perplexity=30, init="pca",
              random_state=corpus_gen.SEED, learning_rate="auto").fit_transform(emb)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for g in GROUPS:
        m = grp == g
        ax.scatter(ts[m, 0], ts[m, 1], s=14, color=CMAP[g], alpha=0.7,
                   edgecolor="white", linewidth=0.3, label=LABELS[g])
    ax.set_xlabel("t-SNE dimension 1"); ax.set_ylabel("t-SNE dimension 2")
    ax.legend(fontsize=7.5, loc="best", framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(BASE / "fig3_tsne.png", bbox_inches="tight"); plt.close()

# ============================================================== FIG 4: ROC + ablation
def fig_roc_ablation():
    acc_group = A["acc_group"]; anomaly = A["anomaly"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    # ROC: composite for three targets
    targets = [("Synthetic vs organic+professional",
                np.isin(acc_group, ["synthetic", "organic", "professional"]),
                acc_group == "synthetic", "#8172B3"),
               ("Human-edited synthetic vs organic",
                np.isin(acc_group, ["mixed", "organic"]),
                acc_group == "mixed", "#C44E52"),
               ("Any machine-origin vs human",
                np.full(len(acc_group), True),
                np.isin(acc_group, ["synthetic", "mixed"]), "#555555")]
    for name, mask, pos, c in targets:
        y = pos[mask].astype(int); s = anomaly[mask]
        fpr, tpr, _ = roc_curve(y, s); auc = roc_auc_score(y, s)
        ax1.plot(fpr, tpr, color=c, lw=1.8, label=f"{name} (AUC={auc:.2f})")
    ax1.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
    ax1.set_title("Composite anomaly score: ROC", fontsize=9)
    ax1.legend(fontsize=6.3, loc="lower right")
    # ablation + single-feature AUC
    fn = ["ppl_mean", "burstiness", "compliance", "stance_std", "convergence"]
    nice = ["Perplexity", "Burstiness", "Compliance", "Stance stab.", "Convergence"]
    single = [R["auc"][f"feature_{f}_vs_both"] for f in fn]
    drop = [R["ablation"][f"drop_{f}"]["auc"] for f in fn]
    y = np.arange(len(fn))
    ax2.barh(y + 0.2, single, height=0.38, color="#8172B3", alpha=0.85, label="Single-feature AUC")
    ax2.barh(y - 0.2, drop, height=0.38, color="#999999", alpha=0.85, label="Composite AUC w/ feature removed")
    ax2.axvline(R["ablation"]["full"], color="black", ls="--", lw=1,
                label=f"Full composite ({R['ablation']['full']:.2f})")
    ax2.set_yticks(y); ax2.set_yticklabels(nice, fontsize=8)
    ax2.set_xlim(0.45, 1.02); ax2.set_xlabel("AUC")
    ax2.set_title("Feature contribution", fontsize=9)
    ax2.legend(fontsize=6.0, loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(BASE / "fig4_roc_ablation.png", bbox_inches="tight"); plt.close()

# ============================================================== FIG 5: lexical vs semantic
def fig_nondup():
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    for g in GROUPS:
        v = R["nonduplication"][g]
        ax.scatter(v["mean_jaccard"], v["mean_embed_cosine"], s=130, color=CMAP[g],
                   edgecolor="black", linewidth=0.6, zorder=3, label=LABELS[g])
        ax.annotate(LABELS[g], (v["mean_jaccard"], v["mean_embed_cosine"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=7.2)
    ax.set_xlabel("Mean pairwise lexical overlap (Jaccard)")
    ax.set_ylabel("Mean pairwise semantic similarity (cosine)")
    ax.set_title("Coordinated originality:\nlow lexical overlap, high semantic convergence", fontsize=9)
    ax.set_xlim(0, 0.22); ax.set_ylim(0.25, 0.68)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(BASE / "fig5_nonduplication.png", bbox_inches="tight"); plt.close()

for f in (fig_pipeline, fig_features, fig_tsne, fig_roc_ablation, fig_nondup):
    f(); print("ok", f.__name__)
print("Figures written.")
