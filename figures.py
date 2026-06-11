"""figures.py -- publication figures for the CCSF simulation.

Generates the six manuscript figures (300 dpi PNG; Figure 1 also as SVG) plus
fingerprint_profiles.csv, all from results.json / arrays.npz. All plotting
randomness (jitter) is seeded.

  fig1_pipeline.png/.svg    CCSF pipeline schematic
  fig2_features.png         account-level feature distributions (boxplots)
  fig3_fingerprint.png      standardised fingerprint profiles by group  (+ CSV)
  fig4_tsne.png             t-SNE of post embeddings (visualisation only)
  fig5_roc_ablation.png     composite ROC + single-feature/ablation AUCs
  fig6_nonduplication.png   coordinated originality (Jaccard vs cosine)
"""
import json, csv
import numpy as np
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
INK, GREY, PURPLE, EDGE = "#2B2F38", "#6B7280", "#6A4FA3", "#4A4F5A"

# ============================================================== FIG 1: pipeline schematic
def fig_pipeline():
    """Minimal journal-style pipeline figure (canonical generator for manuscript
    Figure 1). Communicates: de-identified aggregates -> four fingerprint
    signals (stance variability exploratory only) -> composite score -> human
    expert review; cluster-level triage, not an individual-level detector."""
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    boxes = [
        ("Synthetic ANDS\ndiscourse corpus", "524 posts · 78 accounts\n8 policy topics", "-", EDGE),
        ("De-identified\naccount-level\naggregates", "no usernames or\npost text retained", "-", EDGE),
        ("Four fingerprint\nsignals", "computed per\naccount", "-", PURPLE),
        ("Composite\nfingerprint score", "equal-weight mean\nof standardised\nsignals", "-", PURPLE),
        ("Human expert\nreview", "high-scoring clusters\nonly; no automated\nlabelling", "--", PURPLE),
    ]
    w, gap, y0, h = 17.8, 2.4, 28.5, 16
    x = 0.6
    cx3 = None
    for i, (title, sub, ls, ec) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.4",
                     lw=1.35, edgecolor=ec, facecolor="#FBFBFC", linestyle=ls))
        nl = title.count("\n")
        ax.text(x + w/2, y0 + h - (3.4 if nl < 2 else 4.2), title, ha="center", va="center",
                fontsize=10.0, weight="bold", color=INK)
        ax.text(x + w/2, y0 + (3.6 if sub.count("\n") > 1 else 3.0), sub, ha="center",
                va="center", fontsize=7.9, style="italic", color=GREY)
        if i == 2:
            cx3 = x + w/2
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.45, y0 + h/2), (x + w + gap - 0.45, y0 + h/2),
                         arrowstyle="-|>", mutation_scale=15, lw=1.5, color=EDGE))
        x += w + gap
    # signals panel under box 3
    pw, px, py, ph = 58, 21, 6.2, 18.5
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0.4",
                 lw=1.15, edgecolor=PURPLE, facecolor="#FFFFFF"))
    ax.plot([cx3, cx3], [y0 - 0.5, py + ph + 0.5], ls=":", lw=1.25, color=PURPLE)
    ax.text(px + pw/2, py + ph - 2.3, "Fingerprint signals · hypothesised coordinated pattern",
            ha="center", va="center", fontsize=9.0, weight="bold", color=PURPLE)
    sigs = [("Language-model perplexity", "▼ lower", PURPLE, INK),
            ("Sentence-length burstiness", "▼ lower", PURPLE, INK),
            ("Compliance-lexicon density", "▲ higher", PURPLE, INK),
            ("Semantic convergence", "▲ higher", PURPLE, INK),
            ("Stance variability — exploratory only, not in composite", "(at chance)", GREY, GREY)]
    yy = py + ph - 5.4
    for name, d, dc, nc in sigs:
        ital = "italic" if dc is GREY else "normal"
        ax.text(px + 3.2, yy, name, ha="left", va="center", fontsize=8.6, color=nc, style=ital)
        ax.text(px + pw - 3.2, yy, d, ha="right", va="center", fontsize=8.6,
                weight="bold" if dc is PURPLE else "normal", color=dc, style=ital)
        yy -= 2.85
    ax.text(50, 2.0, "CCSF is a privacy-preserving, cluster-level triage fingerprint — not an individual-level detector.",
            ha="center", va="center", fontsize=8.9, style="italic", color=INK)
    plt.tight_layout()
    plt.savefig(BASE / "fig1_pipeline.png", bbox_inches="tight")
    plt.savefig(BASE / "fig1_pipeline.svg", bbox_inches="tight")
    plt.close()

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

# ============================================================== FIG 3: fingerprint profiles
FP_FEATS = [("acc_ppl_mean", "Perplexity\n(expected lower)"),
            ("acc_burst", "Burstiness\n(expected lower)"),
            ("acc_compliance", "Compliance density\n(expected higher)"),
            ("acc_stance_std", "Stance variability\n(exploratory)"),
            ("acc_conv", "Semantic convergence\n(expected higher)")]

def fig_fingerprint():
    """Standardised fingerprint profile by group: mean z-score (95% CI) of the
    four primary signals plus the exploratory stance signal, z-scored across
    all 78 accounts exactly as in analysis.py. This figure shows the
    fingerprint itself (the joint pattern), rather than classification
    accuracy. Also writes fingerprint_profiles.csv."""
    acc_group = A["acc_group"]
    X = np.column_stack([A[k] for k, _ in FP_FEATS]).astype(float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)   # population SD, as StandardScaler
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = np.arange(len(FP_FEATS))
    offs = {"organic": -0.10, "professional": -0.034, "mixed": 0.034, "synthetic": 0.10}
    style = {"organic": (1.4, 0.9, 4.5, 2), "professional": (1.4, 0.9, 4.5, 2),
             "mixed": (1.4, 0.9, 4.5, 2), "synthetic": (2.4, 1.0, 6.0, 4)}
    rows_csv = []
    for g in GROUPS:
        m = acc_group == g
        mu = Z[m].mean(axis=0)
        se = Z[m].std(axis=0, ddof=1) / np.sqrt(m.sum())
        ci = 1.96 * se
        lw, al, ms, zo = style[g]
        ax.errorbar(xs + offs[g], mu, yerr=ci, fmt="-o", color=CMAP[g], lw=lw, alpha=al,
                    ms=ms, capsize=3, capthick=1.0, label=LABELS[g], zorder=zo)
        for (k, _), mz, lo_, hi_ in zip(FP_FEATS, mu, mu - ci, mu + ci):
            rows_csv.append(dict(group=g, feature=k.replace("acc_", ""),
                                 mean_z=round(float(mz), 4), ci_lo=round(float(lo_), 4),
                                 ci_hi=round(float(hi_), 4), n=int(m.sum())))
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in FP_FEATS], fontsize=8.0)
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(-1.65, 1.75)
    ax.set_ylabel("Group mean z-score (95% CI)")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(BASE / "fig3_fingerprint.png", bbox_inches="tight"); plt.close()
    with open(BASE / "fingerprint_profiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "feature", "mean_z", "ci_lo", "ci_hi", "n"])
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)

# ============================================================== FIG 4: t-SNE of embeddings
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
    plt.savefig(BASE / "fig4_tsne.png", bbox_inches="tight"); plt.close()

# ============================================================== FIG 5: ROC + ablation
def fig_roc_ablation():
    acc_group = A["acc_group"]; anomaly = A["anomaly"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 4.15))
    targets = [("Synthetic vs organic + professional",
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
        ax1.plot(fpr, tpr, color=c, lw=1.8, label=f"{name} (AUC={auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
    ax1.set_title("Composite fingerprint score (four signals): ROC", fontsize=8.7)
    # legends below the panels so they never overlap the data
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), fontsize=6.9,
               frameon=False, handlelength=1.6)
    # right panel: single-feature AUCs for the 4 composite signals + the
    # exploratory stance signal; ablation (grey) bars only for composite signals
    fn4 = ["ppl_mean", "burstiness", "compliance", "convergence"]
    rows = fn4 + ["stance_std"]
    nice = ["Perplexity", "Burstiness", "Compliance", "Convergence",
            "Stance variability\n(exploratory)"]
    single = [R["auc"][f"feature_{f}_vs_both"] for f in rows]
    drop = [R["ablation"][f"drop_{f}"]["auc"] for f in fn4]
    y = np.arange(len(rows))
    ax2.barh(y + 0.2, single, height=0.38, color="#8172B3", alpha=0.85, label="Single-feature AUC")
    ax2.barh(y[:4] - 0.2, drop, height=0.38, color="#999999", alpha=0.85,
             label="Composite AUC with signal removed")
    ax2.axvline(R["ablation"]["full"], color="black", ls="--", lw=1,
                label=f"Full composite (AUC={R['ablation']['full']:.3f})")
    ax2.set_yticks(y); ax2.set_yticklabels(nice, fontsize=7.6)
    ax2.set_xlim(0.45, 1.02); ax2.set_xlabel("AUC")
    ax2.set_title("Signal contribution", fontsize=9)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), fontsize=6.9,
               frameon=False, handlelength=1.6)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.30, wspace=0.34)
    plt.savefig(BASE / "fig5_roc_ablation.png", bbox_inches="tight"); plt.close()

# ============================================================== FIG 6: coordinated originality
def fig_nondup():
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    # per-group label offsets so annotations never overlap the markers
    OFFS = {"organic": (9, 4, "left"), "professional": (9, -3, "left"),
            "mixed": (-9, -5, "right"), "synthetic": (9, 5, "left")}
    for g in GROUPS:
        v = R["nonduplication"][g]
        ax.scatter(v["mean_jaccard"], v["mean_embed_cosine"], s=130, color=CMAP[g],
                   edgecolor="black", linewidth=0.6, zorder=4)
        dx, dy, ha = OFFS[g]
        ax.annotate(LABELS[g], (v["mean_jaccard"], v["mean_embed_cosine"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.4, ha=ha, zorder=4)
    # dashed ellipse around the two machine-origin groups: the coordinated-
    # originality signature (paraphrased but semantically aligned content)
    sj = R["nonduplication"]["synthetic"]; mj = R["nonduplication"]["mixed"]
    cxe = (sj["mean_jaccard"] + mj["mean_jaccard"]) / 2
    cye = (sj["mean_embed_cosine"] + mj["mean_embed_cosine"]) / 2
    ax.add_patch(Ellipse((cxe, cye), 0.055, 0.085, fill=False, ls="--", lw=1.1,
                 edgecolor=PURPLE, zorder=2))
    ax.annotate("coordinated originality:\nparaphrased but on-message",
                (cxe, cye + 0.046), xytext=(cxe - 0.052, cye + 0.078),
                fontsize=7.4, color=PURPLE, ha="center", style="italic",
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.8))
    ax.set_xlabel("Mean pairwise lexical overlap (Jaccard)\nhigher = more shared wording")
    ax.set_ylabel("Mean pairwise semantic similarity (cosine)\nhigher = more shared meaning")
    ax.set_xlim(0, 0.22); ax.set_ylim(0.25, 0.70)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(BASE / "fig6_nonduplication.png", bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    for f in (fig_pipeline, fig_features, fig_fingerprint, fig_tsne,
              fig_roc_ablation, fig_nondup):
        f(); print("ok", f.__name__)
    print("Figures written.")
