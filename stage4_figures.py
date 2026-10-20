#!/usr/bin/env python3
"""Stage 4 figure (campaign-size dilution and cross-repertoire transfer) from frozen outputs.

Writes stage4_outputs/figures/stage4_fig_dilution_transfer.{png,tif} at 300 dpi, 180 mm wide.
Colour-blind-validated categorical slots; every series also has a distinct marker and line style,
so identity never relies on colour alone.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else "stage4_outputs"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
S = {  # model: (label, colour, marker, linestyle)
    "ccsf_lr": ("CCSF-LR (4 account-level)", "#2a78d6", "o", "-"),
    "relational_only": ("Relational-only (TF-IDF)", "#eb6834", "s", "--"),
    "hybrid": ("Hybrid (CCSF + relational)", "#1baf7a", "D", "-"),
    "minilm_semantic_only": ("MiniLM semantic-only", "#eda100", "^", ":"),
    "content_tfidf_lr": ("Supervised TF-IDF content classifier", "#6250d6", "v", "-."),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

D = pd.read_csv(os.path.join(OUT, "stage4_dilution_draws.csv"))
D = D[D.background == "heldout_32"]
M = pd.read_csv(os.path.join(OUT, "stage4_minilm_dilution_draws.csv"))
R = json.load(open(os.path.join(OUT, "stage4_results.json")))

fig, axes = plt.subplots(1, 3, figsize=(7.09, 4.0), gridspec_kw={"width_ratios": [1.1, 1.1, 1.25]})
ks = [1, 2, 3, 5, 10, 20]
xpos = {k: i for i, k in enumerate(ks)}


def curve(ax, df, model, metric):
    g = df[df.model == model].groupby("k")[metric]
    m = g.mean()
    lab, col, mk, ls = S[model]
    x = [xpos[k] for k in m.index]
    ax.plot(x, m.values, color=col, marker=mk, ls=ls, lw=1.6, ms=5, label=lab,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3)


for ax, metric, title, ylab in [(axes[0], "auc_all", "A  Campaign-size dilution", "ROC AUC vs all controls"),
                                (axes[1], "r_precision", "B  Top-k precision", "R-precision")]:
    for model in ["ccsf_lr", "relational_only", "hybrid"]:
        curve(ax, D, model, metric)
    curve(ax, M, "minilm_semantic_only", metric)
    if metric == "auc_all":
        ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_ylim(0.3, 1.02)
    else:
        ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks[:-1]] + ["20*"])
    ax.set_xlabel("Campaign accounts in batch (k)")
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)

# panel C: cross-repertoire transfer, both directions, plus leave-one-family-out reference
ax = axes[2]
models = ["ccsf_lr", "relational_only", "hybrid", "content_tfidf_lr"]
dirs = list(R["R4_cross_repertoire"].keys())
short = {dirs[0]: "Unseen:\npolicy-\nbalance", dirs[1]: "Unseen:\nconsumer-\naccess"}
logo = {"ccsf_lr": R["R3_baselines"]["pooled"]["ccsf_lr"],
        "relational_only": None,
        "hybrid": R["R3_baselines"]["pooled"]["hybrid"],
        "content_tfidf_lr": R["R3_baselines"]["pooled"]["content_tfidf_lr"]}
groups = ["Shared\nrepertoire\n(LOGO)"] + [short[d] for d in dirs]
for j, model in enumerate(models):
    lab, col, mk, _ = S[model]
    xs, ys, lo, hi = [], [], [], []
    for gi, gname in enumerate(groups):
        if gi == 0:
            e = logo[model]
            if e is None:
                s3 = json.load(open("stage3_outputs/stage3_results.json"))["models"]["relational_only"]["pooled"]["all"]
                e = {"auc_all": s3["roc_auc"], "auc_all_ci": s3["roc_auc_ci"]}
        else:
            e = R["R4_cross_repertoire"][dirs[gi - 1]]["pooled"][model]
        xs.append(gi + (j - 1.5) * 0.17); ys.append(e["auc_all"])
        lo.append(e["auc_all"] - e["auc_all_ci"][0]); hi.append(e["auc_all_ci"][1] - e["auc_all"])
    ax.vlines(xs, np.array(ys) - np.array(lo), np.array(ys) + np.array(hi), colors=col,
              linestyles="solid", lw=1.3, zorder=2)
    ax.plot(xs, ys, mk, color=col, ms=5, markeredgecolor="white", markeredgewidth=0.6,
            label=lab, zorder=3)
ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=8)
ax.set_ylim(0.45, 1.02); ax.set_ylabel("Pooled ROC AUC (95% CI)")
ax.set_title("C  Repertoire transfer", loc="left", fontsize=9, fontweight="bold", color=INK)
ax.grid(axis="y", color=GRID, lw=0.6)

h1, l1 = axes[0].get_legend_handles_labels()
h3, l3 = axes[2].get_legend_handles_labels()
handles = h1 + [h3[l3.index(S["content_tfidf_lr"][0])]]
labels = l1 + [S["content_tfidf_lr"][0]]
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, -0.01), handlelength=2.4, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.2)
png = os.path.join(FIG, "stage4_fig_dilution_transfer.png")
fig.savefig(png, dpi=300)
Image.open(png).convert("RGB").save(png.replace(".png", ".tif"), dpi=(300, 300), compression="tiff_lzw")
print("wrote", png)
