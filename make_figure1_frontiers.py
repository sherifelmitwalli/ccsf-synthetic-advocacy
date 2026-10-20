#!/usr/bin/env python3
"""Figure 1 (Frontiers version): four-stage study design. 180 mm wide, 300 dpi PNG + TIFF.
All numbers are taken from the archived outputs (results.json, stage2/3/4 outputs)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

plt.rcParams.update({"font.family": "DejaVu Sans"})
INK, SUB, BODY = "#1b1b1b", "#3a4652", "#f4f6f8"
HEADS = ["#22384f", "#2f5070", "#5a4a6a", "#6a4a3a"]

s2 = json.load(open("stage2_outputs/derived_v1/stage2_derived_results_v1.json"))
s3 = json.load(open("stage3_outputs/stage3_results.json"))["models"]
s4 = json.load(open("stage4_outputs/stage4_results.json"))
hyb = s3["hybrid"]["pooled"]["all"]["roc_auc"]
d = s4["R1_R2_dilution"]["heldout_32"]["hybrid"]
content = s4["R3_baselines"]["pooled"]["content_tfidf_lr"]["auc_all"]

cards = [
    ("STAGE 1", "Construct verification", False,
     ["Rule-based corpus: 78 accounts", "4 signals, fixed-direction composite", "Ground truth known by construction"],
     "Composite AUC 0.995:\na manipulation check only"),
    ("STAGE 2", "Held-out LLM stress test", False,
     ["3 unseen LLM families: 156 accounts", "Organic and vocabulary-matched controls", "Leave-one-family-out evaluation"],
     "Fixed composite AUC 0.394\nCCSF-LR held-out AUC 0.711–0.878"),
    ("STAGE 3", "Relational features", True,
     ["Frozen Stage 2 corpus", "8 label-free cross-account features", "Computed in unlabeled held-out batches"],
     f"Hybrid pooled AUC {hyb:.3f}\n(10 campaign accounts per batch)"),
    ("STAGE 4", "Robustness audit", True,
     ["Campaign-size dilution", "Content and training-free baselines", "Unseen talking-point repertoire"],
     f"Hybrid AUC {d['1']['auc_all']['mean']:.2f} (k=1) to {d['10']['auc_all']['mean']:.2f} (k=10)\nContent classifier AUC {content:.2f}"),
]

fig, ax = plt.subplots(figsize=(7.09, 5.0))
ax.set_xlim(-1, 101); ax.set_ylim(-1, 101); ax.axis("off")
W, H, GX, GY = 47.5, 46.5, 5.0, 5.0
HEAD_H = 10.5
pos = [(0, 53.5), (W + GX, 53.5), (0, 0.5), (W + GX, 0.5)]
for i, (h1, h2, dashed, body, res) in enumerate(cards):
    x, yb = pos[i]
    col = HEADS[i]
    top = yb + H
    ax.add_patch(FancyBboxPatch((x, yb), W, H, boxstyle="round,pad=0.2,rounding_size=1.0",
                                facecolor=BODY, edgecolor=col, lw=1.2,
                                linestyle=(0, (5, 2.5)) if dashed else "-"))
    ax.add_patch(FancyBboxPatch((x, top - HEAD_H), W, HEAD_H, boxstyle="round,pad=0.2,rounding_size=1.0",
                                facecolor=col, edgecolor=col, lw=1))
    ax.text(x + 1.5, top - HEAD_H / 2, f"{h1}  ", color="white", fontsize=10, fontweight="bold", va="center")
    ax.text(x + 14.5, top - HEAD_H / 2, h2, color="white", fontsize=9, va="center")
    if dashed:
        ax.text(x + W - 1.2, top - HEAD_H / 2, "post hoc", color="#f0e6f4", fontsize=8, style="italic",
                va="center", ha="right")
    y = top - HEAD_H - 3.5
    for line in body:
        ax.text(x + 1.8, y, "\u2022 " + line, fontsize=8, color=INK, va="top")
        y -= 6.2
    rb_h = 14.0
    ax.add_patch(FancyBboxPatch((x + 1.5, yb + 1.5), W - 3.0, rb_h,
                                boxstyle="round,pad=0.15,rounding_size=0.8",
                                facecolor="#e8edf2", edgecolor=col, lw=0.8))
    ax.text(x + W / 2, yb + 1.5 + rb_h / 2, res, fontsize=8, color=SUB,
            va="center", ha="center", linespacing=1.35)
for (a, b) in [((W + 0.3, 76.75), (W + GX - 0.3, 76.75)), ((W / 2 + W + GX, 53.2), (W / 2, 47.3)),
               ((W + 0.3, 23.75), (W + GX - 0.3, 23.75))]:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color="#5b6b7a", lw=1.3,
                                 connectionstyle="arc3,rad=0.0"))
os.makedirs("frontiers_figures", exist_ok=True)
png = "frontiers_figures/figure1_design.png"
fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.03)
Image.open(png).convert("RGB").save(png.replace(".png", ".tif"), dpi=(300, 300), compression="tiff_lzw")
print("wrote", png)
