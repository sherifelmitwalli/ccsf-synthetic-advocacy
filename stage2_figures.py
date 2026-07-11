"""Stage 2 manuscript figures, generated from stage2_outputs/stage2_results.json.

Produces (pixel dimensions match the manuscript slots they occupy):
  stage2_outputs/fig1_two_stage_design.png      (2548 x 1208 @ 300 dpi)
  stage2_outputs/fig3_stage2_forest.png         (2121 x 1239 @ 300 dpi)
  stage2_outputs/fig4_stage2_transformations.png(1538 x 1268 @ 300 dpi)

Run after `stage2_run.py evaluate`:  python stage2_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "stage2_outputs"
DPI = 300

FAMILY_LABELS = {
    "alibaba_qwen": "Alibaba Qwen\n(Qwen3.6-35B-A3B)",
    "google_gemini": "Google Gemini\n(Gemini 3.1 Flash Lite)",
    "bytedance_seed": "ByteDance Seed\n(Seed-2.0-Mini)",
}
GEN_TO_FAMILY = {"qwen_3_6_35b": "alibaba_qwen", "gemini_flash_lite": "google_gemini",
                 "seed_2_mini": "bytedance_seed"}
FIXED_COMPOSITE_POOLED_AUC = 0.394  # stage 1 fixed-direction composite, applied without retraining


def _box(ax, x, y, w, h, text, fc, fontsize=8.5, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=fc, edgecolor="#444444", linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color="#444444", linewidth=1.1))


def fig1_two_stage() -> None:
    fig, ax = plt.subplots(figsize=(2548 / DPI, 1208 / DPI), dpi=DPI)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    c_corpus, c_feat, c_score, c_res = "#dbe8f6", "#e8e2f4", "#fdeeda", "#e3f1e3"

    ax.text(0.245, 0.965, "Stage 1: construct verification", ha="center",
            fontsize=8, fontweight="bold")
    ax.text(0.7475, 0.965, "Stage 2: independent multigenerator validation (preliminary)",
            ha="center", fontsize=8, fontweight="bold")
    ax.axvline(0.497, color="#999999", linewidth=0.9, linestyle=(0, (4, 3)), ymin=0.02, ymax=0.94)

    # ---- stage 1 column
    _box(ax, 0.03, 0.74, 0.43, 0.15,
         "Seeded rule-based simulation corpus\n78 accounts, 524 posts; generator deliberately\ninstantiates the measured regularities", c_corpus, fontsize=7.5)
    _box(ax, 0.03, 0.52, 0.43, 0.15,
         "4 fixed account-level signals\nperplexity | burstiness | framing density | convergence", c_feat, fontsize=7.5)
    _box(ax, 0.03, 0.30, 0.43, 0.15,
         "Fixed-direction equal-weight composite\n(no training)", c_score, fontsize=7.5)
    _box(ax, 0.03, 0.06, 0.43, 0.17,
         "Manipulation check (positive control):\nnear-ceiling separation expected by construction\nAUC 0.995 - not a performance estimate", c_res, fontsize=7.5)
    for y1, y2 in ((0.74, 0.67), (0.52, 0.45), (0.30, 0.23)):
        _arrow(ax, 0.245, y1, 0.245, y2)

    # ---- stage 2 column
    _box(ax, 0.52, 0.74, 0.455, 0.15,
         "3 unseen LLM generator families (dated revisions)\nGemini 3.1 Flash Lite | Qwen3.6-35B-A3B | Seed-2.0-Mini\n20 coordinated + 20 organic-style\n+ 12 vocabulary-matched professional accounts per family", c_corpus, fontsize=6.8)
    _box(ax, 0.52, 0.55, 0.455, 0.12,
         "Automated transformations (account level, no human editing)\nLLM paraphrase (4th family) | seeded noise | restructuring", c_feat, fontsize=6.8)
    _box(ax, 0.52, 0.36, 0.455, 0.12,
         "Same 4 fixed signals, same instruments as stage 1", c_feat, fontsize=7.5)
    _box(ax, 0.52, 0.145, 0.215, 0.145,
         "Analysis A\nfixed-direction composite,\napplied without retraining", c_score, fontsize=6.8)
    _box(ax, 0.76, 0.145, 0.215, 0.145,
         "Analysis B: CCSF-LR\nlogistic reweighting trained\nonly on the 2\ndevelopment families", c_score, fontsize=6.8)
    _box(ax, 0.52, 0.01, 0.215, 0.10,
         "Did not transfer\npooled AUC 0.394", "#f6dddd", fontsize=7.2, bold=True)
    _box(ax, 0.76, 0.01, 0.215, 0.10,
         "Held-out-family AUC\n0.711-0.878 (preliminary)", c_res, fontsize=7.2, bold=True)
    _arrow(ax, 0.7475, 0.74, 0.7475, 0.67)
    _arrow(ax, 0.7475, 0.55, 0.7475, 0.48)
    _arrow(ax, 0.63, 0.36, 0.63, 0.29)
    _arrow(ax, 0.865, 0.36, 0.865, 0.29)
    _arrow(ax, 0.6275, 0.145, 0.6275, 0.11)
    _arrow(ax, 0.8675, 0.145, 0.8675, 0.11)

    fig.savefig(OUT / "fig1_two_stage_design.png", dpi=DPI, facecolor="white",
                bbox_inches=None)
    plt.close(fig)


def fig3_forest(results: dict) -> None:
    splits = {s["held_out_value"]: s for s in results["holdouts"]["generator_id"]["splits"]}
    order = ["qwen_3_6_35b", "gemini_flash_lite", "seed_2_mini"]
    short = {"qwen_3_6_35b": "Qwen3.6-35B-A3B", "gemini_flash_lite": "Gemini 3.1 Flash Lite",
             "seed_2_mini": "Seed-2.0-Mini"}
    rows = [(short[g], splits[g]["roc_auc"], splits[g]["roc_auc_ci"], splits[g]["pr_auc"]) for g in order]
    mean_auc = sum(r[1] for r in rows) / len(rows)

    fig, ax = plt.subplots(figsize=(2121 / DPI, 1239 / DPI), dpi=DPI)
    ys = [4, 3, 2]
    for y, (label, auc, ci, pr) in zip(ys, rows):
        ax.plot(ci, [y, y], color="#2b5d8a", linewidth=2)
        ax.plot([ci[0], ci[0]], [y - 0.09, y + 0.09], color="#2b5d8a", linewidth=2)
        ax.plot([ci[1], ci[1]], [y - 0.09, y + 0.09], color="#2b5d8a", linewidth=2)
        ax.plot(auc, y, "o", color="#2b5d8a", markersize=8, zorder=5)
        ax.plot(pr, y - 0.28, "s", color="#7a7a7a", markersize=6, zorder=5)
        ax.text(1.015, y, f"{auc:.3f} ({ci[0]:.3f}-{ci[1]:.3f})", va="center", fontsize=7.5)
        ax.text(1.015, y - 0.28, f"PR {pr:.3f}", va="center", fontsize=7, color="#555555")
    ax.plot(mean_auc, 1.2, "D", color="#2b5d8a", markersize=7, fillstyle="none", markeredgewidth=1.6)
    ax.text(1.015, 1.2, f"{mean_auc:.3f} (no CI calculated)", va="center", fontsize=7.5)
    ax.plot(FIXED_COMPOSITE_POOLED_AUC, 0.4, "X", color="#a33c3c", markersize=9)
    ax.text(1.015, 0.4, f"{FIXED_COMPOSITE_POOLED_AUC:.3f} (no CI)", va="center", fontsize=7.5)

    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1)
    ax.text(0.502, 4.62, "chance", fontsize=7.5, color="#666666")
    ax.set_yticks(ys + [1.2, 0.4])
    ax.set_yticklabels([r[0] for r in rows] +
                       ["CCSF-LR mean\n(unweighted)", "Fixed-direction CCSF\n(stage 1 weights, pooled)"], fontsize=8)
    ax.set_xlim(0.2, 1.0); ax.set_ylim(0, 5)
    ax.set_xlabel("Area under the curve (held-out generator family)", fontsize=9)
    ax.scatter([], [], marker="o", color="#2b5d8a", label="ROC AUC (95% CI)")
    ax.scatter([], [], marker="s", color="#7a7a7a", label="PR AUC (point estimate)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.35), fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.24, right=0.78, top=0.96, bottom=0.14)
    fig.savefig(OUT / "fig3_stage2_forest.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def fig4_transformations(results: dict) -> None:
    # n accounts per family-transformation cell, from the feature table
    with (OUT / "stage2_account_features.csv").open(encoding="utf-8", newline="") as handle:
        feats = list(csv.DictReader(handle))
    order_t = ["none", "llm_paraphrase", "noise_insertion", "sentence_restructure"]
    labels_t = ["untransformed", "LLM paraphrase", "typographic noise", "sentence restructuring"]
    fams = ["alibaba_qwen", "google_gemini", "bytedance_seed"]
    colors = {"alibaba_qwen": "#2b5d8a", "google_gemini": "#4d8a4d", "bytedance_seed": "#a3703c"}
    markers = {"alibaba_qwen": "o", "google_gemini": "s", "bytedance_seed": "^"}
    breakdown = {e["held_out_family"]: e["by_transformation"]
                 for e in results["leave_one_family_out_breakdown"]}

    fig, ax = plt.subplots(figsize=(1538 / DPI, 1268 / DPI), dpi=DPI)
    for j, fam in enumerate(fams):
        xs, aucs = [], []
        for i, t in enumerate(order_t):
            cell = breakdown[fam].get(t)
            if cell is None:
                continue
            xs.append(i + (j - 1) * 0.18)
            aucs.append(cell["roc_auc"])
        ax.plot(xs, aucs, marker=markers[fam], linestyle="-", linewidth=1,
                color=colors[fam], markersize=6,
                label=FAMILY_LABELS[fam].split("\n")[0].replace("Alibaba ", "").replace("Google ", "").replace("ByteDance ", ""))
    counts = {t: len({f["account_id"] for f in feats
                      if f["transformation_id"] == t and f["generator_family"] == fams[0]})
              for t in order_t}
    labels_t = [f"{lab}\n(n={counts[t]}/family)" for lab, t in zip(labels_t, order_t)]
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
    ax.text(3.28, 0.512, "chance", fontsize=6.5, color="#666666", ha="right")
    ax.set_xticks(range(len(order_t)))
    ax.set_xticklabels(labels_t, fontsize=6.8)
    ax.set_ylim(0.35, 1.05)
    ax.set_ylabel("ROC AUC within held-out family\n(CCSF-LR, exploratory point estimates)", fontsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", title="Held-out family", title_fontsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.16)
    fig.savefig(OUT / "fig4_stage2_transformations.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def main() -> None:
    results = json.loads((OUT / "stage2_results.json").read_text(encoding="utf-8"))
    fig1_two_stage()
    fig3_forest(results)
    fig4_transformations(results)
    from PIL import Image
    for f in ("fig1_two_stage_design.png", "fig3_stage2_forest.png", "fig4_stage2_transformations.png"):
        print(f, Image.open(OUT / f).size)


if __name__ == "__main__":
    main()
