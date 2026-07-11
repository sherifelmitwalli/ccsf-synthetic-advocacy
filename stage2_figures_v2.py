"""Revised Stage 2 manuscript and appendix figures (v2).

Reads only frozen Stage 2 outputs and the versioned derivative results
(stage2_outputs/derived_v1/stage2_derived_results_v1.json). Writes to
stage2_outputs/derived_v1/figures/. No original output is modified.

Manuscript figures
  fig1_v2_two_stage_design.png   - two-stage design schematic (updated wording)
  fig3_v2_stage2_forest.png      - per-family CCSF-LR vs fixed-direction CCSF,
                                   both with stratified bootstrap 95% CIs
  fig4_v2_transformations.png    - transformation robustness with EQUIVALENT
                                   groups (5 coordinated vs 5 organic per cell;
                                   professional controls excluded), per family
                                   and pooled (with CI)

Multimedia appendix figures
  ma3_coefficients.png           - standardized CCSF-LR coefficients per split
  ma4_calibration.png            - reliability curves + Brier per held-out family
  ma5_originality_stage2.png     - campaign-level lexical vs semantic similarity,
                                   Stage 2 groups with Stage 1 reference points
  ma6_secondary_holdouts.png     - leave-one-prompt-family/topic/transformation-out

Run: python stage2_figures_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
S2 = HERE / "stage2_outputs"
DER = S2 / "derived_v1"
FIG = DER / "figures"
DPI = 300

FAM_SHORT = {"alibaba_qwen": "Qwen3.6-35B-A3B", "google_gemini": "Gemini 3.1 Flash Lite",
             "bytedance_seed": "Seed-2.0-Mini"}
GEN_TO_FAM = {"qwen_3_6_35b": "alibaba_qwen", "gemini_flash_lite": "google_gemini",
              "seed_2_mini": "bytedance_seed"}
FAM_TO_GEN = {v: k for k, v in GEN_TO_FAM.items()}
BLUE, RED, GREY = "#2b5d8a", "#a33c3c", "#7a7a7a"


def _box(ax, x, y, w, h, text, fc, fontsize=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=fc, edgecolor="#444444", linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color="#444444", linewidth=1.1))


def fig1(d):
    fc = d["fixed_composite"]
    lo = min(s["roc_auc"] for s in _logo_splits())
    hi = max(s["roc_auc"] for s in _logo_splits())
    fig, ax = plt.subplots(figsize=(2548 / DPI, 1208 / DPI), dpi=DPI)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    c_corpus, c_feat, c_score, c_res = "#dbe8f6", "#e8e2f4", "#fdeeda", "#e3f1e3"
    ax.text(0.245, 0.965, "Stage 1: construct verification", ha="center",
            fontsize=8, fontweight="bold")
    ax.text(0.7475, 0.965, "Stage 2: held-out multigenerator stress test (preliminary)",
            ha="center", fontsize=8, fontweight="bold")
    ax.axvline(0.497, color="#999999", linewidth=0.9, linestyle=(0, (4, 3)), ymin=0.02, ymax=0.94)
    _box(ax, 0.03, 0.74, 0.43, 0.15,
         "Seeded rule-based simulation corpus\n78 accounts, 524 posts; generator deliberately\n"
         "instantiates the measured regularities", c_corpus, fontsize=7.5)
    _box(ax, 0.03, 0.52, 0.43, 0.15,
         "4 fixed account-level signals\nperplexity | burstiness | framing density | convergence",
         c_feat, fontsize=7.5)
    _box(ax, 0.03, 0.30, 0.43, 0.15,
         "Fixed-direction equal-weight composite\n(no training)", c_score, fontsize=7.5)
    _box(ax, 0.03, 0.06, 0.43, 0.17,
         "Manipulation check (positive control):\nnear-ceiling separation expected by construction\n"
         "AUC 0.995 - not a performance estimate", c_res, fontsize=7.5)
    for y1, y2 in ((0.74, 0.67), (0.52, 0.45), (0.30, 0.23)):
        _arrow(ax, 0.245, y1, 0.245, y2)
    _box(ax, 0.52, 0.74, 0.455, 0.15,
         "3 unseen LLM generator families (dated revisions)\n"
         "Gemini 3.1 Flash Lite | Qwen3.6-35B-A3B | Seed-2.0-Mini\n"
         "20 coordinated + 20 organic-style\n+ 12 vocabulary-matched professional accounts per family",
         c_corpus, fontsize=6.8)
    _box(ax, 0.52, 0.55, 0.455, 0.12,
         "Automated transformations (account level, no human editing)\n"
         "LLM paraphrase (4th family) | seeded noise | restructuring", c_feat, fontsize=6.8)
    _box(ax, 0.52, 0.36, 0.455, 0.12,
         "Same 4 fixed signals, same instruments as stage 1", c_feat, fontsize=7.5)
    _box(ax, 0.52, 0.145, 0.215, 0.145,
         "Analysis A\nfixed-direction composite,\napplied without retraining", c_score, fontsize=6.8)
    _box(ax, 0.76, 0.145, 0.215, 0.145,
         "Analysis B: CCSF-LR\nlogistic reweighting trained\nonly on the 2\ndevelopment families",
         c_score, fontsize=6.8)
    _box(ax, 0.52, 0.01, 0.215, 0.10,
         f"Did not transfer\npooled AUC {fc['pooled']['roc_auc']:.3f}\n"
         f"(95% CI {fc['pooled']['roc_auc_ci'][0]:.3f}-{fc['pooled']['roc_auc_ci'][1]:.3f})",
         "#f6dddd", fontsize=6.6, bold=True)
    _box(ax, 0.76, 0.01, 0.215, 0.10,
         f"Held-out-family ROC AUC\n{lo:.3f}-{hi:.3f} (preliminary)", c_res, fontsize=6.8, bold=True)
    _arrow(ax, 0.7475, 0.74, 0.7475, 0.67)
    _arrow(ax, 0.7475, 0.55, 0.7475, 0.48)
    _arrow(ax, 0.63, 0.36, 0.63, 0.29)
    _arrow(ax, 0.865, 0.36, 0.865, 0.29)
    _arrow(ax, 0.6275, 0.145, 0.6275, 0.11)
    _arrow(ax, 0.8675, 0.145, 0.8675, 0.11)
    fig.savefig(FIG / "fig1_v2_two_stage_design.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def _logo_splits():
    frozen = json.loads((S2 / "stage2_eval_leave_one_generator_id_out.json").read_text())
    return frozen["splits"]


def fig3(d):
    splits = {s["held_out_value"]: s for s in _logo_splits()}
    fc = d["fixed_composite"]
    order = ["qwen_3_6_35b", "gemini_flash_lite", "seed_2_mini"]
    fig, ax = plt.subplots(figsize=(2121 / DPI, 1350 / DPI), dpi=DPI)
    ys = [5.0, 4.0, 3.0]
    for y, g in zip(ys, order):
        s = splits[g]
        fam = GEN_TO_FAM[g]
        f = fc["per_family"][fam]
        # CCSF-LR
        ax.plot(s["roc_auc_ci"], [y, y], color=BLUE, linewidth=2)
        for xci in s["roc_auc_ci"]:
            ax.plot([xci, xci], [y - 0.09, y + 0.09], color=BLUE, linewidth=2)
        ax.plot(s["roc_auc"], y, "o", color=BLUE, markersize=8, zorder=5)
        ax.text(1.015, y, f"{s['roc_auc']:.3f} ({s['roc_auc_ci'][0]:.3f}-{s['roc_auc_ci'][1]:.3f})",
                va="center", fontsize=7.2, color=BLUE)
        # fixed composite same family
        yf = y - 0.38
        ax.plot(f["roc_auc_ci"], [yf, yf], color=RED, linewidth=1.6, alpha=0.9)
        for xci in f["roc_auc_ci"]:
            ax.plot([xci, xci], [yf - 0.07, yf + 0.07], color=RED, linewidth=1.6)
        ax.plot(f["roc_auc"], yf, "X", color=RED, markersize=7, zorder=5)
        ax.text(1.015, yf, f"{f['roc_auc']:.3f} ({f['roc_auc_ci'][0]:.3f}-{f['roc_auc_ci'][1]:.3f})",
                va="center", fontsize=7.2, color=RED)
        # PR AUC
        ax.plot(s["pr_auc"], y + 0.30, "s", color=GREY, markersize=5, zorder=5)
        ax.text(1.015, y + 0.30, f"PR {s['pr_auc']:.3f}", va="center", fontsize=6.6, color="#555555")
    mean_auc = sum(splits[g]["roc_auc"] for g in order) / 3
    ax.plot(mean_auc, 1.9, "D", color=BLUE, markersize=7, fillstyle="none", markeredgewidth=1.6)
    ax.text(1.015, 1.9, f"{mean_auc:.3f} (unweighted mean)", va="center", fontsize=7.2)
    pooled = fc["pooled"]
    ax.plot(pooled["roc_auc_ci"], [1.1, 1.1], color=RED, linewidth=1.6)
    for xci in pooled["roc_auc_ci"]:
        ax.plot([xci, xci], [1.03, 1.17], color=RED, linewidth=1.6)
    ax.plot(pooled["roc_auc"], 1.1, "X", color=RED, markersize=8, zorder=5)
    ax.text(1.015, 1.1,
            f"{pooled['roc_auc']:.3f} ({pooled['roc_auc_ci'][0]:.3f}-{pooled['roc_auc_ci'][1]:.3f})",
            va="center", fontsize=7.2, color=RED)
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1)
    ax.text(0.503, 5.62, "chance", fontsize=7.2, color="#666666")
    ax.set_yticks(ys + [1.9, 1.1])
    ax.set_yticklabels([FAM_SHORT[GEN_TO_FAM[g]] for g in order] +
                       ["CCSF-LR mean\n(unweighted)", "Fixed-direction CCSF\n(pooled, 156 accounts)"],
                       fontsize=7.6)
    ax.set_xlim(0.15, 1.0); ax.set_ylim(0.5, 5.9)
    ax.set_xlabel("ROC AUC in the held-out generator family", fontsize=9)
    ax.scatter([], [], marker="o", color=BLUE, label="CCSF-LR ROC AUC (95% CI)")
    ax.scatter([], [], marker="X", color=RED, label="Fixed-direction CCSF ROC AUC (95% CI)")
    ax.scatter([], [], marker="s", color=GREY, label="CCSF-LR PR AUC (point estimate)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.995), fontsize=6.6, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.77, top=0.96, bottom=0.14)
    fig.savefig(FIG / "fig3_v2_stage2_forest.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def fig4(d):
    te = d["transformation_equivalent_groups"]
    order_t = ["none", "llm_paraphrase", "noise_insertion", "sentence_restructure"]
    labels_t = ["untransformed", "LLM paraphrase", "typographic noise", "sentence\nrestructuring"]
    fams = ["alibaba_qwen", "google_gemini", "bytedance_seed"]
    colors = {"alibaba_qwen": BLUE, "google_gemini": "#4d8a4d", "bytedance_seed": "#a3703c"}
    markers = {"alibaba_qwen": "o", "google_gemini": "s", "bytedance_seed": "^"}
    fig, ax = plt.subplots(figsize=(1650 / DPI, 1300 / DPI), dpi=DPI)
    for j, fam in enumerate(fams):
        xs = [i + (j - 1) * 0.15 for i in range(4)]
        aucs = [te["per_family"][fam][t]["roc_auc"] for t in order_t]
        ax.scatter(xs, aucs, marker=markers[fam], color=colors[fam], s=34, zorder=5,
                   label=FAM_SHORT[fam])
    for i, t in enumerate(order_t):
        cell = te["pooled"][t]
        ax.plot([i + 0.32, i + 0.32], cell["roc_auc_ci"], color="black", linewidth=1.6)
        for ci in cell["roc_auc_ci"]:
            ax.plot([i + 0.27, i + 0.37], [ci, ci], color="black", linewidth=1.6)
        ax.plot(i + 0.32, cell["roc_auc"], "D", color="black", markersize=6, zorder=6)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
    ax.text(3.55, 0.512, "chance", fontsize=6.5, color="#666666", ha="right")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{lab}\n(5 vs 5 per family)" for lab in labels_t], fontsize=6.6)
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("ROC AUC, coordinated vs organic-style\n(held-out-family predictions)",
                  fontsize=7.5)
    handles, labels = ax.get_legend_handles_labels()
    import matplotlib.lines as mlines
    handles.append(mlines.Line2D([], [], color="black", marker="D", linestyle="-",
                                 markersize=5, label="pooled 15 vs 15 (95% CI)"))
    ax.legend(handles=handles, fontsize=6.2, frameon=False, loc="lower right",
              title="Held-out family", title_fontsize=6.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.15)
    fig.savefig(FIG / "fig4_v2_transformations.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def ma3(d):
    co = d["logo_coefficients"]["splits"]
    feats = ["ppl_mean", "burstiness", "commercial_policy_framing", "convergence"]
    feat_labels = {"ppl_mean": "Perplexity (mean)", "burstiness": "Burstiness (CV)",
                   "commercial_policy_framing": "Framing density", "convergence": "Convergence"}
    stage1_dir = {"ppl_mean": -1, "burstiness": -1, "commercial_policy_framing": 1,
                  "convergence": 1}
    order = ["qwen_3_6_35b", "gemini_flash_lite", "seed_2_mini"]
    colors = {"qwen_3_6_35b": BLUE, "gemini_flash_lite": "#4d8a4d", "seed_2_mini": "#a3703c"}
    fig, ax = plt.subplots(figsize=(2000 / DPI, 1300 / DPI), dpi=DPI)
    for fi, f in enumerate(feats):
        base = len(feats) - fi
        for j, g in enumerate(order):
            e = co[g]
            y = base + (1 - j) * 0.18
            ci = e["bootstrap_ci"][f]
            ax.plot(ci, [y, y], color=colors[g], linewidth=1.6)
            ax.plot(e["coefficients"][f], y, "o", color=colors[g], markersize=5, zorder=5)
        ax.annotate("", xy=(0.55 * stage1_dir[f], base + 0.34),
                    xytext=(0.05 * stage1_dir[f], base + 0.34),
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=1.1))
        ax.text(0.30 * stage1_dir[f], base + 0.40, "stage 1 hypothesis", fontsize=5.8,
                color="#888888", ha="center")
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_yticks([len(feats) - i for i in range(len(feats))])
    ax.set_yticklabels([feat_labels[f] for f in feats], fontsize=8)
    ax.set_xlabel("Standardized logistic coefficient (per development-set SD),\n"
                  "positive = higher value predicts coordination", fontsize=7.5)
    for g in order:
        ax.plot([], [], "o-", color=colors[g], label=f"held out: {FAM_SHORT[GEN_TO_FAM[g]]}")
    ax.legend(fontsize=6.4, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.96, bottom=0.20)
    fig.savefig(FIG / "ma3_coefficients.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def ma4(d):
    cal = d["calibration"]["splits"]
    order = ["qwen_3_6_35b", "gemini_flash_lite", "seed_2_mini"]
    colors = {"qwen_3_6_35b": BLUE, "gemini_flash_lite": "#4d8a4d", "seed_2_mini": "#a3703c"}
    fig, ax = plt.subplots(figsize=(1600 / DPI, 1450 / DPI), dpi=DPI)
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1, label="perfect calibration")
    for g in order:
        bins = cal[g]["calibration_bins"]
        ax.plot([b["mean_predicted"] for b in bins], [b["fraction_positive"] for b in bins],
                "o-", color=colors[g], markersize=4.5, linewidth=1.3,
                label=f"{FAM_SHORT[GEN_TO_FAM[g]]} (Brier {cal[g]['brier_score']:.3f})")
    ax.set_xlabel("Mean predicted probability of coordination (quantile bins)", fontsize=8)
    ax.set_ylabel("Observed fraction coordinated", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=6.6, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.13)
    fig.savefig(FIG / "ma4_calibration.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def ma5(d):
    co = d["coordinated_originality_stage2"]["groups"]
    s1 = json.loads((HERE / "results.json").read_text())["nonduplication"]
    fig, ax = plt.subplots(figsize=(1750 / DPI, 1400 / DPI), dpi=DPI)
    fam_color = {"alibaba_qwen": BLUE, "google_gemini": "#4d8a4d", "bytedance_seed": "#a3703c"}
    for key, v in co.items():
        fam, grp = key.split("::")
        c = fam_color[fam]
        if grp == "organic":
            ax.plot(v["mean_jaccard"], v["mean_embed_cosine"], "s", color=c, markersize=7,
                    fillstyle="none", markeredgewidth=1.6)
        elif grp == "professional":
            ax.plot(v["mean_jaccard"], v["mean_embed_cosine"], "^", color=c, markersize=8,
                    fillstyle="none", markeredgewidth=1.6)
        else:
            ax.plot(v["mean_jaccard"], v["mean_embed_cosine"], "o", color=c, markersize=7)
    s1_style = {"synthetic": ("o", "Stage 1 coordinated"), "organic": ("s", "Stage 1 organic"),
                "professional": ("^", "Stage 1 professional")}
    for grp, (mk, _) in s1_style.items():
        v = s1[grp]
        ax.plot(v["mean_jaccard"], v["mean_embed_cosine"], mk, color="#aaaaaa", markersize=7,
                alpha=0.9, zorder=2)
        ax.annotate(_, (v["mean_jaccard"], v["mean_embed_cosine"]),
                    textcoords="offset points", xytext=(6, -3), fontsize=6, color="#888888")
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="", color="#444444",
                      label="Stage 2 coordinated campaign"),
        mlines.Line2D([], [], marker="s", linestyle="", color="#444444", fillstyle="none",
                      label="Stage 2 organic-style control"),
        mlines.Line2D([], [], marker="^", linestyle="", color="#444444", fillstyle="none",
                      label="Stage 2 professional control"),
        mlines.Line2D([], [], marker="o", linestyle="", color="#aaaaaa",
                      label="Stage 1 reference (grey)")]
    for fam, c in fam_color.items():
        handles.append(mlines.Line2D([], [], marker="o", linestyle="", color=c,
                                     label=FAM_SHORT[fam]))
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper left", ncol=1)
    ax.set_xlabel("Mean pairwise lexical overlap (Jaccard) within group", fontsize=8)
    ax.set_ylabel("Mean pairwise semantic similarity (cosine) within group", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.13)
    fig.savefig(FIG / "ma5_originality_stage2.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def ma6():
    panels = [("prompt_family", "Held-out prompt family"),
              ("topic", "Held-out topic"),
              ("transformation_id", "Held-out transformation")]
    short = {"advertising and promotion rules for vaping products": "advertising and promotion rules",
             "enforcement of youth access laws for vaping products": "youth access law enforcement",
             "flavour restrictions for nicotine vaping products": "flavour restrictions",
             "product safety and manufacturing standards for vaping devices": "product safety standards",
             "retail licensing for nicotine product sellers": "retail licensing",
             "taxation of alternative nicotine products": "taxation",
             "the role of vaping products in smoking cessation services": "role in smoking cessation",
             "use of vaping products in public places": "use in public places",
             "llm_paraphrase": "LLM paraphrase", "noise_insertion": "typographic noise",
             "none": "untransformed", "sentence_restructure": "sentence restructuring",
             "direct_talking_points": "direct talking points",
             "grassroots_persona": "grassroots persona",
             "indirect_policy_discussion": "indirect policy discussion"}
    heights = [3, 8, 4]
    fig, axes = plt.subplots(3, 1, figsize=(1900 / DPI, 2500 / DPI), dpi=DPI,
                             gridspec_kw={"height_ratios": heights})
    for ax, (field, title) in zip(axes, panels):
        data = json.loads((S2 / f"stage2_eval_leave_one_{field}_out.json").read_text())
        splits = data["splits"]
        ys = list(range(len(splits), 0, -1))
        for y, s in zip(ys, splits):
            ax.plot(s["roc_auc_ci"], [y, y], color=BLUE, linewidth=1.6)
            ax.plot(s["roc_auc"], y, "o", color=BLUE, markersize=5, zorder=5)
            ax.text(1.03, y, f"{s['roc_auc']:.3f}", va="center", fontsize=6.4)
        ax.axvline(0.5, color="#888888", linestyle="--", linewidth=0.9)
        ax.set_yticks(ys)
        ax.set_yticklabels([short.get(s["held_out_value"], s["held_out_value"]) for s in splits],
                           fontsize=6.8)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0.4, len(splits) + 0.6)
        ax.set_title(title, fontsize=8, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("ROC AUC (stratified bootstrap 95% CI)", fontsize=7.5)
    fig.subplots_adjust(left=0.30, right=0.88, top=0.97, bottom=0.06, hspace=0.45)
    fig.savefig(FIG / "ma6_secondary_holdouts.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = json.loads((DER / "stage2_derived_results_v1.json").read_text())
    fig1(d); fig3(d); fig4(d); ma3(d); ma4(d); ma5(d); ma6()
    for p in sorted(FIG.glob("*.png")):
        print(p.name)


if __name__ == "__main__":
    main()
