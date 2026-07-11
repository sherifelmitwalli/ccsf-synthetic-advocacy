#!/usr/bin/env python3
"""
Stage 3 figures (exploratory, post hoc, synthetic). Grayscale-readable:
distinct marker shapes + hatching, not colour alone. Reads only Stage 3 JSON
outputs. Seed-free (deterministic layout). Writes to stage3_outputs/figures/.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "stage3_outputs/figures"; os.makedirs(OUT, exist_ok=True)
R = json.load(open("stage3_outputs/stage3_results.json"))
C = json.load(open("stage3_outputs/stage3_model_comparisons.json"))
M = R["models"]
FAMS = ["google_gemini", "alibaba_qwen", "bytedance_seed"]
FLAB = {"google_gemini": "Gemini", "alibaba_qwen": "Qwen", "bytedance_seed": "Seed"}
MODELS = ["fixed_ccsf", "ccsf_lr", "relational_only", "hybrid"]
MLAB = {"fixed_ccsf": "Fixed CCSF", "ccsf_lr": "CCSF-LR",
        "relational_only": "Relational-only", "hybrid": "Hybrid"}
MARK = {"fixed_ccsf": "o", "ccsf_lr": "s", "relational_only": "^", "hybrid": "D"}
GREY = {"fixed_ccsf": "0.0", "ccsf_lr": "0.35", "relational_only": "0.55", "hybrid": "0.15"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.dpi": 320, "font.family": "DejaVu Sans",
                     "axes.edgecolor": "0.3", "axes.linewidth": 0.8})


def fig1_forest():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True)
    for ax, endpoint, title in zip(axes, ["all", "organic_only"],
                                   ["A. Coordinated vs all controls",
                                    "B. Coordinated vs organic-style controls"]):
        yt = []; yl = []
        row = 0
        for fam in FAMS:
            for mi, mod in enumerate(MODELS):
                e = M[mod]["per_family"][endpoint][fam]
                y = row
                ax.errorbar(e["roc_auc"], y,
                            xerr=[[e["roc_auc"] - e["roc_auc_ci"][0]],
                                  [e["roc_auc_ci"][1] - e["roc_auc"]]],
                            fmt=MARK[mod], color=GREY[mod], mfc=GREY[mod],
                            ecolor=GREY[mod], capsize=2.5, ms=6, lw=1.3,
                            label=MLAB[mod] if (fam == FAMS[0]) else None)
                yt.append(y); yl.append(f"{FLAB[fam]} · {MLAB[mod]}")
                row += 1
            row += 1  # gap between families
        ax.axvline(0.5, ls="--", color="0.5", lw=1)
        ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.5)
        ax.set_xlim(0.1, 1.03); ax.invert_yaxis()
        ax.set_xlabel("Held-out ROC AUC (95% CI)")
        ax.set_title(title, fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage3_fig1_forest.png", bbox_inches="tight")
    plt.close(fig)


def fig2_paired():
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    rows = [("all", "relational_only_minus_ccsf_lr", "Relational − CCSF-LR (all)"),
            ("all", "hybrid_minus_ccsf_lr", "Hybrid − CCSF-LR (all)"),
            ("organic_only", "relational_only_minus_ccsf_lr", "Relational − CCSF-LR (organic)"),
            ("organic_only", "hybrid_minus_ccsf_lr", "Hybrid − CCSF-LR (organic)")]
    marks = ["^", "D", "^", "D"]
    for i, (ep, key, lab) in enumerate(rows):
        v = C[ep][key]
        ax.errorbar(v["delta_auc"], i,
                    xerr=[[v["delta_auc"] - v["ci"][0]], [v["ci"][1] - v["delta_auc"]]],
                    fmt=marks[i], color="0.15", capsize=3, ms=7, lw=1.5)
        ax.text(v["delta_auc"], i + 0.18, f"{v['delta_auc']:+.3f}", ha="center", fontsize=8)
    ax.axvline(0, ls="--", color="0.4", lw=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows])
    ax.invert_yaxis(); ax.set_xlabel("Paired bootstrap ΔROC AUC (pooled held-out, 95% CI)")
    ax.set_title("Stage 3: paired AUC differences vs CCSF-LR (▲ relational-only, ◆ hybrid)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage3_fig2_paired_delta.png", bbox_inches="tight")
    plt.close(fig)


def fig3_coefs():
    # mean coefficient across the 3 dev fits, with min/max whisker; hybrid model
    feats = list(R["coefficients"]["hybrid"]["google_gemini"].keys())
    means = []; los = []; his = []
    for f in feats:
        vals = [R["coefficients"]["hybrid"][h][f]["coef"] for h in FAMS]
        means.append(np.mean(vals)); los.append(min(vals)); his.append(max(vals))
    order = np.argsort(means)
    feats = [feats[i] for i in order]; means = np.array(means)[order]
    los = np.array(los)[order]; his = np.array(his)[order]
    ccsf = {"ppl_mean", "burstiness", "commercial_policy_framing", "convergence"}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, f in enumerate(feats):
        mk = "s" if f in ccsf else "^"
        ax.errorbar(means[i], i, xerr=[[means[i] - los[i]], [his[i] - means[i]]],
                    fmt=mk, color="0.15", capsize=3, ms=6.5)
    ax.axvline(0, ls="--", color="0.4", lw=1)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=8.5)
    ax.set_xlabel("Standardized logistic coefficient (mean of 3 dev fits; whisker = range)")
    ax.set_title("Stage 3 hybrid coefficients (■ account-level CCSF, ▲ relational)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage3_fig3_coefficients.png", bbox_inches="tight")
    plt.close(fig)


def fig4_calibration():
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1, label="Perfect calibration")
    for mod, mk in [("ccsf_lr", "s"), ("hybrid", "D")]:
        xs = []; ys = []
        for fam in FAMS:
            for b in M[mod]["per_family"]["all"][fam]["calibration_bins"]:
                xs.append(b["mean_predicted"]); ys.append(b["fraction_positive"])
        idx = np.argsort(xs)
        xs = np.array(xs)[idx]; ys = np.array(ys)[idx]
        # bin into deciles for readability
        ax.plot(xs, ys, mk + "-", color=GREY[mod], ms=5, lw=1.1, alpha=0.9,
                label=f"{MLAB[mod]} (Brier {np.mean([M[mod]['per_family']['all'][f]['brier'] for f in FAMS]):.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction coordinated")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Stage 3 calibration (pooled held-out, all controls)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage3_fig4_calibration.png", bbox_inches="tight")
    plt.close(fig)


def fig5_graph_illustration():
    # explanatory synthetic sketch only (clearly labelled), not evidence
    import networkx as nx
    rng = np.random.default_rng(20260604)
    G = nx.Graph()
    coord = [f"c{i}" for i in range(6)]; org = [f"o{i}" for i in range(8)]
    for i in range(len(coord)):
        for j in range(i + 1, len(coord)):
            if rng.random() < 0.8:
                G.add_edge(coord[i], coord[j])
    for a in org:
        if rng.random() < 0.5:
            G.add_edge(a, rng.choice(org))
    G.add_nodes_from(coord + org)
    pos = nx.spring_layout(G, seed=1, k=0.6)
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.6", width=1)
    nx.draw_networkx_nodes(G, pos, nodelist=coord, node_shape="D", node_color="0.15",
                           node_size=260, label="Coordinated (illustrative)", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=org, node_shape="o", node_color="0.8",
                           edgecolors="0.2", node_size=210, label="Organic-style (illustrative)", ax=ax)
    ax.legend(scatterpoints=1, fontsize=8, loc="lower left")
    ax.set_title("Illustrative mutual-kNN relational graph\n(synthetic sketch — explanatory only, NOT evidence)",
                 fontsize=9.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage3_fig5_graph_illustration.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_forest(); fig2_paired(); fig3_coefs(); fig4_calibration(); fig5_graph_illustration()
    print("wrote figures to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
