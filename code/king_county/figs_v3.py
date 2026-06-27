"""figs_v3.py - cluster pipeline + regime-shift slopegraph (real audit numbers)."""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
OUT = "/home/claude/ictai/figs/"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})

# ---------------- cluster pipeline (no PROCESS 1/2/3) ----------------
def pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.9)); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 62)

    def cluster(x, y, w, h, title, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                     fc=color, ec="#36506a", lw=1.4, alpha=0.30))
        ax.text(x + w/2, y + h - 2.6, title, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color="#16222e")

    def sub(x, y, w, h, lines, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.4",
                     fc=color, ec="#46627c", lw=0.9))
        for i, ln in enumerate(lines):
            ax.text(x + w/2, y + h - 3.2 - i*3.4, ln, ha="center", va="center",
                    fontsize=6.0, color="#16222e")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=13, lw=1.6, color="#2c4considerate"[:7] if False else "#33485c"))

    # Cluster A: Data foundation
    cluster(1, 34, 30, 26, "Data foundation", "#bcd4ef")
    sub(3, 49, 26, 8.5, ["King County, WA  (553,655 sales,", "318,981 properties, 1999-2025)"], "#e8f1fb")
    sub(3, 36, 26, 11, ["clean + engineer 57 features;", "property-disjoint temporal split;", "zero repeat-sale leakage"], "#e8f1fb")
    # Cluster B: Deployed model
    cluster(35, 34, 28, 26, "Deployed valuation model", "#c5e6d2")
    sub(37, 49, 24, 8.5, ["LightGBM + XGBoost + CatBoost"], "#e6f6ec")
    sub(37, 36, 24, 11, ["SLSQP convex blend", "0.67 / 0.22 / 0.11", "deployed f,  R\u00b2 = 0.826"], "#e6f6ec")
    # Cluster C: Explanation methods
    cluster(67, 34, 32, 26, "Explanation methods on f", "#fbe2c6")
    sub(69, 49, 28, 8.5, ["KernelSHAP, Shapley sampling,"], "#fdf0dd")
    sub(69, 36, 28, 11, ["LIME, occlusion, TreeSHAP;", "attributions explain the same", "deployed function f(x)"], "#fdf0dd")
    # arrows across top row
    arrow(31, 47, 35, 47); arrow(63, 47, 67, 47)
    # down to bottom row
    arrow(83, 34, 83, 28)
    # Cluster D: dual-regime faithfulness audit (center-bottom, wide)
    cluster(34, 3, 65, 25, "Dual-regime faithfulness audit", "#e3d4f2")
    sub(36, 16, 30, 9.5, ["MARGINAL perturbation (OOD):", "features -> background mean"], "#f0e7f9")
    sub(68, 16, 29, 9.5, ["ON-MANIFOLD perturbation:", "conditional donor replacement"], "#f0e7f9")
    sub(36, 4.5, 61, 9.5, ["six axes: deletion, insertion, comprehensiveness, infidelity, stability, cost",
                            "=> rank-based axes flip across regimes; infidelity is stable; TreeSHAP unfaithful in both"], "#f0e7f9")
    # Cluster E: output (left-bottom)
    cluster(1, 3, 31, 25, "Trustworthy output", "#f5cfcf")
    sub(3, 14, 27, 11.5, ["audit-validity warning;", "infidelity-grounded selection;", "regime-robust consensus (FWC)"], "#fbe4e4")
    sub(3, 4.5, 27, 8.0, ["served via Flask API + chatbot"], "#fbe4e4")
    arrow(34, 15, 32, 15)
    plt.tight_layout(pad=0.2)
    plt.savefig(OUT + "fig_pipeline_v3.pdf", bbox_inches="tight"); plt.close()
    print("pipeline_v3 saved")

# ---------------- regime-shift slopegraph (real numbers) ----------------
def regime_shift():
    methods = ["kernelshap", "samplingshap", "lime", "occlusion", "treeshap", "consensus"]
    disp = {"kernelshap": "KernelSHAP", "samplingshap": "Shapley samp.", "lime": "LIME",
            "occlusion": "Occlusion", "treeshap": "TreeSHAP", "consensus": "Consensus"}
    deletion = {"marginal": {"kernelshap":0.085,"samplingshap":0.055,"lime":0.031,"occlusion":0.041,"treeshap":0.090,"consensus":-0.003},
                "onman":    {"kernelshap":0.112,"samplingshap":0.014,"lime":0.006,"occlusion":0.287,"treeshap":0.012,"consensus":0.077}}
    infid = {"marginal": {"kernelshap":0.937,"samplingshap":0.904,"lime":1.461,"occlusion":1.906,"treeshap":8.318,"consensus":0.987},
             "onman":    {"kernelshap":1.137,"samplingshap":1.249,"lime":1.579,"occlusion":2.265,"treeshap":8.383,"consensus":1.323}}
    def ranks(dic):
        order = sorted(methods, key=lambda m: dic[m])  # lower=better
        return {m: order.index(m)+1 for m in methods}
    colors = {"kernelshap":"#1f77b4","samplingshap":"#2ca02c","lime":"#ff7f0e",
              "occlusion":"#d62728","treeshap":"#9467bd","consensus":"#111111"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for ax, (data, name) in zip(axes, [(deletion, "Deletion rank (rank-based axis)"),
                                        (infid, "Infidelity rank (calibrated axis)")]):
        rM, rO = ranks(data["marginal"]), ranks(data["onman"])
        for m in methods:
            lw = 3.2 if m in ("occlusion", "treeshap", "consensus") else 1.6
            ax.plot([0, 1], [rM[m], rO[m]], "-o", color=colors[m], lw=lw, ms=6,
                    alpha=0.95 if lw > 2 else 0.6)
            ax.text(-0.04, rM[m], disp[m], ha="right", va="center", fontsize=6.6, color=colors[m])
            ax.text(1.04, rO[m], disp[m], ha="left", va="center", fontsize=6.6, color=colors[m])
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(6.6, 0.4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["marginal\n(OOD)", "on-manifold"], fontsize=7.5)
        ax.set_yticks(range(1, 7)); ax.set_ylabel("rank (1 = most faithful)", fontsize=7.5)
        ax.set_title(name, fontsize=8.5, fontweight="bold")
        ax.tick_params(labelsize=7); ax.grid(axis="y", ls=":", alpha=0.4)
    axes[0].text(0.5, 6.45, "rankings reshuffle: occlusion 3->6, TreeSHAP 6->2",
                 ha="center", fontsize=6.4, color="#b00", style="italic")
    axes[1].text(0.5, 6.45, "nearly stable: only the top two swap",
                 ha="center", fontsize=6.4, color="#060", style="italic")
    plt.tight_layout(pad=0.4)
    plt.savefig(OUT + "fig_regime_shift.pdf", bbox_inches="tight"); plt.close()
    print("regime_shift saved")

pipeline(); regime_shift(); print("done")
