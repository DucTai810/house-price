"""fig_pipeline_v4.py - clean cluster pipeline: short text, bold arrows, clear flow."""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = "/home/claude/ictai/figs/"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})

fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 60)

INK = "#15222e"
def box(x, y, w, h, color, title, line, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=2.4",
                 fc=color, ec="#3a5570", lw=1.3))
    ax.text(x + w/2, y + h - 5.0, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=INK)
    ax.text(x + w/2, y + h - 12.0, line, ha="center", va="center", fontsize=9.5, color=INK)
    if sub:
        ax.text(x + w/2, y + h - 17.5, sub, ha="center", va="center", fontsize=9.5, color=INK)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=13, lw=1.6, color="#26384a",
                 shrinkA=2, shrinkB=2))

# top row, left to right
box(1.5, 38, 28, 20, "#cfe0f2", "Data foundation", "King County, WA", "leakage-free split")
box(36, 38, 28, 20, "#cfe9d6", "Deployed model", "LGBM + XGB + CatBoost", "blend,  R\u00b2 = 0.826")
box(70.5, 38, 28, 20, "#fbe3c6", "Explanation methods", "5 explainers on f", "SHAP, LIME, Occl, Tree")
arrow(29.5, 48, 36, 48)
arrow(64, 48, 70.5, 48)

# down from explanation methods to the audit
arrow(84.5, 38, 84.5, 31)

# bottom row: audit (right) -> output (left)
box(36, 9, 62.5, 22, "#e6d7f4", "", "", None)
ax.text(67.2, 27.2, "Dual-regime faithfulness audit", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=INK)
# two regime chips
ax.add_patch(FancyBboxPatch((39, 17), 27, 6.2, boxstyle="round,pad=0.3,rounding_size=1.4",
             fc="#f2ecfa", ec="#7a5ea8", lw=1.0))
ax.text(52.5, 20.1, "Marginal  (OOD)", ha="center", va="center", fontsize=9.5, color=INK)
ax.add_patch(FancyBboxPatch((68, 17), 27, 6.2, boxstyle="round,pad=0.3,rounding_size=1.4",
             fc="#f2ecfa", ec="#7a5ea8", lw=1.0))
ax.text(81.5, 20.1, "On-manifold  (donor)", ha="center", va="center", fontsize=9.5, color=INK)
ax.text(67.2, 12.4, "6 axes:  ranks flip,  infidelity stable", ha="center", va="center",
        fontsize=9.0, color=INK, style="italic")

box(1.5, 9, 28, 22, "#f6d2d2", "Trustworthy output", "regime-robust consensus", "API + chatbot")
arrow(36, 20, 29.5, 20)

plt.tight_layout(pad=0.2)
plt.savefig(OUT + "fig_pipeline_v4.pdf", bbox_inches="tight"); plt.close()
print("pipeline_v4 saved")
