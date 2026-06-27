"""figs.py - publication figures for the multi-XAI faithfulness paper."""
import pickle, numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import helpers as H

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.labelsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
})
C = H.load_cache(); MF = C["MF"]
FIG = "figs/"

# palette
CB = {"kernelshap": "#1f77b4", "samplingshap": "#2ca02c", "lime": "#d62728",
      "occlusion": "#9467bd", "treeshap": "#8c564b", "consensus": "#ff7f0e"}
LABEL = {"kernelshap": "KernelSHAP", "samplingshap": "SamplingSHAP", "lime": "LIME",
         "occlusion": "Occlusion", "treeshap": "TreeSHAP", "consensus": "Consensus (ours)"}

DISP = {"area": "area", "year": "year", "log_sqft": "log_sqft",
        "submarket": "submarket", "grade_sqft": "grade×sqft",
        "log_total_living_sqft": "log_living_sqft", "quality_score": "quality_score",
        "area_submarket": "area×submarket", "view_lakewash": "view_lake",
        "year_sq": "year²", "stories": "stories", "grade": "grade",
        "premium_location_score": "premium_loc", "log_above_ground_sqft": "log_above_sqft",
        "noise_traffic": "noise_traffic", "wfnt": "waterfront", "log_age": "log_age"}
def disp(f): return DISP.get(f, f)

# ---------------------------------------------------------------- Fig 1: pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.1, 2.55)); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    stages = [
        ("INPUT", "#dbe9f6", [
            "553,655 transactions", "318,981 properties", "1999–2025, 57 features"]),
        ("PROCESS — Model", "#d8f0d8", [
            "Temporal property-", "disjoint split (0 leak)", "LGBM+XGB+CatBoost",
            "convex blend (SLSQP)"]),
        ("PROCESS — Explain", "#fde9d0", [
            "5 XAI methods on f", "KernelSHAP, Sampling,", "LIME, Occlusion, Tree",
            "Faithfulness audit"]),
        ("OUTPUT", "#f6d8d8", [
            "Best model R²=0.826", "Disagreement map", "Faithfulness-weighted",
            "Consensus explanation"]),
    ]
    xs = [2, 27, 52, 77]; w = 21
    for (title, color, lines), x in zip(stages, xs):
        box = FancyBboxPatch((x, 4), w, 32, boxstyle="round,pad=0.4,rounding_size=2",
                             fc=color, ec="#333333", lw=1.1)
        ax.add_patch(box)
        ax.text(x + w/2, 32, title, ha="center", va="center", fontsize=8.4, fontweight="bold")
        for i, ln in enumerate(lines):
            ax.text(x + w/2, 26 - i*5.0, ln, ha="center", va="center", fontsize=7.0)
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x + w + 0.5, 20), (x + w + 4.3, 20),
                     arrowstyle="-|>", mutation_scale=13, lw=1.6, color="#333333"))
    plt.tight_layout(pad=0.2); plt.savefig(FIG + "fig_pipeline.pdf", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 2: models
def fig_models():
    res = pd.read_csv("outputs/model_results.csv")
    res = res.sort_values("R2")
    with open("outputs/artifacts.pkl", "rb") as f: A = pickle.load(f)
    meta = pd.read_parquet("outputs/test_meta.parquet")
    yactual = meta["price"].to_numpy(); ypred = H.log_to_price(A["ens_test_log"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    short = {"Convex-Weighted Blend Ensemble": "Convex Ensemble",
             "Multiple Linear Regression": "Multiple Linear",
             "Simple Linear Regression": "Simple Linear",
             "HistGradientBoosting": "HistGB"}
    names = [short.get(m, m) for m in res["Model"]]
    cols = ["#ff7f0e" if m == "Convex-Weighted Blend Ensemble" else "#4C72B0" for m in res["Model"]]
    a1.barh(names, res["R2"], color=cols, edgecolor="#222", lw=0.5)
    a1.set_xlabel("Test $R^2$ (price scale)"); a1.set_title("(a) Model comparison")
    a1.set_xlim(-0.2, 0.95)
    for i, (r, rm) in enumerate(zip(res["R2"], res["RMSLE"])):
        a1.text(max(r, 0) + 0.01, i, f"{r:.2f}", va="center", fontsize=6.8)

    rng = np.random.RandomState(1); idx = rng.choice(len(yactual), 4000, replace=False)
    lim = np.percentile(yactual[idx], 99)
    a2.scatter(yactual[idx]/1e6, ypred[idx]/1e6, s=5, alpha=0.25, color="#ff7f0e", edgecolors="none")
    a2.plot([0, lim/1e6], [0, lim/1e6], "--", color="#333", lw=1)
    a2.set_xlim(0, lim/1e6); a2.set_ylim(0, lim/1e6)
    a2.set_xlabel("Actual (M USD)"); a2.set_ylabel("Predicted (M USD)")
    a2.set_title("(b) Ensemble: actual vs predicted")
    a2.text(0.05*lim/1e6, 0.9*lim/1e6, "$R^2$=0.826\nRMSLE=0.200", fontsize=7.5)
    plt.tight_layout(pad=0.4); plt.savefig(FIG + "fig_models.pdf", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 3: disagreement
def fig_disagree():
    gi = pd.read_csv("outputs/global_importance.csv", index_col=0)
    methods = ["kernelshap", "samplingshap", "lime", "occlusion", "treeshap"]
    top = gi[methods].apply(lambda c: c / c.sum())  # normalize each method
    order = top.mean(axis=1).sort_values(ascending=False).index[:12]
    M = top.loc[order, methods].T.values
    ag = np.load("outputs/agreement.npz", allow_pickle=True); S = ag["spearman"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [1.55, 1]})
    im = a1.imshow(M, aspect="auto", cmap="YlOrRd")
    a1.set_xticks(range(len(order))); a1.set_xticklabels([disp(f) for f in order], rotation=55, ha="right", fontsize=6.6)
    a1.set_yticks(range(len(methods))); a1.set_yticklabels([LABEL[m] for m in methods], fontsize=7.2)
    a1.set_title("(a) Normalized feature importance by method")
    fig.colorbar(im, ax=a1, fraction=0.025, pad=0.02)
    # mark top-1 per method
    for r in range(len(methods)):
        c = int(np.argmax(M[r])); a1.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False, ec="#111", lw=1.6))

    im2 = a2.imshow(S, cmap="RdYlGn", vmin=0.3, vmax=1)
    a2.set_xticks(range(len(methods))); a2.set_xticklabels([LABEL[m].split()[0] for m in methods], rotation=55, ha="right", fontsize=6.6)
    a2.set_yticks(range(len(methods))); a2.set_yticklabels([LABEL[m].split()[0] for m in methods], fontsize=6.6)
    for i in range(len(methods)):
        for j in range(len(methods)):
            a2.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center", fontsize=6.3,
                    color="black")
    a2.set_title("(b) Rank agreement (Spearman)")
    fig.colorbar(im2, ax=a2, fraction=0.045, pad=0.03)
    plt.tight_layout(pad=0.4); plt.savefig(FIG + "fig_disagree.pdf", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 4: radar
def fig_radar():
    au = pd.read_csv("outputs/final_audit_table.csv", index_col=0)
    metrics = ["DeletionAUC", "InsertionAUC", "Comprehensiveness", "Infidelity", "Stability", "Runtime_s"]
    higher_better = {"DeletionAUC": False, "InsertionAUC": True, "Comprehensiveness": True,
                     "Infidelity": False, "Stability": False, "Runtime_s": False}
    norm = au[metrics].copy()
    for m in metrics:
        v = au[m].values; lo, hi = v.min(), v.max()
        s = (v - lo) / (hi - lo + 1e-12)
        norm[m] = s if higher_better[m] else 1 - s
    labels = ["Deletion\n(↓)", "Insertion\n(↑)", "Comprehens.\n(↑)", "Infidelity\n(↓)", "Stability\n(↓)", "Speed\n(↓rt)"]
    ang = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(3.5, 3.4), subplot_kw=dict(polar=True))
    for m in ["kernelshap", "lime", "treeshap", "consensus"]:
        vals = norm.loc[m, metrics].tolist(); vals += vals[:1]
        lw = 2.6 if m == "consensus" else 1.3
        ax.plot(ang, vals, color=CB[m], lw=lw, label=LABEL[m])
        if m == "consensus": ax.fill(ang, vals, color=CB[m], alpha=0.18)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    ax.set_title("Faithfulness audit (outer = better)", fontsize=9, pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=6.6, frameon=False)
    plt.tight_layout(); plt.savefig(FIG + "fig_radar.pdf", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 5: del/ins curves
def fig_curves():
    # recompute curves quickly for plotting (cheap subset)
    import ensemble_predict as E
    bg = pd.read_parquet("outputs/xai_bg.parquet").to_numpy().astype("float32")
    ex = pd.read_parquet("outputs/xai_explain.parquet").to_numpy().astype("float32")
    bg_mean = bg.mean(0).astype("float32"); n, p = ex.shape
    mu = float(E.predict_price(bg).mean()); gx = E.predict_price(ex) - mu
    valid = np.abs(gx) > 1000
    cz = np.load("outputs/consensus_attr.npz", allow_pickle=True)
    methods = {"kernelshap": None, "lime": None, "consensus": cz["attr"]}
    for m in ["kernelshap", "lime"]:
        methods[m] = np.load(f"outputs/attr_{m}.npz")["attr"]
    def mask_eval(keep):
        Xm = np.where(keep, ex, bg_mean[None, :]).astype("float32"); return E.predict_price(Xm) - mu
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.7))
    frac = np.arange(p+1)/p
    for m, attr in methods.items():
        order = np.argsort(-np.abs(attr), axis=1)
        keepd = np.ones((n, p), bool); keepi = np.zeros((n, p), bool)
        dcur = [np.nanmean((mask_eval(keepd)/gx)[valid])]; icur = [np.nanmean((mask_eval(keepi)/gx)[valid])]
        for t in range(p):
            idx = order[:, t]; keepd[np.arange(n), idx] = False; keepi[np.arange(n), idx] = True
            dcur.append(np.nanmean((mask_eval(keepd)/gx)[valid])); icur.append(np.nanmean((mask_eval(keepi)/gx)[valid]))
        lw = 2.4 if m == "consensus" else 1.4
        a1.plot(frac, dcur, color=CB[m], lw=lw, label=LABEL[m])
        a2.plot(frac, icur, color=CB[m], lw=lw, label=LABEL[m])
    a1.set_title("(a) Deletion curve (steeper = better)"); a1.set_xlabel("fraction removed (most important first)")
    a1.set_ylabel("normalized $g(x)$"); a1.legend(fontsize=6.8, frameon=False)
    a2.set_title("(b) Insertion curve (higher = better)"); a2.set_xlabel("fraction inserted")
    a2.set_ylabel("normalized $g(x)$"); a2.legend(fontsize=6.8, frameon=False)
    plt.tight_layout(pad=0.4); plt.savefig(FIG + "fig_curves.pdf", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 6: consensus
def fig_consensus():
    au = pd.read_csv("outputs/final_audit_table.csv", index_col=0)
    gi = pd.read_csv("outputs/global_importance.csv", index_col=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={"width_ratios": [1, 1.2]})
    order = au["MeanRank_all"].sort_values()
    cols = [CB.get(m, "#888") for m in order.index]
    a1.barh([LABEL[m] for m in order.index], order.values, color=cols, edgecolor="#222", lw=0.5)
    a1.invert_yaxis(); a1.set_xlabel("mean rank across 6 axes (lower = better)")
    a1.set_title("(a) Overall explainer quality")
    for i, v in enumerate(order.values): a1.text(v+0.05, i, f"{v:.2f}", va="center", fontsize=6.8)

    cons = gi["consensus"].sort_values(ascending=False).head(10)[::-1]
    a2.barh([disp(f) for f in cons.index], cons.values/1e3, color="#ff7f0e", edgecolor="#222", lw=0.5)
    a2.set_xlabel("mean |attribution| (USD, thousands)")
    a2.set_title("(b) Consensus global importance")
    plt.tight_layout(pad=0.4); plt.savefig(FIG + "fig_consensus.pdf", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 7: local waterfall
def fig_waterfall():
    cz = np.load("outputs/consensus_attr.npz", allow_pickle=True)
    attr = cz["attr"]; fx = cz["fx"]; mu = float(cz["mu"])
    i = int(np.argmax(fx))  # a high-value property
    phi = attr[i]; base = mu
    top = np.argsort(-np.abs(phi))[:9]
    order = top[np.argsort(-phi[top])]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    running = base; ys = []
    ax.axvline(base/1e6, color="#888", ls="--", lw=0.8)
    for k, j in enumerate(order):
        val = phi[j]; color = "#2ca02c" if val > 0 else "#d62728"
        ax.barh(k, val/1e6, left=running/1e6, color=color, edgecolor="#222", lw=0.4)
        running += val; ys.append(disp(MF[j]))
    ax.set_yticks(range(len(order))); ax.set_yticklabels(ys, fontsize=6.8); ax.invert_yaxis()
    ax.set_xlabel("price contribution (M USD)")
    ax.set_title(f"Local consensus explanation\n(base ${base/1e3:.0f}k → pred ${fx[i]/1e6:.2f}M)", fontsize=8.5)
    plt.tight_layout(); plt.savefig(FIG + "fig_waterfall.pdf", bbox_inches="tight"); plt.close()

for fn in [fig_pipeline, fig_models, fig_disagree, fig_radar, fig_curves, fig_consensus, fig_waterfall]:
    fn(); print("ok", fn.__name__)
print("All figures generated.")
