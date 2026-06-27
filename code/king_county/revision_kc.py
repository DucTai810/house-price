"""revision_kc.py - reviewer-driven fixes on King County, reusing cached data:
(1) bootstrap CIs for cross-regime rank correlations,
(2) blend-aware TreeSHAP (weighted sum over the three base learners) for a fair comparison,
(3) on-manifold OOD verification via distance to the nearest background neighbour."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import shap
import ensemble_predict as E
import helpers as H

RNG = np.random.RandomState(11)
C = H.load_cache(); MF = C["MF"]; p = len(MF)
bg = pd.read_parquet("outputs/xai_bg.parquet").to_numpy().astype("float32")
ex = pd.read_parquet("outputs/xai_explain.parquet").to_numpy().astype("float32")
bg_mean = bg.mean(0).astype("float32"); n = len(ex)
fx = E.predict_price(ex).astype("float64"); mu = float(E.predict_price(bg).mean()); gx = fx - mu
valid = np.abs(gx) > 1000.0
SC = np.vstack([bg, ex]).std(0).astype("float64"); SC[SC < 1e-9] = 1.0; bg_s = bg / SC

AG = ["kernelshap", "samplingshap", "lime", "occlusion"]
attrs = {m: np.load(f"outputs/attr_{m}.npz")["attr"].astype("float64") for m in AG + ["treeshap"]}

# ---------- (2) blend-aware TreeSHAP: weighted sum of per-model TreeSHAP on residual ----------
def lgbm_frame(arr):
    df = pd.DataFrame(arr, columns=MF)
    for c in E.CAT: df[c] = pd.Categorical(df[c], categories=E.FINAL_CATS[c])
    return df
ts_blend = np.zeros((n, p))
for name in E.order:
    mdl = E._models[name]; w = E.weights[name]
    Xin = lgbm_frame(ex) if name == "LightGBM" else ex
    sv = np.array(shap.TreeExplainer(mdl).shap_values(Xin))
    ts_blend += w * sv
attrs["treeshap_blend"] = ts_blend
ALL = AG + ["treeshap", "treeshap_blend"]

def donor(keep):
    Xm = ex.copy()
    for i in range(n):
        k = keep[i]
        j = int(np.argmin(((bg_s[:, k] - (ex[i] / SC)[k][None, :]) ** 2).sum(1))) if k.sum() else \
            int(np.argmin(((bg_s - ex[i] / SC) ** 2).sum(1)))
        Xm[i, ~k] = bg[j, ~k]
    return Xm.astype("float32")
def meval(keep, reg): return E.predict_price(np.where(keep, ex, bg_mean[None, :]).astype("float32") if reg == "M" else donor(keep)).astype("float64")

def deletion_per_instance(attr, reg):
    o = np.argsort(-np.abs(attr), 1); keep = np.ones((n, p), bool); cur = [meval(keep, reg) - mu]
    for t in range(p):
        keep[np.arange(n), o[:, t]] = False; cur.append(meval(keep, reg) - mu)
    cnorm = np.array(cur).T / gx[:, None]      # (n, p+1)
    return cnorm.mean(1)                        # per-instance AUC
def infidelity_per_instance(attr, reg, npert=30, frac=0.3):
    s = attr.sum(1); sc = np.where(np.abs(s) > 1e-9, gx / np.where(s == 0, 1e-9, s), 1.0); ac = attr * sc[:, None]
    acc = np.zeros(n)
    for _ in range(npert):
        m = RNG.rand(n, p) < frac
        acc += ((ac * m).sum(1) - (fx - meval(~m, reg))) ** 2 / gx ** 2
    return acc / npert

# precompute per-instance metric values for every method x regime
print("computing per-instance metrics (cached attributions reused)...")
delM = {m: deletion_per_instance(attrs[m], "M") for m in ALL}
delO = {m: deletion_per_instance(attrs[m], "C") for m in ALL}
infM = {m: infidelity_per_instance(attrs[m], "M") for m in ALL}
infO = {m: infidelity_per_instance(attrs[m], "C") for m in ALL}

# consensus from inverse mean-infidelity (marginal) over agnostic methods
def cal(a):
    s = a.sum(1); sc = np.where(np.abs(s) > 0.05 * np.abs(a).sum(1), gx / np.where(s == 0, 1e-9, s), 1.0); return a * sc[:, None]
wts = np.array([1.0 / np.nanmean(infM[m][valid]) for m in AG]); wts /= wts.sum()
cons = sum(wt * cal(attrs[m]) for wt, m in zip(wts, AG))
for d, reg in [(delM, "M"), (delO, "C")]: d["consensus"] = deletion_per_instance(cons, reg)
for d, reg in [(infM, "M"), (infO, "C")]: d["consensus"] = infidelity_per_instance(cons, reg)
METHODS = AG + ["treeshap", "treeshap_blend", "consensus"]

def rankcorr(metric_M, metric_O, idx):
    mM = np.array([np.nanmean(metric_M[m][idx]) for m in METHODS])
    mO = np.array([np.nanmean(metric_O[m][idx]) for m in METHODS])
    return spearmanr(mM, mO).correlation

# ---------- (1) bootstrap CIs ----------
vidx = np.where(valid)[0]; B = 2000
bd, bi = [], []
for _ in range(B):
    s = RNG.choice(vidx, len(vidx), replace=True)
    bd.append(rankcorr(delM, delO, s)); bi.append(rankcorr(infM, infO, s))
bd, bi = np.array(bd), np.array(bi)
def ci(a): return np.nanpercentile(a, 2.5), np.nanmean(a), np.nanpercentile(a, 97.5)
print("\n=== (1) cross-regime rank correlation, bootstrap 95% CI (King County) ===")
print("deletion  : mean %.2f  CI [%.2f, %.2f]" % (ci(bd)[1], ci(bd)[0], ci(bd)[2]))
print("infidelity: mean %.2f  CI [%.2f, %.2f]" % (ci(bi)[1], ci(bi)[0], ci(bi)[2]))
print("P(infidelity corr > deletion corr) = %.3f" % np.mean(bi > bd))

# ---------- (2) blend-aware TreeSHAP infidelity ----------
print("\n=== (2) TreeSHAP fairness: infidelity (lower=better) ===")
for m in ["treeshap", "treeshap_blend"]:
    print(f"{m:16s} M={np.nanmean(infM[m][valid]):.3f}  OM={np.nanmean(infO[m][valid]):.3f}")
for m in AG + ["consensus"]:
    print(f"{m:16s} M={np.nanmean(infM[m][valid]):.3f}  OM={np.nanmean(infO[m][valid]):.3f}")

# ---------- (3) on-manifold OOD verification ----------
def nn_dist(X):
    d = np.zeros(len(X))
    for i in range(len(X)):
        d[i] = np.sqrt(((bg_s - X[i] / SC) ** 2).sum(1).min())
    return d
# perturb 50% of features under each regime, compare distance-to-manifold
mask = RNG.rand(n, p) < 0.5; keep = ~mask
Xmarg = np.where(keep, ex, bg_mean[None, :]).astype("float32")
Xon = donor(keep)
dm, do, dx = nn_dist(Xmarg), nn_dist(Xon), nn_dist(ex.astype("float32"))
print("\n=== (3) distance to nearest background neighbour (lower = more in-distribution) ===")
print("original instances : %.2f" % dx.mean())
print("marginal perturbed : %.2f" % dm.mean())
print("on-manifold perturbed: %.2f" % do.mean())
print("on-manifold is closer to the manifold than marginal for %.0f%% of points" % (100 * np.mean(do < dm)))

pd.DataFrame({"boot_deletion": bd, "boot_infidelity": bi}).to_csv("outputs/bootstrap_rankcorr.csv", index=False)
print("\nsaved outputs/bootstrap_rankcorr.csv")
