"""stability.py - max-sensitivity / stability of explanations under input noise."""
import time, numpy as np, pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import ensemble_predict as E
import helpers as H

C = H.load_cache(); MF = C["MF"]; p = len(MF)
CAT = C["CAT"]; cat_idx = [MF.index(c) for c in CAT]
num_idx = [i for i in range(p) if i not in cat_idx]
bg = pd.read_parquet("outputs/xai_bg.parquet").to_numpy().astype("float32")
ex = pd.read_parquet("outputs/xai_explain.parquet").to_numpy().astype("float32")
bg_mean = bg.mean(axis=0).astype("float32")
lime_train = pd.read_parquet("outputs/xai_lime_train.parquet").to_numpy().astype("float32")

cz = np.load("outputs/consensus_attr.npz", allow_pickle=True)
ag = list(cz["ag"]); fw = dict(zip(ag, cz["fw"]))

K, P = 8, 3                          # instances, perturbations each
sigma = 0.03                         # noise scale (fraction of per-feature std)
fstd = ex[:, num_idx].std(axis=0) + 1e-6
RNG = np.random.RandomState(11)
sel = np.arange(K)                   # first K explained instances

# explainers (model-agnostic on f)
ksh = shap.KernelExplainer(E.predict_price, bg)
ssh = shap.SamplingExplainer(E.predict_price, bg)
lim = LimeTabularExplainer(lime_train, mode="regression", feature_names=MF,
        categorical_features=cat_idx, discretize_continuous=False, random_state=42)
tre = shap.TreeExplainer(E._models["LightGBM"])

def explain(method, X):
    if method == "kernelshap":
        return np.asarray(ksh.shap_values(X, nsamples=200, l1_reg=0.0))
    if method == "samplingshap":
        return np.asarray(ssh.shap_values(X, nsamples=1200))
    if method == "lime":
        out = np.zeros((len(X), p))
        for i in range(len(X)):
            e = lim.explain_instance(X[i], E.predict_price, num_features=p, num_samples=1000)
            for fi, w in e.local_exp[0]:
                out[i, fi] = w
        return out
    if method == "occlusion":
        fx = E.predict_price(X); out = np.zeros((len(X), p))
        for j in range(p):
            Xo = X.copy(); Xo[:, j] = bg_mean[j]
            out[:, j] = fx - E.predict_price(Xo)
        return out
    if method == "treeshap":
        return np.asarray(tre.shap_values(E._lgbm_frame(X)))

def rel_change(a, b):
    d = np.linalg.norm(a - b); n = np.linalg.norm(a)
    return d / (n + 1e-9)

methods = ["kernelshap", "samplingshap", "lime", "occlusion", "treeshap"]
base = {m: explain(m, ex[sel]) for m in methods}
base_cons = sum(fw[m] * base[m] for m in ag)

stab = {m: [] for m in methods + ["consensus"]}
t0 = time.time()
for pi in range(P):
    noise = np.zeros_like(ex[sel])
    noise[:, num_idx] = RNG.randn(K, len(num_idx)) * sigma * fstd
    Xp = (ex[sel] + noise).astype("float32")
    pert = {m: explain(m, Xp) for m in methods}
    pert_cons = sum(fw[m] * pert[m] for m in ag)
    for i in range(K):
        for m in methods:
            stab[m].append(rel_change(base[m][i], pert[m][i]))
        stab["consensus"].append(rel_change(base_cons[i], pert_cons[i]))
print(f"stability compute {time.time()-t0:.0f}s")

summary = {m: float(np.mean(v)) for m, v in stab.items()}
pd.Series(summary, name="Stability").to_csv("outputs/stability.csv")
for m in methods + ["consensus"]:
    print(f"{m:13s} stability(rel L2 change) = {summary[m]:.3f}")
