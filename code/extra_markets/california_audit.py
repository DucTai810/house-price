"""california_audit.py - replicate the dual-regime faithfulness audit on the
California Housing dataset (Pace & Barry, 1997) to test generalization."""
import os
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
import shap
from lime.lime_tabular import LimeTabularExplainer

RNG = np.random.RandomState(7)
HERE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(HERE, "california.csv")).dropna().reset_index(drop=True)
# standard engineered features for this dataset
df["rooms_per_household"] = df.total_rooms / df.households
df["bedrooms_per_room"] = df.total_bedrooms / df.total_rooms
df["population_per_household"] = df.population / df.households
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
y = np.log(df.pop("median_house_value").to_numpy())
X = df.astype("float64")
FEAT = list(X.columns); p = len(FEAT)
Xtr, Xte, ytr, yte = (X.iloc[:16000].to_numpy(), X.iloc[16000:].to_numpy(),
                      y[:16000], y[16000:])
# validation slice for blend weights
Xv, yv = Xtr[12000:], ytr[12000:]
Xf, yf = Xtr[:12000], ytr[:12000]

lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1).fit(Xf, yf)
xgbm = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, verbosity=0).fit(Xf, yf)
catm = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=6, verbose=0).fit(Xf, yf)
P = lambda M, A: M.predict(A)
Vv = np.vstack([P(lgbm, Xv), P(xgbm, Xv), P(catm, Xv)]).T
def obj(w): return np.sqrt(np.mean((yv - Vv @ w) ** 2))
cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
res = minimize(obj, np.ones(3)/3, bounds=[(0,1)]*3, constraints=cons, method="SLSQP")
w = res.x; w[w < 0] = 0; w /= w.sum()
print("blend weights LGBM/XGB/Cat:", np.round(w, 3))

def predict_log(A):
    A = np.asarray(A, dtype="float64")
    if A.ndim == 1: A = A[None, :]
    return w[0]*P(lgbm, A) + w[1]*P(xgbm, A) + w[2]*P(catm, A)

def f_price(A):
    return np.exp(np.clip(predict_log(A), 6, 16))

pred_log_te = predict_log(Xte); pred_te = np.exp(np.clip(pred_log_te, 6, 16)); true_te = np.exp(yte)
ss = 1 - np.sum((true_te - pred_te)**2) / np.sum((true_te - true_te.mean())**2)
rmsle = np.sqrt(np.mean((yte - pred_log_te) ** 2))
print(f"test RMSLE = {rmsle:.4f}, R2 (price) = {ss:.3f},  RMSE = {np.sqrt(np.mean((true_te-pred_te)**2)):,.0f}")

# audit setup
ti = RNG.choice(len(Xte), 40, replace=False)
ex = Xte[ti].astype("float64")
bg = Xf[RNG.choice(len(Xf), 40, replace=False)].astype("float64")
bg_mean = bg.mean(0); fx = f_price(ex); mu = float(f_price(bg).mean()); gx = fx - mu
valid = np.abs(gx) > 1000
SC = np.vstack([bg, ex]).std(0); SC[SC < 1e-9] = 1.0; bg_s = bg / SC
n = len(ex)

# attributions
A = {}
ks = shap.KernelExplainer(f_price, bg); A["kernelshap"] = np.array(ks.shap_values(ex, nsamples=200, l1_reg=0.0))
sm = shap.SamplingExplainer(f_price, bg); A["samplingshap"] = np.array(sm.shap_values(ex, nsamples=800))
lt = LimeTabularExplainer(bg, mode="regression", feature_names=FEAT, discretize_continuous=False, random_state=1)
lime_a = np.zeros((n, p))
for i in range(n):
    e = lt.explain_instance(ex[i], f_price, num_features=p, num_samples=800)
    for fi, val in e.local_exp[0]: lime_a[i, fi] = val
A["lime"] = lime_a
occ = np.zeros((n, p))
for j in range(p):
    Xo = ex.copy(); Xo[:, j] = bg_mean[j]; occ[:, j] = fx - f_price(Xo)
A["occlusion"] = occ
te = shap.TreeExplainer(lgbm); A["treeshap"] = np.array(te.shap_values(ex))  # LightGBM log-residual only

def donor(keep):
    Xm = ex.copy()
    for i in range(n):
        k = keep[i]
        j = int(np.argmin(((bg_s[:, k] - (ex[i]/SC)[k][None,:])**2).sum(1))) if k.sum() else \
            int(np.argmin(((bg_s - ex[i]/SC)**2).sum(1)))
        Xm[i, ~k] = bg[j, ~k]
    return Xm
def meval(keep, reg):
    return f_price(np.where(keep, ex, bg_mean[None,:]) if reg=="M" else donor(keep))
def deletion(attr, reg):
    order = np.argsort(-np.abs(attr), 1); keep = np.ones((n,p), bool); curve=[(meval(keep,reg)-mu)]
    for t in range(p):
        keep[np.arange(n), order[:,t]] = False; curve.append(meval(keep,reg)-mu)
    cur = np.array(curve).T / gx[:,None]
    return float(np.nanmean(cur[valid].mean(1)))
def infidelity(attr, reg, npert=30, frac=0.3):
    s = attr.sum(1); sc = np.where(np.abs(s)>1e-9, gx/np.where(s==0,1e-9,s), 1.0); ac = attr*sc[:,None]; v=[]
    for _ in range(npert):
        m = RNG.rand(n,p) < frac
        actual = fx - meval(~m, reg); pred = (ac*m).sum(1)
        v.append(((pred-actual)**2/gx**2)[valid])
    return float(np.nanmean(np.concatenate(v)))

# consensus (inverse-infidelity on marginal)
infM = {m: infidelity(A[m],"M") for m in ["kernelshap","samplingshap","lime","occlusion"]}
inv = np.array([1/infM[m] for m in infM]); fw = dict(zip(infM, inv/inv.sum()))
def cal(a):
    s=a.sum(1); sc=np.where(np.abs(s)>0.05*np.abs(a).sum(1), gx/np.where(s==0,1e-9,s),1.0); return a*sc[:,None]
A["consensus"] = sum(fw[m]*cal(A[m]) for m in fw)
order_m = ["kernelshap","samplingshap","lime","occlusion","treeshap","consensus"]
print("\nmethod        del_M  del_OM | inf_M  inf_OM")
res_rows=[]
for m in order_m:
    dM, dO = deletion(A[m],"M"), deletion(A[m],"C")
    iM = infM.get(m, infidelity(A[m],"M")); iO = infidelity(A[m],"C")
    res_rows.append((m,dM,dO,iM,iO))
    print(f"{m:13s} {dM:6.3f} {dO:6.3f} | {iM:6.3f} {iO:6.3f}")
def rank(idx): 
    order = sorted(res_rows, key=lambda r:r[idx]); return [r[0] for r in order]
print("\ndeletion rank  M:", " > ".join(rank(1)))
print("deletion rank OM:", " > ".join(rank(2)))
print("infidel  rank  M:", " > ".join(rank(3)))
print("infidel  rank OM:", " > ".join(rank(4)))
pd.DataFrame(res_rows, columns=["method","del_M","del_OM","inf_M","inf_OM"]).to_csv(os.path.join(HERE, "california_audit.csv"), index=False)
print("saved california_audit.csv")
