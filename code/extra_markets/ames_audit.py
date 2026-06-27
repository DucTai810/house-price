"""ames_audit.py - dual-regime faithfulness audit on Ames, Iowa (De Cock 2011)."""
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
df = pd.read_csv(os.path.join(HERE, "ames.csv"))
y = np.log(df.pop("SalePrice").to_numpy())
num = df.select_dtypes(include=[np.number]).drop(columns=["Id"]).fillna(df.median(numeric_only=True))
# pick top-20 numeric features by a quick LightGBM importance for a fast, strong model
imp = lgb.LGBMRegressor(n_estimators=300, verbose=-1).fit(num, y)
top = list(pd.Series(imp.feature_importances_, index=num.columns).sort_values(ascending=False).head(20).index)
X = num[top].astype("float64"); FEAT = top; p = len(FEAT)
n_all = len(X)
idx = RNG.permutation(n_all)
tr, va, te = idx[:1000], idx[1000:1200], idx[1200:]
Xf, yf = X.iloc[tr].to_numpy(), y[tr]
Xv, yv = X.iloc[va].to_numpy(), y[va]
Xte, yte = X.iloc[te].to_numpy(), y[te]

lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1).fit(Xf, yf)
xgbm = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, verbosity=0).fit(Xf, yf)
catm = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5, verbose=0).fit(Xf, yf)
P = lambda M, A: M.predict(A)
Vv = np.vstack([P(lgbm, Xv), P(xgbm, Xv), P(catm, Xv)]).T
res = minimize(lambda w: np.sqrt(np.mean((yv - Vv @ w) ** 2)), np.ones(3)/3,
               bounds=[(0,1)]*3, constraints=({"type":"eq","fun":lambda w: w.sum()-1},), method="SLSQP")
w = np.clip(res.x, 0, None); w /= w.sum()
print("blend LGBM/XGB/Cat:", np.round(w, 3))

def predict_log(A):
    A = np.asarray(A, dtype="float64");  A = A[None,:] if A.ndim==1 else A
    return w[0]*P(lgbm,A)+w[1]*P(xgbm,A)+w[2]*P(catm,A)

def f_price(A):
    return np.exp(np.clip(predict_log(A), 9, 15))

pred_log = predict_log(Xte); pred, true = np.exp(np.clip(pred_log, 9, 15)), np.exp(yte)
rmsle = np.sqrt(np.mean((yte - pred_log) ** 2))
print(f"test RMSLE = {rmsle:.4f}, R2 = {1-np.sum((true-pred)**2)/np.sum((true-true.mean())**2):.3f}, RMSE = {np.sqrt(np.mean((true-pred)**2)):,.0f}")

ti = RNG.choice(len(Xte), 40, replace=False); ex = Xte[ti]
bg = Xf[RNG.choice(len(Xf), 40, replace=False)]
bg_mean = bg.mean(0); fx = f_price(ex); mu = float(f_price(bg).mean()); gx = fx-mu
valid = np.abs(gx) > 1000
SC = np.vstack([bg, ex]).std(0); SC[SC<1e-9]=1.0; bg_s = bg/SC; n=len(ex)

A = {}
A["kernelshap"] = np.array(shap.KernelExplainer(f_price, bg).shap_values(ex, nsamples=200, l1_reg=0.0))
A["samplingshap"] = np.array(shap.SamplingExplainer(f_price, bg).shap_values(ex, nsamples=600))
lt = LimeTabularExplainer(bg, mode="regression", feature_names=FEAT, discretize_continuous=False, random_state=1)
la = np.zeros((n,p))
for i in range(n):
    for fi,val in lt.explain_instance(ex[i], f_price, num_features=p, num_samples=600).local_exp[0]: la[i,fi]=val
A["lime"]=la
occ=np.zeros((n,p))
for j in range(p):
    Xo=ex.copy(); Xo[:,j]=bg_mean[j]; occ[:,j]=fx-f_price(Xo)
A["occlusion"]=occ
A["treeshap"]=np.array(shap.TreeExplainer(lgbm).shap_values(ex))

def donor(keep):
    Xm=ex.copy()
    for i in range(n):
        k=keep[i]
        j=int(np.argmin(((bg_s[:,k]-(ex[i]/SC)[k][None,:])**2).sum(1))) if k.sum() else int(np.argmin(((bg_s-ex[i]/SC)**2).sum(1)))
        Xm[i,~k]=bg[j,~k]
    return Xm
def meval(keep,reg): return f_price(np.where(keep,ex,bg_mean[None,:]) if reg=="M" else donor(keep))
def deletion(attr,reg):
    o=np.argsort(-np.abs(attr),1); keep=np.ones((n,p),bool); cur=[meval(keep,reg)-mu]
    for t in range(p): keep[np.arange(n),o[:,t]]=False; cur.append(meval(keep,reg)-mu)
    c=np.array(cur).T/gx[:,None]; return float(np.nanmean(c[valid].mean(1)))
def infidelity(attr,reg,npert=30,frac=0.3):
    s=attr.sum(1); sc=np.where(np.abs(s)>1e-9, gx/np.where(s==0,1e-9,s),1.0); ac=attr*sc[:,None]; v=[]
    for _ in range(npert):
        m=RNG.rand(n,p)<frac; v.append((((ac*m).sum(1)-(fx-meval(~m,reg)))**2/gx**2)[valid])
    return float(np.nanmean(np.concatenate(v)))
def cal(a):
    s=a.sum(1); sc=np.where(np.abs(s)>0.05*np.abs(a).sum(1), gx/np.where(s==0,1e-9,s),1.0); return a*sc[:,None]
infM={m:infidelity(A[m],"M") for m in ["kernelshap","samplingshap","lime","occlusion"]}
inv=np.array([1/infM[m] for m in infM]); fw=dict(zip(infM, inv/inv.sum()))
A["consensus"]=sum(fw[m]*cal(A[m]) for m in fw)
order=["kernelshap","samplingshap","lime","occlusion","treeshap","consensus"]
rows=[]
print("method        del_M  del_OM | inf_M  inf_OM")
for m in order:
    dM,dO=deletion(A[m],"M"),deletion(A[m],"C"); iM=infM.get(m,infidelity(A[m],"M")); iO=infidelity(A[m],"C")
    rows.append((m,dM,dO,iM,iO)); print(f"{m:13s} {dM:6.3f} {dO:6.3f} | {iM:6.3f} {iO:6.3f}")
from scipy.stats import spearmanr
rd=spearmanr([r[1] for r in rows],[r[2] for r in rows]).correlation
ri=spearmanr([r[3] for r in rows],[r[4] for r in rows]).correlation
print(f"\nAmes deletion rank-corr(M,OM)={rd:+.2f}   infidelity rank-corr(M,OM)={ri:+.2f}")
pd.DataFrame(rows,columns=["method","del_M","del_OM","inf_M","inf_OM"]).to_csv(os.path.join(HERE, "ames_audit.csv"),index=False)
print("saved ames_audit.csv")
