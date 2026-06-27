"""boot_extra.py - bootstrap CIs for cross-regime rank correlations on California and Ames."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
import shap
from lime.lime_tabular import LimeTabularExplainer

def run_market(name, csv, target_col, prep, seed=7, n_inst=40, n_bg=40):
    RNG = np.random.RandomState(seed)
    X, y, FEAT = prep(csv)
    p = len(FEAT); idx = RNG.permutation(len(X))
    ntr = int(len(X)*0.7); nva = int(len(X)*0.1)
    tr, va, te = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]
    Xf, yf = X[tr], y[tr]; Xv, yv = X[va], y[va]; Xte, yte = X[te], y[te]
    lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1).fit(Xf, yf)
    xgbm = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, verbosity=0).fit(Xf, yf)
    catm = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5, verbose=0).fit(Xf, yf)
    P = lambda M,A: M.predict(A)
    Vv = np.vstack([P(lgbm,Xv),P(xgbm,Xv),P(catm,Xv)]).T
    res = minimize(lambda w: np.sqrt(np.mean((yv-Vv@w)**2)), np.ones(3)/3, bounds=[(0,1)]*3,
                   constraints=({"type":"eq","fun":lambda w: w.sum()-1},), method="SLSQP")
    w = np.clip(res.x,0,None); w/=w.sum()
    def f(A):
        A=np.asarray(A,dtype="float64"); A=A[None,:] if A.ndim==1 else A
        return np.exp(np.clip(w[0]*P(lgbm,A)+w[1]*P(xgbm,A)+w[2]*P(catm,A),5,16))
    ti = RNG.choice(len(Xte), n_inst, replace=False); ex = Xte[ti]
    bg = Xf[RNG.choice(len(Xf), n_bg, replace=False)]
    bg_mean = bg.mean(0); fx=f(ex); mu=float(f(bg).mean()); gx=fx-mu
    valid = np.abs(gx)>1e-6*np.abs(mu)
    SC = np.vstack([bg,ex]).std(0); SC[SC<1e-9]=1.0; bg_s=bg/SC; n=len(ex)
    A={}
    A["kernelshap"]=np.array(shap.KernelExplainer(f,bg).shap_values(ex,nsamples=160,l1_reg=0.0))
    A["samplingshap"]=np.array(shap.SamplingExplainer(f,bg).shap_values(ex,nsamples=400))
    lt=LimeTabularExplainer(bg,mode="regression",feature_names=FEAT,discretize_continuous=False,random_state=1)
    la=np.zeros((n,p))
    for i in range(n):
        for fi,val in lt.explain_instance(ex[i],f,num_features=p,num_samples=400).local_exp[0]: la[i,fi]=val
    A["lime"]=la
    occ=np.zeros((n,p))
    for j in range(p):
        Xo=ex.copy(); Xo[:,j]=bg_mean[j]; occ[:,j]=fx-f(Xo)
    A["occlusion"]=occ
    A["treeshap"]=np.array(shap.TreeExplainer(lgbm).shap_values(ex))
    def donor(keep):
        Xm=ex.copy()
        for i in range(n):
            k=keep[i]
            j=int(np.argmin(((bg_s[:,k]-(ex[i]/SC)[k][None,:])**2).sum(1))) if k.sum() else int(np.argmin(((bg_s-ex[i]/SC)**2).sum(1)))
            Xm[i,~k]=bg[j,~k]
        return Xm
    def meval(keep,reg): return f(np.where(keep,ex,bg_mean[None,:]) if reg=="M" else donor(keep))
    def deli(attr,reg):
        o=np.argsort(-np.abs(attr),1); keep=np.ones((n,p),bool); cur=[meval(keep,reg)-mu]
        for t in range(p): keep[np.arange(n),o[:,t]]=False; cur.append(meval(keep,reg)-mu)
        return (np.array(cur).T/gx[:,None]).mean(1)   # per-instance
    def infi(attr,reg,T=30,frac=0.3):
        s=attr.sum(1); sc=np.where(np.abs(s)>1e-9,gx/np.where(s==0,1e-9,s),1.0); ac=attr*sc[:,None]; acc=np.zeros(n)
        for _ in range(T):
            m=RNG.rand(n,p)<frac; acc+=(((ac*m).sum(1)-(fx-meval(~m,reg)))**2/gx**2)
        return acc/T
    def cal(a):
        s=a.sum(1); sc=np.where(np.abs(s)>0.05*np.abs(a).sum(1),gx/np.where(s==0,1e-9,s),1.0); return a*sc[:,None]
    AG=["kernelshap","samplingshap","lime","occlusion"]
    infM={m:infi(A[m],"M") for m in AG}
    wts=np.array([1/np.nanmean(infM[m][valid]) for m in AG]); wts/=wts.sum()
    A["consensus"]=sum(wt*cal(A[m]) for wt,m in zip(wts,AG))
    M=AG+["treeshap","consensus"]
    delM={m:deli(A[m],"M") for m in M}; delO={m:deli(A[m],"C") for m in M}
    infMM={m:(infM[m] if m in infM else infi(A[m],"M")) for m in M}; infO={m:infi(A[m],"C") for m in M}
    vidx=np.where(valid)[0]
    def rc(dM,dO,s): 
        a=[np.nanmean(dM[m][s]) for m in M]; b=[np.nanmean(dO[m][s]) for m in M]; return spearmanr(a,b).correlation
    B=2000; bd=[]; bi=[]
    for _ in range(B):
        s=RNG.choice(vidx,len(vidx),replace=True)
        bd.append(rc(delM,delO,s)); bi.append(rc(infMM,infO,s))
    bd=np.array(bd); bi=np.array(bi)
    ci=lambda a:(np.nanpercentile(a,2.5),np.nanmean(a),np.nanpercentile(a,97.5))
    print(f"\n=== {name}  (blend {np.round(w,3)}) ===")
    print("deletion   mean %.2f  CI [%.2f, %.2f]"%(ci(bd)[1],ci(bd)[0],ci(bd)[2]))
    print("infidelity mean %.2f  CI [%.2f, %.2f]"%(ci(bi)[1],ci(bi)[0],ci(bi)[2]))
    print("P(infid > del) = %.3f"%np.mean(bi>bd))

def prep_ca(csv):
    df=pd.read_csv(csv); y=np.log(df.pop("median_house_value").to_numpy() if "median_house_value" in df else df.pop(df.columns[-1]).to_numpy())
    df=df.select_dtypes(include=[np.number]).fillna(df.median(numeric_only=True))
    return df.to_numpy().astype("float64"), y, list(df.columns)
def prep_ames(csv):
    df=pd.read_csv(csv); y=np.log(df.pop("SalePrice").to_numpy())
    num=df.select_dtypes(include=[np.number]).drop(columns=["Id"]).fillna(df.median(numeric_only=True))
    imp=lgb.LGBMRegressor(n_estimators=300,verbose=-1).fit(num,y)
    top=list(pd.Series(imp.feature_importances_,index=num.columns).sort_values(ascending=False).head(20).index)
    return num[top].to_numpy().astype("float64"), y, top

import os
run_market("California","/home/claude/extra_data/california.csv","",prep_ca,seed=7)
run_market("Ames","/home/claude/extra_data/ames.csv","SalePrice",prep_ames,seed=7)
