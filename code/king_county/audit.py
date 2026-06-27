"""audit.py - multi-method explanation faithfulness audit + consensus + agreement."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import ensemble_predict as E
import helpers as H

C = H.load_cache(); MF = C["MF"]; p = len(MF)
bg = pd.read_parquet("outputs/xai_bg.parquet").to_numpy().astype("float32")
ex = pd.read_parquet("outputs/xai_explain.parquet").to_numpy().astype("float32")
bg_mean = bg.mean(axis=0).astype("float32")
n = len(ex)
RNG = np.random.RandomState(7)

AGNOSTIC = ["kernelshap", "samplingshap", "lime", "occlusion"]
ALL = AGNOSTIC + ["treeshap"]
attrs, runtimes = {}, {}
for m in ALL:
    z = np.load(f"outputs/attr_{m}.npz")
    attrs[m] = z["attr"].astype("float64")
    runtimes[m] = float(z["runtime_per"])

fx = E.predict_price(ex).astype("float64")
mu = float(E.predict_price(bg).mean())
gx = fx - mu                      # quantity attributions explain
valid = np.abs(gx) > 1000.0       # skip near-baseline instances for ratio metrics

def mask_eval(keep_bool):
    """keep_bool: (n,p) bool. returns f on masked instances."""
    Xm = np.where(keep_bool, ex, bg_mean[None, :]).astype("float32")
    return E.predict_price(Xm).astype("float64")

def curves(attr):
    """deletion & insertion normalized AUC using |attr| ordering."""
    order = np.argsort(-np.abs(attr), axis=1)            # most->least important
    del_norm = np.zeros((n, p + 1)); ins_norm = np.zeros((n, p + 1))
    keep_del = np.ones((n, p), dtype=bool)               # deletion: start full
    keep_ins = np.zeros((n, p), dtype=bool)              # insertion: start empty
    g_full = gx.copy()
    del_norm[:, 0] = mask_eval(keep_del) - mu
    ins_norm[:, 0] = mask_eval(keep_ins) - mu
    for t in range(p):
        idx = order[:, t]
        keep_del[np.arange(n), idx] = False
        keep_ins[np.arange(n), idx] = True
        del_norm[:, t + 1] = mask_eval(keep_del) - mu
        ins_norm[:, t + 1] = mask_eval(keep_ins) - mu
    with np.errstate(divide="ignore", invalid="ignore"):
        dN = del_norm / g_full[:, None]
        iN = ins_norm / g_full[:, None]
    del_auc = np.nanmean(dN[valid].mean(axis=1))
    ins_auc = np.nanmean(iN[valid].mean(axis=1))
    # comprehensiveness @ top-20%
    k = max(1, int(0.20 * p))
    keep_c = np.ones((n, p), dtype=bool)
    for i in range(n):
        keep_c[i, order[i, :k]] = False
    g_rem = mask_eval(keep_c) - mu
    comp = np.nanmean(1 - (g_rem[valid] / g_full[valid]))
    return del_auc, ins_auc, comp

def infidelity(attr, n_pert=40, frac=0.3):
    """calibrated additive infidelity vs random subset perturbations."""
    s = attr.sum(axis=1)
    scale = np.where(np.abs(s) > 1e-9, gx / np.where(s == 0, 1e-9, s), 1.0)
    attr_cal = attr * scale[:, None]
    vals = []
    for _ in range(n_pert):
        m = RNG.rand(n, p) < frac          # perturbed-to-baseline mask
        keep = ~m
        actual = fx - mask_eval(keep)      # f(x)-f(x_pert)
        pred = (attr_cal * m).sum(axis=1)  # attribution-predicted change
        with np.errstate(divide="ignore", invalid="ignore"):
            vals.append(((pred - actual) ** 2 / (gx ** 2))[valid])
    return float(np.nanmean(np.concatenate(vals)))

def evaluate(attr):
    d, i, c = curves(attr)
    return {"DeletionAUC": d, "InsertionAUC": i, "Comprehensiveness": c,
            "Infidelity": infidelity(attr)}

# ---- pass 1: evaluate the 5 individual methods ----
rows = {}
for m in ALL:
    r = evaluate(attrs[m]); r["Runtime_s"] = runtimes[m]; rows[m] = r
    print(f"{m:13s} del={r['DeletionAUC']:.3f} ins={r['InsertionAUC']:.3f} "
          f"comp={r['Comprehensiveness']:.3f} infid={r['Infidelity']:.4f} rt={r['Runtime_s']:.2f}")

# ---- faithfulness weights: inverse infidelity (additive fidelity to deployed model) ----
inv = np.array([1.0 / rows[m]["Infidelity"] for m in AGNOSTIC])
fw = dict(zip(AGNOSTIC, inv / inv.sum()))
print("\nFaithfulness weights (inverse-infidelity):", {k: round(v, 3) for k, v in fw.items()})

# ---- consensus: faithfulness-weighted average of additively-calibrated attributions ----
def calibrate(a):
    """scale each instance's attribution so sum == g(x) (guarded)."""
    s = a.sum(axis=1)
    scale = np.where(np.abs(s) > 0.05 * np.abs(a).sum(axis=1), gx / np.where(s == 0, 1e-9, s), 1.0)
    return a * scale[:, None]
cons = np.zeros((n, p))
for m in AGNOSTIC:
    cons += fw[m] * calibrate(attrs[m])   # convex weights -> consensus sum ~ g(x)
attrs["consensus"] = cons
r = evaluate(cons); r["Runtime_s"] = sum(runtimes[m] for m in AGNOSTIC)  # cost = sum of members
rows["consensus"] = r
print(f"{'consensus':13s} del={r['DeletionAUC']:.3f} ins={r['InsertionAUC']:.3f} "
      f"comp={r['Comprehensiveness']:.3f} infid={r['Infidelity']:.4f}")

res = pd.DataFrame(rows).T[["DeletionAUC", "InsertionAUC", "Comprehensiveness", "Infidelity", "Runtime_s"]]
res.to_csv("outputs/faithfulness_audit.csv")

# ---- agreement matrices (global importance) ----
glob = {m: np.abs(attrs[m]).mean(axis=0) for m in ALL}
glob["consensus"] = np.abs(attrs["consensus"]).mean(axis=0)
methods_agree = ALL
S = np.zeros((len(methods_agree),) * 2); J = np.zeros_like(S)
def topk(v, k=10): return set(np.argsort(-v)[:k])
for a in range(len(methods_agree)):
    for b in range(len(methods_agree)):
        va, vb = glob[methods_agree[a]], glob[methods_agree[b]]
        S[a, b] = spearmanr(va, vb).correlation
        ja, jb = topk(va), topk(vb)
        J[a, b] = len(ja & jb) / len(ja | jb)
np.savez("outputs/agreement.npz", spearman=S, jaccard=J, methods=np.array(methods_agree))
np.savez("outputs/consensus_attr.npz", attr=cons, fw=np.array([fw[m] for m in AGNOSTIC]),
         ag=np.array(AGNOSTIC), gx=gx, fx=fx, mu=mu)
# global importance table (USD for agnostic price-scale methods)
gi = pd.DataFrame({m: glob[m] for m in ALL + ["consensus"]}, index=MF)
gi.to_csv("outputs/global_importance.csv")

print("\n=== AGREEMENT (Spearman) rows/cols =", methods_agree)
print(np.round(S, 2))
print("\n=== Top-10 Jaccard ===")
print(np.round(J, 2))
print("\nSaved audit artifacts.")
