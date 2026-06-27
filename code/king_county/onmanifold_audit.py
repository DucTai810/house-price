"""onmanifold_audit.py - recompute the faithfulness audit under TWO perturbation
regimes: (M) marginal background-mean replacement (standard, OOD), and
(C) conditional on-manifold donor replacement (kept-feature matched). Reuses cached
attributions so explainers are not re-run."""
import numpy as np, pandas as pd
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
attrs = {m: np.load(f"outputs/attr_{m}.npz")["attr"].astype("float64") for m in ALL}

fx = E.predict_price(ex).astype("float64")
mu = float(E.predict_price(bg).mean())
gx = fx - mu
valid = np.abs(gx) > 1000.0

# feature scale for matching distances (robust std over bg+ex)
SC = np.concatenate([bg, ex], axis=0).std(axis=0).astype("float64")
SC[SC < 1e-6] = 1.0
bg_s = bg / SC[None, :]

def donor_fill(keep_bool):
    """on-manifold: replace ~keep features by a bg donor matched on KEPT features."""
    Xm = ex.copy()
    for i in range(n):
        k = keep_bool[i]
        if k.sum() == 0:                      # nothing kept -> nearest overall donor
            j = int(np.argmin(((bg_s - ex[i] / SC) ** 2).sum(axis=1)))
        else:
            d = ((bg_s[:, k] - (ex[i] / SC)[k][None, :]) ** 2).sum(axis=1)
            j = int(np.argmin(d))
        Xm[i, ~k] = bg[j, ~k]
    return Xm.astype("float32")

def mask_eval(keep_bool, regime):
    if regime == "M":
        Xm = np.where(keep_bool, ex, bg_mean[None, :]).astype("float32")
    else:
        Xm = donor_fill(keep_bool)
    return E.predict_price(Xm).astype("float64")

def curves(attr, regime):
    order = np.argsort(-np.abs(attr), axis=1)
    del_n = np.zeros((n, p + 1)); ins_n = np.zeros((n, p + 1))
    keep_d = np.ones((n, p), bool); keep_i = np.zeros((n, p), bool)
    del_n[:, 0] = mask_eval(keep_d, regime) - mu
    ins_n[:, 0] = mask_eval(keep_i, regime) - mu
    for t in range(p):
        idx = order[:, t]
        keep_d[np.arange(n), idx] = False
        keep_i[np.arange(n), idx] = True
        del_n[:, t + 1] = mask_eval(keep_d, regime) - mu
        ins_n[:, t + 1] = mask_eval(keep_i, regime) - mu
    with np.errstate(divide="ignore", invalid="ignore"):
        dN = del_n / gx[:, None]; iN = ins_n / gx[:, None]
    da = np.nanmean(dN[valid].mean(axis=1)); ia = np.nanmean(iN[valid].mean(axis=1))
    k = max(1, int(0.20 * p)); keep_c = np.ones((n, p), bool)
    for i in range(n):
        keep_c[i, order[i, :k]] = False
    g_rem = mask_eval(keep_c, regime) - mu
    comp = np.nanmean(1 - (g_rem[valid] / gx[valid]))
    return da, ia, comp

def infidelity(attr, regime, n_pert=30, frac=0.3):
    s = attr.sum(axis=1)
    scale = np.where(np.abs(s) > 1e-9, gx / np.where(s == 0, 1e-9, s), 1.0)
    ac = attr * scale[:, None]; vals = []
    for _ in range(n_pert):
        m = RNG.rand(n, p) < frac; keep = ~m
        actual = fx - mask_eval(keep, regime)
        pred = (ac * m).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            vals.append(((pred - actual) ** 2 / (gx ** 2))[valid])
    return float(np.nanmean(np.concatenate(vals)))

def calibrate(a):
    s = a.sum(axis=1)
    scale = np.where(np.abs(s) > 0.05 * np.abs(a).sum(axis=1), gx / np.where(s == 0, 1e-9, s), 1.0)
    return a * scale[:, None]

# consensus from inverse-infidelity weights computed on the MARGINAL regime (as deployed)
infM = {m: infidelity(attrs[m], "M") for m in AGNOSTIC}
inv = np.array([1.0 / infM[m] for m in AGNOSTIC]); fw = dict(zip(AGNOSTIC, inv / inv.sum()))
cons = np.zeros((n, p))
for m in AGNOSTIC:
    cons += fw[m] * calibrate(attrs[m])
attrs["consensus"] = cons
METHODS = ALL + ["consensus"]

rows = []
for regime, tag in [("M", "marginal/OOD"), ("C", "on-manifold")]:
    for m in METHODS:
        d, i, c = curves(attrs[m], regime)
        inf = infM[m] if (regime == "M" and m in infM) else infidelity(attrs[m], regime)
        rows.append({"regime": tag, "method": m, "DeletionAUC": d, "InsertionAUC": i,
                     "Comprehensiveness": c, "Infidelity": inf})
        print(f"[{tag:12s}] {m:13s} del={d:.3f} ins={i:.3f} comp={c:.3f} infid={inf:.3f}")
df = pd.DataFrame(rows)
df.to_csv("outputs/onmanifold_audit.csv", index=False)

# ranking shift on infidelity and deletion
print("\n=== ranking by Infidelity (lower=better) ===")
for tag in ["marginal/OOD", "on-manifold"]:
    sub = df[df.regime == tag].sort_values("Infidelity")
    print(f"{tag:12s}:", " > ".join(sub.method.tolist()))
print("\n=== ranking by DeletionAUC (lower=better) ===")
for tag in ["marginal/OOD", "on-manifold"]:
    sub = df[df.regime == tag].sort_values("DeletionAUC")
    print(f"{tag:12s}:", " > ".join(sub.method.tolist()))
print("\nweights:", {k: round(v, 3) for k, v in fw.items()})
print("saved outputs/onmanifold_audit.csv")
