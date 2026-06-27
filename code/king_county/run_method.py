"""run_method.py <method> - compute attributions for all explained instances."""
import sys, time, numpy as np, pandas as pd
import ensemble_predict as E
import helpers as H

method = sys.argv[1]
C = H.load_cache(); MF = C["MF"]
bg = pd.read_parquet("outputs/xai_bg.parquet").to_numpy().astype("float32")
ex = pd.read_parquet("outputs/xai_explain.parquet").to_numpy().astype("float32")
bg_mean = bg.mean(axis=0)
n, p = ex.shape

t0 = time.time()
if method == "kernelshap":
    import shap
    expl = shap.KernelExplainer(E.predict_price, bg)
    attr = np.asarray(expl.shap_values(ex, nsamples=200, l1_reg=0.0))
    base = float(np.asarray(expl.expected_value).reshape(-1)[0])

elif method == "samplingshap":
    import shap
    expl = shap.SamplingExplainer(E.predict_price, bg)
    attr = np.asarray(expl.shap_values(ex, nsamples=1200))
    base = float(np.asarray(expl.expected_value).reshape(-1)[0])

elif method == "lime":
    from lime.lime_tabular import LimeTabularExplainer
    lime_train = pd.read_parquet("outputs/xai_lime_train.parquet").to_numpy().astype("float32")
    cat_idx = [MF.index(c) for c in C["CAT"]]
    expl = LimeTabularExplainer(lime_train, mode="regression", feature_names=MF,
                                categorical_features=cat_idx, discretize_continuous=False,
                                random_state=42, verbose=False)
    attr = np.zeros((n, p))
    for i in range(n):
        e = expl.explain_instance(ex[i], E.predict_price, num_features=p, num_samples=1000)
        for fidx, w in e.local_exp[0]:
            attr[i, fidx] = w
    base = float(np.mean([E.predict_price(bg).mean()]))

elif method == "occlusion":
    fx = E.predict_price(ex)
    attr = np.zeros((n, p))
    for j in range(p):
        ex_occ = ex.copy(); ex_occ[:, j] = bg_mean[j]
        attr[:, j] = fx - E.predict_price(ex_occ)
    base = float(E.predict_price(bg).mean())

elif method == "treeshap":
    import shap
    # TreeSHAP on the dominant LightGBM base learner (log-residual target)
    frame = E._lgbm_frame(ex)
    expl = shap.TreeExplainer(E._models["LightGBM"])
    attr = np.asarray(expl.shap_values(frame))
    base = float(np.asarray(expl.expected_value).reshape(-1)[0])

else:
    raise SystemExit("unknown method")

rt = time.time() - t0
np.savez(f"outputs/attr_{method}.npz", attr=attr, base=base,
         runtime_total=rt, runtime_per=rt / n)
print(f"{method}: {rt:.1f}s total, {rt/n:.2f}s/inst | attr {attr.shape} | "
      f"mean|attr| top feat = {MF[int(np.argmax(np.abs(attr).mean(0)))]}")
