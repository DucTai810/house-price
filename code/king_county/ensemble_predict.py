"""ensemble_predict.py - deployed ensemble price function f(X_2d)->USD."""
import pickle, numpy as np, pandas as pd
import helpers as H

_C = H.load_cache()
MF, CAT = _C["MF"], _C["CAT"]
FINAL_CATS = _C["FINAL_CATS"]
final_trend = _C["final_trend"]
_cat_idx = [MF.index(c) for c in CAT]

with open("outputs/artifacts.pkl", "rb") as f:
    _A = pickle.load(f)
weights = _A["weights"]; order = _A["order"]

_models = {}
for n in order:
    with open(f"outputs/model_{n}.pkl", "rb") as fh:
        _models[n] = pickle.load(fh)

_year_idx = MF.index("year")


def _lgbm_frame(arr):
    df = pd.DataFrame(arr, columns=MF)
    for c in CAT:
        df[c] = pd.Categorical(df[c], categories=FINAL_CATS[c])
    return df


def predict_price(arr):
    """arr: (n, 57) float array of MODEL_FEATURES. returns USD price (n,)."""
    arr = np.asarray(arr, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    years = arr[:, _year_idx]
    resid = np.zeros(len(arr), dtype="float64")
    for n in order:
        w = weights[n]
        if n == "LightGBM":
            resid += w * _models[n].predict(_lgbm_frame(arr))
        else:
            resid += w * _models[n].predict(arr)
    log_pred = resid + H.trend_values(final_trend, years)
    return H.log_to_price(log_pred)


def predict_lgbm_residual(arr):
    """LightGBM residual (log scale) for TreeSHAP reference."""
    arr = np.asarray(arr, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return _models["LightGBM"].predict(_lgbm_frame(arr))


if __name__ == "__main__":
    import time
    Xte = pd.read_parquet("outputs/X_test.parquet").to_numpy()[:200]
    t = time.time()
    p = predict_price(Xte)
    print("f(200 rows):", round(time.time()-t, 2), "s | sample USD:", p[:3].round(0))
