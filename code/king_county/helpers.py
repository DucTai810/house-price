"""helpers.py - shared loaders and trend reconstruction."""
import pickle, numpy as np, pandas as pd

def load_cache():
    with open("outputs/cache.pkl", "rb") as f:
        return pickle.load(f)

def trend_values(coef_intercept, years):
    coef, intercept = coef_intercept
    years = np.asarray(years)
    return np.asarray(coef)[0] * (years - 2012) + intercept

def log_to_price(v):
    return np.exp(np.clip(np.asarray(v), 8, 18))

def lgbm_frame(frame, cats, CAT):
    out = frame.copy()
    for c in CAT:
        out[c] = pd.Categorical(out[c], categories=cats[c])
    return out

def reg_metrics(name, yt_log, yp_log):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ytp, ypp = log_to_price(yt_log), log_to_price(yp_log)
    return {"Model": name, "MSE": mean_squared_error(ytp, ypp),
            "RMSE": mean_squared_error(ytp, ypp) ** 0.5,
            "MAE": mean_absolute_error(ytp, ypp),
            "RMSLE": mean_squared_error(yt_log, yp_log) ** 0.5,
            "R2": r2_score(ytp, ypp)}
