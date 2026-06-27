"""
train_models.py
Trains LightGBM, XGBoost, CatBoost under the strict temporal property-disjoint
design, learns convex blend weights via SLSQP on the 2023 validation set,
evaluates every model on the 2024-2025 benchmark, and saves all artifacts
needed for the multi-method explanation audit. Real metrics only.
"""
import time, json, pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import data_pipeline as dp

RS = dp.RANDOM_STATE
t0 = time.time()
d = dp.build()
train_df, test_df, tune_df, val_df = d["train_df"], d["test_df"], d["tune_df"], d["validation_df"]
MF, CAT = d["MODEL_FEATURES"], d["CATEGORICAL_FEATURES"]

# ---- targets (market-trend-detrended residual on log scale) ----
tune_trend = dp.fit_market_trend(tune_df, dp.TUNE_END_YEAR)
final_trend = dp.fit_market_trend(train_df, dp.TRAIN_END_YEAR)

def res(df, trend):
    return df["log_price"].to_numpy() - dp.market_trend_values(trend, df["year"])

y_tune_res = res(tune_df, tune_trend)
y_train_res = res(train_df, final_trend)
y_val_log = val_df["log_price"].to_numpy()
y_test_log = test_df["log_price"].to_numpy()
y_test_price = dp.log_to_price(y_test_log)

FINAL_CATS = {c: sorted(train_df[c].dropna().unique().tolist()) for c in CAT}
TUNE_CATS = {c: sorted(tune_df[c].dropna().unique().tolist()) for c in CAT}

def lgbm_frame(frame, cats):
    out = frame.copy()
    for c in CAT:
        out[c] = pd.Categorical(out[c], categories=cats[c])
    return out

def predict_log(model, X, years, trend, kind, cats=FINAL_CATS):
    Xe = lgbm_frame(X, cats) if kind == "lgbm" else X
    return model.predict(Xe) + dp.market_trend_values(trend, years)

def metrics(name, yt_log, yp_log):
    ytp, ypp = dp.log_to_price(yt_log), dp.log_to_price(yp_log)
    return {"Model": name, "MSE": mean_squared_error(ytp, ypp),
            "RMSE": mean_squared_error(ytp, ypp) ** 0.5,
            "MAE": mean_absolute_error(ytp, ypp),
            "RMSLE": mean_squared_error(yt_log, yp_log) ** 0.5,
            "R2": r2_score(ytp, ypp)}

def make_lgbm():
    return LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=255,
        min_child_samples=40, max_bin=255, subsample=0.85, colsample_bytree=0.90,
        reg_alpha=0.03, reg_lambda=0.50, random_state=RS, n_jobs=-1, verbosity=-1)

def make_xgb():
    return XGBRegressor(n_estimators=900, learning_rate=0.035, max_depth=0,
        grow_policy="lossguide", max_leaves=255, min_child_weight=8, subsample=0.85,
        colsample_bytree=0.90, reg_alpha=0.03, reg_lambda=0.70,
        objective="reg:squarederror", tree_method="hist", max_bin=255,
        random_state=RS, n_jobs=-1)

def make_cat(iters):
    return CatBoostRegressor(iterations=iters, depth=8, learning_rate=0.05,
        l2_leaf_reg=5, random_strength=0.25, bootstrap_type="Bernoulli",
        subsample=0.85, rsm=0.90, loss_function="RMSE", random_seed=RS,
        thread_count=-1, verbose=False, allow_writing_files=False)

X_tune = tune_df[MF].astype("float32")
X_train = train_df[MF].astype("float32")
X_val = val_df[MF].astype("float32")
X_test = test_df[MF].astype("float32")

val_pred_log, test_pred_log, metrics_reg, fitted = {}, {}, {}, {}

# ---------- LightGBM ----------
t = time.time()
lgbm_tune = make_lgbm(); lgbm_tune.fit(lgbm_frame(X_tune, TUNE_CATS), y_tune_res, categorical_feature=CAT)
val_pred_log["LightGBM"] = predict_log(lgbm_tune, X_val, val_df["year"], tune_trend, "lgbm", TUNE_CATS)
lgbm = make_lgbm(); lgbm.fit(lgbm_frame(X_train, FINAL_CATS), y_train_res, categorical_feature=CAT)
test_pred_log["LightGBM"] = predict_log(lgbm, X_test, test_df["year"], final_trend, "lgbm")
fitted["LightGBM"] = lgbm
print(f"LightGBM done {time.time()-t:.0f}s")

# ---------- XGBoost ----------
t = time.time()
xgb_tune = make_xgb(); xgb_tune.fit(X_tune.to_numpy(), y_tune_res)
val_pred_log["XGBoost"] = xgb_tune.predict(X_val.to_numpy()) + dp.market_trend_values(tune_trend, val_df["year"])
xgb = make_xgb(); xgb.fit(X_train.to_numpy(), y_train_res)
test_pred_log["XGBoost"] = xgb.predict(X_test.to_numpy()) + dp.market_trend_values(final_trend, test_df["year"])
fitted["XGBoost"] = xgb
print(f"XGBoost done {time.time()-t:.0f}s")

# ---------- CatBoost ----------
t = time.time()
cat_tune_n = min(180000, len(X_tune))
cti = pd.Series(np.arange(len(X_tune))).sample(cat_tune_n, random_state=RS).to_numpy()
cb_tune = make_cat(500)
cb_tune.fit(X_tune.to_numpy()[cti], y_tune_res[cti])
val_pred_log["CatBoost"] = cb_tune.predict(X_val.to_numpy()) + dp.market_trend_values(tune_trend, val_df["year"])
cat_final_n = min(320000, len(X_train))
cfi = pd.Series(np.arange(len(X_train))).sample(cat_final_n, random_state=RS).to_numpy()
cb = make_cat(600)
cb.fit(X_train.to_numpy()[cfi], y_train_res[cfi])
test_pred_log["CatBoost"] = cb.predict(X_test.to_numpy()) + dp.market_trend_values(final_trend, test_df["year"])
fitted["CatBoost"] = cb
print(f"CatBoost done {time.time()-t:.0f}s")

for n in ["LightGBM", "XGBoost", "CatBoost"]:
    metrics_reg[n] = metrics(n, y_test_log, test_pred_log[n])

# ---------- Convex-weighted blend (SLSQP on validation) ----------
order = ["LightGBM", "XGBoost", "CatBoost"]
Vm = np.column_stack([val_pred_log[n] for n in order])
Tm = np.column_stack([test_pred_log[n] for n in order])

def obj(w):
    return mean_squared_error(y_val_log, Vm @ np.asarray(w)) ** 0.5

w0 = np.repeat(1/3, 3)
opt = minimize(obj, w0, method="SLSQP", bounds=[(0, 1)]*3,
               constraints={"type": "eq", "fun": lambda v: np.sum(v) - 1.0},
               options={"maxiter": 200, "ftol": 1e-10})
w = np.clip(opt.x if opt.success else w0, 0, 1); w = w / w.sum()
weights = dict(zip(order, w))
ens_test_log = Tm @ w
metrics_reg["Convex-Weighted Blend Ensemble"] = metrics("Convex-Weighted Blend Ensemble", y_test_log, ens_test_log)

results = pd.DataFrame(metrics_reg.values()).sort_values(["RMSLE", "RMSE", "R2"],
                                                         ascending=[True, True, False]).reset_index(drop=True)
print("\n=== TEST METRICS (strict 2024-2025) ===")
print(results.to_string(index=False))
print("\nConvex weights:", {k: round(v, 4) for k, v in weights.items()})
print("Validation blend RMSLE:", round(obj(w), 5))

# ---- per-year and per-history audit for ensemble ----
def subset_metrics(mask, label):
    m = metrics("ens", y_test_log[mask], ens_test_log[mask])
    m["subset"] = label; m["rows"] = int(mask.sum())
    return m

audit = []
for yr in sorted(test_df["year"].unique()):
    audit.append(subset_metrics(test_df["year"].to_numpy() == yr, f"year_{yr}"))
for st in test_df["property_history_status"].unique():
    audit.append(subset_metrics(test_df["property_history_status"].to_numpy() == st, st))
audit_df = pd.DataFrame(audit)
print("\n=== SUBSET AUDIT (ensemble) ===")
print(audit_df[["subset", "rows", "RMSLE", "RMSE", "MAE", "R2"]].to_string(index=False))

# ---- save artifacts ----
results.to_csv("outputs/model_results.csv", index=False)
audit_df.to_csv("outputs/subset_audit.csv", index=False)
with open("outputs/artifacts.pkl", "wb") as f:
    pickle.dump({"weights": weights, "order": order, "MF": MF, "CAT": CAT,
                 "FINAL_CATS": FINAL_CATS,
                 "final_trend_coef": (final_trend.coef_.tolist(), float(final_trend.intercept_)),
                 "ens_test_log": ens_test_log, "test_pred_log": test_pred_log,
                 "y_test_log": y_test_log}, f)
for n, m in fitted.items():
    with open(f"outputs/model_{n}.pkl", "wb") as f:
        pickle.dump(m, f)
test_df[["year", "area", "submarket", "log_price", "price", "property_history_status"]].to_csv(
    "outputs/test_meta.csv", index=False)
X_test.to_parquet("outputs/X_test.parquet") if hasattr(X_test, "to_parquet") else X_test.to_csv("outputs/X_test.csv", index=False)
# also save a background pool (2023 rows from training) and the test feature matrix sample
bg_pool = X_train.loc[train_df["year"].to_numpy() == dp.TRAIN_END_YEAR]
bg_pool.to_csv("outputs/bg_pool_2023.csv", index=False)
print(f"\nTOTAL train time {time.time()-t0:.0f}s. Artifacts saved.")
