"""step4_ensemble.py - baselines + convex blend + audits + artifacts."""
import time, pickle, numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import helpers as H
C = H.load_cache(); MF, CAT = C["MF"], C["CAT"]
Xtr = pd.read_parquet("outputs/X_train.parquet")
Xv = pd.read_parquet("outputs/X_val.parquet")
Xte = pd.read_parquet("outputs/X_test.parquet")
y_train_res, y_val_log, y_test_log = C["y_train_res"], C["y_val_log"], C["y_test_log"]
y_train_log = y_train_res + H.trend_values(C["final_trend"], C["train_years"])

results = {}
val_log_all, test_log_all = {}, {}

# boosting predictions from steps 1-3
for n in ["LightGBM", "XGBoost", "CatBoost"]:
    z = np.load(f"outputs/pred_{n}.npz")
    val_log_all[n] = z["val_log"]; test_log_all[n] = z["test_log"]
    results[n] = H.reg_metrics(n, y_test_log, z["test_log"])

t0 = time.time()
# ---- Simple Linear Regression (single structural predictor) ----
slr = LinearRegression().fit(Xtr[["log_total_living_sqft"]], y_train_log)
results["Simple Linear Regression"] = H.reg_metrics(
    "Simple Linear Regression", y_test_log, slr.predict(Xte[["log_total_living_sqft"]]))

# ---- Multiple Linear Regression (all 57 features) ----
mlr = LinearRegression().fit(Xtr, y_train_log)
results["Multiple Linear Regression"] = H.reg_metrics(
    "Multiple Linear Regression", y_test_log, mlr.predict(Xte))

# ---- HistGradientBoosting (residual+trend) ----
hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=0.5, max_bins=255, random_state=42)
hgb.fit(Xtr, y_train_res)
results["HistGradientBoosting"] = H.reg_metrics("HistGradientBoosting", y_test_log,
        hgb.predict(Xte) + H.trend_values(C["final_trend"], C["test_years"]))

# ---- Random Forest (residual+trend, subsampled) ----
ridx = pd.Series(np.arange(len(Xtr))).sample(min(120000, len(Xtr)), random_state=42).to_numpy()
rf = RandomForestRegressor(n_estimators=60, max_depth=22, min_samples_leaf=2,
        max_features=0.75, max_samples=0.75, random_state=42, n_jobs=-1)
rf.fit(Xtr.iloc[ridx], y_train_res[ridx])
results["Random Forest"] = H.reg_metrics("Random Forest", y_test_log,
        rf.predict(Xte) + H.trend_values(C["final_trend"], C["test_years"]))
print(f"baselines {time.time()-t0:.0f}s")

# ---- Convex-weighted blend (SLSQP on validation) ----
order = ["LightGBM", "XGBoost", "CatBoost"]
Vm = np.column_stack([val_log_all[n] for n in order])
Tm = np.column_stack([test_log_all[n] for n in order])
def obj(w): return np.sqrt(np.mean((y_val_log - Vm @ np.asarray(w)) ** 2))
opt = minimize(obj, np.repeat(1/3, 3), method="SLSQP", bounds=[(0, 1)]*3,
               constraints={"type": "eq", "fun": lambda v: np.sum(v) - 1.0},
               options={"maxiter": 300, "ftol": 1e-12})
w = np.clip(opt.x, 0, 1); w = w / w.sum()
weights = dict(zip(order, w))
ens_test_log = Tm @ w
results["Convex-Weighted Blend Ensemble"] = H.reg_metrics(
    "Convex-Weighted Blend Ensemble", y_test_log, ens_test_log)

res_df = pd.DataFrame(results.values()).sort_values(["RMSLE", "RMSE", "R2"],
        ascending=[True, True, False]).reset_index(drop=True)
res_df.to_csv("outputs/model_results.csv", index=False)
print("\n=== FULL MODEL COMPARISON (strict 2024-2025) ===")
print(res_df.to_string(index=False))
print("\nConvex weights:", {k: round(v, 4) for k, v in weights.items()},
      "| val blend RMSLE:", round(obj(w), 5))

# improvement vs MLR
mlr_r = results["Multiple Linear Regression"]; ens_r = results["Convex-Weighted Blend Ensemble"]
print("RMSLE reduction vs MLR: %.2f%%" % (100*(mlr_r["RMSLE"]-ens_r["RMSLE"])/mlr_r["RMSLE"]))
print("RMSE reduction vs MLR:  %.2f%%" % (100*(mlr_r["RMSE"]-ens_r["RMSE"])/mlr_r["RMSE"]))

# ---- subset audits for ensemble ----
meta = pd.read_parquet("outputs/test_meta.parquet")
def sub(mask, label):
    m = H.reg_metrics(label, y_test_log[mask], ens_test_log[mask])
    m["rows"] = int(mask.sum()); return m
aud = []
for yr in sorted(meta["year"].unique()):
    aud.append(sub(meta["year"].to_numpy() == yr, f"year_{yr}"))
for st in meta["property_history_status"].unique():
    aud.append(sub(meta["property_history_status"].to_numpy() == st, st))
aud_df = pd.DataFrame(aud)
aud_df.to_csv("outputs/subset_audit.csv", index=False)
print("\n=== SUBSET AUDIT (ensemble) ===")
print(aud_df[["Model", "rows", "RMSLE", "RMSE", "MAE", "R2"]].to_string(index=False))

with open("outputs/artifacts.pkl", "wb") as f:
    pickle.dump({"weights": weights, "order": order,
                 "ens_test_log": ens_test_log, "test_pred_log": test_log_all,
                 "y_test_log": y_test_log}, f)
print("\nSaved artifacts.")
