"""step2_xgb.py"""
import time, pickle, numpy as np, pandas as pd
from xgboost import XGBRegressor
import helpers as H
C = H.load_cache(); MF, CAT = C["MF"], C["CAT"]
Xt = pd.read_parquet("outputs/X_tune.parquet").to_numpy()
Xtr = pd.read_parquet("outputs/X_train.parquet").to_numpy()
Xv = pd.read_parquet("outputs/X_val.parquet").to_numpy()
Xte = pd.read_parquet("outputs/X_test.parquet").to_numpy()

def mk():
    return XGBRegressor(n_estimators=900, learning_rate=0.035, max_depth=0,
        grow_policy="lossguide", max_leaves=255, min_child_weight=8, subsample=0.85,
        colsample_bytree=0.90, reg_alpha=0.03, reg_lambda=0.70,
        objective="reg:squarederror", tree_method="hist", max_bin=255,
        random_state=42, n_jobs=-1)

t = time.time()
tune = mk(); tune.fit(Xt, C["y_tune_res"])
val_log = tune.predict(Xv) + H.trend_values(C["tune_trend"], C["val_years"])
final = mk(); final.fit(Xtr, C["y_train_res"])
test_log = final.predict(Xte) + H.trend_values(C["final_trend"], C["test_years"])
np.savez("outputs/pred_XGBoost.npz", val_log=val_log, test_log=test_log)
with open("outputs/model_XGBoost.pkl", "wb") as f:
    pickle.dump(final, f)
print(f"XGBoost {time.time()-t:.0f}s", H.reg_metrics("XGBoost", C["y_test_log"], test_log))
