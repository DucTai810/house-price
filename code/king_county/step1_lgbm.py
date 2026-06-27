"""step1_lgbm.py"""
import time, pickle, numpy as np, pandas as pd
from lightgbm import LGBMRegressor
import helpers as H
C = H.load_cache(); MF, CAT = C["MF"], C["CAT"]
Xt = pd.read_parquet("outputs/X_tune.parquet")
Xtr = pd.read_parquet("outputs/X_train.parquet")
Xv = pd.read_parquet("outputs/X_val.parquet")
Xte = pd.read_parquet("outputs/X_test.parquet")

def mk():
    return LGBMRegressor(n_estimators=900, learning_rate=0.03, num_leaves=255,
        min_child_samples=40, max_bin=255, subsample=0.85, colsample_bytree=0.90,
        reg_alpha=0.03, reg_lambda=0.50, random_state=42, n_jobs=-1, verbosity=-1)

t = time.time()
tune = mk(); tune.fit(H.lgbm_frame(Xt, C["TUNE_CATS"], CAT), C["y_tune_res"], categorical_feature=CAT)
val_log = tune.predict(H.lgbm_frame(Xv, C["TUNE_CATS"], CAT)) + H.trend_values(C["tune_trend"], C["val_years"])
final = mk(); final.fit(H.lgbm_frame(Xtr, C["FINAL_CATS"], CAT), C["y_train_res"], categorical_feature=CAT)
test_log = final.predict(H.lgbm_frame(Xte, C["FINAL_CATS"], CAT)) + H.trend_values(C["final_trend"], C["test_years"])
np.savez("outputs/pred_LightGBM.npz", val_log=val_log, test_log=test_log)
with open("outputs/model_LightGBM.pkl", "wb") as f:
    pickle.dump(final, f)
print(f"LightGBM {time.time()-t:.0f}s", H.reg_metrics("LightGBM", C["y_test_log"], test_log))
