"""step3_cat.py"""
import time, pickle, numpy as np, pandas as pd
from catboost import CatBoostRegressor
import helpers as H
C = H.load_cache(); MF, CAT = C["MF"], C["CAT"]
Xt = pd.read_parquet("outputs/X_tune.parquet").to_numpy()
Xtr = pd.read_parquet("outputs/X_train.parquet").to_numpy()
Xv = pd.read_parquet("outputs/X_val.parquet").to_numpy()
Xte = pd.read_parquet("outputs/X_test.parquet").to_numpy()

def mk(it):
    return CatBoostRegressor(iterations=it, depth=8, learning_rate=0.05, l2_leaf_reg=5,
        random_strength=0.25, bootstrap_type="Bernoulli", subsample=0.85, rsm=0.90,
        loss_function="RMSE", random_seed=42, thread_count=-1, verbose=False,
        allow_writing_files=False)

t = time.time()
ti = pd.Series(np.arange(len(Xt))).sample(min(180000, len(Xt)), random_state=42).to_numpy()
tune = mk(500); tune.fit(Xt[ti], C["y_tune_res"][ti])
val_log = tune.predict(Xv) + H.trend_values(C["tune_trend"], C["val_years"])
fi = pd.Series(np.arange(len(Xtr))).sample(min(320000, len(Xtr)), random_state=42).to_numpy()
final = mk(600); final.fit(Xtr[fi], C["y_train_res"][fi])
test_log = final.predict(Xte) + H.trend_values(C["final_trend"], C["test_years"])
np.savez("outputs/pred_CatBoost.npz", val_log=val_log, test_log=test_log)
with open("outputs/model_CatBoost.pkl", "wb") as f:
    pickle.dump(final, f)
print(f"CatBoost {time.time()-t:.0f}s", H.reg_metrics("CatBoost", C["y_test_log"], test_log))
