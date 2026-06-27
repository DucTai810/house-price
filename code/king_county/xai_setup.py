"""xai_setup.py - fixed background and explained instances for all methods."""
import numpy as np, pandas as pd
import helpers as H
C = H.load_cache(); MF = C["MF"]
RS = 42
N_BG = 40        # background coalitions reference (2023)
N_EXPLAIN = 60   # explained test instances
N_LIME_TRAIN = 5000

bg = pd.read_parquet("outputs/bg_pool_2023.parquet")
bg_sample = bg.sample(min(N_BG, len(bg)), random_state=RS).reset_index(drop=True)

Xte = pd.read_parquet("outputs/X_test.parquet").reset_index(drop=True)
# stratified by price decile for representativeness
meta = pd.read_parquet("outputs/test_meta.parquet")
price = meta["price"].to_numpy()
deciles = pd.qcut(price, 10, labels=False, duplicates="drop")
rng = np.random.RandomState(RS)
idx = []
per = max(1, N_EXPLAIN // 10)
for dval in np.unique(deciles):
    pool = np.where(deciles == dval)[0]
    idx.extend(rng.choice(pool, min(per, len(pool)), replace=False))
idx = np.array(sorted(idx))[:N_EXPLAIN]
explain = Xte.iloc[idx].reset_index(drop=True)

# LIME training reference (mix of train history)
Xtr = pd.read_parquet("outputs/X_train.parquet")
lime_train = Xtr.sample(min(N_LIME_TRAIN, len(Xtr)), random_state=RS).reset_index(drop=True)

bg_sample.to_parquet("outputs/xai_bg.parquet")
explain.to_parquet("outputs/xai_explain.parquet")
lime_train.to_parquet("outputs/xai_lime_train.parquet")
np.savez("outputs/xai_explain_idx.npz", idx=idx, price=price[idx])
print("bg", bg_sample.shape, "explain", explain.shape, "lime_train", lime_train.shape)
print("explain price range USD:", int(price[idx].min()), "-", int(price[idx].max()))
