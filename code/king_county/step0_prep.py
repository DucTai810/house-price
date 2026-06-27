"""step0_prep.py - build and cache all splits/targets once."""
import pickle, numpy as np, pandas as pd
import data_pipeline as dp

d = dp.build()
train_df, test_df, tune_df, val_df = d["train_df"], d["test_df"], d["tune_df"], d["validation_df"]
MF, CAT = d["MODEL_FEATURES"], d["CATEGORICAL_FEATURES"]

tune_trend = dp.fit_market_trend(tune_df, dp.TUNE_END_YEAR)
final_trend = dp.fit_market_trend(train_df, dp.TRAIN_END_YEAR)

def res(df, trend):
    return df["log_price"].to_numpy() - dp.market_trend_values(trend, df["year"])

cache = {
    "MF": MF, "CAT": CAT,
    "FINAL_CATS": {c: sorted(train_df[c].dropna().unique().tolist()) for c in CAT},
    "TUNE_CATS": {c: sorted(tune_df[c].dropna().unique().tolist()) for c in CAT},
    "tune_trend": (tune_trend.coef_.tolist(), float(tune_trend.intercept_)),
    "final_trend": (final_trend.coef_.tolist(), float(final_trend.intercept_)),
    "y_tune_res": res(tune_df, tune_trend),
    "y_train_res": res(train_df, final_trend),
    "y_val_log": val_df["log_price"].to_numpy(),
    "y_test_log": test_df["log_price"].to_numpy(),
    "tune_years": tune_df["year"].to_numpy(),
    "train_years": train_df["year"].to_numpy(),
    "val_years": val_df["year"].to_numpy(),
    "test_years": test_df["year"].to_numpy(),
    "pre_purge_overlap": d["pre_purge_overlap"], "purged_rows": d["purged_rows"],
    "n_unique_props": d["n_unique_props"], "n_unique_tx": d["n_unique_tx"],
}
for nm, df in [("tune", tune_df), ("train", train_df), ("val", val_df), ("test", test_df)]:
    df[MF].astype("float32").to_parquet(f"outputs/X_{nm}.parquet")
test_df[["year", "area", "submarket", "log_price", "price", "property_history_status"]].to_parquet("outputs/test_meta.parquet")
# 2023 background pool for SHAP
train_df.loc[train_df["year"].to_numpy() == dp.TRAIN_END_YEAR, MF].astype("float32").to_parquet("outputs/bg_pool_2023.parquet")
with open("outputs/cache.pkl", "wb") as f:
    pickle.dump(cache, f)
print("Cached. tune", len(tune_df), "train", len(train_df), "val", len(val_df), "test", len(test_df))
