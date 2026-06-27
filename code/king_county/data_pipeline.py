"""
data_pipeline.py
Reproduces the King County feature engineering and the strict temporal,
property-disjoint (PINX) split used by the deployed valuation ensemble.
All numbers reported in the paper are produced by executing this code.
"""
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TRAIN_END_YEAR = 2023
TEST_START_YEAR = 2024
TUNE_END_YEAR = 2022
VALIDATION_YEAR = 2023


def add_engineered_features(frame):
    data = frame.copy()
    data["sale_nbr_missing"] = data["sale_nbr"].isna().astype("int8")
    data["sale_nbr"] = data["sale_nbr"].fillna(1).clip(1, 10)
    data["total_living_sqft"] = data["sqft_1"] + data["sqft_fbsmt"]
    data["log_total_living_sqft"] = np.log1p(data["total_living_sqft"].clip(lower=0))
    data["log_above_ground_sqft"] = np.log1p(data["sqft_1"].clip(lower=0))
    data["log_basement_sqft"] = np.log1p(data["sqft_fbsmt"].clip(lower=0))
    data["has_basement"] = (data["sqft_fbsmt"] > 0).astype("int8")
    data["basement_ratio"] = data["sqft_fbsmt"] / (data["total_living_sqft"] + 1)
    data["finished_basement_score"] = data["sqft_fbsmt"] * data["fbsmt_grade"]
    data["has_garage"] = (data["gara_sqft"] > 0).astype("int8")
    data["log_garage_sqft"] = np.log1p(data["gara_sqft"].clip(lower=0))
    data["garage_ratio"] = data["gara_sqft"] / (data["total_living_sqft"] + 1)
    data["beds_capped"] = data["beds"].clip(0, 10)
    data["bath_capped"] = data["total_bath"].clip(0, 8)
    data["bath_per_bed"] = data["total_bath"] / data["beds"].clip(lower=1)
    data["sqft_per_bed"] = data["total_living_sqft"] / data["beds"].clip(lower=1)
    data["sqft_per_bath"] = data["total_living_sqft"] / data["total_bath"].clip(lower=0.5)
    data["grade_sqft"] = data["grade"] * data["log_total_living_sqft"]
    data["quality_score"] = data["grade"] * (1 + 0.1 * data["condition"])
    view_cols = ["view_olympics", "view_cascades", "view_territorial",
                 "view_sound", "view_lakewash", "view_other"]
    data["view_score"] = data[view_cols].sum(axis=1)
    data["view_max"] = data[view_cols].max(axis=1)
    data["view_count"] = (data[view_cols] > 0).sum(axis=1)
    data["has_view"] = (data["view_score"] > 0).astype("int8")
    data["has_waterfront"] = (data["wfnt"] > 0).astype("int8")
    data["premium_location_score"] = data["view_score"] + 2 * data["wfnt"] + data["greenbelt"]
    data["log_age"] = np.log1p(data["age_at_sale"].clip(lower=0))
    data["age_sq"] = (data["age_at_sale"].clip(0, 150) / 100.0) ** 2
    data["reno_age_interaction"] = data["is_reno"] * data["log_age"]
    data["year_centered"] = data["year"] - 2012
    data["year_sq"] = (data["year_centered"] / 10.0) ** 2
    data["post_2012"] = (data["year"] >= 2013).astype("int8")
    data["post_2020"] = (data["year"] >= 2021).astype("int8")
    data["area_submarket"] = data["area"].astype("int32") * 100 + data["submarket"].astype("int32")
    return data


def fit_market_trend(frame, end_year, start_year=2009):
    annual = (frame.loc[frame["year"] <= end_year]
              .groupby("year")["log_price"].median().loc[start_year:end_year])
    model = LinearRegression()
    model.fit((annual.index.to_numpy() - 2012).reshape(-1, 1), annual.to_numpy())
    return model


def market_trend_values(model, years):
    years = np.asarray(years)
    return model.predict((years - 2012).reshape(-1, 1))


def log_to_price(values):
    return np.exp(np.clip(np.asarray(values), 8, 18))


CATEGORICAL_FEATURES = ["area", "submarket", "area_submarket"]


def build():
    csv_path = Path("kingco_cleaned_v2.csv")
    df = pd.read_csv(csv_path, dtype={"sale_id": "string", "pinx": "string"})
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.reset_index(drop=True)
    df = add_engineered_features(df)
    df["price"] = log_to_price(df["log_price"])

    TARGET_LOG_COL = "log_price"
    GROUP_COL = "pinx"
    TIME_COL = "year"
    NON_FEATURE = {TARGET_LOG_COL, "price", GROUP_COL, "sale_id",
                   "sale_nbr", "sale_nbr_missing"}
    MODEL_FEATURES = [c for c in df.columns if c not in NON_FEATURE]

    historical = df[df[TIME_COL] <= TRAIN_END_YEAR].copy()
    test_df = df[df[TIME_COL] >= TEST_START_YEAR].copy()
    test_groups = set(test_df[GROUP_COL])
    hist_groups = set(historical[GROUP_COL])
    pre_purge_overlap = hist_groups & test_groups
    test_df["property_history_status"] = np.where(
        test_df[GROUP_COL].isin(hist_groups),
        "Previously observed before purge", "First appearance in 2024-2025")

    train_df = historical[~historical[GROUP_COL].isin(test_groups)].copy()
    validation_df = df[(df[TIME_COL] == VALIDATION_YEAR)
                       & (~df[GROUP_COL].isin(test_groups))].copy()
    validation_groups = set(validation_df[GROUP_COL])
    tune_df = df[(df[TIME_COL] <= TUNE_END_YEAR)
                 & (~df[GROUP_COL].isin(test_groups | validation_groups))].copy()

    assert not (set(train_df[GROUP_COL]) & test_groups)
    assert train_df[TIME_COL].max() < test_df[TIME_COL].min()

    out = dict(
        df=df, train_df=train_df, test_df=test_df, tune_df=tune_df,
        validation_df=validation_df, MODEL_FEATURES=MODEL_FEATURES,
        CATEGORICAL_FEATURES=CATEGORICAL_FEATURES, GROUP_COL=GROUP_COL,
        TIME_COL=TIME_COL, TARGET_LOG_COL=TARGET_LOG_COL,
        pre_purge_overlap=len(pre_purge_overlap),
        purged_rows=len(historical) - len(train_df),
        n_unique_props=df[GROUP_COL].nunique(),
        n_unique_tx=df["sale_id"].nunique(),
    )
    return out


if __name__ == "__main__":
    d = build()
    print("n transactions:", len(d["df"]))
    print("n unique properties:", d["n_unique_props"])
    print("n features:", len(d["MODEL_FEATURES"]))
    print("tune:", len(d["tune_df"]), "| val:", len(d["validation_df"]),
          "| train:", len(d["train_df"]), "| test:", len(d["test_df"]))
    print("pre-purge overlap props:", d["pre_purge_overlap"],
          "| purged rows:", d["purged_rows"])
    print(d["test_df"]["property_history_status"].value_counts().to_string())
