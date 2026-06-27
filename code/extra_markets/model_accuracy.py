"""Compute deployed-model accuracy metrics for the extra housing markets.

This is the small companion to the California/Ames XAI audits. It trains the
same three base learners and convex blend, then reports RMSLE, RMSE, and R2 so
the paper table does not need to use n/a for the extra markets.
"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb


HERE = os.path.dirname(__file__)
RNG = np.random.RandomState(7)


def price_metrics(market, model, y_log, pred_log, clip_bounds):
    low, high = clip_bounds
    true_price = np.exp(y_log)
    pred_price = np.exp(np.clip(pred_log, low, high))
    err = true_price - pred_price
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rmsle = float(np.sqrt(np.mean((y_log - pred_log) ** 2)))
    r2 = float(1 - np.sum(err ** 2) / np.sum((true_price - true_price.mean()) ** 2))
    return {"Market": market, "Model": model, "RMSLE": rmsle, "RMSE": rmse, "R2": r2}


def convex_weights(y_val, val_pred):
    opt = minimize(lambda w: np.sqrt(np.mean((y_val - val_pred @ w) ** 2)),
                   np.ones(val_pred.shape[1]) / val_pred.shape[1],
                   bounds=[(0, 1)] * val_pred.shape[1],
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},),
                   method="SLSQP")
    w = np.clip(opt.x, 0, None)
    return w / w.sum()


def evaluate_market(market, yte, pred_logs, weights, clip_bounds):
    rows = []
    for name, pred_log in pred_logs.items():
        rows.append(price_metrics(market, name, yte, pred_log, clip_bounds))
    order = list(pred_logs)
    blend_log = sum(weights[i] * pred_logs[name] for i, name in enumerate(order))
    rows.append(price_metrics(market, "Convex blend", yte, blend_log, clip_bounds))
    best = min((r for r in rows if r["Model"] != "Convex blend"), key=lambda r: r["RMSLE"])
    rows.append({"Market": market, "Model": "best single learner",
                 "RMSLE": best["RMSLE"], "RMSE": best["RMSE"], "R2": best["R2"]})
    return rows


def california():
    df = pd.read_csv(os.path.join(HERE, "california.csv")).dropna().reset_index(drop=True)
    df["rooms_per_household"] = df.total_rooms / df.households
    df["bedrooms_per_room"] = df.total_bedrooms / df.total_rooms
    df["population_per_household"] = df.population / df.households
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

    y = np.log(df.pop("median_house_value").to_numpy())
    X = df.astype("float64")
    Xtr, Xte, ytr, yte = (X.iloc[:16000].to_numpy(), X.iloc[16000:].to_numpy(),
                          y[:16000], y[16000:])
    Xv, yv = Xtr[12000:], ytr[12000:]
    Xf, yf = Xtr[:12000], ytr[:12000]

    models = {
        "LightGBM": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                      num_leaves=31, verbose=-1).fit(Xf, yf),
        "XGBoost": xgb.XGBRegressor(n_estimators=400, learning_rate=0.05,
                                    max_depth=6, verbosity=0).fit(Xf, yf),
        "CatBoost": CatBoostRegressor(iterations=400, learning_rate=0.05,
                                      depth=6, verbose=0).fit(Xf, yf),
    }
    val_pred = np.vstack([m.predict(Xv) for m in models.values()]).T
    weights = convex_weights(yv, val_pred)
    pred_logs = {name: model.predict(Xte) for name, model in models.items()}
    return evaluate_market("California", yte, pred_logs, weights, (6, 16)), weights


def ames():
    df = pd.read_csv(os.path.join(HERE, "ames.csv"))
    y = np.log(df.pop("SalePrice").to_numpy())
    num = df.select_dtypes(include=[np.number]).drop(columns=["Id"])
    num = num.fillna(num.median(numeric_only=True))

    imp = lgb.LGBMRegressor(n_estimators=300, verbose=-1).fit(num, y)
    top = list(pd.Series(imp.feature_importances_, index=num.columns)
               .sort_values(ascending=False).head(20).index)
    X = num[top].astype("float64")

    idx = RNG.permutation(len(X))
    tr, va, te = idx[:1000], idx[1000:1200], idx[1200:]
    Xf, yf = X.iloc[tr].to_numpy(), y[tr]
    Xv, yv = X.iloc[va].to_numpy(), y[va]
    Xte, yte = X.iloc[te].to_numpy(), y[te]

    models = {
        "LightGBM": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                      num_leaves=31, verbose=-1).fit(Xf, yf),
        "XGBoost": xgb.XGBRegressor(n_estimators=400, learning_rate=0.05,
                                    max_depth=5, verbosity=0).fit(Xf, yf),
        "CatBoost": CatBoostRegressor(iterations=400, learning_rate=0.05,
                                      depth=5, verbose=0).fit(Xf, yf),
    }
    val_pred = np.vstack([m.predict(Xv) for m in models.values()]).T
    weights = convex_weights(yv, val_pred)
    pred_logs = {name: model.predict(Xte) for name, model in models.items()}
    return evaluate_market("Ames (Iowa)", yte, pred_logs, weights, (9, 15)), weights


if __name__ == "__main__":
    ca_rows, ca_w = california()
    am_rows, am_w = ames()
    out = pd.DataFrame(ca_rows + am_rows)
    out.to_csv(os.path.join(HERE, "model_accuracy.csv"), index=False)

    print("California weights LGBM/XGB/Cat:", np.round(ca_w, 3))
    print("Ames weights LGBM/XGB/Cat:", np.round(am_w, 3))
    print(out.to_string(index=False, formatters={
        "RMSLE": "{:.4f}".format,
        "RMSE": "{:,.0f}".format,
        "R2": "{:.4f}".format,
    }))
    print("saved model_accuracy.csv")
