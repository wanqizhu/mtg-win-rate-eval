#!/usr/bin/env python3
"""Leave-one-set-out feature model for GIH WR deviation from set mean."""
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUT = "/workspace/analysis/out"

META = {"name", "name_full", "collector_number", "cn_int", "type_line", "mana_cost", "color_identity", "oracle",
        "rarity_str", "layout", "scry_set", "printed_name", "set", "format", "in_main_set", "is_guest"}


def load():
    h = pd.read_csv(f"{OUT}/historical_features.csv").copy()
    h = h[(h.in_main_set == 1) & (h.t_basic == 0) & (h.gih >= 200)].copy()
    means = h[h.gih >= 500].groupby("set").gih_wr.mean()
    h["set_mean"] = h["set"].map(means)
    h["y"] = h["gih_wr"] - h["set_mean"]
    h["w"] = np.minimum(h["gih"], 5000) / 5000
    return h


def feature_cols(df, fra):
    stat_cols = {c for c in df.columns if c.startswith(("gp", "oh", "gd", "gih", "gns", "iwd", "deck_copies", "set_"))}
    cols = [c for c in fra.columns if c not in META and c not in stat_cols and c in df.columns]
    return cols


def evaluate(pred, y, w):
    rmse = np.sqrt(np.average((pred - y) ** 2, weights=w))
    mae = np.average(np.abs(pred - y), weights=w)
    return rmse, mae, pearsonr(pred, y)[0], spearmanr(pred, y)[0]


def main():
    h = load()
    fra = pd.read_csv(f"{OUT}/fra_features.csv")
    cols = feature_cols(h, fra)
    print(len(cols), "features")
    sets = sorted(h["set"].unique())
    res = {"lgb": np.zeros(len(h)), "ridge": np.zeros(len(h)), "rarity": np.zeros(len(h))}
    X = h[cols].astype(float)
    for s in sets:
        tr = h["set"] != s
        te = ~tr
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=15,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5, verbose=-1)
        m.fit(X[tr], h.y[tr], sample_weight=h.w[tr])
        res["lgb"][te.values] = m.predict(X[te])
        r = make_pipeline(StandardScaler(), Ridge(alpha=30))
        r.fit(X[tr].fillna(0), h.y[tr], ridge__sample_weight=h.w[tr])
        res["ridge"][te.values] = r.predict(X[te].fillna(0))
        rm = h[tr].groupby("rarity_str").y.mean()
        res["rarity"][te.values] = h[te].rarity_str.map(rm).values
    for k, p in res.items():
        print(k, "ALL rmse %.4f mae %.4f r %.3f rho %.3f" % evaluate(p, h.y.values, h.w.values))
        for s in sets:
            m = (h["set"] == s).values
            print("   ", s, "rmse %.4f mae %.4f r %.3f rho %.3f" % evaluate(p[m], h.y.values[m], h.w.values[m]))
    h["pred_lgb"] = res["lgb"]
    h["pred_ridge"] = res["ridge"]
    h[["set", "name", "rarity_str", "gih", "gih_wr", "y", "pred_lgb", "pred_ridge"]].to_csv(f"{OUT}/ml_loso.csv", index=False)
    # final fit on all, predict FRA
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=15,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5, verbose=-1)
    m.fit(X, h.y, sample_weight=h.w)
    r = make_pipeline(StandardScaler(), Ridge(alpha=30))
    r.fit(X.fillna(0), h.y, ridge__sample_weight=h.w)
    XF = fra[cols].astype(float)
    fra["ml_lgb"] = m.predict(XF)
    fra["ml_ridge"] = r.predict(XF.fillna(0))
    fra[["name", "rarity_str", "type_line", "mana_cost", "ml_lgb", "ml_ridge"]].to_csv(f"{OUT}/fra_ml.csv", index=False)
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
    print(imp.head(30))
    coef = pd.Series(r.named_steps["ridge"].coef_, index=cols).sort_values()
    print(coef.head(15)); print(coef.tail(15))


if __name__ == "__main__":
    main()
