#!/usr/bin/env python3
"""Measure how the main-set ML model mis-predicts bonus-sheet / special-guest cards.

The ML model is trained only on main-set cards, where printed rarity is a strong
power signal. Special-guest cards carry a sheet rarity (all FRA guests are
"mythic") that says little about their limited power. Historical sets contain 407
out-of-main-set cards with real 17lands results; predicting them with LOSO models
quantifies the bias, which is then removed from FRA guest predictions.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ml_model import feature_cols, load

warnings.filterwarnings("ignore")
OUT = "/workspace/analysis/out"


def main():
    h = load()
    allh = pd.read_csv(f"{OUT}/historical_features.csv")
    o = allh[(allh.in_main_set == 0) & (allh.t_basic == 0) & (allh.gih >= 500)].copy()
    o["y"] = o.gih_wr - o.set.map(h[h.gih >= 500].groupby("set").gih_wr.mean())
    o = o[o.y.notna()]
    fra = pd.read_csv(f"{OUT}/fra_features.csv")
    cols = feature_cols(h, fra)
    o["pred_lgb"] = np.nan
    for s in sorted(o.set.unique()):
        tr = h["set"] != s
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=15,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5, verbose=-1)
        m.fit(h.loc[tr, cols].astype(float), h.y[tr], sample_weight=h.w[tr])
        te = o.set == s
        o.loc[te, "pred_lgb"] = m.predict(o.loc[te, cols].astype(float))
    o["dev"] = o.y * 100
    o["lgb"] = o.pred_lgb * 100
    o["resid"] = o.dev - o.lgb
    w = np.minimum(o.gih, 5000) / 5000
    print(f"n={len(o)} r(lgb, actual)={pearsonr(o.lgb, o.dev)[0]:.3f}")
    print(f"mean actual dev {o.dev.mean():.2f}, mean lgb {o.lgb.mean():.2f}, weighted mean residual {np.average(o.resid, weights=w):.2f}")
    print(o.groupby("rarity_str")[["dev", "lgb", "resid"]].agg(["mean", "count"]).round(2).to_string())
    b = np.polyfit(o.lgb, o.dev, 1, w=w)
    print(f"actual = {b[0]:.3f} * lgb + {b[1]:.2f}")
    o[["set", "name", "rarity_str", "type_line", "gih", "dev", "lgb", "resid"]].to_csv(f"{OUT}/guest_calibration.csv", index=False)


if __name__ == "__main__":
    main()
