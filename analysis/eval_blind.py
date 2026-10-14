#!/usr/bin/env python3
"""Compare blind expert ratings with realised GIH WR and the LOSO ML model."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

code = sys.argv[1] if len(sys.argv) > 1 else "SOS"
OUT = "/workspace/analysis/out"
b = pd.read_csv(f"/workspace/analysis/blind/{code.lower()}_blind_ratings.csv")
ml = pd.read_csv(f"{OUT}/ml_loso.csv")
ml = ml[ml["set"] == code]
h = pd.read_csv(f"{OUT}/historical_features.csv")
h = h[h["set"] == code][["name", "rarity_str", "gih", "gih_wr", "set_mean" if "set_mean" in h else "set_wr", "type_line", "cmc", "color_identity"]]
d = b.merge(ml[["name", "y", "pred_lgb", "pred_ridge"]], on="name", how="left").merge(h, on="name", how="left", suffixes=("", "_h"))
missing = d[d.gih_wr.isna()]
if len(missing):
    print("unmatched:", missing.name.tolist())
d = d[d.gih_wr.notna() & (d.gih >= 200)].copy()
d["actual"] = d.gih_wr * 100
set_mean = d["actual"].mean()
d["pred_c"] = d.pred - d.pred.mean() + set_mean
d["ml"] = d.pred_lgb * 100 + set_mean
d["err"] = d.actual - d.pred_c
print(f"n={len(d)} actual mean {set_mean:.2f} sd {d.actual.std():.2f}; my mean {d.pred.mean():.2f} sd {d.pred.std():.2f}")
for col in ["pred_c", "ml"]:
    r = pearsonr(d[col], d.actual)[0]
    rho = spearmanr(d[col], d.actual)[0]
    mae = np.mean(np.abs(d[col] - d.actual))
    rmse = np.sqrt(np.mean((d[col] - d.actual) ** 2))
    print(f"{col}: r={r:.3f} rho={rho:.3f} MAE={mae:.2f} RMSE={rmse:.2f}")
slope = np.polyfit(d.pred_c, d.actual, 1)
print("calibration actual = %.3f*pred + %.2f" % tuple(slope))
from sklearn.linear_model import LinearRegression
X = d[["pred_c", "ml"]].values
lr = LinearRegression().fit(X, d.actual)
print("combo coefs", lr.coef_, lr.intercept_, "r", pearsonr(lr.predict(X), d.actual)[0])
for r in ["common", "uncommon", "rare", "mythic"]:
    m = d.rarity_str == r
    print(r, m.sum(), "mean err %.2f" % d.err[m].mean(), "r %.3f" % pearsonr(d.pred_c[m], d.actual[m])[0],
          "actual sd %.2f pred sd %.2f" % (d.actual[m].std(), d.pred_c[m].std()))
d["absdiff"] = d.err.abs()
pd.set_option("display.width", 200)
cols = ["name", "rarity_str", "color_identity", "cmc", "gih", "actual", "pred_c", "ml", "err"]
print(d.sort_values("err")[cols].round(2).to_string())
d.to_csv(f"{OUT}/blind_eval_{code}.csv", index=False)
