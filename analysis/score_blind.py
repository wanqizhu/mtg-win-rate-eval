#!/usr/bin/env python3
"""Score blind manual ratings for a historical set against actual GIH WR and the baseline model's LOSO predictions.

Usage: score_blind.py SET RATINGS_FILE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys

SET = sys.argv[1]
RATINGS = sys.argv[2]

d = pd.read_csv("out/baseline_cv.csv")
d = d[d.set == SET].copy()
rat = {}
for line in open(RATINGS):
    if line.startswith("#") or "|" not in line:
        continue
    n, v = line.strip().rsplit("|", 1)
    rat[n] = v
d["manual"] = d.name.map(lambda n: rat.get(n))
missing = d[d.manual.isna()].name.tolist()
print("unrated:", missing)
d = d[d.manual.notna() & (d.manual != "SEEN")].copy()
d["manual"] = d.manual.astype(float) / 100
print("n scored", len(d))


def report(sub: pd.DataFrame, label: str) -> None:
    out = [label, f"n={len(sub)}"]
    for col in ["manual", "pred_ens"]:
        r = np.corrcoef(sub.rel, sub[col])[0, 1]
        out.append(f"r_{col}={r:.3f}")
    for w in [0.5, 0.65, 0.8]:
        b = w * sub.manual + (1 - w) * sub.pred_ens
        out.append(f"r_blend{w}={np.corrcoef(sub.rel, b)[0, 1]:.3f}")
    print("  ".join(out))


report(d, "ALL")
for r in ["common", "uncommon", "rare", "mythic"]:
    report(d[d.rarity == r], r)
# best linear calibration rel ~ a + b*manual + c*model
X = np.column_stack([np.ones(len(d)), d.manual, d.pred_ens])
coef, *_ = np.linalg.lstsq(X, d.rel, rcond=None)
print("OLS rel ~ 1 + manual + model:", coef.round(3))
X2 = np.column_stack([np.ones(len(d)), d.manual])
c2, *_ = np.linalg.lstsq(X2, d.rel, rcond=None)
print("OLS rel ~ 1 + manual:", c2.round(3), "sd manual", d.manual.std().round(4), "sd rel", d.rel.std().round(4))
for s in [0.6, 0.8, 1.0]:
    for w in [0.6, 0.8]:
        p = s * (w * d.manual + (1 - w) * d.pred_ens)
        print(f"scale {s} w {w}: MAE {(d.rel - p).abs().mean():.4f} RMSE {np.sqrt(((d.rel - p) ** 2).mean()):.4f}")
print("MAE zero", d.rel.abs().mean().round(4), "MAE model", (d.rel - d.pred_ens).abs().mean().round(4))
d["err"] = d.rel - d.manual
pd.set_option("display.width", 200)
print(d.sort_values("err")[["name", "rarity", "colors", "rel", "manual", "pred_ens"]].head(25).to_string())
print(d.sort_values("err")[["name", "rarity", "colors", "rel", "manual", "pred_ens"]].tail(25).to_string())
d["col1"] = d.colors.fillna("").map(lambda c: c if len(c) == 1 else ("Gold" if len(c) > 1 else "C"))
print(d.groupby("col1")[["rel", "manual", "pred_ens", "err"]].mean().round(4))
print(d.groupby("rarity")[["rel", "manual", "pred_ens", "err"]].mean().round(4))
d.to_csv("out/" + SET.lower() + "_blind_scored.csv", index=False)
