#!/usr/bin/env python3
"""Blend blind expert ratings with the LOSO ML model and produce FRA predictions.

Calibration data: the two blind-rated historical sets (SOS, ECL). For each, the
expert ratings were written before any card-level stats of that set were viewed,
and the ML predictions come from models trained without that set (LOSO).
All quantities are deviations (percentage points) from the set's card-mean GIH WR.
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

OUT = "/workspace/analysis/out"
BLIND = "/workspace/analysis/blind"
RARITIES = ["common", "uncommon", "rare", "mythic"]


def prep(df, set_mean=None):
    d = df.copy()
    d["exp"] = d["pred"] - d["pred"].mean()
    d["lgb"] = d["pred_lgb"] * 100
    d["ridge"] = d["pred_ridge"] * 100
    for r in RARITIES[1:]:
        d[f"r_{r}"] = (d["rarity_str"] == r).astype(float)
    if "y" in d:
        d["dev"] = d["y"] * 100
    return d


def design(d, cols):
    return np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])


def fit(d, cols, w=None, lam=0.0):
    X = design(d, cols)
    w = np.ones(len(d)) if w is None else w
    W = np.diag(w)
    pen = lam * np.eye(X.shape[1])
    pen[0, 0] = 0
    return np.linalg.solve(X.T @ W @ X + pen, X.T @ W @ d["dev"].values)


def predict(d, cols, beta):
    return design(d, cols) @ beta


def weights(d):
    return np.minimum(d["gih"].values, 5000) / 5000


MODELS = {
    "ml_lgb": ["lgb"],
    "ml_lgb+ridge": ["lgb", "ridge"],
    "expert": ["exp"],
    "expert+rarity": ["exp", "r_uncommon", "r_rare", "r_mythic"],
    "expert+lgb": ["exp", "lgb"],
    "expert+lgb+rarity": ["exp", "lgb", "r_uncommon", "r_rare", "r_mythic"],
    "expert+lgb+ridge+rarity": ["exp", "lgb", "ridge", "r_uncommon", "r_rare", "r_mythic"],
}


def main():
    sets = {}
    for code in ["SOS", "ECL"]:
        b = pd.read_csv(f"{OUT}/blind_eval_{code}.csv")
        sets[code] = prep(b)

    rows = []
    for name, cols in MODELS.items():
        for train, test in [("SOS", "ECL"), ("ECL", "SOS")]:
            tr, te = sets[train], sets[test]
            beta = fit(tr, cols, weights(tr))
            p = predict(te, cols, beta)
            w = weights(te)
            rmse = np.sqrt(np.average((p - te.dev) ** 2, weights=w))
            mae = np.average(np.abs(p - te.dev), weights=w)
            rows.append(dict(model=name, train=train, test=test, r=pearsonr(p, te.dev)[0], rmse=rmse, mae=mae))
    cv = pd.DataFrame(rows)
    print("Cross-set validation (fit on one blind set, score on the other), gih-weighted errors:")
    print(cv.round(3).to_string(index=False))
    print(cv.groupby("model")[["r", "rmse", "mae"]].mean().sort_values("rmse").round(3).to_string())
    cv.to_csv(f"{OUT}/blend_cv.csv", index=False)

    pooled = pd.concat([sets["SOS"], sets["ECL"]], ignore_index=True)
    final_cols = MODELS["expert+lgb"]
    beta = fit(pooled, final_cols, weights(pooled))
    print("\nPooled final coefficients:", dict(zip(["const"] + final_cols, beta.round(3))))
    for code, d in sets.items():
        p = predict(d, final_cols, beta)
        print(f"  in-sample {code}: r={pearsonr(p, d.dev)[0]:.3f} rmse={np.sqrt(np.mean((p - d.dev) ** 2)):.2f}")

    h = pd.read_csv(f"{OUT}/historical_features.csv")
    h = h[(h.in_main_set == 1) & (h.t_basic == 0) & (h.gih >= 500)]
    set_means = h.groupby("set").gih_wr.mean() * 100
    print("\nHistorical card-mean GIH by set:", set_means.round(2).to_dict())
    fra_set_mean = set_means.mean()
    print(f"FRA set mean assumed: {fra_set_mean:.2f}")

    e = pd.read_csv(f"{BLIND}/fra_expert_ratings.csv")
    m = pd.read_csv(f"{OUT}/fra_ml.csv")
    f = prep(e.merge(m, on="name", how="inner", validate="one_to_one").rename(columns={"ml_lgb": "pred_lgb", "ml_ridge": "pred_ridge"}))
    assert len(f) == len(e) == len(m)
    # Guest-sheet rarity is not a power signal; shift ML by the LOSO residual observed
    # on historical bonus-sheet cards of the same printed rarity (guest_calibration.py).
    gc = pd.read_csv(f"{OUT}/guest_calibration.csv")
    nums = collector_numbers()
    f["is_guest"] = [nums[n][0] != "" for n in f.name]
    shift = {}
    for r, g in gc.groupby("rarity_str"):
        shift[r] = np.average(g.resid, weights=np.minimum(g.gih, 5000) / 5000)
    f.loc[f.is_guest, "lgb"] = f.loc[f.is_guest, "lgb"] + f.loc[f.is_guest, "rarity_str"].map(shift)
    print("Guest ML shift by rarity:", {k: round(v, 2) for k, v in shift.items()})
    f["blend_dev"] = predict(f, final_cols, beta)
    f["pred_gih"] = fra_set_mean + f["blend_dev"]
    f["expert_only"] = fra_set_mean + predict(f, MODELS["expert+rarity"], fit(pooled, MODELS["expert+rarity"], weights(pooled)))
    f["ml_only"] = fra_set_mean + predict(f, MODELS["ml_lgb"], fit(pooled, MODELS["ml_lgb"], weights(pooled)))
    f.to_csv(f"{OUT}/fra_blend_full.csv", index=False)

    print("\nFRA prediction summary by rarity:")
    print(f.groupby("rarity_str")[["pred", "lgb", "pred_gih"]].agg(["mean", "std"]).round(2).to_string())
    print(f"overall mean {f.pred_gih.mean():.2f} sd {f.pred_gih.std():.2f}")
    write_table(f)
    return f


def collector_numbers():
    nums = {}
    for fname, prefix in [("fra.txt", ""), ("fra_guests.txt", "SPG ")]:
        for line in open(f"/workspace/analysis/sets/{fname}"):
            m = re.match(r"^#(\S+) (.+?)  \[", line)
            if m:
                nums[m.group(2).split(" // ")[0]] = (prefix, m.group(1))
    return nums


def write_table(f):
    nums = collector_numbers()
    missing = [n for n in f.name if n not in nums]
    assert not missing, missing
    t = f.copy()
    t["set_part"] = [nums[n][0].strip() or "FRA" for n in t.name]
    t["cn"] = [int(nums[n][1]) for n in t.name]
    t = t.sort_values(["set_part", "cn"], ascending=[True, True])
    t = t[["set_part", "cn", "name", "rarity_str", "mana_cost", "type_line", "pred_gih", "pred", "lgb", "note"]]
    t.columns = ["sheet", "collector_number", "name", "rarity", "mana_cost", "type_line",
                 "predicted_gih_wr", "expert_rating", "ml_lgb_dev", "expert_note"]
    t["ml_lgb_dev"] = t["ml_lgb_dev"].round(2)
    t["predicted_gih_wr"] = t["predicted_gih_wr"].round(1)
    t.to_csv("/workspace/analysis/predictions.csv", index=False)

    lines = ["# Reality Fracture (FRA): predicted 17lands GIH WR (Premier Draft)", "",
             f"{len(t)} cards. Predicted GIH WR in %, sorted by collector number. "
             "Special guests (SPG) listed after the main set. See the PR description for methodology.", "",
             "| # | Card | Rarity | Cost | Type | Pred GIH WR |", "|---|---|---|---|---|---|"]
    for r in t.itertuples():
        cost = r.mana_cost if isinstance(r.mana_cost, str) else ""
        num = r.collector_number if r.sheet == "FRA" else f"SPG {r.collector_number}"
        lines.append(f"| {num} | {r.name} | {r.rarity} | {cost} | {r.type_line} | {r.predicted_gih_wr:.1f} |")
    open("/workspace/analysis/PREDICTIONS.md", "w").write("\n".join(lines) + "\n")
    print(f"wrote predictions.csv and PREDICTIONS.md ({len(t)} rows)")


if __name__ == "__main__":
    main()
