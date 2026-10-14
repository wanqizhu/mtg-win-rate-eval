#!/usr/bin/env python3
"""Train statistical baseline models for GIH WR from card features.
Leave-one-set-out CV on the 10 historical sets; then fit on everything and
predict FRA cards. Outputs work/model_predictions_fra.csv and CV report."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from features import set_features  # noqa: E402

MIN_N = 400
RARITY_NUM = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3, "special": 2}

df = pd.read_csv("/workspace/work/training_cards.csv")
df = df[(df.n_gih >= MIN_N) & (~df.type_line.str.contains("Basic Land", na=False))].copy()
df["set_mean"] = df.groupby("set").gih_wr.transform("mean")
df["y"] = df.gih_wr - df.set_mean  # set-centered target
df["rarity_num"] = df.rarity.map(RARITY_NUM)
df["w"] = np.sqrt(df.n_gih).clip(upper=200)

drop = {"name", "set", "rarity", "text", "type_line", "mana_cost", "oracle_text_raw", "front_name", "card_name_17l",
        "format", "n_gih", "w_gih", "n_oh", "w_oh", "n_gd", "w_gd", "n_gp", "w_gp", "n_gns", "w_gns", "copies_deck",
        "gih_wr", "oh_wr", "gd_wr", "gp_wr", "gns_wr", "iwd", "n_seen", "alsa", "n_taken", "ata", "set_mean", "y", "w",
        "bonus_sheet"}
feat_cols = [c for c in df.columns if c not in drop]
print("n features:", len(feat_cols), "n cards:", len(df))

params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=1)


def fit_lgb(tr: pd.DataFrame, rounds: int = 600) -> lgb.Booster:
    ds = lgb.Dataset(tr[feat_cols].astype(float), tr.y, weight=tr.w)
    return lgb.train(params, ds, num_boost_round=rounds)


def fit_text(tr: pd.DataFrame):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xt = vec.fit_transform(tr.text.fillna(""))
    Xn = csr_matrix(tr[["rarity_num", "cmc", "t_creature", "pt_sum"]].fillna(0).to_numpy(dtype=float))
    X = hstack([Xt, Xn * 0.3]).tocsr()
    m = Ridge(alpha=3.0).fit(X, tr.y, sample_weight=tr.w)
    return vec, m


def pred_text(model, te: pd.DataFrame) -> np.ndarray:
    vec, m = model
    Xt = vec.transform(te.text.fillna(""))
    Xn = csr_matrix(te[["rarity_num", "cmc", "t_creature", "pt_sum"]].fillna(0).to_numpy(dtype=float))
    return m.predict(hstack([Xt, Xn * 0.3]).tocsr())


# ---- Leave-one-set-out CV ----
rows = []
oof = pd.Series(index=df.index, dtype=float)
oof_lgb = pd.Series(index=df.index, dtype=float)
oof_txt = pd.Series(index=df.index, dtype=float)
for s in sorted(df.set.unique()):
    tr, te = df[df.set != s], df[df.set == s]
    m1 = fit_lgb(tr)
    p1 = m1.predict(te[feat_cols].astype(float))
    m2 = fit_text(tr)
    p2 = pred_text(m2, te)
    p = 0.6 * p1 + 0.4 * p2
    oof.loc[te.index] = p; oof_lgb.loc[te.index] = p1; oof_txt.loc[te.index] = p2
    base = np.zeros(len(te))
    # rarity-only baseline
    rb = tr.groupby("rarity").y.mean()
    pr = te.rarity.map(rb).fillna(0).to_numpy()
    def mae(a): return float(np.mean(np.abs(a - te.y)))
    def corr(a): return float(np.corrcoef(a, te.y)[0, 1])
    rows.append({"set": s, "n": len(te), "mae_zero": mae(base), "mae_rarity": mae(pr), "corr_rarity": corr(pr),
                 "mae_lgb": mae(p1), "corr_lgb": corr(p1), "mae_txt": mae(p2), "corr_txt": corr(p2),
                 "mae_blend": mae(p), "corr_blend": corr(p)})
    print(rows[-1])
rep = pd.DataFrame(rows)
print(rep.round(4))
print("mean:", rep.drop(columns="set").mean().round(4).to_dict())
rep.to_csv("/workspace/work/model_cv_report.csv", index=False)
df["oof_pred"] = oof + df.set_mean
df["oof_lgb"] = oof_lgb + df.set_mean
df["oof_txt"] = oof_txt + df.set_mean
df[["set", "name", "rarity", "cmc", "gih_wr", "n_gih", "oof_pred", "oof_lgb", "oof_txt"]].to_csv("/workspace/work/model_oof_predictions.csv", index=False)

# ---- Fit on all and predict FRA ----
m1 = fit_lgb(df)
m2 = fit_text(df)
fra = set_features("FRA", booster_only=False)
fra = fra[~fra.type_line.str.contains("Basic Land", na=False)].copy()
fra["rarity_num"] = fra.rarity.map(RARITY_NUM)
for c in feat_cols:
    if c not in fra.columns:
        fra[c] = 0
p1 = m1.predict(fra[feat_cols].astype(float))
p2 = pred_text(m2, fra)
FRA_MEAN = 0.558
fra["model_lgb"] = p1 + FRA_MEAN
fra["model_txt"] = p2 + FRA_MEAN
fra["model_blend"] = 0.6 * p1 + 0.4 * p2 + FRA_MEAN
imp = pd.Series(m1.feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
print(imp.head(30).round(0))
imp.to_csv("/workspace/work/model_feature_importance.csv")
fra[["name", "rarity", "mana_cost", "cmc", "type_line", "model_lgb", "model_txt", "model_blend"]].sort_values("model_blend", ascending=False).to_csv("/workspace/work/model_predictions_fra.csv", index=False)
print(fra[["name", "rarity", "model_blend"]].sort_values("model_blend", ascending=False).to_string())
