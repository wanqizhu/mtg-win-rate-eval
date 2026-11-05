import warnings; warnings.filterwarnings("ignore")
#!/usr/bin/env python3
"""Train GIH-WR models on 10 historical sets with leave-one-set-out CV, then score TRK.
Target = card GIH WR minus set-level GIH-weighted mean (so the model predicts relative strength).
"""
import numpy as np, pandas as pd, json, sys
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix
sys.path.insert(0, "/tmp/trk")
from features import featurize

MIN_GIH = 500
h = pd.read_csv("/tmp/trk/hist/all_cards.csv")
h = h[(h.gih >= MIN_GIH) & (h.in_set_file)].copy()
h = h[~h.type_line.str.startswith("Basic Land")]
h["set_mean"] = h.groupby("set").apply(lambda g: np.average(g.gih_wr, weights=g.gih)).reindex(h.set).values
h["rel"] = h.gih_wr - h.set_mean
h["w"] = np.sqrt(h.gih).clip(upper=200)
X = featurize(h)
trk = pd.read_csv("/tmp/trk/trk_attrs.csv")
trk = trk[~trk.type_line.str.startswith("Basic Land")].reset_index(drop=True)
Xt = featurize(trk)
Xt = Xt.reindex(columns=X.columns, fill_value=0)

sets = sorted(h.set.unique())
y = h.rel.values; w = h.w.values

def lgb_model():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=15,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, verbose=-1)

def ridge_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=60.0))

def text_model_fit_predict(tr_text, tr_y, tr_w, te_text, tr_meta, te_meta):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english")
    A = vec.fit_transform(tr_text); B = vec.transform(te_text)
    A = hstack([A, csr_matrix(tr_meta)]); B = hstack([B, csr_matrix(te_meta)])
    m = Ridge(alpha=3.0); m.fit(A, tr_y, sample_weight=tr_w)
    return m.predict(B)

meta_cols = ["rar_common", "rar_uncommon", "rar_rare", "rar_mythic", "cmc", "type_Creature", "type_Land", "multicolor", "colorless"]
oof = pd.DataFrame(index=h.index, columns=["lgb", "ridge", "text"], dtype=float)
for s in sets:
    tr = h.set != s; te = ~tr
    m1 = lgb_model(); m1.fit(X[tr], y[tr], sample_weight=w[tr]); oof.loc[te, "lgb"] = m1.predict(X[te])
    m2 = ridge_model(); m2.fit(X[tr], y[tr], ridge__sample_weight=w[tr]); oof.loc[te, "ridge"] = m2.predict(X[te])
    oof.loc[te, "text"] = text_model_fit_predict(h.oracle_text.fillna("")[tr], y[tr], w[tr], h.oracle_text.fillna("")[te], X.loc[tr, meta_cols].values, X.loc[te, meta_cols].values)
oof["blend"] = 0.45 * oof.lgb + 0.25 * oof.ridge + 0.30 * oof.text

def report(pred, name):
    err = pred - y
    mae = np.average(np.abs(err), weights=w); rmse = np.sqrt(np.average(err ** 2, weights=w))
    corr = np.corrcoef(pred, y)[0, 1]
    print(f"{name:8s} wMAE={mae:.4f} wRMSE={rmse:.4f} corr={corr:.3f}")
    return mae
print("Leave-one-set-out CV (target = GIH WR relative to set mean); baseline std:", round(np.sqrt(np.average(y**2, weights=w)), 4))
for c in ["lgb", "ridge", "text", "blend"]:
    report(oof[c].values, c)
print("\nPer-rarity blend wMAE / corr:")
for r in ["common", "uncommon", "rare", "mythic"]:
    m = (h.rarity == r).values
    print(r, round(np.average(np.abs(oof.blend.values[m] - y[m]), weights=w[m]), 4), round(np.corrcoef(oof.blend.values[m], y[m])[0, 1], 3))
print("\nPer-set blend corr:")
for s in sets:
    m = (h.set == s).values
    print(s, round(np.corrcoef(oof.blend.values[m], y[m])[0, 1], 3))
h["oof_blend"] = oof.blend.values
h[["set", "name", "rarity", "gih", "gih_wr", "rel", "oof_blend"]].to_csv("/tmp/trk/hist/oof_predictions.csv", index=False)

# fit on all data, score TRK
m1 = lgb_model(); m1.fit(X, y, sample_weight=w)
m2 = ridge_model(); m2.fit(X, y, ridge__sample_weight=w)
pt = text_model_fit_predict(h.oracle_text.fillna(""), y, w, trk.oracle_text.fillna(""), X[meta_cols].values, Xt[meta_cols].values)
trk["m_lgb"] = m1.predict(Xt); trk["m_ridge"] = m2.predict(Xt); trk["m_text"] = pt
trk["m_blend"] = 0.45 * trk.m_lgb + 0.25 * trk.m_ridge + 0.30 * trk.m_text
imp = pd.Series(m1.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop LGB features:", imp.head(25).to_dict())
trk[["name", "rarity", "colors", "mana_cost", "type_line", "m_lgb", "m_ridge", "m_text", "m_blend"]].to_csv("/tmp/trk/trk_model_baseline.csv", index=False)
print(trk.sort_values("m_blend", ascending=False)[["name", "rarity", "mana_cost", "m_blend"]].head(30).round(3).to_string())
print(trk.sort_values("m_blend")[["name", "rarity", "mana_cost", "m_blend"]].head(20).round(3).to_string())
