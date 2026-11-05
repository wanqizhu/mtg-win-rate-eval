#!/usr/bin/env python3
"""LOSO (leave-one-set-out) evaluation of GIH WR models + final ensemble training.
Target: GIH win rate centered on the set's overall game win rate."""
import json, re, sys
import numpy as np
from pathlib import Path
sys.path.insert(0,"analysis")
from features import feature_vector, FEATURE_NAMES, clean_text
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

data=json.loads(Path("analysis/historical_gih.json").read_text())
rows=[r for r in data["rows"] if r["gih"]>=100]
sets=sorted(set(r["set"] for r in rows))
base={s:data["report"][s]["base_wr"] for s in sets}
print("sets",sets); print("bases",{k:round(v,4) for k,v in base.items()})
print("rows gih>=100:",len(rows))

# feature matrices
X=[]; names=None
for r in rows:
    v,names=feature_vector(r, names); X.append(v)
X=np.array(X); print("structured dims", X.shape)
y=np.array([r["gih_wr"]-base[r["set"]] for r in rows])
w=np.array([min(r["gih"],5000) for r in rows], dtype=float)
texts=[clean_text(r) for r in rows]

def eval_model(make, label):
    preds=np.zeros(len(rows))
    for s in sets:
        tr=[i for i,r in enumerate(rows) if r["set"]!=s]; te=[i for i,r in enumerate(rows) if r["set"]==s]
        m=make(); m.fit(tr); preds[te]=m.predict(te)
    rmse=np.sqrt(np.mean((preds-y)**2)); mae=np.mean(np.abs(preds-y))
    # weighted rmse by gih
    wrmse=np.sqrt(np.average((preds-y)**2, weights=w))
    pr=pearsonr(preds,y).statistic; sr=spearmanr(preds,y).statistic
    per={}
    for s in sets:
        idx=[i for i,r in enumerate(rows) if r["set"]==s]
        per[s]=round(float(np.sqrt(np.mean((preds[idx]-y[idx])**2))),4)
    print(f"{label:28s} RMSE={rmse:.4f} wRMSE={wrmse:.4f} MAE={mae:.4f} r={pr:.3f} rho={sr:.3f} per-set={per}")
    return preds

class RidgeWrap:
    def __init__(self, alpha=10.0, use_text=True, text_alpha=None):
        self.alpha=alpha; self.use_text=use_text
    def fit(self, idx):
        self.vec=TfidfVectorizer(max_features=6000, ngram_range=(1,2), min_df=3, sublinear_tf=True)
        Xt=self.vec.fit_transform([texts[i] for i in idx]) if self.use_text else None
        Xs=X[idx]
        self.scaler=StandardScaler(); Xs_s=self.scaler.fit_transform(Xs)
        M=sparse.hstack([sparse.csr_matrix(Xs_s), Xt*1.0]) if self.use_text else sparse.csr_matrix(Xs_s)
        self.model=Ridge(alpha=self.alpha)
        self.model.fit(M, y[idx], sample_weight=w[idx])
        return self
    def predict(self, idx):
        Xs_s=self.scaler.transform(X[idx])
        Xt=self.vec.transform([texts[i] for i in idx]) if self.use_text else None
        M=sparse.hstack([sparse.csr_matrix(Xs_s), Xt*1.0]) if self.use_text else sparse.csr_matrix(Xs_s)
        return self.model.predict(M)

class GBRWrap:
    def fit(self, idx):
        self.model=HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=3,
                        l2_regularization=1.0, min_samples_leaf=20, random_state=0)
        self.model.fit(X[idx], y[idx], sample_weight=w[idx]); return self
    def predict(self, idx): return self.model.predict(X[idx])

class RFWrap:
    def fit(self, idx):
        self.model=RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.4, n_jobs=-1, random_state=0)
        self.model.fit(X[idx], y[idx], sample_weight=w[idx]); return self
    def predict(self, idx): return self.model.predict(X[idx])

p_ridge_struct = eval_model(lambda: RidgeWrap(30.0, use_text=False), "ridge structured a=30")
p_ridge_both5 = eval_model(lambda: RidgeWrap(5.0, use_text=True), "ridge struct+text a=5")
p_ridge_both20 = eval_model(lambda: RidgeWrap(20.0, use_text=True), "ridge struct+text a=20")
p_ridge_both50 = eval_model(lambda: RidgeWrap(50.0, use_text=True), "ridge struct+text a=50")
p_gbr = eval_model(lambda: GBRWrap(), "histGBR structured")
p_rf = eval_model(lambda: RFWrap(), "RF structured")

# simple baselines
def baseline_rarity(idx_tr, idx_te):
    # mean centered wr by rarity from train
    from collections import defaultdict
    d=defaultdict(list)
    for i in idx_tr: d[rows[i]["rarity"]].append(y[i])
    m={k:np.average(v, weights=w[[j for j in idx_tr if rows[j]["rarity"]==k]]) for k,v in d.items()}
    return np.array([m.get(rows[i]["rarity"],0) for i in idx_te])
preds=np.zeros(len(rows))
for s in sets:
    tr=[i for i,r in enumerate(rows) if r["set"]!=s]; te=[i for i,r in enumerate(rows) if r["set"]==s]
    preds[te]=baseline_rarity(tr,te)
print(f"{'rarity-only baseline':28s} RMSE={np.sqrt(np.mean((preds-y)**2)):.4f} r={pearsonr(preds,y).statistic:.3f}")
print(f"{'zero baseline':28s} RMSE={np.sqrt(np.mean(y**2)):.4f}")

pred_ens = 0.45*p_ridge_both20 + 0.25*p_gbr + 0.15*p_rf + 0.15*p_ridge_struct
print(f"{'ENSEMBLE (fixed weights)':28s} RMSE={np.sqrt(np.mean((pred_ens-y)**2)):.4f} wRMSE={np.sqrt(np.average((pred_ens-y)**2,weights=w)):.4f} r={pearsonr(pred_ens,y).statistic:.3f} rho={spearmanr(pred_ens,y).statistic:.3f}")
for s in sets:
    idx=[i for i,r in enumerate(rows) if r["set"]==s]
    print(" ",s, round(float(np.sqrt(np.mean((pred_ens[idx]-y[idx])**2))),4), "r=",round(float(pearsonr(pred_ens[idx],y[idx]).statistic),3))

np.savez("analysis/loso_preds.npz", y=y, w=w, ridge_both20=p_ridge_both20, ridge_struct=p_ridge_struct, gbr=p_gbr, rf=p_rf, ens=pred_ens,
         names=np.array([r["set"]+"/"+r["name"] for r in rows]))
