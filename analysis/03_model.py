"""Train GIH model on historical sets, predict TRK. Ridge + TF-IDF + structured features."""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

ANALYSIS = Path("/agent/repos/mtg-win-rate-eval/analysis")

# Load GIH
gih = json.loads((ANALYSIS/"gih_by_set.json").read_text())
# Load features
feat = pd.read_pickle(ANALYSIS/"features.pkl")
feat["front"] = feat["name"]  # already front-face deduped

# Build training frame
rows=[]
for code, res in gih.items():
    setcode = code  # e.g. FRA
    fsub = feat[feat["set"]==setcode].set_index("front")
    for c in res["cards"]:
        name=c["card"]
        # normalize: game uses front face for split cards; scryfall front matches
        # also strip whitespace
        if name in fsub.index:
            f = fsub.loc[name]
            # f may be DataFrame if duplicate index (shouldn't happen)
            if isinstance(f, pd.DataFrame):
                f=f.iloc[0]
            rows.append({
                "set":setcode,"name":name,"gih":c["gih"],"wins":c["wins"],"gih_wr":c["gih_wr"],
                **{k:f[k] for k in f.index if k not in ("name","set")}
            })
        else:
            # try front-face match for "//" cards: game name is front of scryfall full name
            # fsub index already front, so if not found, it's a bonus-sheet card not in set extract
            pass

train=pd.DataFrame(rows)
print("training rows:", train.shape, "sets:", train["set"].unique())
# filter low sample
train=train[train["gih"]>=100].copy()
print("after gih>=100:", train.shape)
train.to_pickle(ANALYSIS/"train.pkl")

# Features
STRUCT = [c for c in feat.columns if c not in ("name","set","oracle_text","type_line","mana_cost","colors","rarity","front")]
print("struct features:", len(STRUCT))
# One-hot rarity already have rarity_ord; add dummies
# TF-IDF on oracle text + type line
train["text_full"] = (train["type_line"].fillna("")+" "+train["oracle_text"].fillna("")).str.lower()
train["text_full"] = train["text_full"].str.replace(r"\(.*?\)"," ",regex=True)  # drop reminder text

vec = TfidfVectorizer(max_features=1500, ngram_range=(1,2), min_df=5, stop_words="english")
X_tfidf = vec.fit_transform(train["text_full"])
print("tfidf shape", X_tfidf.shape)

import scipy.sparse as sp
X_struct = train[STRUCT].copy()
# bools to float
for c in X_struct.columns:
    if X_struct[c].dtype==bool: X_struct[c]=X_struct[c].astype(float)
X_struct = X_struct.fillna(0).to_numpy(dtype=float)
# standardize struct? Ridge handles; scale by std
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_struct_s=scaler.fit_transform(X_struct)
X_all=sp.hstack([sp.csr_matrix(X_struct_s), X_tfidf]).tocsr()
y=train["gih_wr"].to_numpy()
w=np.sqrt(train["gih"].to_numpy())

# Leave-one-set-out CV with Ridge
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
groups=train["set"].to_numpy()
logo=LeaveOneGroupOut()
preds=np.zeros_like(y)
for tr,te in logo.split(X_all,y,groups):
    m=Ridge(alpha=30.0)
    m.fit(X_all[tr],y[tr],sample_weight=w[tr])
    preds[te]=m.predict(X_all[te])
rmse=np.sqrt(mean_squared_error(y,preds))
corr=np.corrcoef(y,preds)[0,1]
print(f"LOSO Ridge RMSE={rmse:.4f} corr={corr:.4f}")
# per-set corr
for s in sorted(train["set"].unique()):
    m=(train["set"]==s).to_numpy()
    c=np.corrcoef(y[m],preds[m])[0,1]
    r=np.sqrt(mean_squared_error(y[m],preds[m]))
    print(f"  {s}: n={(m).sum()} rmse={r:.4f} corr={c:.4f} mean_true={y[m].mean():.4f} mean_pred={preds[m].mean():.4f}")

# Save CV preds
train["cv_pred"]=preds
train.to_pickle(ANALYSIS/"train_cv.pkl")

# Final fit on all
final=Ridge(alpha=30.0)
final.fit(X_all,y,sample_weight=w)
print("final fit coef norm", np.linalg.norm(final.coef_))

# Predict TRK
trk=feat[feat["set"]=="TRK"].copy().reset_index(drop=True)
trk["text_full"]=(trk["type_line"].fillna("")+" "+trk["oracle_text"].fillna("")).str.lower()
trk["text_full"]=trk["text_full"].str.replace(r"\(.*?\)"," ",regex=True)
X_trk_tfidf=vec.transform(trk["text_full"])
X_trk_struct=trk[STRUCT].copy()
for c in X_trk_struct.columns:
    if X_trk_struct[c].dtype==bool: X_trk_struct[c]=X_trk_struct[c].astype(float)
X_trk_struct=X_trk_struct.fillna(0).to_numpy(dtype=float)
X_trk_struct_s=scaler.transform(X_trk_struct)
X_trk_all=sp.hstack([sp.csr_matrix(X_trk_struct_s), X_trk_tfidf]).tocsr()
trk_pred=final.predict(X_trk_all)
trk["pred_gih"]=trk_pred
trk[["name","rarity","mana_cost","type_line","pred_gih"]].sort_values("pred_gih",ascending=False).head(20).to_string()
print(trk[["name","rarity","mana_cost","type_line","pred_gih"]].sort_values("pred_gih",ascending=False).head(25).to_string())
print(trk[["name","rarity","mana_cost","type_line","pred_gih"]].sort_values("pred_gih").head(15).to_string())
trk.to_pickle(ANALYSIS/"trk_pred_raw.pkl")
# Save vectorizer/scaler/model for reuse
import pickle
pickle.dump({"vec":vec,"scaler":scaler,"struct_cols":STRUCT,"model":final}, open(ANALYSIS/"model_bundle.pkl","wb"))
print("saved")
