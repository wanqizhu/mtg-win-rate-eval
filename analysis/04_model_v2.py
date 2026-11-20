"""V2: drop OM1, richer features, Ridge alpha tuning, GBM ensemble, calibration."""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error

ANALYSIS = Path("/agent/repos/mtg-win-rate-eval/analysis")
gih = json.loads((ANALYSIS/"gih_by_set.json").read_text())
feat = pd.read_pickle(ANALYSIS/"features.pkl")

# Extra engineered features (computed from base cols + text)
def add_engineered(df):
    df=df.copy()
    df["is_basic_land"]=(df["is_basic"]).astype(float)
    df["is_nonbasic_land"]=(df["is_land"] & ~df["is_basic"]).astype(float)
    # shocklands: rare nonbasic land with two basic types? approximate: rare land with text "Shock"? use type_line contains two basics
    df["is_gainland"]=((df["is_land"]) & (df["text_len"]>0) & (df["oracle_text"].str.contains("enters tapped",na=False)) & (df["oracle_text"].str.contains("you gain 1 life",na=False))).astype(float)
    df["pow_def"]=df["power"]-df["toughness"]
    df["is_wall"]=((df["is_creature"]) & (df["power"]<=1)).astype(float)
    df["body_total"]=np.where(df["is_creature"], df["pt_sum"], 0)
    df["cheap"]=((df["cmc"]<=2) & (~df["is_land"])).astype(float)
    df["expensive"]=((df["cmc"]>=6) & (~df["is_land"])).astype(float)
    df["removal"]=((df["destroys"])|(df["exiles"])).astype(float)
    df["card_adv"]=(df["n_draw_trig"]>0).astype(float)
    df["token_maker"]=df["makes_token"].astype(float)
    df["evasion"]=((df["kw_flying"]>0)|(df["kw_menace"]>0)|(df["kw_trample"]>0)|(df["kw_haste"]>0)|(df["kw_unblockable"] if "kw_unblockable" in df.columns else 0)>0).astype(float) if "kw_unblockable" in df.columns else ((df["kw_flying"]>0)|(df["kw_menace"]>0)|(df["kw_trample"]>0)|(df["kw_haste"]>0)).astype(float)
    df["bad_station"]=(df["station_thresh"]>=8).astype(float)
    df["good_station"] = ((df["station_thresh"]>0)&(df["station_thresh"]<=6)).astype(float)
    df["etb_draw"]=((df["etb"]>0)&(df["n_draw_trig"]>0)).astype(float)
    df["x_big"]=((df["cmc"]>=7)).astype(float)
    return df

feat=add_engineered(feat)

STRUCT = [c for c in feat.columns if c not in ("name","set","oracle_text","type_line","mana_cost","colors","rarity")]
print("n struct:",len(STRUCT))

# Build training (drop OM1)
rows=[]
for code,res in gih.items():
    if code=="OM1": continue
    fsub=feat[feat["set"]==code].set_index("name")
    for c in res["cards"]:
        name=c["card"]
        if name in fsub.index:
            f=fsub.loc[name]
            if isinstance(f,pd.DataFrame): f=f.iloc[0]
            rows.append({"set":code,"name":name,"gih":c["gih"],"gih_wr":c["gih_wr"], **{k:f[k] for k in STRUCT}})
train=pd.DataFrame(rows)
train=train[train["gih"]>=100].copy()
print("train",train.shape, train["set"].unique())

train["text_full"]=(train["type_line"].fillna("")+" "+train["oracle_text"].fillna("")).str.lower() if "oracle_text" in train.columns else ""
# need oracle/type: merge from feat
aux=feat[["set","name","oracle_text","type_line"]].drop_duplicates()
train=train.merge(aux,on=["set","name"],how="left")
train["text_full"]=(train["type_line"].fillna("")+" "+train["oracle_text"].fillna("")).str.lower()
train["text_full"]=train["text_full"].str.replace(r"\(.*?\)"," ",regex=True)

vec=TfidfVectorizer(max_features=1200,ngram_range=(1,2),min_df=8,stop_words="english")
X_tfidf=vec.fit_transform(train["text_full"])
X_struct=train[STRUCT].copy()
for c in X_struct.columns:
    if X_struct[c].dtype==bool: X_struct[c]=X_struct[c].astype(float)
X_struct=X_struct.fillna(0).to_numpy(float)
scaler=StandardScaler()
X_struct_s=scaler.fit_transform(X_struct)
X_all=sp.hstack([sp.csr_matrix(X_struct_s),X_tfidf]).tocsr()
y=train["gih_wr"].to_numpy(); w=np.sqrt(train["gih"].to_numpy()); groups=train["set"].to_numpy()

# alpha tuning via LOSO
for alpha in [3,10,30,100]:
    logo=LeaveOneGroupOut()
    preds=np.zeros_like(y)
    for tr,te in logo.split(X_all,y,groups):
        m=Ridge(alpha=alpha); m.fit(X_all[tr],y[tr],sample_weight=w[tr]); preds[te]=m.predict(X_all[te])
    rmse=np.sqrt(mean_squared_error(y,preds)); corr=np.corrcoef(y,preds)[0,1]
    print(f"Ridge alpha={alpha}: RMSE={rmse:.4f} corr={corr:.4f}")

# GBM on structured only (LOSO)
logo=LeaveOneGroupOut()
gpreds=np.zeros_like(y)
for tr,te in logo.split(X_struct,y,groups):
    g=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_depth=3,min_samples_leaf=30,l2_regularization=5.0,random_state=0)
    g.fit(X_struct[tr],y[tr],sample_weight=w[tr])
    gpreds[te]=g.predict(X_struct[te])
print(f"GBM struct-only: RMSE={np.sqrt(mean_squared_error(y,gpreds)):.4f} corr={np.corrcoef(y,gpreds)[0,1]:.4f}")

# Ensemble ridge(10)+gbm
alpha=10
logo=LeaveOneGroupOut()
rpreds=np.zeros_like(y)
for tr,te in logo.split(X_all,y,groups):
    m=Ridge(alpha=alpha); m.fit(X_all[tr],y[tr],sample_weight=w[tr]); rpreds[te]=m.predict(X_all[te])
for lam in [0.5,0.6,0.7,0.8]:
    ens=lam*rpreds+(1-lam)*gpreds
    print(f"ens lam={lam}: RMSE={np.sqrt(mean_squared_error(y,ens)):.4f} corr={np.corrcoef(y,ens)[0,1]:.4f}")

train["ridge_pred"]=rpreds; train["gbm_pred"]=gpreds
train.to_pickle(ANALYSIS/"train_cv_v2.pkl")
