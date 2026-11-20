"""Final ensemble: Ridge(100) on struct+tfidf + GBM on struct. Calibrate per rarity. Predict TRK."""
import json, pickle, re
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error

ANALYSIS=Path("/agent/repos/mtg-win-rate-eval/analysis")
gih=json.loads((ANALYSIS/"gih_by_set.json").read_text())
feat=pd.read_pickle(ANALYSIS/"features.pkl")

def add_engineered(df):
    df=df.copy()
    df["is_basic_land"]=(df["is_basic"]).astype(float)
    df["is_nonbasic_land"]=(df["is_land"] & ~df["is_basic"]).astype(float)
    df["is_gainland"]=((df["is_land"]) & (df["oracle_text"].str.contains("enters tapped",na=False)) & (df["oracle_text"].str.contains("you gain 1 life",na=False))).astype(float)
    df["pow_def"]=df["power"]-df["toughness"]
    df["is_wall"]=((df["is_creature"]) & (df["power"]<=1)).astype(float)
    df["body_total"]=np.where(df["is_creature"],df["pt_sum"],0)
    df["cheap"]=((df["cmc"]<=2)&(~df["is_land"])).astype(float)
    df["expensive"]=((df["cmc"]>=6)&(~df["is_land"])).astype(float)
    df["removal"]=((df["destroys"])|(df["exiles"])).astype(float)
    df["card_adv"]=(df["n_draw_trig"]>0).astype(float)
    df["token_maker"]=df["makes_token"].astype(float)
    df["evasion"]=((df["kw_flying"]>0)|(df["kw_menace"]>0)|(df["kw_trample"]>0)|(df["kw_haste"]>0)).astype(float)
    df["bad_station"]=(df["station_thresh"]>=8).astype(float)
    df["good_station"]=((df["station_thresh"]>0)&(df["station_thresh"]<=6)).astype(float)
    df["etb_draw"]=((df["etb"]>0)&(df["n_draw_trig"]>0)).astype(float)
    df["x_big"]=((df["cmc"]>=7)).astype(float)
    return df
feat=add_engineered(feat)
STRUCT=[c for c in feat.columns if c not in ("name","set","oracle_text","type_line","mana_cost","colors","rarity")]

rows=[]
for code,res in gih.items():
    if code=="OM1": continue
    fsub=feat[feat["set"]==code].set_index("name")
    for c in res["cards"]:
        if c["card"] in fsub.index:
            f=fsub.loc[c["card"]]
            if isinstance(f,pd.DataFrame): f=f.iloc[0]
            rows.append({"set":code,"name":c["card"],"gih":c["gih"],"gih_wr":c["gih_wr"],"rarity":f["rarity"],**{k:f[k] for k in STRUCT}})
train=pd.DataFrame(rows); train=train[train["gih"]>=100].copy()
aux=feat[["set","name","oracle_text","type_line"]].drop_duplicates()
train=train.merge(aux,on=["set","name"],how="left")
train["text_full"]=(train["type_line"].fillna("")+" "+train["oracle_text"].fillna("")).str.lower().str.replace(r"\(.*?\)"," ",regex=True)

vec=TfidfVectorizer(max_features=1200,ngram_range=(1,2),min_df=8,stop_words="english")
X_tfidf=vec.fit_transform(train["text_full"])
Xs=train[STRUCT].copy()
for c in Xs.columns:
    if Xs[c].dtype==bool: Xs[c]=Xs[c].astype(float)
Xs=Xs.fillna(0).to_numpy(float)
scaler=StandardScaler(); Xs_s=scaler.fit_transform(Xs)
X_all=sp.hstack([sp.csr_matrix(Xs_s),X_tfidf]).tocsr()
y=train["gih_wr"].to_numpy(); w=np.sqrt(train["gih"].to_numpy()); groups=train["set"].to_numpy()

# LOSO for ridge100 + gbm
logo=LeaveOneGroupOut()
rp=np.zeros_like(y); gp=np.zeros_like(y)
for tr,te in logo.split(X_all,y,groups):
    m=Ridge(alpha=100); m.fit(X_all[tr],y[tr],sample_weight=w[tr]); rp[te]=m.predict(X_all[te])
for tr,te in logo.split(Xs,y,groups):
    g=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_depth=3,min_samples_leaf=30,l2_regularization=5.0,random_state=0)
    g.fit(Xs[tr],y[tr],sample_weight=w[tr]); gp[te]=g.predict(Xs[te])
ens=0.5*rp+0.5*gp
print(f"ridge100 RMSE={np.sqrt(mean_squared_error(y,rp)):.4f} corr={np.corrcoef(y,rp)[0,1]:.4f}")
print(f"gbm RMSE={np.sqrt(mean_squared_error(y,gp)):.4f} corr={np.corrcoef(y,gp)[0,1]:.4f}")
print(f"ens RMSE={np.sqrt(mean_squared_error(y,ens)):.4f} corr={np.corrcoef(y,ens)[0,1]:.4f}")

# calibration per rarity: regress true on ens
train["ens_cv"]=ens
import pandas as pd
cal={}
for rar in ["common","uncommon","rare","mythic"]:
    m=train["rarity"]==rar
    # slope/intercept of true ~ ens (weighted)
    X=np.column_stack([np.ones(m.sum()),ens[m]])
    W=np.sqrt(train.loc[m,"gih"].to_numpy())
    XtW=X*W[:,None]; yt=y[m]*W
    beta=np.linalg.lstsq(XtW,yt,rcond=None)[0]
    cal[rar]=beta
    pred_cv=beta[0]+beta[1]*ens[m]
    print(f"{rar}: n={m.sum()} intercept={beta[0]:.4f} slope={beta[1]:.4f} resid_rmse={np.sqrt(((((y[m]-pred_cv)**2)*W**2).sum()/(W**2).sum())):.4f}")
print(cal)

# Final fit on all data
ridge=Ridge(alpha=100); ridge.fit(X_all,y,sample_weight=w)
gbm=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_depth=3,min_samples_leaf=30,l2_regularization=5.0,random_state=0)
gbm.fit(Xs,y,sample_weight=w)

# Predict TRK
trk=feat[feat["set"]=="TRK"].copy().reset_index(drop=True)
trk=trk.merge(feat[feat["set"]=="TRK"][["name","oracle_text","type_line"]],on="name",how="left",suffixes=("","_y"))
trk["text_full"]=(trk["type_line"].fillna("")+" "+trk["oracle_text"].fillna("")).str.lower().str.replace(r"\(.*?\)"," ",regex=True)
Xt=vec.transform(trk["text_full"])
Xst=trk[STRUCT].copy()
for c in Xst.columns:
    if Xst[c].dtype==bool: Xst[c]=Xst[c].astype(float)
Xst=Xst.fillna(0).to_numpy(float); Xst_s=scaler.transform(Xst)
X_all_t=sp.hstack([sp.csr_matrix(Xst_s),Xt]).tocsr()
trk["ridge"]=ridge.predict(X_all_t); trk["gbm"]=gbm.predict(Xst)
trk["ens_raw"]=0.5*trk["ridge"]+0.5*trk["gbm"]
# apply rarity calibration
def calib(row):
    b=cal[row["rarity"]]
    return b[0]+b[1]*row["ens_raw"]
trk["pred_calibrated"]=trk.apply(calib,axis=1)
trk.to_pickle(ANALYSIS/"trk_pred_v2.pkl")
print(trk[["name","rarity","mana_cost","pred_calibrated"]].sort_values("pred_calibrated",ascending=False).head(30).to_string())
print("---bottom 20---")
print(trk[["name","rarity","mana_cost","pred_calibrated"]].sort_values("pred_calibrated").head(20).to_string())
print("---by rarity stats---")
print(trk.groupby("rarity")["pred_calibrated"].describe())
pickle.dump({"vec":vec,"scaler":scaler,"STRUCT":STRUCT,"ridge":ridge,"gbm":gbm,"cal":cal},open(ANALYSIS/"model_v2.pkl","wb"))
