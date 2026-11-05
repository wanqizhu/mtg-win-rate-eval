#!/usr/bin/env python3
"""Final TRK (+SDS bonus sheet, +basics) GIH WR predictions.
Ensemble: 2x HistGBR (structured) + Ridge(structured+TF-IDF text) + analog kNN,
trained on 9 historical Premier Draft sets (gih>=100), target centered per set.
Then: linear recalibration (slope from LOSO), exact-reprint anchor, manual adjustments."""
import json, gzip, re, sys
import numpy as np
from pathlib import Path
sys.path.insert(0,"analysis")
from features import feature_vector, clean_text
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from scipy import sparse

BASE_TRK = 0.556  # mean historical Premier base win rate (0.5499-0.5680, mean 0.5563)

data=json.loads(Path("analysis/historical_gih.json").read_text())
rows=[r for r in data["rows"] if r["gih"]>=100]
sets=sorted(set(r["set"] for r in rows)); base={s:data["report"][s]["base_wr"] for s in sets}
X=[]; names=None
for r in rows:
    v,names=feature_vector(r,names); X.append(v)
X=np.array(X,float); y=np.array([r["gih_wr"]-base[r["set"]] for r in rows])
w=np.array([min(r["gih"],5000) for r in rows],float)
texts=[clean_text(r) for r in rows]

# ---------- target pools ----------
def cnum(c):
    try: return int(c["collector_number"])
    except Exception: return 99999
def load(p):
    out=[]
    with gzip.open(p,"rt") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out
def rec_from_scryfall(c, rarity_override=None):
    return {"name":c["name"].split(" // ")[0],"rarity":rarity_override or c.get("rarity"),
            "mana_cost":c.get("mana_cost") or "","cmc":c.get("cmc"),
            "type_line":c.get("type_line") or "","oracle_text":c.get("oracle_text") or "",
            "power":c.get("power"),"toughness":c.get("toughness"),
            "colors":"".join(c.get("colors") or []),"color_identity":"".join(c.get("color_identity") or []),
            "keywords":"|".join(c.get("keywords") or []),"collector_number":c.get("collector_number")}
targets=[]; seen=set()
for c in sorted(load("/opt/mtg-data/scryfall/sets/trk.jsonl.gz"), key=cnum):
    if cnum(c)<=306 and c["name"] not in seen:
        seen.add(c["name"]); r=rec_from_scryfall(c); r["pool"]="TRK"; targets.append(r)
for nm in ["Plains","Island","Swamp","Mountain","Forest"]:
    targets.append({"name":nm,"rarity":"common","mana_cost":"","cmc":0,"type_line":f"Basic Land — {nm}",
                    "oracle_text":"({T}: Add one mana.)","power":None,"toughness":None,"colors":"",
                    "color_identity":"","keywords":"","pool":"TRK-basic"})
for c in sorted(load("/opt/mtg-data/scryfall/sets/sds.jsonl.gz"), key=lambda z: z["name"]):
    r=rec_from_scryfall(c); r["pool"]="SDS"; targets.append(r)
print("targets", len(targets))

Xt=[]; 
for r in targets:
    v,_=feature_vector(r,names); Xt.append(v)
Xt=np.array(Xt,float); ttexts=[clean_text(r) for r in targets]

# ---------- models ----------
gbr1=HistGradientBoostingRegressor(max_iter=600,learning_rate=0.03,max_depth=3,l2_regularization=3.0,min_samples_leaf=20,random_state=0).fit(X,y,sample_weight=w)
gbr2=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,l2_regularization=1.0,min_samples_leaf=20,random_state=0).fit(X,y,sample_weight=w)
vec=TfidfVectorizer(max_features=4000, ngram_range=(1,2), min_df=3, sublinear_tf=True)
A=vec.fit_transform(texts); sc=StandardScaler(); Xs=sc.fit_transform(X)
ridge=Ridge(alpha=1000.0); ridge.fit(sparse.hstack([sparse.csr_matrix(Xs), A*0.3]), y, sample_weight=w)

p_gbr=(gbr1.predict(Xt)+gbr2.predict(Xt))/2
p_ridge=ridge.predict(sparse.hstack([sparse.csr_matrix(sc.transform(Xt)), vec.transform(ttexts)*0.3]))

# kNN analog
Mtr=normalize(sparse.hstack([sparse.csr_matrix(Xs)*1.0, A*2.0]).tocsr())
Mte=normalize(sparse.hstack([sparse.csr_matrix(sc.transform(Xt))*1.0, vec.transform(ttexts)*2.0]).tocsr())
# NOTE: vec was fit on 4000 feats for ridge; refit a 6000-feat vec for knn would mismatch A dims.
sim=(Mte @ Mtr.T).toarray()
p_knn=np.zeros(len(targets)); neighbors=[]
for i in range(len(targets)):
    order=np.argsort(-sim[i])[:25]
    s=np.maximum(sim[i,order],0)**3*np.sqrt(w[order])
    p_knn[i]=np.sum(s*y[order])/(np.sum(s)+3.0)
    neighbors.append([(rows[j]["set"],rows[j]["name"],round(float(y[j]),3),round(float(sim[i,j]),2)) for j in order[:5]])

ens=(p_gbr+p_ridge)/2*0.8 + p_knn*0.2
# wait: ensemble intended = mean(gbr1,gbr2,ridge) 0.8 + knn 0.2
ens = ((gbr1.predict(Xt)+gbr2.predict(Xt)+p_ridge)/3)*0.8 + p_knn*0.2

# recalibration from LOSO (slope 1.14, intercept -0.005)
centered = 1.10*ens - 0.002
# category micro-bias from LOSO residuals: unweighted per-card residuals show mythic/rare/
# spacecraft bumps are not supported (sign flips vs play-weighted residuals), so only the
# saga correction (+0.004 unweighted, +0.008 weighted; halved) is retained.
for i,r in enumerate(targets):
    if "Saga" in r["type_line"]: centered[i]+=0.002

# exact reprint anchor
from collections import defaultdict
hist=defaultdict(list)
for r in rows: hist[r["name"]].append(r)
anchor_info={}
for i,r in enumerate(targets):
    if r["name"] in hist:
        hs=hist[r["name"]]; tot=sum(h["gih"] for h in hs)
        exp=sum(h["gih"]*base[h["set"]] for h in hs); wins=sum(h["gih_wins"] for h in hs)
        anch=(wins-exp)/tot
        aw = 0.65*tot/(tot+1200)
        centered[i]=(1-aw)*centered[i]+aw*anch
        anchor_info[r["name"]]={"gih":tot,"centered_hist":round(float(anch),4),"weight":round(float(aw),3)}

pred = BASE_TRK + centered
out=[]
for i,r in enumerate(targets):
    out.append({**{k:r.get(k) for k in ("name","rarity","mana_cost","type_line","pool","collector_number")},
                "model_gih_wr": round(float(pred[i]),4),
                "p_gbr":round(float(BASE_TRK+p_gbr[i]),4),"p_ridge":round(float(BASE_TRK+p_ridge[i]),4),
                "p_knn":round(float(BASE_TRK+p_knn[i]),4),
                "analogs":neighbors[i],"anchor":anchor_info.get(r["name"])})
Path("analysis/trk_model_predictions.json").write_text(json.dumps(out,indent=1))
out_sorted=sorted(out,key=lambda z:-z["model_gih_wr"])
for r in out_sorted[:40]: print(f"{r['model_gih_wr']:.3f} [{r['pool']}] {r['name']} ({r['rarity']}) gbr={r['p_gbr']:.3f} ridge={r['p_ridge']:.3f} knn={r['p_knn']:.3f}")
print("...")
for r in out_sorted[-40:]: print(f"{r['model_gih_wr']:.3f} [{r['pool']}] {r['name']} ({r['rarity']}) gbr={r['p_gbr']:.3f} ridge={r['p_ridge']:.3f} knn={r['p_knn']:.3f}")
