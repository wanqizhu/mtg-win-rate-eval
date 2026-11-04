#!/usr/bin/env python3
import json, numpy as np, sys
from pathlib import Path
sys.path.insert(0,"analysis")
from features import feature_vector, clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from scipy import sparse
from scipy.stats import pearsonr

data=json.loads(Path("analysis/historical_gih.json").read_text())
rows=[r for r in data["rows"] if r["gih"]>=100]
sets=sorted(set(r["set"] for r in rows))
base={s:data["report"][s]["base_wr"] for s in sets}
X=[]; names=None
for r in rows:
    v,names=feature_vector(r,names); X.append(v)
X=np.array(X,float); y=np.array([r["gih_wr"]-base[r["set"]] for r in rows])
w=np.array([min(r["gih"],5000) for r in rows],float)
texts=[clean_text(r) for r in rows]
vec=TfidfVectorizer(max_features=6000, ngram_range=(1,2), min_df=3, sublinear_tf=True)
Xt=vec.fit_transform(texts)
sc=StandardScaler(); Xs=sc.fit_transform(X)
# combined embedding: structured scaled *1.0 + tfidf *2.0
M=normalize(sparse.hstack([sparse.csr_matrix(Xs)*1.0, Xt*2.0]).tocsr())
sim=(M @ M.T).toarray()
np.fill_diagonal(sim,0)
setarr=np.array([r["set"] for r in rows])
pred=np.zeros(len(rows))
K=25
for i in range(len(rows)):
    mask=setarr!=setarr[i]
    idx=np.where(mask)[0]
    order=idx[np.argsort(-sim[i,idx])[:K]]
    s=np.maximum(sim[i,order],0)**3 * np.sqrt(w[order])
    pred[i]= np.sum(s*y[order])/(np.sum(s)+3.0)  # shrink to 0
def rmse(p): return float(np.sqrt(np.mean((p-y)**2)))
def wrmse(p): return float(np.sqrt(np.average((p-y)**2,weights=w)))
print(f"kNN K={K} RMSE={rmse(pred):.4f} w={wrmse(pred):.4f} r={pearsonr(pred,y).statistic:.3f}")
d=np.load("analysis/tune_preds.npz")
ens = (d['gbr14']+d['gbr15']+d['ridgeB1000_0.3'])/3
for a in [0.15,0.25,0.35,0.5]:
    p=(1-a)*ens+a*pred
    print(f"blend knn a={a} RMSE={rmse(p):.4f} w={wrmse(p):.4f} r={pearsonr(p,y).statistic:.3f}")
np.save("analysis/knn_preds.npy", pred)
