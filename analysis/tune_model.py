#!/usr/bin/env python3
import json, numpy as np, sys
from pathlib import Path
sys.path.insert(0,"analysis")
from features import feature_vector, clean_text
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import sparse
from scipy.stats import pearsonr

data=json.loads(Path("analysis/historical_gih.json").read_text())
rows=[r for r in data["rows"] if r["gih"]>=100]
sets=sorted(set(r["set"] for r in rows))
base={s:data["report"][s]["base_wr"] for s in sets}
X=[]; names=None
for r in rows:
    v,names=feature_vector(r,names); X.append(v)
X=np.array(X); y=np.array([r["gih_wr"]-base[r["set"]] for r in rows])
w=np.array([min(r["gih"],5000) for r in rows],dtype=float)
texts=[clean_text(r) for r in rows]
set_idx={s:[i for i,r in enumerate(rows) if r["set"]==s] for s in sets}

def rmse(p): return float(np.sqrt(np.mean((p-y)**2)))
def wrmse(p): return float(np.sqrt(np.average((p-y)**2,weights=w)))

def loso(make_fit_predict):
    p=np.zeros(len(rows))
    for s in sets:
        te=set_idx[s]; tr=[i for i in range(len(rows)) if i not in set(te)]
        p[te]=make_fit_predict(tr,te)
    return p

# cache tfidf full vocab fit per fold inside functions
def ridge_struct(alpha):
    def fp(tr,te):
        sc=StandardScaler(); m=Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr], sample_weight=w[tr])
        return m.predict(sc.transform(X[te]))
    return fp
def ridge_text(alpha, maxf=4000, scale=1.0):
    def fp(tr,te):
        vec=TfidfVectorizer(max_features=maxf, ngram_range=(1,2), min_df=3, sublinear_tf=True)
        A=vec.fit_transform([texts[i] for i in tr]); B=vec.transform([texts[i] for i in te])
        m=Ridge(alpha=alpha); m.fit(A*scale, y[tr], sample_weight=w[tr]); return m.predict(B*scale)
    return fp
def ridge_both(alpha, scale):
    def fp(tr,te):
        vec=TfidfVectorizer(max_features=4000, ngram_range=(1,2), min_df=3, sublinear_tf=True)
        A=vec.fit_transform([texts[i] for i in tr]); B=vec.transform([texts[i] for i in te])
        sc=StandardScaler(); Xs=sc.fit_transform(X[tr]); Xt=sc.transform(X[te])
        M=sparse.hstack([sparse.csr_matrix(Xs), A*scale]); Mt=sparse.hstack([sparse.csr_matrix(Xt), B*scale])
        m=Ridge(alpha=alpha); m.fit(M, y[tr], sample_weight=w[tr]); return m.predict(Mt)
    return fp

results={}
for a in [3,10,30,100,300]:
    p=loso(ridge_struct(a)); results[f"ridgeS{a}"]=p; print(f"ridgeS a={a} RMSE={rmse(p):.4f} w={wrmse(p):.4f} r={pearsonr(p,y).statistic:.3f}",flush=True)
for a in [100,500,2000,8000]:
    p=loso(ridge_text(a)); results[f"ridgeT{a}"]=p; print(f"ridgeT a={a} RMSE={rmse(p):.4f} w={wrmse(p):.4f} r={pearsonr(p,y).statistic:.3f}",flush=True)
for a,sc in [(100,0.3),(300,0.3),(100,1.0),(1000,0.3)]:
    p=loso(ridge_both(a,sc)); results[f"ridgeB{a}_{sc}"]=p; print(f"ridgeB a={a} scale={sc} RMSE={rmse(p):.4f} w={wrmse(p):.4f} r={pearsonr(p,y).statistic:.3f}",flush=True)

def gbr(**kw):
    def fp(tr,te):
        m=HistGradientBoostingRegressor(random_state=0, **kw); m.fit(X[tr],y[tr],sample_weight=w[tr]); return m.predict(X[te])
    return fp
for kw in [dict(max_iter=300,learning_rate=0.04,max_depth=2,l2_regularization=1.0,min_samples_leaf=30),
           dict(max_iter=600,learning_rate=0.03,max_depth=3,l2_regularization=3.0,min_samples_leaf=20),
           dict(max_iter=300,learning_rate=0.05,max_depth=None,max_leaf_nodes=15,l2_regularization=1.0,min_samples_leaf=20)]:
    p=loso(gbr(**kw)); results[f"gbr{len(results)}"]=p; print(f"gbr {kw} RMSE={rmse(p):.4f} w={wrmse(p):.4f} r={pearsonr(p,y).statistic:.3f}",flush=True)

np.savez("analysis/tune_preds.npz", y=y,w=w, **results,
         names=np.array([r["set"]+"/"+r["name"] for r in rows]))
# try blends of best few by wrmse
ranked=sorted(results.items(), key=lambda kv: wrmse(kv[1]))
print("ranked by wRMSE:", [(k,round(wrmse(v),4)) for k,v in ranked])
top=[v for k,v in ranked[:4]]
for weights in [(0.5,0.5),(0.4,0.3,0.3),(0.34,0.33,0.33),(0.4,0.2,0.2,0.2),(0.25,0.25,0.25,0.25)]:
    ws=weights[:len(top)]; ws=np.array(ws)/sum(ws)
    p=sum(wi*vi for wi,vi in zip(ws,top))
    print("blend",ws,"of",[k for k,v in ranked[:len(ws)]],f"RMSE={rmse(p):.4f} w={wrmse(p):.4f} r={pearsonr(p,y).statistic:.3f}")
