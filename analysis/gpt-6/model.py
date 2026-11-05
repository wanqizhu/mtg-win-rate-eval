"""Whole-set holdouts for rules-text and structured-feature models."""
from pathlib import Path
import json, os, sys
sys.path.insert(0,'/workspace/model-packages')
import numpy as np
import pandas as pd
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
from features import text, feature_matrix

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'

def load(extended=True):
    cards=json.loads((OUT/'cards.json').read_text())
    labels=pd.read_csv(OUT/'historical_labels.csv')
    known=set(zip(labels['set'],labels.name))
    aliases=[]
    for c in cards:
        if (c['set'],c['name']) not in known and c.get('card_faces'):
            front=c['card_faces'][0]['name']
            if (c['set'],front) in known:
                aliases.append({'set':c['set'],'oracle_name':c['name'],'label_name':front})
                c['oracle_name']=c['name'];c['name']=front
    pd.DataFrame(aliases).to_csv(OUT/'face_aliases.csv',index=False)
    meta=pd.DataFrame([{'set':c['set'],'name':c['name'],'rarity':c['rarity'],'type_line':c['type_line']} for c in cards])
    meta=meta.merge(labels,on=['set','name'],how='left')
    f=feature_matrix(cards,extended=extended)
    docs=[c['type_line'].lower()+' '+text(c) for c in cards]
    return cards,meta,f,docs

def run_fold(train,test,cards,meta,f,docs):
    # Each historical environment is centered on its overall game WR. Unknown
    # test-set population baseline is predicted from other historical sets.
    base=meta.iloc[train].groupby('set').set_wr.first().mean()
    y=(meta.iloc[train].gih_wr-meta.iloc[train].set_wr).to_numpy()*100
    n=meta.iloc[train].gih_games.to_numpy()
    w=n/(n+1000)
    a=f.iloc[train].to_numpy(); b=f.iloc[test].to_numpy()
    result={}
    result['baseline']=np.full(len(test),base*100)
    group=meta.iloc[train].assign(y=y).groupby('rarity').y.mean()
    result['rarity_baseline']=np.array([group.get(r,0)+base*100 for r in meta.iloc[test].rarity])
    vect=TfidfVectorizer(ngram_range=(1,3),min_df=2,max_features=24000,sublinear_tf=True,strip_accents='unicode',token_pattern=r'(?u)\b\w+\b')
    ta=vect.fit_transform([docs[i] for i in train]);tb=vect.transform([docs[i] for i in test])
    scaler=StandardScaler();na=scaler.fit_transform(a);nb=scaler.transform(b)
    for scale in [0.15,0.5]:
        xa=hstack([ta,csr_matrix(na*scale)]).tocsr();xb=hstack([tb,csr_matrix(nb*scale)]).tocsr()
        for alpha in [3,15,60]:
            model=Ridge(alpha=alpha,solver='lsqr').fit(xa,y,sample_weight=w)
            result[f'ridge_{scale}_{alpha}']=model.predict(xb)+base*100
    for depth,iters in [(4,1400),(6,1200)]:
        model=CatBoostRegressor(iterations=iters,depth=depth,learning_rate=.025,l2_leaf_reg=12,loss_function='RMSE',random_seed=42,thread_count=3,verbose=False,allow_writing_files=False)
        model.fit(a,y,sample_weight=w)
        result[f'cat_{depth}']=model.predict(b)+base*100
        if meta.iloc[test]['set'].isin(['TRK','SDS']).all():
            pd.DataFrame({'feature':f.columns,'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(OUT/f'importance_cat{depth}.csv',index=False)
    svd=TruncatedSVD(n_components=100,random_state=42)
    sa=svd.fit_transform(ta);sb=svd.transform(tb)
    model=CatBoostRegressor(iterations=1400,depth=5,learning_rate=.025,l2_leaf_reg=15,loss_function='RMSE',random_seed=19,thread_count=3,verbose=False,allow_writing_files=False)
    model.fit(np.column_stack([a,sa]),y,sample_weight=w)
    result['cat_text']=model.predict(np.column_stack([b,sb]))+base*100
    model=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=6,max_features=.8,n_jobs=3,random_state=42).fit(a,y,sample_weight=w)
    result['extra']=model.predict(b)+base*100
    return pd.DataFrame(result,index=test)

def main():
    legacy='--legacy' in sys.argv
    dest=OUT/'initial_model' if legacy else OUT
    dest.mkdir(exist_ok=True)
    cards,meta,f,docs=load(extended=not legacy)
    train_ix=np.where(meta.gih_games.fillna(0).to_numpy()>=100)[0]
    target_ix=np.where(meta.set.isin(['TRK','SDS']))[0]
    f.to_csv(OUT/'features.csv',index=False)
    meta.to_csv(OUT/'model_rows.csv',index=False)
    predictions=[]
    for s in sorted(meta.iloc[train_ix].set.unique()):
        tr=np.array([i for i in train_ix if meta.iloc[i]['set']!=s]);te=np.array([i for i in train_ix if meta.iloc[i]['set']==s])
        path=dest/f'cv_{s}.csv'
        if path.exists():p=pd.read_csv(path,index_col=0).rename(columns={'rarity':'rarity_baseline'})
        else:
            p=run_fold(tr,te,cards,meta,f,docs);p.to_csv(path)
        predictions.append(p)
        keep=meta.iloc[te].gih_games.to_numpy()>=500
        errors={k:round(float(np.sqrt(np.mean((p[k].to_numpy()[keep]-meta.iloc[te].gih_wr.to_numpy()[keep]*100)**2))),3) for k in p}
        print(s,len(tr),len(te),errors,flush=True)
    pred=pd.concat(predictions).sort_index()
    oof=meta.loc[pred.index].join(pred)
    oof.to_csv(dest/'cross_validation.csv',index=False)
    metrics=[]
    for k in pred:
        for scope in ['all','premier','n2000']:
            use=oof.gih_games>=500
            if scope=='premier':use &= oof['format']=='PremierDraft'
            if scope=='n2000':use &= oof.gih_games>=2000
            z=oof[use];e=z[k]-z.gih_wr*100
            metrics.append({'model':k,'scope':scope,'n':len(z),'rmse_pp':np.sqrt(np.mean(e**2)),'mae_pp':abs(e).mean(),'bias_pp':e.mean(),'correlation':z[k].corr(z.gih_wr)})
    pd.DataFrame(metrics).to_csv(dest/'validation_metrics.csv',index=False)
    print(pd.DataFrame(metrics).query('scope=="premier"').sort_values('rmse_pp').to_string(index=False),flush=True)
    p=run_fold(train_ix,target_ix,cards,meta,f,docs)
    meta.loc[p.index].join(p).to_csv(dest/'statistical_predictions.csv',index=False)

if __name__=='__main__':main()
