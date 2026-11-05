"""Leave-one-set-out validation and full-data predictions from supplied text only."""
from pathlib import Path
import argparse, json, os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostRegressor
from features import features, text_input
from prepare import OUT, SETS

def load():
    cards=[json.loads(x) for x in (OUT/'cards.jsonl').read_text().splitlines()]
    by_key={(c['set'],c['name']):c for c in cards}
    by_name={c['name']:c for c in cards if c['set'] not in ['TRK','SDS']}
    for c in cards:
        if c.get('printed_name') and c['set'] not in ['TRK','SDS']:
            by_key[(c['set'],c['printed_name'])]=c
            by_name[c['printed_name']]=c
    hist=pd.read_csv(OUT/'historical_all.csv')
    hist=hist[hist.gih_n>=400].copy()
    hist['card']= [by_key.get((s,n),by_name.get(n)) for s,n in zip(hist['set'],hist.name)]
    missing=hist[hist.card.isna()].drop(columns='card')
    missing.to_csv(OUT/'unmatched_historical.csv',index=False)
    hist=hist[hist.card.notna()].reset_index(drop=True)
    baselines={s:json.loads((OUT/f'summary_{s}.json').read_text())['win_rate']*100 for s in SETS}
    hist['set_baseline']=hist['set'].map(baselines)
    hist['y']=100*hist.gih_wr-hist.set_baseline+55.5
    hist['weight']=hist.gih_n/(hist.gih_n+1800)
    trk=[c for c in cards if c['set']=='TRK']
    return hist,trk,baselines

def matrix(train,test):
    dv=DictVectorizer();num=dv.fit_transform([features(c) for c in train]);vnum=dv.transform([features(c) for c in test])
    scaler=StandardScaler(with_mean=False);num=scaler.fit_transform(num);vnum=scaler.transform(vnum)
    tf=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.97,sublinear_tf=True,token_pattern=r'(?u)\b\w+\b')
    word=tf.fit_transform([text_input(c) for c in train]);vword=tf.transform([text_input(c) for c in test])
    char_tf=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=4,max_features=18000,sublinear_tf=True)
    char=char_tf.fit_transform([text_input(c) for c in train]);vchar=char_tf.transform([text_input(c) for c in test])
    return num,vnum,word,vword,char,vchar,dv,tf

def predict(train,test,y,weights,seed=17):
    num,vnum,word,vword,char,vchar,dv,tf=matrix(train,test)
    pred={}
    for a in [1,4,16,64]:
        x=sparse.hstack([num*.10,word,char*.65],format='csr');vx=sparse.hstack([vnum*.10,vword,vchar*.65],format='csr')
        reg=Ridge(alpha=a,solver='lsqr',tol=1e-6);reg.fit(x,y,sample_weight=weights)
        pred[f'ridge_{a}']=reg.predict(vx)
    for a in [20,100]:
        reg=Ridge(alpha=a,solver='lsqr',tol=1e-6);reg.fit(num,y,sample_weight=weights)
        pred[f'meta_{a}']=reg.predict(vnum)
    svd=TruncatedSVD(n_components=100,random_state=seed)
    z=svd.fit_transform(word);vz=svd.transform(vword)
    x=np.column_stack([num.toarray(),z]);vx=np.column_stack([vnum.toarray(),vz])
    for depth in [4,6]:
        cb=CatBoostRegressor(iterations=1800,depth=depth,learning_rate=.022,l2_leaf_reg=8,loss_function='RMSE',
                             random_seed=seed,verbose=False,thread_count=4,allow_writing_files=False)
        cb.fit(x,y,sample_weight=weights)
        pred[f'cat_{depth}']=cb.predict(vx)
    # Similar text reprints/analogues are especially informative if available.
    sim=(vword@word.T).toarray()
    for k in [3,10]:
        order=np.argsort(-sim,axis=1)[:,:k];w=np.take_along_axis(sim,order,axis=1)**5
        base=np.average(y,weights=weights)
        pred[f'knn_{k}']=(np.sum(w*y[order],axis=1)+base*.15)/(w.sum(axis=1)+.15)
    return pred

def metrics(frame,cols):
    out=[]
    for col in cols:
        for s in ['ALL']+SETS:
            d=frame if s=='ALL' else frame[frame['set']==s]
            if not len(d):continue
            e=d[col]-d.y
            out.append({'model':col,'set':s,'n':len(d),'rmse':np.sqrt(np.mean(e**2)),
                        'mae':np.mean(abs(e)), 'bias':np.mean(e),
                        'weighted_rmse':np.sqrt(np.average(e**2,weights=d.weight)),
                        'correlation':d[[col,'y']].corr().iloc[0,1]})
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',nargs='*',default=SETS);ap.add_argument('--full',action='store_true');args=ap.parse_args()
    h,target,baselines=load()
    print('Historical training rows:',len(h),'Set baselines:',baselines,flush=True)
    for s in args.folds:
        tr=h[h['set']!=s];va=h[h['set']==s]
        pred=predict(tr.card.tolist(),va.card.tolist(),tr.y.to_numpy(),tr.weight.to_numpy())
        out=va.drop(columns='card').copy()
        for name,p in pred.items():out[name]=p
        out.to_csv(OUT/f'validation_{s}.csv',index=False)
        m=metrics(out,list(pred));print(s,m[m['set']=='ALL'][['model','rmse','mae','bias']].round(3).to_string(index=False),flush=True)
    vals=[pd.read_csv(OUT/f'validation_{s}.csv') for s in SETS if (OUT/f'validation_{s}.csv').exists()]
    if vals:
        val=pd.concat(vals,ignore_index=True);val.to_csv(OUT/'validation_all.csv',index=False)
        cols=[x for x in val if x.startswith(('ridge_','meta_','cat_','knn_'))]
        metrics(val,cols).to_csv(OUT/'validation_metrics.csv',index=False)
    if args.full:
        pred=predict(h.card.tolist(),target,h.y.to_numpy(),h.weight.to_numpy())
        out=pd.DataFrame({'name':[c['name'] for c in target]})
        for name,p in pred.items():out[name]=p
        out.to_csv(OUT/'model_predictions.csv',index=False)
        print('Saved full-data target predictions',flush=True)

if __name__=='__main__':main()
