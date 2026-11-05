"""Validate historical analogues, including exact-text reprints, by set holdout."""
from pathlib import Path
import json,itertools
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from model import load,OUT
from features import text

def main():
    cards,meta,f,docs=load()
    # This vocabulary is unsupervised; it includes supplied card text only.
    v=TfidfVectorizer(ngram_range=(1,3),min_df=1,sublinear_tf=True,token_pattern=r'(?u)\b\w+\b')
    z=v.fit_transform([text(c) for c in cards]); sim=cosine_similarity(z)
    # Match actual casting and board role; rule text alone mistakes large creatures
    # for small ones and sorcery speed for instant speed.
    mat=f.to_numpy();cols=list(f.columns)
    a=f.mv.to_numpy();p=f.power.to_numpy();t=f.toughness.to_numpy()
    penalty=.045*np.minimum(abs(a[:,None]-a[None,:]),6)
    penalty+=.018*np.minimum(abs(p[:,None]-p[None,:])+abs(t[:,None]-t[None,:]),10)
    for key in ['type_creature','type_land','type_spacecraft','type_equipment','type_aura']:
        x=f[key].to_numpy();penalty+=.15*abs(x[:,None]-x[None,:])
    adjusted=np.clip(sim-penalty,0,1)
    train=np.where(meta.gih_games.fillna(0)>=100)[0]
    target=np.where(meta['set']=='TRK')[0]
    configs=list(itertools.product([3,7,15],[2,4,8]))
    out=[];neighbors=[]
    for ix in np.r_[train,target]:
        available=train[meta.iloc[train]['set'].to_numpy()!=meta.iloc[ix]['set']]
        baseline=meta.iloc[available].groupby('set').set_wr.first().mean()*100
        y=(meta.iloc[available].gih_wr-meta.iloc[available].set_wr).to_numpy()*100
        n=meta.iloc[available].gih_games.to_numpy();quality=n/(n+1000)
        row={'set':meta.iloc[ix]['set'],'name':meta.iloc[ix]['name'],'index':ix}
        for mode,scores in [('text',sim[ix,available]),('role',adjusted[ix,available])]:
            order=np.argsort(scores)[::-1]
            for k,power in configs:
                take=order[:k];w=(scores[take]**power)*quality[take]+1e-10
                # Small prior weight protects against distant or low-count analogues.
                row[f'{mode}_{k}_{power}']=baseline+(w@y[take])/(w.sum()+.03)
            if mode=='role':
                for rank,j in enumerate(order[:8]):
                    c=meta.iloc[available[j]]
                    neighbors.append({'set':row['set'],'name':row['name'],'rank':rank+1,'analogue_set':c['set'],'analogue':c['name'],'similarity':scores[j],'gih_wr':c.gih_wr,'gih_games':c.gih_games,'relative_wr_pp':y[j]})
        out.append(row)
    pred=pd.DataFrame(out);pred.to_csv(OUT/'analogue_predictions.csv',index=False)
    pd.DataFrame(neighbors).to_csv(OUT/'historical_analogues.csv',index=False)
    q=pred.merge(meta[['set','name','gih_games','gih_wr']],on=['set','name'])
    q=q[(q.gih_games>=500)&(q['set']!='OM1')]
    metrics=[]
    for k in pred.columns[3:]:
        metrics.append({'model':k,'rmse_pp':np.sqrt(np.mean((q[k]-q.gih_wr*100)**2)),'mae_pp':abs(q[k]-q.gih_wr*100).mean()})
    pd.DataFrame(metrics).sort_values('rmse_pp').to_csv(OUT/'analogue_metrics.csv',index=False)
    print(pd.DataFrame(metrics).sort_values('rmse_pp').to_string(index=False))

if __name__=='__main__':main()
