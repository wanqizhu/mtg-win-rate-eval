"""Check sensitivity to low-play cards and observational-noise weighting."""
import json,sys
sys.path.insert(0,'/workspace/model-packages')
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from model import load,OUT

def main():
    cards,m,f,docs=load();ix=np.where(m.gih_games.fillna(0)>=100)[0]
    allp=[]
    for group in list(sorted(m.iloc[ix]['set'].unique()))+['TRK']:
        tr=ix[m.iloc[ix]['set'].to_numpy()!=group]
        te=ix[m.iloc[ix]['set'].to_numpy()==group] if group!='TRK' else np.where(m['set']=='TRK')[0]
        base=m.iloc[tr].groupby('set').set_wr.first().mean()*100
        y=(m.iloc[tr].gih_wr-m.iloc[tr].set_wr)*100;n=m.iloc[tr].gih_games
        a=f.iloc[tr].to_numpy();b=f.iloc[te].to_numpy()
        result=m.iloc[te][['set','name','gih_games','gih_wr']].copy()
        for prior in [0,250,1000]:
            reg=CatBoostRegressor(iterations=1300,depth=5,learning_rate=.025,l2_leaf_reg=15,thread_count=3,verbose=False,allow_writing_files=False,random_seed=123)
            reg.fit(a,y,sample_weight=n/(n+prior));result[f'prior_{prior}']=reg.predict(b)+base
        # This auxiliary outcome measures realized deck inclusion, normalized
        # by total set games; it is always predicted from printed-card features.
        total={x['set']:x['games'] for x in json.loads((OUT/'historical_summary.json').read_text())}
        play=np.log(np.maximum(m.iloc[tr].deck_games,1).to_numpy()/np.array([total[s] for s in m.iloc[tr]['set']]))
        reg=CatBoostRegressor(iterations=1300,depth=5,learning_rate=.025,l2_leaf_reg=15,thread_count=3,verbose=False,allow_writing_files=False,random_seed=123)
        reg.fit(a,play);result['log_playability']=reg.predict(b)
        allp.append(result)
        print(group,{k:float(np.sqrt(np.mean((result[k]-result.gih_wr*100)**2))) for k in ['prior_0','prior_250','prior_1000']},flush=True)
    p=pd.concat(allp);p.to_csv(OUT/'weight_experiments.csv',index=False)

if __name__=='__main__':main()
