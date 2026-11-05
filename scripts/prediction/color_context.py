"""Test whether predicted common quality forecasts archetype strength across sets."""
import itertools,json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from features import features
from environment import fits
from prepare import OUT,SETS

def main():
    cards=[json.loads(x) for x in (OUT/'cards.jsonl').read_text().splitlines()]
    v=pd.read_csv(OUT/'validation_all.csv');p=pd.read_csv(OUT/'model_predictions.csv');p['set']='TRK'
    vals=pd.concat([v,p],ignore_index=True)
    vals['prediction']=.4*vals.ridge_1+.25*vals.ridge_4+.175*vals.cat_4+.175*vals.cat_6
    lookup={(s,n):x for s,n,x in zip(vals['set'],vals.name,vals.prediction)}
    rows=[]
    for s in SETS+['TRK']:
        sc=[c for c in cards if c['set']==s and 'Land' not in c.get('type_line',c.get('card_faces',[{}])[0].get('type_line',''))]
        for c in sc:c['pred']=lookup.get((s,c['name']),lookup.get((s,c.get('printed_name')),np.nan))
        # Baseline: broad card pool, weighted toward common cards.
        base=np.nanmean([c['pred'] for c in sc if c['rarity']=='common'])
        summary=json.loads((OUT/f'summary_{s}.json').read_text()) if s!='TRK' else None
        for pair in [''.join(x) for x in itertools.combinations('WUBRG',2)]:
            pool=[c for c in sc if fits(c,pair) and not np.isnan(c['pred']) and len(c.get('colors',[]))]
            d={'set':s,'colors':pair,'pool_n':len(pool)}
            if summary:
                actual=summary['colors'].get(pair,{'games':0})
                d['games']=actual['games'];d['target']=100*(actual['wr']-summary['win_rate']) if actual['games'] else np.nan
            for rarity in ['common','uncommon','rare']:
                x=[c['pred']-base for c in pool if c['rarity']==rarity]
                d[rarity+'_quality']=np.mean(x) if x else 0
                d[rarity+'_top']=np.mean(sorted(x,reverse=True)[:10]) if x else 0
            fs=[features(c) for c in pool if c['rarity'] in ['common','uncommon']]
            for k in ['type_Creature','kw_flying','kw_defender','kw_beam me up','kw_station','type_Artifact','etb_card','removal','burn','draw','modes','power_eff','toughness_eff']:
                d[k]=np.mean([f[k] for f in fs]) if fs else 0
            for co in 'WUBRG':d[co]=int(co in pair)
            rows.append(d)
    df=pd.DataFrame(rows);train=df[(df['set']!='TRK')&(df.games>=5000)].copy();target=df[df['set']=='TRK'].copy()
    specs={'colors':list('WUBRG'),'quality':['common_quality','common_top','uncommon_quality','uncommon_top']+list('WUBRG'),
           'structural':['common_quality','common_top','uncommon_quality','uncommon_top','type_Creature','kw_flying','etb_card','removal','draw','modes']+list('WUBRG')}
    metrics=[];out=pd.DataFrame({'colors':target.colors})
    for spec,cols in specs.items():
      for alpha in [1,10,100]:
        pred=np.zeros(len(train));y=train.target.to_numpy()
        for s in SETS:
            va=train['set'].eq(s).to_numpy();tr=~va
            scale=StandardScaler();a=scale.fit_transform(train.loc[tr,cols]);b=scale.transform(train.loc[va,cols])
            model=Ridge(alpha=alpha).fit(a,y[tr]);pred[va]=model.predict(b)
        metrics.append({'model':spec,'alpha':alpha,'n':len(train),'rmse':np.sqrt(np.mean((pred-y)**2)),
                        'correlation':np.corrcoef(pred,y)[0,1],'bias':np.mean(pred-y)})
        train[f'pred_{spec}_{alpha}']=pred
        scale=StandardScaler();a=scale.fit_transform(train[cols]);b=scale.transform(target[cols])
        model=Ridge(alpha=alpha).fit(a,y);out[f'{spec}_{alpha}']=model.predict(b)
    metrics.append({'model':'zero','alpha':0,'n':len(train),'rmse':np.sqrt(np.mean(train.target**2)), 'correlation':0,'bias':-train.target.mean()})
    train.to_csv(OUT/'archetype_validation.csv',index=False);df.to_csv(OUT/'archetype_features.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'archetype_validation_metrics.csv',index=False);out.to_csv(OUT/'archetype_predictions.csv',index=False)
    print(pd.DataFrame(metrics).round(3).to_string(index=False));print(out.round(3).to_string(index=False))

if __name__=='__main__':main()
