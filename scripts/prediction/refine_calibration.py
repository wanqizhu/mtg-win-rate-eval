"""Evaluate nonlinear grade calibration and inspect stable blend choices."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from features import features,get_text
from prepare import OUT, SETS

def expert_features(card,g,spec):
    f=features(card);mv=f['mv']
    d={'grade':g,'mv':mv,'land':float(f['type_Land'])}
    if spec in ['hinges','impact','details']:
        d.update(low=max(1.5-g,0),high=max(g-3.7,0))
    if spec=='cubic':d.update(square=g*g/5,cube=g*g*g/25)
    if spec in ['impact','details']:
        text=get_text(card).lower()
        entry_effect=f['etb'] and (f['etb_card'] or f['counters'] or f['burn'] or f['shrink'] or f['tap_down'] or f['fight'] or f['life_gain'] or f['ramp'] or f['anthem'] or f['kw_explores'] or ('look at' in text and 'hand' in text))
        immediate=entry_effect or f['kw_haste'] or f['kw_cycling'] or 'at the beginning of combat on your turn' in text or 'enter as a copy' in text
        d['expensive_no_entry']=(max(mv-3,0) if f['type_Creature'] and not immediate else 0)
        # A disabling Aura affects the board immediately even without an explicit ETB trigger.
        disabling_aura=f['type_Aura'] and (f['negative_aura'] or f['shrink'] or f['removal'])
        d['setup']=float(f['type_Enchantment'] and not f['etb'] and not f['type_Saga'] and not f['type_Creature'] and not disabling_aura)
        d['gold']=float(f['n_required_colors']>=2)
    if spec=='details':
        for col in ['rarity','type_Artifact','type_Creature','global_removal','etb_card','kw_flying','token_power','counterspell','draw','splashable']:
            if col=='rarity':continue
            d[col]=float(f[col])
    return d

def main():
    ex=pd.read_csv(OUT/'expert_validation.csv')
    cards=[json.loads(x) for x in (OUT/'calibration_cards.jsonl').read_text().splitlines()]
    by_key={(c['set'],c.get('hist_name',c['name'])):c for c in cards}
    report=[];models={};predictions=ex[['set','name','y','grade','weight','machine']].copy()
    for spec in ['linear','hinges','cubic','impact','details']:
        fs=[expert_features(by_key[(s,n)],g,spec) for s,n,g in zip(ex['set'],ex.name,ex.grade)]
        x=pd.DataFrame(fs).astype(float)
        for reg in [1,4,16]:
            for wtype in ['equal','reliability']:
                weights=np.ones(len(ex)) if wtype=='equal' else ex.weight.to_numpy()
                pred=np.zeros(len(ex))
                for s in SETS:
                    va=(ex['set']==s).to_numpy();tr=~va
                    model=Ridge(alpha=reg).fit(x[tr],ex.y[tr],sample_weight=weights[tr]);pred[va]=model.predict(x[va])
                label=f'{spec}_{reg}_{wtype}';predictions[label]=pred
                for a in [.65,.8,1.0]:
                    p=a*pred+(1-a)*ex.machine.to_numpy();e=p-ex.y
                    report.append({'spec':spec,'alpha':reg,'weights':wtype,'blend':a,'rmse':np.sqrt(np.mean(e**2)),
                                   'weighted_rmse':np.sqrt(np.average(e**2,weights=ex.weight)),
                                   'mae':np.mean(abs(e)),'bias':e.mean()})
                model=Ridge(alpha=reg).fit(x,ex.y,sample_weight=weights)
                models[label]={'spec':spec,'intercept':model.intercept_,'coefficients':dict(zip(x.columns,map(float,model.coef_)))}
    report=pd.DataFrame(report);report.to_csv(OUT/'refined_calibration_metrics.csv',index=False)
    predictions.to_csv(OUT/'refined_calibration_oof.csv',index=False)
    (OUT/'refined_calibration_models.json').write_text(json.dumps(models,indent=2)+'\n')
    print(report.sort_values('rmse').head(25).round(3).to_string(index=False))
    print('WEIGHTED');print(report.sort_values('weighted_rmse').head(15).round(3).to_string(index=False))

if __name__=='__main__':main()
