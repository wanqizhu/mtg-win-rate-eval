"""Calibrate frozen human grades and select blends without target outcomes."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from prepare import OUT, SETS

def main():
    vals=pd.read_csv(OUT/'validation_all.csv')
    methods=['ridge_1','ridge_4','cat_4','cat_6','knn_3']
    vals['machine']=.40*vals.ridge_1+.25*vals.ridge_4+.175*vals.cat_4+.175*vals.cat_6
    cards=[json.loads(x) for x in (OUT/'calibration_cards.jsonl').read_text().splitlines()]
    grades=json.loads((OUT/'expert_calibration_grades.json').read_text())['grades']
    ex=pd.DataFrame([{'set':c['set'],'name':c.get('hist_name',c['name']),'grade':grades[str(i)],'rarity':c['rarity'],
                      'mv':c.get('cmc',0),'land':'Land' in c['type_line'],
                      'ncolor':len(c.get('colors',[])),'common':c['rarity']=='common','mythic':c['rarity']=='mythic',
                      'rare':c['rarity']=='rare','uncommon':c['rarity']=='uncommon'} for i,c in enumerate(cards)])
    ex=ex.merge(vals,on=['set','name'],validate='one_to_one')
    print('Calibration rows',len(ex))
    # Grades intentionally receive very few fitted degrees of freedom.
    specs={'linear':['grade'], 'rarity':['grade','rare','mythic'],
           'mana':['grade','mv','land'], 'all':['grade','mv','land','rare','mythic']}
    summary=[]
    for spec,cols in specs.items():
        ex[f'expert_{spec}']=np.nan
        for s in SETS:
            tr=ex['set']!=s;va=~tr
            reg=Ridge(alpha=10).fit(ex.loc[tr,cols].astype(float),ex.loc[tr,'y'],sample_weight=ex.loc[tr,'weight'])
            ex.loc[va,f'expert_{spec}']=reg.predict(ex.loc[va,cols].astype(float))
        for a in [0,.25,.5,.65,.8,1]:
            p=a*ex[f'expert_{spec}']+(1-a)*ex.machine
            e=p-ex.y;summary.append({'spec':spec,'human_weight':a,'rmse':np.sqrt(np.mean(e**2)),
                                     'weighted_rmse':np.sqrt(np.average(e**2,weights=ex.weight)),
                                     'mae':np.mean(abs(e)),'bias':e.mean(),'n':len(e)})
    pd.DataFrame(summary).to_csv(OUT/'expert_validation_metrics.csv',index=False)
    ex.to_csv(OUT/'expert_validation.csv',index=False)
    print(pd.DataFrame(summary).sort_values('weighted_rmse').round(3).to_string(index=False))
    best={}
    for spec,cols in specs.items():
        reg=Ridge(alpha=10).fit(ex[cols].astype(float),ex.y,sample_weight=ex.weight)
        best[spec]={'features':cols,'intercept':float(reg.intercept_),'coefficients':dict(zip(cols,map(float,reg.coef_)))}
    (OUT/'expert_calibration_models.json').write_text(json.dumps(best,indent=2)+'\n')
    print(json.dumps(best,indent=2))
    for s,d in vals.groupby('set'):
        print(s,'n=',len(d),'mean=',round(d.y.mean(),2),'rmse=',round(np.sqrt(np.mean((d.machine-d.y)**2)),3),'bias=',round((d.machine-d.y).mean(),3))

if __name__=='__main__':main()
