"""Combine held-out models and evaluate qualitative judgments without target labels."""
import json
import numpy as np,pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from model import OUT,HERE

def rmse(y,p):return float(np.sqrt(np.mean((np.array(y)-np.array(p))**2)))

def main():
    old=pd.read_csv(OUT/'initial_model/cross_validation.csv')
    new=pd.read_csv(OUT/'cross_validation.csv')
    a=pd.read_csv(OUT/'analogue_predictions.csv')
    near=pd.read_csv(OUT/'historical_analogues.csv');near=near[near['rank']==1]
    d=new.merge(old[['set','name','ridge_0.15_3','cat_6','cat_text']],on=['set','name'],suffixes=('','_v1'))
    d=d.merge(a[['set','name','role_15_2']],on=['set','name']).merge(near[['set','name','similarity','relative_wr_pp']],on=['set','name'])
    d['truth']=d.gih_wr*100
    configs={
      'equal4':{'ridge_0.15_3_v1':.25,'cat_6_v1':.25,'cat_6':.25,'cat_text':.25},
      'equal3':{'ridge_0.15_3_v1':1/3,'cat_6':1/3,'cat_text':1/3},
      'ridge_cat':{'ridge_0.15_3_v1':.5,'cat_6':.5},
      'cat_weighted':{'ridge_0.15_3_v1':.25,'cat_6':.5,'cat_text':.25},
      'with_analogue':{'ridge_0.15_3_v1':.3,'cat_6':.3,'cat_text':.2,'role_15_2':.2},
    }
    metrics=[]
    use=(d.gih_games>=500)&(d['format']=='PremierDraft')
    for name,weights in configs.items():
        d[name]=sum(d[k]*v for k,v in weights.items())
        metrics.append({'model':name,'rmse_pp':rmse(d.loc[use,'truth'],d.loc[use,name])})
    # Compare near-exact reprints only; distant analogues are unreliable.
    best=min(metrics,key=lambda x:x['rmse_pp'])['model']
    for cutoff in [.65,.8,.95]:
        for mix in [.25,.5,.75]:
            col=f'exact_{cutoff}_{mix}'
            d[col]=d[best]+np.where(d.similarity>=cutoff,mix*(d.relative_wr_pp+d.baseline-d[best]),0)
            metrics.append({'model':col,'rmse_pp':rmse(d.loc[use,'truth'],d.loc[use,col]),'affected':int(((d.similarity>=cutoff)&use).sum())})
    pd.DataFrame(metrics).sort_values('rmse_pp').to_csv(OUT/'ensemble_metrics.csv',index=False)
    print(pd.DataFrame(metrics).sort_values('rmse_pp').to_string(index=False))
    # The best exact-analogue tweak differs by <0.001 pp: keep the simpler blend.
    chosen=best
    d['statistical']=d[chosen]
    sample=[]
    for label,js,cs in [('calibration','calibration_sample.json','calibration_ratings.csv'),('audit','audit_sample.json','audit_ratings.csv')]:
        s=pd.DataFrame(json.loads((OUT/js).read_text()))[['set','name']].join(pd.read_csv(HERE/cs).set_index('index'))
        s=s.merge(d,on=['set','name']);s['sample']=label;sample.append(s)
    s=pd.concat(sample,ignore_index=True)
    # Global blend weights are fit with one whole sample-set held out at a time.
    audit=[]
    for weight in [0,.15,.25,.35,.5,.65,.8,1]:
        p=weight*s.qualitative_wr+(1-weight)*s.statistical
        for group in ['calibration','audit','combined']:
            q=s['sample']==group if group!='combined' else np.ones(len(s),dtype=bool)
            audit.append({'human_weight':weight,'sample':group,'rmse_pp':rmse(s.loc[q,'truth'],p[q]),'bias_pp':float((p[q]-s.loc[q,'truth']).mean())})
    cv=np.zeros(len(s));cvblend=np.zeros(len(s));fits=[]
    for group in s['set'].unique():
        tr=s['set']!=group;te=~tr
        x=s[['qualitative_wr','statistical']].to_numpy()-55.7;y=s.truth.to_numpy()-55.7
        m=Ridge(alpha=60).fit(x[tr],y[tr])
        cv[te]=m.predict(x[te])+55.7
        candidates=np.linspace(0,1,101)
        errs=[rmse(s.loc[tr,'truth'],w*s.loc[tr,'qualitative_wr']+(1-w)*s.loc[tr,'statistical']) for w in candidates]
        w=candidates[np.argmin(errs)]
        cvblend[te]=w*s.loc[te,'qualitative_wr']+(1-w)*s.loc[te,'statistical']
        fits.append({'held_out':group,'coef':m.coef_.tolist(),'intercept':float(m.intercept_),'blend_weight':float(w)})
    s['calibrated_cv']=cv;s['selected_blend_cv']=cvblend
    fit=Ridge(alpha=60).fit(s[['qualitative_wr','statistical']].to_numpy()-55.7,s.truth.to_numpy()-55.7)
    report={'base_ensemble':best,'chosen_statistical':chosen,'base_weights':configs[best], 'qualitative_sample_n':len(s),'ridge_coefficients':fit.coef_.tolist(),'ridge_intercept_centered':float(fit.intercept_),'ridge_cv_rmse_pp':rmse(s.truth,cv),'selected_blend_cv_rmse_pp':rmse(s.truth,cvblend),'statistical_sample_rmse_pp':rmse(s.truth,s.statistical),'folds':fits}
    (OUT/'calibration_report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame(audit).to_csv(OUT/'qualitative_blend_metrics.csv',index=False)
    s.to_csv(OUT/'calibrated_sample_predictions.csv',index=False)
    d.to_csv(OUT/'ensemble_cross_validation.csv',index=False)
    print(json.dumps(report,indent=2));print(pd.DataFrame(audit).query('sample=="combined"').to_string(index=False))

if __name__=='__main__':main()
