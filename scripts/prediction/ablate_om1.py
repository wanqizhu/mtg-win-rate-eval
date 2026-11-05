"""Check the effect of including the historical Pick Two format."""
import numpy as np
import pandas as pd
from model import load,predict
from prepare import OUT,SETS

def main():
    h,target,_=load();frames=[]
    for s in SETS:
        if s=='OM1':continue
        tr=h[~h['set'].isin([s,'OM1'])];va=h[h['set']==s]
        pred=predict(tr.card.tolist(),va.card.tolist(),tr.y.to_numpy(),tr.weight.to_numpy())
        out=va.drop(columns='card').copy()
        for k,v in pred.items():out[k]=v
        out['machine']=.4*out.ridge_1+.25*out.ridge_4+.175*out.cat_4+.175*out.cat_6
        frames.append(out);print(s,round(np.sqrt(np.mean((out.machine-out.y)**2)),3),flush=True)
    out=pd.concat(frames,ignore_index=True);out.to_csv(OUT/'validation_without_OM1.csv',index=False)
    with_om=pd.read_csv(OUT/'validation_all.csv');with_om=with_om[with_om['set']!='OM1'].copy()
    with_om['machine']=.4*with_om.ridge_1+.25*with_om.ridge_4+.175*with_om.cat_4+.175*with_om.cat_6
    report=[]
    for label,v in [('include_OM1',with_om),('exclude_OM1',out)]:
        for s in ['ALL']+[x for x in SETS if x!='OM1']:
            d=v if s=='ALL' else v[v['set']==s];e=d.machine-d.y
            report.append({'training':label,'test_set':s,'n':len(d),'rmse':np.sqrt(np.mean(e**2)),
                           'mae':np.mean(abs(e)),'weighted_rmse':np.sqrt(np.average(e**2,weights=d.weight))})
    pd.DataFrame(report).to_csv(OUT/'om1_ablation_metrics.csv',index=False)
    tr=h[h['set']!='OM1'];pred=predict(tr.card.tolist(),target,tr.y.to_numpy(),tr.weight.to_numpy())
    out=pd.DataFrame({'name':[c['name'] for c in target]})
    for k,v in pred.items():out[k]=v
    out.to_csv(OUT/'model_predictions_without_OM1.csv',index=False)

if __name__=='__main__':main()
