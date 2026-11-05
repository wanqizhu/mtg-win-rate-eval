"""Produce the single prediction table and fully decomposed audit artifacts."""
import itertools,json,hashlib,platform
from pathlib import Path
import numpy as np
import pandas as pd
from refine_calibration import expert_features
from environment import fits
from prepare import OUT,REPO,SETS

def machine(d):return .4*d.ridge_1+.25*d.ridge_4+.175*d.cat_4+.175*d.cat_6

def main():
    settings=json.loads((OUT/'final_settings.json').read_text())
    cards=[json.loads(x) for x in (OUT/'cards.jsonl').read_text().splitlines() if json.loads(x)['set']=='TRK']
    cards.sort(key=lambda c:int(c['collector_number']))
    notes=pd.read_csv(OUT/'target_expert.tsv',sep='\t').set_index('collector_number')
    models=json.loads((OUT/'refined_calibration_models.json').read_text());cal=models[settings['expert_calibrator']]
    pred=pd.read_csv(OUT/'model_predictions.csv').set_index('name')
    if settings['machine_variant']=='exclude_OM1':pred=pd.read_csv(OUT/'model_predictions_without_OM1.csv').set_index('name')
    elif settings['machine_variant']=='average_OM1':
        other=pd.read_csv(OUT/'model_predictions_without_OM1.csv').set_index('name');pred=(pred+other)/2
    arches=pd.read_csv(OUT/'archetype_predictions.csv').set_index('colors')[settings['color_context_model']]
    arches=arches-arches.mean()
    baseline=np.mean([json.loads((OUT/f'summary_{s}.json').read_text())['win_rate']*100 for s in SETS if s!='OM1'])
    rows=[]
    for c in cards:
        i=int(c['collector_number']);note=notes.loc[i];f=expert_features(c,float(note.grade),cal['spec'])
        expert=cal['intercept']+sum(f[k]*w for k,w in cal['coefficients'].items())
        mch=float(machine(pred.loc[c['name']]))
        ew=settings['wording_priority_expert_weight'] if note.expert_priority else settings['ordinary_expert_weight']
        if 'Land' in c['type_line']:
            ci=set(c.get('color_identity',[]));pairs=[p for p in arches.index if ci<=set(p)] if ci else list(arches.index)
        else:pairs=[p for p in arches.index if fits(c,p)]
        if not pairs:
            ci=set(c.get('colors',[]));pairs=[p for p in arches.index if len(ci&set(p))==2] or list(arches.index)
        context=settings['color_context_shrinkage']*float(arches.loc[pairs].mean())
        adjustment=settings['manual_adjustments_pp'].get(str(i),{'delta':0,'reason':''})
        value=ew*expert+(1-ew)*mch+(baseline-55.5)+context+adjustment['delta']
        basic='Basic Land' in c['type_line']
        if basic:value=baseline+context
        rows.append({'collector_number':i,'name':c['name'],'rarity':c['rarity'],'mana_value':c['cmc'],
                     'grade':float(note.grade),'expert_pp':expert,'machine_pp':mch,'expert_weight':ew,
                     'baseline_shift_pp':baseline-55.5,'color_context_pp':context,'manual_adjustment_pp':adjustment['delta'],
                     'basic_land_baseline_override':basic,'prediction_pp':value,
                     'rationale':note.rationale,'adjustment_reason':adjustment['reason']})
    audit=pd.DataFrame(rows)
    audit.to_csv(OUT/'prediction_components.csv',index=False,float_format='%.6f')
    final=pd.DataFrame({'card_name':audit.name,'predicted_gih_win_rate':np.round(audit.prediction_pp/100,3)})
    dest=REPO/'predictions.csv';dest.parent.mkdir(exist_ok=True)
    final.to_csv(dest,index=False,float_format='%.3f')
    summary={'model_name':settings['model_name'],'cards':len(final),'unique_names':final.card_name.nunique(),
             'fraction_units':True,'baseline_percent':baseline,'minimum_percent':audit.prediction_pp.min(),
             'maximum_percent':audit.prediction_pp.max(),'mean_percent':audit.prediction_pp.mean(),
             'standard_deviation_pp':audit.prediction_pp.std(),'sha256':hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),
             'scope':'311 unique English TRK base cards, collectors 1–311, including five basics; cosmetics collapsed. SDS is paper-only in the frozen data and its set object has no Arena code.',
             'machine_variant':settings['machine_variant']}
    (OUT/'prediction_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print(audit.nlargest(15,'prediction_pp')[['name','prediction_pp']].round(2).to_string(index=False))

if __name__=='__main__':main()
