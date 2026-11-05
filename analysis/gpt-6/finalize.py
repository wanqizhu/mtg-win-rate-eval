"""Create the sole submission table and an auditable component decomposition."""
import json,gzip,hashlib,csv,sys
from pathlib import Path
import numpy as np,pandas as pd
from model import OUT,HERE

REPO=HERE.parents[1]

def main():
    cards=json.loads((OUT/'cards.json').read_text())
    target=[c for c in cards if c['set']=='TRK']
    scope=pd.DataFrame([{'name':c['name'],'number':int(c['collector_number']),'rarity':c['rarity'],'mana_value':c['cmc'],'colors':''.join(c.get('colors',[])),'type_line':c['type_line']} for c in target])
    old=pd.read_csv(OUT/'initial_model/statistical_predictions.csv')
    new=pd.read_csv(OUT/'statistical_predictions.csv')
    d=scope.merge(old[['name','set','ridge_0.15_3']],on='name',validate='one_to_one')
    d=d.merge(new[['name','cat_6']],on='name',validate='one_to_one')
    d=d.merge(pd.read_csv(HERE/'target_ratings.tsv',sep='\t'),on='number',validate='one_to_one')
    d['statistical_wr']=.5*d['ridge_0.15_3']+.5*d.cat_6
    d['human_weight']=.35
    d['reviewed_qualitative_wr']=d.qualitative_wr
    d['review_reason']=''
    adjustments=pd.read_csv(HERE/'review_adjustments.tsv',sep='\t').set_index('number')
    for i,row in d.iterrows():
        if row.number in adjustments.index:
            a=adjustments.loc[row.number]
            d.loc[i,'reviewed_qualitative_wr']=a.revised_qualitative_wr
            d.loc[i,'human_weight']=a.human_weight
            d.loc[i,'review_reason']=a.reason
    d['unreviewed_blend']=.65*d.statistical_wr+.35*d.qualitative_wr
    d['predicted_gih_win_rate']=((1-d.human_weight)*d.statistical_wr+d.human_weight*d.reviewed_qualitative_wr)/100
    d=d.sort_values('number')
    assert len(d)==311 and d.name.is_unique and d.number.tolist()==list(range(1,312))
    assert d.predicted_gih_win_rate.between(0,1).all() and d.predicted_gih_win_rate.notna().all()
    d[['name','predicted_gih_win_rate']].to_csv(REPO/'predictions.csv',index=False,float_format='%.4f')
    d.to_csv(OUT/'prediction_components.csv',index=False,float_format='%.6f')
    pd.DataFrame([{'set':'SDS','name':c['name'],'reason':'Supplied printing is paper-only; no Arena entry in games metadata.'} for c in cards if c['set']=='SDS']).to_csv(OUT/'excluded_scope.csv',index=False)
    facts={'model':'GPT-6','branch':'gpt-6','target_set':'TRK','target_format':'PremierDraft','rate_units':'fraction: 0.5821 means 58.21%','prediction_rows':len(d),'unique_names':d.name.nunique(),'all_printing_variants_collapsed':True,'source_TRK_printings':576,'source_TRK_distinct_names':311,'source_SDS_distinct_names':30,'basic_lands_included':5,'qualitative_reviews':311,'manual_review_adjustments':len(adjustments),'blend_human_weight':.35,'minimum_prediction':float(d.predicted_gih_win_rate.min()),'maximum_prediction':float(d.predicted_gih_win_rate.max()),'mean_prediction':float(d.predicted_gih_win_rate.mean()),'submission_sha256':hashlib.file_digest((REPO/'predictions.csv').open('rb'),'sha256').hexdigest()}
    (OUT/'submission_manifest.json').write_text(json.dumps(facts,indent=2)+'\n')
    # Independently recompute selected full-file labels with Python csv, including
    # basic land, common removal, mythic and a low-play build-around.
    root=Path('/workspace/mtg-data')
    file=root/'17lands/sets/FRA/game_data_public.FRA.PremierDraft.csv.gz'
    selected=['Forest','Memory Trap','Emrakul, the Exigent Doom','Gardenize']
    counts={s:[0,0] for s in selected};games=wins=0
    with gzip.open(file,'rt') as handle:
        reader=csv.DictReader(handle)
        for row in reader:
            won=row['won'].lower() in ['true','1'];games+=1;wins+=won
            assert row['expansion']=='FRA'
            for name in selected:
                present=int(row['opening_hand_'+name] or 0)+int(row['drawn_'+name] or 0)>0
                if present:counts[name][0]+=1;counts[name][1]+=won
    labels=pd.read_csv(OUT/'labels_FRA.csv').set_index('name')
    for name,(n,w) in counts.items():
        assert n==labels.loc[name,'gih_games'] and w==labels.loc[name,'gih_wins']
    checks={'coverage':'passed','finite_probabilities':'passed','duplicate_names':'none','independent_FRA_games':games,'independent_FRA_wins':wins,'independently_verified_card_counts':counts,'target_outcomes_used':False}
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(facts,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
