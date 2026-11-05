"""Independent checks of coverage, units, reproducibility and historical aggregation."""
import csv,gzip,hashlib,json
import numpy as np
import pandas as pd
from prepare import DATA,REPO,OUT,SETS

def main():
    cards=[json.loads(x) for x in gzip.open(DATA/'scryfall'/'sets'/'trk.jsonl.gz','rt')]
    target={c['name'] for c in cards}
    p=pd.read_csv(REPO/'predictions.csv')
    assert list(p.columns)==['card_name','predicted_gih_win_rate']
    assert len(p)==len(target)==311 and p.card_name.is_unique
    assert set(p.card_name)==target
    assert p.predicted_gih_win_rate.between(0,1).all() and p.notna().all().all()
    audit=pd.read_csv(OUT/'prediction_components.csv')
    assert p.card_name.tolist()==audit.name.tolist()
    assert np.allclose(p.predicted_gih_win_rate,np.round(audit.prediction_pp/100,3))
    notes=pd.read_csv(OUT/'target_expert.tsv',sep='\t')
    assert set(notes.collector_number)==set(range(1,312)) and notes.collector_number.is_unique
    assert notes.rationale.notna().all() and notes.grade.between(0,5).all()
    assert {c['name'] for c in cards if 'arena' in c['games']}==target
    sds=[json.loads(x) for x in gzip.open(DATA/'scryfall'/'sets'/'sds.jsonl.gz','rt')]
    assert len(sds)==30 and all('arena' not in c['games'] for c in sds)
    h=pd.read_csv(OUT/'historical_all.csv');assert set(h['set'])==set(SETS)
    assert (h.gih_wins<=h.gih_n).all() and (h.gih_n<=h.deck_n).all()
    checked={};names=['The Theorist, Jace Beleren','Unflinching Hortimancer','Forest']
    tallies={n:[0,0] for n in names};rows=0;duplicate_exposures=0
    source=DATA/'17lands'/'sets'/'FRA'/'game_data_public.FRA.PremierDraft.csv.gz'
    with gzip.open(source,'rt',newline='') as handle:
        reader=csv.reader(handle);head=next(reader)
        indexes={n:[head.index('opening_hand_'+n),head.index('drawn_'+n)] for n in names}
        win=head.index('won');exp=head.index('expansion');event=head.index('event_type')
        for row in reader:
            assert row[exp]=='FRA' and row[event]=='PremierDraft';rows+=1
            won=row[win].lower() in ['true','1']
            for n,(a,b) in indexes.items():
                count=int(row[a])+int(row[b]);seen=count>0
                tallies[n][0]+=seen;tallies[n][1]+=seen and won
                duplicate_exposures+=count>1
    for n,(count,wins) in tallies.items():
        actual=h[(h['set']=='FRA')&(h.name==n)].iloc[0]
        assert count==actual.gih_n and wins==actual.gih_wins
        checked[n]={'gih_n':count,'gih_wins':wins}
    leakage=json.loads((DATA/'manifests'/'leakage_report.json').read_text())
    assert leakage['star_trek_seventeenlands_rows']==0 and leakage['star_trek_seventeenlands_files']==0
    report={'checks_passed':True,'target_cards':len(p),'sds_excluded':len(sds),'independent_source':'FRA PremierDraft',
            'independently_recounted_rows':rows,'checked_card_counts':checked,
            'multiple_copy_exposures_in_checked_cards':duplicate_exposures,
            'gih_definition':'One exposure per game if opening_hand + drawn > 0; tutored-only exposure excluded, following supplied demo.',
            'frozen_leakage_report_clean':True,
            'prediction_sha256':hashlib.file_digest((REPO/'predictions.csv').open('rb'),'sha256').hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
