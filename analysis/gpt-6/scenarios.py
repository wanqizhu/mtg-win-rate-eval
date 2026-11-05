"""Deck-support calculations, not a purported full-game win-rate simulator."""
from pathlib import Path
import json,re
import numpy as np
from scipy.stats import hypergeom
from features import text

OUT=Path(__file__).resolve().parent/'artifacts'

def main():
    rng=np.random.default_rng(20261105)
    report={'seed':20261105,'iterations':100000,'searches':[],'land_names':[],'federation':[],'dilemmas':[]}
    for name,nlook in [('Rutherford',6),('Spock',4),('Hoshi',5)]:
        for hits in [6,8,10,12,14]:
            report['searches'].append({'card':name,'library':35,'hits_remaining':hits,'look':nlook,'hit_probability':float(1-hypergeom.pmf(0,35,hits,nlook))})
    # Condition on having made N land drops. These are mana-base scenarios,
    # not forecasts of the probability of surviving or reaching that turn.
    for basics in [(9,8),(7,6),(6,5),(5,4)]:
        unique=17-sum(basics)
        lands=np.array([0]*basics[0]+[1]*basics[1]+list(range(2,2+unique)))
        perm=np.argsort(rng.random((100000,17)),axis=1)
        for drops in [3,4,5,6,7]:
            names=lands[perm[:,:drops]]
            distinct=(np.sort(names,axis=1)[:,1:]!=np.sort(names,axis=1)[:,:-1]).sum(axis=1)+1
            report['land_names'].append({'basic_counts':list(basics),'unique_nonbasics':unique,'land_drops':drops,'expected_distinct':float(distinct.mean()),'p_at_least_4':float((distinct>=4).mean()),'p_at_least_5':float((distinct>=5).mean()),'p_at_least_7':float((distinct>=7).mean())})
    cards=[c for c in json.loads((OUT/'cards.json').read_text()) if c['set']=='TRK']
    def playable(c,pair):
        symbols=re.findall(r'\{([^}]+)\}',c.get('mana_cost',''))
        return all(not set(s).intersection('WUBRG') or set(s).intersection(pair) for s in symbols)
    for pair in ['WU','WB','WR','WG','UB','UR','UG','BR','BG','RG']:
        creatures=[c for c in cards if 'Creature' in c['type_line'] and c['rarity'] in ['common','uncommon'] and playable(c,pair)]
        # Availability-weighted samples of creatures, without pretending this is
        # actual draft collation or a model of survival through combat/removal.
        weights=np.array([2.5 if c['rarity']=='common' else 1 for c in creatures]);weights/=weights.sum()
        types=[set(c['type_line'].split(' — ')[-1].split()) for c in creatures]
        for size in [1,2,3,4]:
            picked=rng.choice(len(creatures),(20000,size),p=weights)
            counts=np.array([len(set().union(*(types[i] for i in row if 'Borg' not in types[i]))) for row in picked])
            report['federation'].append({'colors':pair,'creatures_on_board':size,'mean_types':float(counts.mean()),'p_at_least_5':float((counts>=5).mean()),'p_at_least_10':float((counts>=10).mean())})
        pool=[c for c in cards if c['rarity'] in ['common','uncommon'] and playable(c,pair) and 'Land' not in c['type_line']]
        report['dilemmas'].append({'colors':pair,'common_modal':sum('choose one' in text(c) and c['rarity']=='common' for c in pool),'uncommon_modal':sum('choose one' in text(c) and c['rarity']=='uncommon' for c in pool),'common_artifact_creatures':sum('Artifact Creature' in c['type_line'] and c['rarity']=='common' for c in pool),'common_beam_spells':sum('beam me up' in text(c) and c['rarity']=='common' for c in pool)})
    report['modal_density']=[{'modal_cards_in_40':k,'cards_seen':seen,'p_at_least_one':float(1-hypergeom.pmf(0,40,k,seen)),'p_at_least_two':float(1-hypergeom.cdf(1,40,k,seen))} for k in [3,5,7,9] for seen in [10,13,16]]
    (OUT/'mechanic_scenarios.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'land_names':report['land_names'],'dilemmas':report['dilemmas']},indent=2))

if __name__=='__main__':main()
