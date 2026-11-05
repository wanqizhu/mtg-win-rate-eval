"""Mechanic census and controlled consistency simulations, not full-game win-rate simulations."""
import collections,itertools,json,re
import numpy as np
import pandas as pd
from features import features,get_text
from prepare import OUT

def mana_colors(c):
    import re
    cost=c.get('mana_cost','')
    required=[]
    for s in re.findall(r'\{([^}]+)\}',cost):
        if any(k in s for k in 'WUBRG'):required.append(set(s)&set('WUBRG'))
    return required

def fits(c,pair):return all(bool(s&set(pair)) for s in mana_colors(c))

def token_bodies(c):
    nums={'a':1,'an':1,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'ten':10}
    out=[]
    for count,power in re.findall(r'create (a|an|one|two|three|four|five|six|ten) (?:tapped )?(\d+)/\d+',get_text(c).lower()):
        out += [int(power)]*nums[count]
    return out

def pool(cards,pair,kind):
    e=pd.read_csv(OUT/'target_expert.tsv',sep='\t').set_index('collector_number')
    p=[];w=[]
    for c in cards:
        f=features(c)
        if f['type_Land'] or not fits(c,pair):continue
        if bool(f['type_Creature'])!=(kind=='creature'):continue
        grade=float(e.loc[int(c['collector_number']),'grade'])
        if grade<1.9:continue
        rarity={'common':4.,'uncommon':1.5,'rare':.22,'mythic':.11}[c['rarity']]
        p.append(c);w.append(rarity*np.exp(.30*(grade-2.7)))
    return p,np.array(w)/sum(w)

def census(cards):
    rows=[]
    for pair in ['W','U','B','R','G']+[''.join(x) for x in itertools.combinations('WUBRG',2)]:
        p=[c for c in cards if fits(c,pair) and 'Land' not in c['type_line']]
        fs=[features(c) for c in p]
        rows.append({'colors':pair,'nonlands':len(p),'creatures':sum(f['type_Creature'] for f in fs),
                     'artifact_creatures':sum(f['type_Creature'] and f['type_Artifact'] for f in fs),
                     'ships':sum(f['type_Spacecraft'] for f in fs),
                     'modal':sum(f['modes']>0 for f in fs),'repeatable_modal':sum(f['modal_repeat']>0 for f in fs),
                     'beam':sum(f['kw_beam me up'] for f in fs),
                     'life_gain_sources':sum(f['life_gain']>0 or f['kw_lifelink'] or f['food'] for f in fs),
                     'creature_etb':sum(f['type_Creature'] and f['etb']>0 for f in fs),
                     'commons':sum(c['rarity']=='common' for c in p),
                     'common_creatures':sum(c['rarity']=='common' and f['type_Creature'] for c,f in zip(p,fs)),
                     'removal_candidates':sum(bool(f['removal'] or f['burn'] or f['shrink'] or f['fight'] or f['negative_aura']) for f in fs)})
    pd.DataFrame(rows).to_csv(OUT/'mechanic_census.csv',index=False)

def land_trials(rng,n=40000):
    rows=[]
    for nb in [2,3,4,6]:
      for extra_basic in [False,True]:
       for token in [False,True]:
        names=[0]*((17-nb)//2)+[1]*((18-nb)//2)+list(range(2,2+nb))
        if extra_basic:names[0]=20
        deck=np.array(names+[-1]*23)
        counts={t:[] for t in [3,5,7,9]}
        lands_cast={t:[] for t in counts}
        for i in range(n):
            shuffled=rng.permutation(deck);hand=list(shuffled[:7]);played=[]
            for t in range(1,10):
                if t>1:hand.append(shuffled[5+t])
                lands=[x for x in hand if x>=0]
                if lands:
                    fresh=[x for x in lands if x not in played]
                    choice=(fresh or lands)[0];hand.remove(choice);played.append(choice)
                if token and t==4:played.append(30)
                if t in counts:counts[t].append(len(set(played)));lands_cast[t].append(len(played))
        for t,vals in counts.items():
            a=np.array(vals)
            rows.append({'nonbasic_names':nb,'third_basic':extra_basic,'planet_on_turn4':token,'turn':t,
                         'trials':n,'mean_land_count':np.mean(lands_cast[t]),'mean_names':a.mean(),
                         **{f'p_names_ge_{k}':np.mean(a>=k) for k in [3,4,5,6,7]}})
    pd.DataFrame(rows).to_csv(OUT/'land_name_simulation.csv',index=False)

def type_trials(cards,rng,n=20000):
    rows=[]
    for pair in ['WU','WG','UG','WB','UB','BG']:
        p,w=pool(cards,pair,'creature')
        for allies in [1,2,3,4]:
            vals=[]
            for _ in range(n):
                board=rng.choice(len(p),size=allies,p=w)
                # Hoshi/Saavik provides two non-Borg types itself.
                types={'Human','Scientist'} if pair in ['WU','UG','UB'] else {'Vulcan','Advisor'}
                for i in board:
                    ty=p[i]['type_line'].split('—')[-1].strip().split()
                    if 'Borg' not in ty:types.update(ty)
                vals.append(len(types))
            a=np.array(vals)
            rows.append({'colors':pair,'other_creatures':allies,'trials':n,'mean_federation':a.mean(),
                         'p_ge_4':np.mean(a>=4),'p_ge_5':np.mean(a>=5),'p_ge_10':np.mean(a>=10),'p_ge_15':np.mean(a>=15)})
    pd.DataFrame(rows).to_csv(OUT/'federation_simulation.csv',index=False)

def station_trials(cards,rng,n=3500):
    ships={c['name']:c for c in cards if 'Spacecraft' in c['type_line']}
    cases=[('WU','Oberth-Class Starship'),('WU','Suurok-Class Science Vessel'),('WU','Enterprise NX-01'),
           ('WU','U.S.S. Enterprise-A, Free Spirit'),('WU','U.S.S. Cerritos, Following Up'),
           ('WU','U.S.S. Enterprise-D, Galaxy-Class'),('BR','D-7 Battle Cruiser'),('UG','U.S.S. Discovery, at Black Alert')]
    rows=[]
    for pair,name in cases:
      ship=ships[name];sf=features(ship);threshold=int(sf['station_max']);cost=int(sf['mv'])
      if name=='D-7 Battle Cruiser':threshold=3 # its first and most important engine
      p,w=pool(cards,pair,'creature')
      pf=[features(c) for c in p];pt=[token_bodies(c) if f['etb'] else [] for c,f in zip(p,pf)]
      for survival in [1.,.8]:
        ready=[];charge=[];castturn=[]
        for trial in range(n):
            # Condition on having the ship. Other 39 cards: 17 land, 15 bodies, 7 other spells.
            body_idx=rng.choice(len(p),size=15,p=w)
            deck=np.array([-1]*17+list(map(int,body_idx))+[-2]*7);deck=rng.permutation(deck)
            hand=list(deck[:6]);mana=0;bodies=[];counters=0;deployed=False;spent=0;ct=None;rt=None
            for turn in range(1,11):
                if turn>1:hand.append(deck[4+turn])
                bodies=[b for b in bodies if rng.random()<survival]
                if -1 in hand:hand.remove(-1);mana+=1
                available=mana
                if not deployed and available>=cost:
                    available-=cost;deployed=True;ct=turn
                    # Entry tokens can station immediately, despite summoning sickness.
                    bodies += token_bodies(ship)
                affordable=sorted([i for i in hand if i>=0 and p[i].get('cmc',0)<=available],key=lambda i:p[i].get('cmc',0))
                for i in affordable:
                    f=pf[i];mc=int(f['mv'])
                    if mc>available:continue
                    available-=mc;hand.remove(i)
                    power=f['power']
                    if f['power_unknown']:power=3
                    bodies.append(max(power,0));bodies += pt[i]
                if deployed and counters<threshold:
                    # Upper-bound charging: all bodies available, no attack reserved.
                    for power in sorted(bodies,reverse=True):
                        counters+=power;spent+=power
                        if counters>=threshold:break
                    if counters>=threshold:rt=turn;break
            ready.append(rt or 11);charge.append(spent);castturn.append(ct or 11)
        a=np.array(ready);c=np.array(castturn)
        rows.append({'colors':pair,'ship':name,'threshold':threshold,'creature_survival_per_turn':survival,
                     'trials':n,'p_cast_by_5':np.mean(c<=5),'p_ready_by_5':np.mean(a<=5),
                     'p_ready_by_7':np.mean(a<=7),'median_ready_turn_capped11':np.median(a),
                     'mean_charge_power_spent':np.mean(charge),'median_turns_to_ready_if_ready':np.median((a-c)[a<=10])})
    pd.DataFrame(rows).to_csv(OUT/'station_simulation.csv',index=False)

def dilemmas():
    from scipy.stats import hypergeom
    rows=[]
    for m in [4,6,8,10]:
        for turn in [4,6,8]:
            # Condition on drawing one payoff card: 39 cards remain.
            drawn=7+turn-1-1
            rows.append({'modal_cards_in_deck':m,'turn':turn,'other_cards_seen':drawn,
                         'expected_modal_seen':drawn*m/39,'p_at_least_one':hypergeom.sf(0,39,m,drawn),
                         'p_at_least_two':hypergeom.sf(1,39,m,drawn)})
    pd.DataFrame(rows).to_csv(OUT/'dilemma_consistency.csv',index=False)

def main():
    cards=[json.loads(x) for x in (OUT/'cards.jsonl').read_text().splitlines() if json.loads(x)['set']=='TRK']
    rng=np.random.default_rng(20261105)
    census(cards);land_trials(rng);type_trials(cards,rng);station_trials(cards,rng);dilemmas()
    print('Saved mechanic census and controlled scenario simulations.')

if __name__=='__main__':main()
