#!/usr/bin/env python3
"""Train final ensemble on all historical data; predict FRA + SPG."""
import gzip, json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import statistics

rows=json.load(open("/workspace/analysis/train_rows.json"))
def extra_features(r):
    txt=(r['oracle_combined'] or ''); low=txt.lower(); t=(r['type_line'] or ''); cmc=r['cmc']
    out={}
    words=r['text_words']
    out['is_vanilla_like']=int(r['is_creature']==1 and words<=4 and r['n_keywords']<=1)
    out['has_etb']=int('when' in low and ('enters' in low or 'enter ' in low))
    out['removal_uncond']=int(bool(re.search(r'destroy target (creature|permanent|planeswalker)',low)) and ('nonland' in low or 'any target' in low or 'target creature' in low and 'with ' not in low))
    out['removal_cond']=int(('destroy target' in low or 'exile target' in low) and not out['removal_uncond'])
    out['is_sweeper']=int(('destroy all' in low or 'exile all' in low or 'all creatures' in low))
    out['draw_n']=len(re.findall(r'draw (a card|two cards|three cards|cards)',low))
    out['draw_any']=r['has_draw']
    out['is_counterspell']=int('counter target' in low)
    out['is_burn']=int('damage to any target' in low or re.search(r'deals \d+ damage',low) is not None)
    try:
        p=float(r['power']) if r['power']!=-1 else None
        t_=float(r['toughness']) if r['toughness']!=-1 else None
    except: p=t_=None
    if p is not None and t_ is not None and r['is_creature']:
        out['stat_eff']=(p+t_)-2.5*cmc if cmc>0 else (p+t_)
        out['power_gt_cmc']=int(p>=cmc)
    else:
        out['stat_eff']=0.0; out['power_gt_cmc']=0
    out['cmc_ge6']=int(cmc>=6); out['cmc_le2_creature']=int(cmc<=2 and r['is_creature']==1)
    out['is_aura']=int('Aura' in t); out['is_equipment']=int('Equipment' in t); out['is_saga']=int('Saga' in t)
    out['mythic_bomb']=int(r['rarity']=='mythic' and (cmc>=5 or r['is_planeswalker']))
    out['rare_bomb']=int(r['rarity']=='rare' and (out['is_sweeper'] or r['has_draw'] or cmc>=5))
    out['cantrip']=int((r['is_instant'] or r['is_sorcery']) and r['has_draw'] and cmc<=2)
    return out
for r in rows: r.update(extra_features(r))
def clean_text(t):
    t=t.lower(); t=re.sub(r'\([^)]*\)',' ',t); t=re.sub(r'[^a-z0-9\s\+\-/]',' ',t); t=re.sub(r'\s+',' ',t)
    return t.strip()
for r in rows: r['_clean']=clean_text(r['oracle_combined']+' '+(r['type_line'] or ''))

NUM_KEYS=['cmc','log_cmc','n_colors','is_multicolor','is_colorless_nonland','W','U','B','R','G',
'is_creature','is_land','is_instant','is_sorcery','is_enchantment','is_artifact','is_planeswalker',
'is_legendary','is_basic','power','toughness','pt_sum','has_pt','n_keywords','text_len','text_words',
'has_flying','has_trample','has_haste','has_vigilance','has_deathtouch','has_menace','has_reach',
'has_lifelink','has_ward','has_flash','has_draw','has_destroy','has_exile','has_counter','has_token',
'has_treasure','has_surveil','has_scry','has_mill','has_sacrifice','has_discard','has_lifegain',
'has_damage','has_fight','has_search','has_return_gy','has_copy','has_proliferate','has_cycling',
'has_flashback','has_landfall','has_prowess','has_flying_hate','n_removal_words',
'is_vanilla_like','has_etb','removal_uncond','removal_cond','is_sweeper','draw_n','draw_any',
'is_counterspell','is_burn','stat_eff','power_gt_cmc','cmc_ge6','cmc_le2_creature','is_aura','is_equipment',
'is_saga','mythic_bomb','rare_bomb','cantrip']
RAR=['common','uncommon','rare','mythic']
set_mean={s: statistics.mean([r['gih_wr'] for r in rows if r['set']==s]) for s in set(r['set'] for r in rows)}
global_mean=statistics.mean([r['gih_wr'] for r in rows])
print("global mean", round(global_mean,4))

def num_matrix_row(r):
    return [float(r[k]) for k in NUM_KEYS]+[1.0 if r['rarity']==rar else 0.0 for rar in RAR]

Xnum_all=np.array([num_matrix_row(r) for r in rows])
y_all=np.array([r['gih_wr']-set_mean[r['set']] for r in rows])
w_all=np.array([np.sqrt(r['gih']) for r in rows])

vect=TfidfVectorizer(max_features=2000, ngram_range=(1,2), min_df=4, sublinear_tf=True)
Xt_all=vect.fit_transform([r['_clean'] for r in rows])
sc=StandardScaler(with_mean=False)
Xn_all=sc.fit_transform(sparse.csr_matrix(Xnum_all))
X_all=sparse.hstack([Xn_all,Xt_all]).tocsr()
ridge=Ridge(alpha=30.0); ridge.fit(X_all,y_all,sample_weight=w_all)
hgb=HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=4, min_samples_leaf=20, l2_regularization=5.0, random_state=0)
hgb.fit(Xnum_all,y_all,sample_weight=w_all)
print("trained")

# featurize FRA
import numpy as np
def full_text(card):
    parts=[]
    if card.get('oracle_text'): parts.append(card['oracle_text'])
    for f in (card.get('card_faces') or []):
        if f.get('oracle_text'): parts.append(f['oracle_text'])
    return "\n".join(parts)
def parse_pt(v):
    import re as _re
    if v is None: return None
    s=str(v).strip()
    if s in ('*','?',''): return None
    m=_re.match(r'^-?\d+', s)
    if m: return float(m.group(0))
    return None
def feats(card):
    t=(card.get('type_line') or ''); txt=full_text(card); low=txt.lower()
    cmc=float(card.get('cmc') or 0); mc=card.get('mana_cost') or ''
    import re as _re
    pips=_re.findall(r'\{([^}]+)\}', mc)
    colors=card.get('colors') or []
    rar=card.get('rarity')
    pow_=parse_pt(card.get('power')); tou=parse_pt(card.get('toughness'))
    is_creature='Creature' in t; is_land='Land' in t
    is_instant='Instant' in t; is_sorcery='Sorcery' in t
    is_ench='Enchantment' in t; is_art='Artifact' in t; is_pw='Planeswalker' in t
    is_legend='Legendary' in t; is_basic='Basic Land' in t
    kw=card.get('keywords') or []
    d={}
    d['cmc']=cmc; d['log_cmc']=float(np.log1p(cmc)); d['n_colors']=len(colors)
    d['is_multicolor']=int(len(colors)>=2); d['is_colorless_nonland']=int(len(colors)==0 and not is_land)
    for cc in ['W','U','B','R','G']: d[cc]=int(cc in colors)
    d['rarity']=rar
    d['is_creature']=int(is_creature); d['is_land']=int(is_land)
    d['is_instant']=int(is_instant); d['is_sorcery']=int(is_sorcery)
    d['is_enchantment']=int(is_ench); d['is_artifact']=int(is_art); d['is_planeswalker']=int(is_pw)
    d['is_legendary']=int(is_legend); d['is_basic']=int(is_basic)
    d['power']=pow_ if pow_ is not None else -1; d['toughness']=tou if tou is not None else -1
    d['pt_sum']=(pow_+tou) if (pow_ is not None and tou is not None) else -1
    d['has_pt']=int(pow_ is not None); d['n_keywords']=len(kw)
    d['text_len']=len(txt); d['text_words']=len(txt.split())
    for k,term in [('has_flying','flying'),('has_trample','trample'),('has_haste','haste'),('has_vigilance','vigilance'),
                   ('has_deathtouch','deathtouch'),('has_menace','menace'),('has_reach','reach'),('has_lifelink','lifelink'),
                   ('has_ward','ward'),('has_flash','flash')]:
        d[k]=int(term in low)
    d['has_draw']=int(_re.search(r'\bdraw\b', low) is not None)
    for k,term in [('has_destroy','destroy'),('has_exile','exile'),('has_token','token'),('has_treasure','treasure'),
                   ('has_surveil','surveil'),('has_scry','scry'),('has_sacrifice','sacrifice'),('has_discard','discard'),
                   ('has_damage','damage'),('has_copy',' copy'),('has_proliferate','proliferate'),('has_cycling','cycling'),
                   ('has_flashback','flashback'),('has_landfall','landfall'),('has_prowess','prowess')]:
        d[k]=int(term in low)
    d['has_counter']=int('counter target' in low or 'counter all' in low)
    d['has_mill']=int(' mill' in low or low.startswith('mill'))
    d['has_lifegain']=int('you gain' in low and 'life' in low)
    d['has_fight']=int(' fight' in low)
    d['has_search']=int('search your library' in low)
    d['has_return_gy']=int('return' in low and 'graveyard' in low)
    d['n_removal_words']=low.count('destroy')+low.count('exile')
    d['has_flying_hate']=int('with flying' in low)
    # extras
    words=d['text_words']
    d['is_vanilla_like']=int(is_creature and words<=4 and len(kw)<=1)
    d['has_etb']=int('when' in low and ('enters' in low or 'enter ' in low))
    d['removal_uncond']=int(bool(_re.search(r'destroy target (creature|permanent|planeswalker)',low)) and ('nonland' in low or 'any target' in low or 'target creature' in low and 'with ' not in low))
    d['removal_cond']=int(('destroy target' in low or 'exile target' in low) and not d['removal_uncond'])
    d['is_sweeper']=int(('destroy all' in low or 'exile all' in low or 'all creatures' in low))
    d['draw_n']=len(_re.findall(r'draw (a card|two cards|three cards|cards)',low))
    d['draw_any']=d['has_draw']
    d['is_counterspell']=int('counter target' in low)
    d['is_burn']=int('damage to any target' in low or _re.search(r'deals \d+ damage',low) is not None)
    if pow_ is not None and tou is not None and is_creature:
        d['stat_eff']=(pow_+tou)-2.5*cmc if cmc>0 else (pow_+tou)
        d['power_gt_cmc']=int(pow_>=cmc)
    else:
        d['stat_eff']=0.0; d['power_gt_cmc']=0
    d['cmc_ge6']=int(cmc>=6); d['cmc_le2_creature']=int(cmc<=2 and is_creature)
    d['is_aura']=int('Aura' in t); d['is_equipment']=int('Equipment' in t); d['is_saga']=int('Saga' in t)
    d['mythic_bomb']=int(rar=='mythic' and (cmc>=5 or is_pw))
    d['rare_bomb']=int(rar=='rare' and (d['is_sweeper'] or d['has_draw'] or cmc>=5))
    d['cantrip']=int((is_instant or is_sorcery) and d['has_draw'] and cmc<=2)
    d['oracle_combined']=txt; d['type_line']=t
    d['_clean']=clean_text(txt+' '+t)
    return d

with gzip.open("/opt/mtg-data/scryfall/sets/fra.jsonl.gz",'rt') as f:
    fra_all=[json.loads(l) for l in f if l.strip()]
by_oracle={}
for c in fra_all: by_oracle.setdefault(c['oracle_id'],[]).append(c)
def cn2(c):
    try: return int(str(c['collector_number']).split('/')[0])
    except: return 9999
canon=sorted([sorted(v,key=cn2)[0] for v in by_oracle.values()], key=cn2)
spg=[json.loads(l) for l in open("/opt/mtg-data/scryfall/sets/fra-special-guests.jsonl") if l.strip()]

FRA_MEAN=global_mean  # 0.5573 prior; refined later
out=[]
for card in canon+spg:
    f=feats(card)
    xn=np.array([[float(f[k]) for k in NUM_KEYS]+[1.0 if f['rarity']==rar else 0.0 for rar in RAR]])
    xt=vect.transform([f['_clean']])
    xxn=sc.transform(sparse.csr_matrix(xn))
    x=sparse.hstack([xxn,xt]).tocsr()
    pr=float(ridge.predict(x)[0]); ph=float(hgb.predict(xn)[0])
    ens=0.6*pr+0.4*ph+FRA_MEAN
    full=card['name']
    short=full.split('//')[0].strip() if '//' in full else full
    out.append({"oracle_id":card.get('oracle_id'),"name_full":full,"name_17lands":short,
                "set":card.get('set'),"rarity":card.get('rarity'),"mana_cost":card.get('mana_cost'),
                "cmc":card.get('cmc'),"type_line":card.get('type_line'),
                "collector_number":str(card.get('collector_number')),
                "base_pred":round(ens,4),"ridge":round(pr+FRA_MEAN,4),"hgb":round(ph+FRA_MEAN,4)})
out_sorted=sorted(out,key=lambda x:-x['base_pred'])
for o in out_sorted[:25]:
    print(o['base_pred'], o['rarity'], o['name_full'][:55], o['mana_cost'], (o['type_line'] or '')[:40])
print("...")
for o in out_sorted[-12:]:
    print(o['base_pred'], o['rarity'], o['name_full'][:55], o['mana_cost'], (o['type_line'] or '')[:40])
json.dump(out, open("/workspace/analysis/fra_base.json","w"), indent=1)
print("saved", len(out))
