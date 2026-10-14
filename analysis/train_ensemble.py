#!/usr/bin/env python3
"""Final ensemble LOSO: Ridge(30, num+tfidf) + HGB(num)."""
import json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import sparse
import statistics

rows=json.load(open("/workspace/analysis/train_rows.json"))
# recompute extras (same as train_v2)
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
for r in rows:
    r.update(extra_features(r))
def clean_text(t):
    t=t.lower(); t=re.sub(r'\([^)]*\)',' ',t); t=re.sub(r'[^a-z0-9\s\+\-/]',' ',t); t=re.sub(r'\s+',' ',t)
    return t.strip()
for r in rows:
    r['_clean']=clean_text(r['oracle_combined']+' '+(r['type_line'] or ''))

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
SETS=sorted(set(r['set'] for r in rows))
set_mean={s: statistics.mean([r['gih_wr'] for r in rows if r['set']==s]) for s in SETS}

def num_matrix(idxs):
    return np.array([[float(rows[i][k]) for k in NUM_KEYS]+[1.0 if rows[i]['rarity']==rar else 0.0 for rar in RAR] for i in idxs])

results=[]
for held in SETS:
    tr=[i for i,r in enumerate(rows) if r['set']!=held]
    te=[i for i,r in enumerate(rows) if r['set']==held]
    ytr=np.array([rows[i]['gih_wr']-set_mean[rows[i]['set']] for i in tr])
    wtr=np.array([np.sqrt(rows[i]['gih']) for i in tr])
    yte=np.array([rows[i]['gih_wr'] for i in te])
    # ridge
    vect=TfidfVectorizer(max_features=2000, ngram_range=(1,2), min_df=4, sublinear_tf=True)
    Xt_tr=vect.fit_transform([rows[i]['_clean'] for i in tr])
    Xt_te=vect.transform([rows[i]['_clean'] for i in te])
    sc=StandardScaler(with_mean=False)
    Xn_tr=sc.fit_transform(sparse.csr_matrix(num_matrix(tr)))
    Xn_te=sc.transform(sparse.csr_matrix(num_matrix(te)))
    Xr_tr=sparse.hstack([Xn_tr,Xt_tr]).tocsr(); Xr_te=sparse.hstack([Xn_te,Xt_te]).tocsr()
    m=Ridge(alpha=30.0); m.fit(Xr_tr,ytr,sample_weight=wtr)
    pr=m.predict(Xr_te)
    # hgb
    h=HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=4, min_samples_leaf=20, l2_regularization=5.0, random_state=0)
    h.fit(num_matrix(tr),ytr,sample_weight=wtr)
    ph=h.predict(num_matrix(te))
    for w in [0.5,0.6,0.7]:
        ens=w*pr+(1-w)*ph+set_mean[held]
        rmse=np.sqrt(mean_squared_error(yte,ens))
        results.append((held,w,rmse))
import collections
for w in [0.5,0.6,0.7]:
    rs=[r for h,ww,r in results if ww==w]
    print(f"w_ridge={w} mean_rmse={np.mean(rs):.4f}")
# detail for w=0.6
print("--- w=0.6 per set ---")
for held in SETS:
    for h,w,r in results:
        if h==held and w==0.6:
            # baseline
            yte=np.array([rows[i]['gih_wr'] for i,r in enumerate(rows) if r['set']==held])
            base=np.sqrt(mean_squared_error(yte,[set_mean[held]]*len(yte)))
            print(held, f"ens={r:.4f} base={base:.4f}")
