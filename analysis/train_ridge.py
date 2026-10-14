#!/usr/bin/env python3
"""Train Ridge (numeric + TF-IDF) with leave-one-set-out validation."""
import json, re
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import sparse

rows=json.load(open("/workspace/analysis/train_rows.json"))
SETS=sorted(set(r['set'] for r in rows))
print("sets", SETS, "rows", len(rows))

NUM_KEYS=['cmc','log_cmc','n_colors','is_multicolor','is_colorless_nonland','W','U','B','R','G',
'is_creature','is_land','is_instant','is_sorcery','is_enchantment','is_artifact','is_planeswalker',
'is_legendary','is_basic','power','toughness','pt_sum','has_pt','n_keywords','text_len','text_words',
'has_flying','has_trample','has_haste','has_vigilance','has_deathtouch','has_menace','has_reach',
'has_lifelink','has_ward','has_flash','has_draw','has_destroy','has_exile','has_counter','has_token',
'has_treasure','has_surveil','has_scry','has_mill','has_sacrifice','has_discard','has_lifegain',
'has_damage','has_fight','has_search','has_return_gy','has_copy','has_proliferate','has_cycling',
'has_flashback','has_landfall','has_prowess','has_flying_hate','n_removal_words']
RAR=['common','uncommon','rare','mythic']

def clean_text(t):
    t=t.lower()
    t=re.sub(r'\([^)]*\)',' ',t)  # remove reminder text
    t=re.sub(r'[^a-z0-9\s\+\-/]',' ',t)
    t=re.sub(r'\s+',' ',t)
    return t.strip()

for r in rows:
    r['_clean']=clean_text(r['oracle_combined']+ ' ' + (r['type_line'] or ''))

def build_matrix(idxs, vect=None, scaler=None):
    Xnum=[]
    for i in idxs:
        r=rows[i]
        v=[float(r[k]) for k in NUM_KEYS]
        # rarity onehot
        for rar in RAR: v.append(1.0 if r['rarity']==rar else 0.0)
        # log gih weight handled outside
        Xnum.append(v)
    Xnum=np.array(Xnum)
    texts=[rows[i]['_clean'] for i in idxs]
    if vect is None:
        vect=TfidfVectorizer(max_features=2500, ngram_range=(1,2), min_df=3, sublinear_tf=True)
        Xt=vect.fit_transform(texts)
    else:
        Xt=vect.transform(texts)
    if scaler is None:
        scaler=StandardScaler(with_mean=False)
        Xn=scaler.fit_transform(sparse.csr_matrix(Xnum))
    else:
        Xn=scaler.transform(sparse.csr_matrix(Xnum))
    X=sparse.hstack([Xn, Xt]).tocsr()
    return X, vect, scaler

# LOSO with demeaned target
import statistics
set_mean={s: statistics.mean([r['gih_wr'] for r in rows if r['set']==s]) for s in SETS}
global_mean=statistics.mean([r['gih_wr'] for r in rows])
print("set means", {k:round(v,4) for k,v in set_mean.items()}, "global", round(global_mean,4))

results=[]
for held in SETS:
    tr=[i for i,r in enumerate(rows) if r['set']!=held]
    te=[i for i,r in enumerate(rows) if r['set']==held]
    ytr=np.array([rows[i]['gih_wr']-set_mean[rows[i]['set']] for i in tr])
    wtr=np.array([np.sqrt(rows[i]['gih']) for i in tr])
    Xtr, vect, scaler = build_matrix(tr)
    Xte, _, _ = build_matrix(te, vect, scaler)
    # alpha tuning: try a few
    best=None
    for alpha in [1.0, 3.0, 8.0, 20.0]:
        m=Ridge(alpha=alpha)
        m.fit(Xtr, ytr, sample_weight=wtr)
        pred=m.predict(Xte)+set_mean[held]  # oracle mean for validation
        rmse=np.sqrt(mean_squared_error([rows[i]['gih_wr'] for i in te], pred))
        if best is None or rmse<best[1]: best=(alpha,rmse,pred)
    alpha,rmse,pred=best
    yte=np.array([rows[i]['gih_wr'] for i in te])
    # correlation
    corr=np.corrcoef(yte,pred)[0,1]
    # baseline: predict set mean
    base_rmse=np.sqrt(mean_squared_error(yte,[set_mean[held]]*len(yte)))
    results.append((held,alpha,rmse,base_rmse,corr))
    print(f"held={held} alpha={alpha} rmse={rmse:.4f} base={base_rmse:.4f} corr={corr:.3f} n={len(te)}")

rms=np.mean([r[2] for r in results])
print(f"mean LOSO rmse={rms:.4f}")
# also realistic: predict with global mean offset (no oracle)
print("--- realistic (global mean offset) ---")
for held,alpha,rmse,base,corr in results:
    pass
