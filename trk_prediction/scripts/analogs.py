import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0,'/tmp/trk')
h=pd.read_csv('hist/all_cards.csv'); h=h[(h.gih>=500)&(h.in_set_file)&(~h.type_line.str.startswith('Basic Land'))].copy()
h['set_mean']=h.groupby('set').apply(lambda g: np.average(g.gih_wr,weights=g.gih)).reindex(h.set).values
h['rel']=h.gih_wr-h.set_mean
t=pd.read_csv('trk_attrs.csv'); t=t[~t.type_line.str.startswith('Basic Land')].reset_index(drop=True)
def doc(r):
    tl=str(r.type_line); txt=str(r.oracle_text) if isinstance(r.oracle_text,str) else ''
    import re
    txt=re.sub(r'\([^)]*\)','',txt)
    return f"{r.rarity} {tl} cmc{int(r.cmc) if pd.notna(r.cmc) else 0} pow{r.power} tou{r.toughness} "+txt
vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True)
A=vec.fit_transform(h.apply(doc,axis=1)); B=vec.transform(t.apply(doc,axis=1))
S=cosine_similarity(B,A)
rows=[]
with open('trk_analogs.txt','w') as f:
    for i,r in t.iterrows():
        top=np.argsort(-S[i])[:6]
        f.write(f"\n### {r['name']} [{r.rarity}] {r.mana_cost} {r.type_line} {r.power}/{r.toughness}\n    {str(r.oracle_text).replace(chr(10),' | ')[:300]}\n")
        for j in top:
            hr=h.iloc[j]
            f.write(f"   {S[i,j]:.2f} {hr.set} {hr['name']} [{hr.rarity}] {hr.mana_cost} {hr.type_line} {hr.power}/{hr.toughness} GIH={hr.gih_wr:.3f} rel={hr.rel:+.3f} n={int(hr.gih)}\n      {str(hr.oracle_text).replace(chr(10),' | ')[:200]}\n")
        rows.append({'name':r['name'],'analog_rel_mean':np.average([h.iloc[j].rel for j in top[:3]],weights=[S[i,j] for j in top[:3]])})
pd.DataFrame(rows).to_csv('trk_analog_scores.csv',index=False)
