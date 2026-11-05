import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, sys, re
sys.path.insert(0,'/tmp/trk')
from features import featurize
pd.set_option('display.width',260); pd.set_option('display.max_rows',2000); pd.set_option('display.max_colwidth',90)
h=pd.read_csv('hist/all_cards.csv'); h=h[(h.gih>=500)&(h.in_set_file)].copy()
h['set_mean']=h.groupby('set').apply(lambda g: np.average(g.gih_wr,weights=g.gih)).reindex(h.set).values
h['rel']=(h.gih_wr-h.set_mean).round(3)
h['txt']=h.oracle_text.fillna('').str.replace('\n',' | ').str.replace(r'\([^)]*\)','',regex=True).str.slice(0,110)
F=featurize(h); h=pd.concat([h,F.add_prefix('f_')],axis=1)
h['gih_wr']=h.gih_wr.round(3); h['pr']=h.play_rate.round(3)
cols=['set','name','rarity','mana_cost','type_line','power','toughness','gih_wr','rel','pr','txt']
def show(mask,title,sort='rel'):
    print(f"\n==== {title} (n={mask.sum()}) mean rel={h[mask].rel.mean():+.3f}")
    print(h[mask].sort_values(sort,ascending=False)[cols].to_string(index=False))
if __name__=='__main__':
    q=sys.argv[1]
    show(h.name.str.contains(q,case=False)|h.oracle_text.fillna('').str.contains(q,case=False), q)
