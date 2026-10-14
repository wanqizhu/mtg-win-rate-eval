#!/usr/bin/env python3
"""Query historical cards by regex over oracle text / type / cost, print GIH WR."""
import re, sys, pandas as pd
df = pd.read_csv('/workspace/work/training_cards.csv')
df = df[df.n_gih >= 300]
pat = sys.argv[1]
extra = sys.argv[2] if len(sys.argv) > 2 else None
m = df[df.text.fillna('').str.contains(pat, regex=True, case=False)]
if extra:
    m = m.query(extra)
m = m.sort_values('gih_wr', ascending=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 110)
m = m.assign(gih=(100*m.gih_wr).round(1), n=m.n_gih.astype(int), alsa=m.alsa.round(1), text=m.oracle_text_raw.str.replace('\n', ' | ').str.slice(0, 110))
print(m[['set', 'name', 'rarity', 'mana_cost', 'type_line', 'gih', 'n', 'alsa', 'text']].to_string(index=False))
print('N=', len(m), 'mean GIH=', round(100*m.gih_wr.mean(), 2))
