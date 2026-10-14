#!/usr/bin/env python3
"""Dump a historical set's card texts alongside 17lands stats for manual review."""
import gzip, json, sys, pandas as pd
s = sys.argv[1]; fmt = sys.argv[2] if len(sys.argv) > 2 else 'PremierDraft'
st = pd.read_csv(f'/workspace/work/stats/{s}_{fmt}_card_stats.csv')
cards = [json.loads(l) for l in gzip.open(f'/opt/mtg-data/scryfall/sets/{s.lower()}.jsonl.gz', 'rt') if l.strip()]
def cnum(c):
    d = ''.join(ch for ch in c['collector_number'] if ch.isdigit()); return int(d) if d else 0
cards = [c for c in cards if c['collector_number'].isdigit()]
cards.sort(key=cnum)
seen = set(); main = []
for c in cards:
    if c['name'] in seen: continue
    seen.add(c['name']); main.append(c)
stm = {r['name']: r for r in st.to_dict('records')}
out = []
for c in main:
    r = stm.get(c['name']) or stm.get(c['name'].split(' // ')[0])
    if r is None:
        continue
    sline = f"GIH {100*r['gih_wr']:.1f}% (n={int(r['n_gih'])}) OH {100*r['oh_wr']:.1f} IWD {100*r['iwd']:+.1f} ALSA {r['alsa']:.1f} GP {100*r['gp_wr']:.1f}"
    out.append(f"#{c['collector_number']} [{c['rarity'].upper()[0]}] {c['name']}  {c.get('mana_cost','')}  {c['type_line']}" + (f"  {c['power']}/{c['toughness']}" if 'power' in c else '') + f"   || {sline}")
    if 'card_faces' in c:
        for f in c['card_faces']:
            out.append(f"  == {f['name']} {f.get('mana_cost','')} {f['type_line']}" + (f" {f['power']}/{f['toughness']}" if 'power' in f else ''))
            out.append('     ' + f.get('oracle_text', '').replace('\n', '\n     '))
    else:
        out.append('     ' + (c.get('oracle_text') or '').replace('\n', '\n     '))
    out.append('')
open(f'/workspace/work/{s.lower()}_cards_with_stats.txt', 'w').write('\n'.join(out))
print(len(main), 'cards written')
