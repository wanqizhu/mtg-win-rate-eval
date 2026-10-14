#!/usr/bin/env python3
"""Recompute GiH correctly (numeric zero handling) + join to Scryfall features mapping.
Saves per-set tables restricted to Scryfall set names (including first-face mapping for DFCs)."""
import gzip, csv, json
from pathlib import Path
from collections import defaultdict

SETS = ["HOB","MSH","SOS","TMT","ECL","TLA","OM1","EOE","FIN","TDM"]

def game_file(s):
    base = Path(f"/opt/mtg-data/17lands/sets/{s}")
    c = base / f"game_data_public.{s}.PremierDraft.csv.gz"
    if c.exists(): return c
    c2 = base / f"game_data_public.{s}.PickTwoDraft.csv.gz"
    if c2.exists(): return c2
    cands = list(base.glob("game_data_public.*.csv.gz"))
    return cands[0]

def is_hit(v):
    if v is None: return False
    s = v.strip()
    if s in ("", "0"): return False
    try:
        return float(s) > 0
    except ValueError:
        return True  # non-numeric non-empty -> present

def load_scryfall_names(code):
    with gzip.open(f"/opt/mtg-data/scryfall/sets/{code.lower()}.jsonl.gz",'rt') as f:
        cards=[json.loads(l) for l in f if l.strip()]
    full=set(c['name'] for c in cards)
    first={}
    for c in cards:
        n=c['name']
        ff=n.split('//')[0].strip() if '//' in n else n
        first.setdefault(ff, []).append(c)
    return cards, full, first

for s in SETS:
    p = game_file(s)
    print(f"=== {s} ===", flush=True)
    _, full, first = load_scryfall_names(s)
    games=0; wins=0
    seen=defaultdict(lambda:[0,0])
    with gzip.open(p,'rt',newline='') as h:
        r=csv.reader(h)
        header=next(r)
        won_idx=header.index('won')
        oh=[(i,c[len('opening_hand_'):]) for i,c in enumerate(header) if c.startswith('opening_hand_')]
        dr=[(i,c[len('drawn_'):]) for i,c in enumerate(header) if c.startswith('drawn_')]
        oh_map=defaultdict(list); dr_map=defaultdict(list)
        for i,c in oh: oh_map[c].append(i)
        for i,c in dr: dr_map[c].append(i)
        cards=set(oh_map)|set(dr_map)
        for row in r:
            games+=1
            won=row[won_idx] in ('1','True','true','TRUE')
            if won: wins+=1
            for card in cards:
                hit=False
                for i in oh_map.get(card,()):
                    if is_hit(row[i]): hit=True; break
                if not hit:
                    for i in dr_map.get(card,()):
                        if is_hit(row[i]): hit=True; break
                if hit:
                    seen[card][0]+=1
                    if won: seen[card][1]+=1
    # map to scryfall
    matched=0; unmatched=[]
    out=[]
    for card,(gih,w) in seen.items():
        wr=w/gih if gih else 0
        # determine scryfall full name
        if card in full:
            fname=card; status='full'
        elif card in first:
            # first-face; map to full name (if multiple, pick one? use first-face key)
            cands=first[card]
            fname=cands[0]['name'] if len(cands)==1 else card+ ' [MULTI]'
            status='first-face'
        else:
            fname=card; status='non-set (bonus/token/other)'
        out.append({"card_17lands":card,"scryfall_name":fname,"match":status,"gih":gih,"gih_wr":wr})
    out.sort(key=lambda x:-x["gih_wr"])
    n_match=sum(1 for x in out if x['match']!='non-set (bonus/token/other)')
    print(f"games={games} base_wr={wins/games:.4f} game_cards={len(out)} matched={n_match}")
    # show non-set top examples
    nons=[x for x in out if x['match'].startswith('non-set')]
    print(f"  non-set cards: {len(nons)} e.g.", [(x['card_17lands'],x['gih'],round(x['gih_wr'],4)) for x in sorted(nons,key=lambda x:-x['gih'])[:8]])
    # matched extremes (min 200)
    m200=[x for x in out if x['match']!='non-set (bonus/token/other)' and x['gih']>=200]
    m200s=sorted(m200,key=lambda x:-x['gih_wr'])
    print("  matched top:", [(x['card_17lands'],x['gih'],round(x['gih_wr'],4)) for x in m200s[:6]])
    print("  matched bot:", [(x['card_17lands'],x['gih'],round(x['gih_wr'],4)) for x in m200s[-5:]])
    Path(f"/workspace/analysis/gih2_{s}.json").write_text(json.dumps(
        {"set":s,"games":games,"wins":wins,"cards":out}, indent=1))
print("done")
