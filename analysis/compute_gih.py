#!/usr/bin/env python3
"""Compute GiH win rates for all 10 historical sets (PremierDraft game data)."""
import gzip, csv, json
from pathlib import Path
from collections import defaultdict

SETS = ["HOB","MSH","SOS","TMT","ECL","TLA","OM1","EOE","FIN","TDM"]
# OM1 has no PremierDraft; uses PickTwoDraft. Handle per-set file choice.
# Map set -> game file to use (prefer PremierDraft, fallback PickTwoDraft)
import os

def find_game_file(setcode):
    base = Path(f"/opt/mtg-data/17lands/sets/{setcode}")
    # prefer PremierDraft game
    cand = base / f"game_data_public.{setcode}.PremierDraft.csv.gz"
    if cand.exists():
        return cand
    cand2 = base / f"game_data_public.{setcode}.PickTwoDraft.csv.gz"
    if cand2.exists():
        return cand2
    # OM1 PickTwoTradDraft?
    for p in base.glob("game_data_public.*.csv.gz"):
        return p
    return None

for s in SETS:
    print(s, find_game_file(s))

def compute_gih(path):
    games=0; wins=0
    seen=defaultdict(lambda:[0,0])
    with gzip.open(path,'rt',newline='') as h:
        r=csv.reader(h)
        header=next(r)
        won_idx=header.index('won')
        oh_cols=[(i,c[len('opening_hand_'):]) for i,c in enumerate(header) if c.startswith('opening_hand_')]
        dr_map=defaultdict(list)
        for i,c in enumerate(header):
            if c.startswith('drawn_'):
                dr_map[c[len('drawn_'):]].append(i)
        oh_map=defaultdict(list)
        for i,card in oh_cols:
            oh_map[card].append(i)
        cards=set(list(oh_map.keys())+list(dr_map.keys()))
        for row in r:
            games+=1
            won=row[won_idx] in ('1','True','true','TRUE')
            if won: wins+=1
            for card in cards:
                hit=False
                for i in oh_map.get(card,()):
                    v=row[i]
                    if v not in ('','0'):
                        hit=True; break
                if not hit:
                    for i in dr_map.get(card,()):
                        v=row[i]
                        if v not in ('','0'):
                            hit=True; break
                if hit:
                    seen[card][0]+=1
                    if won: seen[card][1]+=1
    return games, wins, seen

out={}
for s in SETS:
    p=find_game_file(s)
    print(f"=== {s} {p} ===", flush=True)
    games,wins,seen=compute_gih(p)
    rows=[]
    for card,(gih,w) in seen.items():
        rows.append((card,gih,w/gih if gih else 0))
    rows.sort(key=lambda x:-x[2])
    # stats
    import statistics
    gihs=[g for _,g,_ in rows]
    wrs=[w for _,_,w in rows]
    print(f"games={games} winrate={wins/games:.4f} ncards={len(rows)}")
    # distribution of gih counts
    rows_by_n=sorted(rows,key=lambda x:-x[1])
    print("top5 by gih:", [(c,g,round(w,4)) for c,g,w in [(r[0],r[1],r[2]) for r in rows_by_n[:5]]])
    print("highest WR (min 200 gih):", [(c,g,round(w,4)) for c,g,w in [(c,g,w) for c,g,w in rows if g>=200][:8]])
    print("lowest WR (min 200 gih):", [(c,g,round(w,4)) for c,g,w in [(c,g,w) for c,g,w in rows if g>=200][-5:]])
    # save
    Path(f"/workspace/analysis/gih_{s}.json").write_text(json.dumps(
        {"set":s,"file":str(p),"games":games,"wins":wins,
         "cards":[{"card":c,"gih":g,"gih_wr":w} for c,g,w in rows]}, indent=1))
print("done")
