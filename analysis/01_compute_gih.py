"""Compute GIH win rates for all historical sets. Efficient row-wise scan."""
import gzip, csv, json
from pathlib import Path
from collections import defaultdict

ROOT = Path("/opt/mtg-data")
OUT = Path("/agent/repos/mtg-win-rate-eval/analysis/gih_by_set.json")

# Map set -> (game file pattern, format label)
SETS = ["FRA","HOB","MSH","SOS","TMT","ECL","TLA","EOE","FIN"]
# OM1 has no PremierDraft; use PickTwoDraft
SETS_FMT = {s: "PremierDraft" for s in SETS}
SETS_FMT["OM1"] = "PickTwoDraft"

def compute(code, fmt):
    p = ROOT/"17lands"/"sets"/code/f"game_data_public.{code}.{fmt}.csv.gz"
    with gzip.open(p,"rt",newline="") as h:
        r = csv.reader(h)
        header = next(r)
        won_i = header.index("won")
        # GIH columns: opening_hand_* and drawn_*
        idx_by_card = defaultdict(list)
        for i,name in enumerate(header):
            if name.startswith("opening_hand_"):
                idx_by_card[name[len("opening_hand_"):]].append(i)
            elif name.startswith("drawn_"):
                idx_by_card[name[len("drawn_"):]].append(i)
        cards = list(idx_by_card.keys())
        col_idx = [idx_by_card[c] for c in cards]
        gih = [0]*len(cards)
        wins = [0]*len(cards)
        n=0; w=0
        for row in r:
            n+=1
            won = row[won_i] in ("1","True","true","TRUE")
            if won: w+=1
            # for each card, check if any of its cols nonzero
            # micro-opt: local vars
            for j, idxs in enumerate(col_idx):
                hit=False
                for ii in idxs:
                    v=row[ii]
                    if v!="" and v!="0":
                        hit=True; break
                if hit:
                    gih[j]+=1
                    if won: wins[j]+=1
    out=[]
    for j,c in enumerate(cards):
        g=gih[j]
        wr=wins[j]/g if g else None
        out.append({"card":c,"gih":g,"wins":wins[j],"gih_wr":wr})
    out.sort(key=lambda x: (x["gih_wr"] or 0), reverse=True)
    return {"code":code,"format":fmt,"games":n,"win_rate":w/n if n else None,"cards":out}

all_res={}
for code,fmt in SETS_FMT.items():
    print(f"computing {code} {fmt}...", flush=True)
    res=compute(code,fmt)
    print(f"  games={res['games']} win_rate={res['win_rate']:.4f} n_cards={len(res['cards'])}", flush=True)
    all_res[code]=res

OUT.write_text(json.dumps(all_res,indent=1))
print(f"wrote {OUT}")
