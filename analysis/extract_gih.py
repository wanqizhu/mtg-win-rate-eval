#!/usr/bin/env python3
"""Compute 17lands Premier Draft GIH win rates per card for historical sets,
joined to Scryfall objects (main set file; bonus-sheet cards via default-cards bulk).
Cached per set under analysis/cache/."""
import gzip, json, csv
from pathlib import Path
from collections import defaultdict

ROOT = Path("/opt/mtg-data")
SETS = ["FRA","HOB","MSH","SOS","TMT","ECL","TLA","EOE","FIN"]  # OM1 has no PremierDraft files
CACHE = Path("analysis/cache"); CACHE.mkdir(parents=True, exist_ok=True)

def cnum(c):
    try: return int(c["collector_number"])
    except Exception: return 99999

def load_scryfall_file(p):
    out=[]
    with gzip.open(p,"rt") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def card_record(c, source):
    txt = c.get("oracle_text") or ""
    mc = c.get("mana_cost") or ""
    tl = c.get("type_line") or ""
    power, tough = c.get("power"), c.get("toughness")
    if c.get("card_faces"):
        faces=c["card_faces"]
        if not txt: txt="\n//\n".join(f.get("oracle_text","") for f in faces)
        if not mc: mc=faces[0].get("mana_cost","")
        if not tl: tl=faces[0].get("type_line","")
        if power is None: power=faces[0].get("power"); tough=faces[0].get("toughness")
    return {"rarity":c.get("rarity"),"mana_cost":mc,"cmc":c.get("cmc"),
            "type_line":tl,"oracle_text":txt,"power":power,"toughness":tough,
            "colors":"".join(c.get("colors") or []),
            "color_identity":"".join(c.get("color_identity") or []),
            "keywords":"|".join(c.get("keywords") or []),
            "collector_number":c.get("collector_number"),"layout":c.get("layout"),
            "source":source,"set_code":c.get("set")}

def set_lookup(code):
    cards = load_scryfall_file(ROOT/"scryfall"/"sets"/f"{code.lower()}.jsonl.gz")
    byname={}
    for c in sorted(cards, key=cnum):
        front=c["name"].split(" // ")[0]
        byname.setdefault(front, card_record(c,"main"))
    return byname

def gih_for_set(code):
    cache = CACHE/f"{code}_gih.json"
    if cache.exists(): return json.loads(cache.read_text())
    p = ROOT/"17lands"/"sets"/code/f"game_data_public.{code}.PremierDraft.csv.gz"
    seen=defaultdict(lambda:[0,0]); games=0; wins=0
    with gzip.open(p,"rt",newline="") as f:
        r=csv.reader(f); h=next(r); won_i=h.index("won")
        cols={}
        for i,n in enumerate(h):
            for pre in ("opening_hand_","drawn_"):
                if n.startswith(pre): cols.setdefault(n[len(pre):],[]).append(i)
        for row in r:
            games+=1; won=row[won_i] in {"1","True","true","TRUE"}
            if won: wins+=1
            for card, idxs in cols.items():
                if any(row[i] not in ("","0") for i in idxs):
                    seen[card][0]+=1
                    if won: seen[card][1]+=1
    out={"games":games,"wins":wins,"cards":{k:{"gih":v[0],"wins":v[1]} for k,v in seen.items()}}
    cache.write_text(json.dumps(out)); return out

def main():
    all_data={}
    needed_bonus=set()
    per_set={}
    for code in SETS:
        g=gih_for_set(code); sf=set_lookup(code)
        per_set[code]=(g,sf)
        for name in g["cards"]:
            if name not in sf: needed_bonus.add(name)
        print(code, g["games"], round(g["wins"]/g["games"],4), len(g["cards"]), "bonus_needed_cum", len(needed_bonus), flush=True)
    # one pass over default-cards for bonus names (prefer Arena, english, not oversized)
    bonus={}
    if needed_bonus:
        with gzip.open(ROOT/"scryfall"/"default-cards.jsonl.gz","rt") as f:
            for line in f:
                c=json.loads(line)
                front=c["name"].split(" // ")[0]
                if front in needed_bonus:
                    rec=card_record(c,"bonus")
                    prev=bonus.get(front)
                    # prefer a printing whose games include arena and rarity mythic/rare special guest style; keep first arena printing
                    if prev is None and "arena" in (c.get("games") or []):
                        bonus[front]=rec
        print("bonus resolved", len(bonus), "of", len(needed_bonus), "missing", sorted(needed_bonus-set(bonus))[:20])
    rows=[]; report={}
    for code,(g,sf) in per_set.items():
        unmatched=[]
        for name,d in g["cards"].items():
            rec = sf.get(name) or bonus.get(name)
            if rec is None: unmatched.append(name); continue
            rows.append({"set":code,"name":name,"gih":d["gih"],"gih_wins":d["wins"],
                         "gih_wr": d["wins"]/d["gih"] if d["gih"] else None, **rec})
        report[code]={"games":g["games"],"base_wr":g["wins"]/g["games"],
                      "cards":len(g["cards"]),"unmatched":unmatched}
    Path("analysis/historical_gih.json").write_text(json.dumps({"rows":rows,"report":report},indent=1))
    print("rows",len(rows)); print(json.dumps(report,indent=1)[:2000])

if __name__=="__main__":
    main()
