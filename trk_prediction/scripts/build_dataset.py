#!/usr/bin/env python3
"""Join 17lands per-card stats with Scryfall attributes for the 10 historical sets,
and build the equivalent attribute table for TRK. Output: /tmp/trk/hist/all_cards.csv, /tmp/trk/trk_attrs.csv"""
import gzip, json, re, glob
import pandas as pd

SCRY = "/opt/mtg-data/scryfall/sets"
SETS = {"FRA": "PremierDraft", "HOB": "PremierDraft", "MSH": "PremierDraft", "SOS": "PremierDraft",
        "TMT": "PremierDraft", "ECL": "PremierDraft", "TLA": "PremierDraft", "OM1": "PickTwoDraft",
        "EOE": "PremierDraft", "FIN": "PremierDraft"}

def cn(c):
    return int(re.sub(r"\D", "", c["collector_number"]) or 0)

def load_set(code):
    cards = [json.loads(l) for l in gzip.open(f"{SCRY}/{code.lower()}.jsonl.gz", "rt") if l.strip()]
    cards.sort(key=cn)
    out = {}
    for c in cards:
        if code == "TRK" and not (c["collector_number"].isdigit() and cn(c) <= 311):
            continue
        name = c["name"]
        if name in out:
            continue
        out[name] = c
        if "//" in name:
            out.setdefault(name.split(" // ")[0], c)
    return out

def attrs(c):
    faces = c.get("card_faces") or [c]
    front = faces[0]
    text = "\n".join((f.get("oracle_text") or "") for f in faces)
    mana = c.get("mana_cost") or front.get("mana_cost") or ""
    colors = c.get("colors")
    if colors is None:
        colors = front.get("colors", [])
    return {
        "rarity": c["rarity"], "mana_cost": mana, "cmc": c.get("cmc", front.get("cmc", 0)),
        "colors": "".join(sorted(colors)), "color_identity": "".join(sorted(c.get("color_identity", []))),
        "type_line": c["type_line"], "power": front.get("power"), "toughness": front.get("toughness"),
        "loyalty": front.get("loyalty"), "keywords": ";".join(c.get("keywords", [])),
        "oracle_text": text, "layout": c["layout"], "produced_mana": "".join(c.get("produced_mana", []) or []),
        "collector_number": c["collector_number"],
    }

BULK = {}
def load_bulk(needed):
    for l in gzip.open("/opt/mtg-data/scryfall/default-cards.jsonl.gz", "rt"):
        c = json.loads(l)
        if c.get("lang") != "en" or c["name"] not in needed:
            continue
        prev = BULK.get(c["name"])
        # prefer paper printings; among same digital-ness prefer most recent
        key = (0 if c.get("digital") else 1, c["released_at"])
        if prev is None or key > (0 if prev.get("digital") else 1, prev["released_at"]):
            BULK[c["name"]] = c

needed = set()
for code, event in SETS.items():
    stats = pd.read_csv(f"/tmp/trk/hist/{code}_{event}_cards.csv")
    needed |= set(stats["name"])
load_bulk(needed)
print("bulk matched", len(BULK))

rows = []
for code, event in SETS.items():
    stats = pd.read_csv(f"/tmp/trk/hist/{code}_{event}_cards.csv")
    scry = load_set(code)
    matched = 0
    for _, r in stats.iterrows():
        c = scry.get(r["name"])
        in_set = c is not None
        if c is None:
            c = BULK.get(r["name"])
        if c is None:
            continue
        matched += 1
        d = r.to_dict(); d.update(attrs(c)); d["scry_name"] = c["name"]; d["in_set_file"] = in_set; d["print_set"] = c["set"]
        rows.append(d)
    print(code, "stats rows", len(stats), "scryfall booster cards", len(scry), "matched", matched)
hist = pd.DataFrame(rows)
hist.to_csv("/tmp/trk/hist/all_cards.csv", index=False)

trk = load_set("TRK")
trows = []
for name, c in trk.items():
    if name != c["name"]:
        continue
    d = {"set": "TRK", "name": name}; d.update(attrs(c)); trows.append(d)
pd.DataFrame(trows).to_csv("/tmp/trk/trk_attrs.csv", index=False)
print("TRK cards", len(trows))
