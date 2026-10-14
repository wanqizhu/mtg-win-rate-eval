#!/usr/bin/env python3
"""Dump a Scryfall set (booster-relevant printings, one per name) as readable text."""
import gzip
import json
import re
import sys
from pathlib import Path

SCRY = Path("/opt/mtg-data/scryfall/sets")


def cn(c):
    m = re.match(r"(\d+)", c["collector_number"])
    return int(m.group(1)) if m else 99999


def load(code):
    if code == "fra-special-guests":
        lines = (SCRY / "fra-special-guests.jsonl").read_text().splitlines()
    else:
        lines = gzip.open(SCRY / f"{code}.jsonl.gz", "rt").read().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def faces(c):
    if "card_faces" in c:
        return c["card_faces"]
    return [c]


def fmt(c):
    out = [f"#{c['collector_number']} {c['name']}  [{c['rarity']}]"]
    for f in faces(c):
        pt = f" {f.get('power')}/{f.get('toughness')}" if f.get("power") is not None else ""
        loy = f" loyalty {f.get('loyalty')}" if f.get("loyalty") else ""
        out.append(f"  {f.get('name')} {f.get('mana_cost','')} | {f.get('type_line','')}{pt}{loy}")
        txt = (f.get("oracle_text") or "").replace("\n", "\n    ")
        out.append(f"    {txt}")
    return "\n".join(out)


def main_set(code, max_cn=None):
    cards = load(code)
    seen = set()
    res = []
    for c in sorted(cards, key=cn):
        if c["name"] in seen:
            continue
        if max_cn and cn(c) > max_cn:
            continue
        if "Basic Land" in c.get("type_line", ""):
            continue
        seen.add(c["name"])
        res.append(c)
    return res


if __name__ == "__main__":
    code = sys.argv[1]
    max_cn = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for c in main_set(code, max_cn):
        print(fmt(c))
