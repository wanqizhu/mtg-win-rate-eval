#!/usr/bin/env python3
"""Join historical 17lands card stats with Scryfall card objects.

Writes analysis/out/hist_joined.csv (one row per set+card) and reports name-match coverage.
"""

from __future__ import annotations

import json

import pandas as pd

from cards_lib import cn_int, faces, full_text, load_set, oracle

SETS = ["FRA", "HOB", "MSH", "SOS", "TMT", "ECL", "TLA", "OM1", "EOE", "FIN"]
OUT = "out"


def all_printings_by_name() -> dict[str, dict]:
    """Index the full default-cards bulk by name and front-face name for bonus-sheet lookups."""
    import gzip

    idx: dict[str, dict] = {}
    with gzip.open("/opt/mtg-data/scryfall/default-cards.jsonl.gz", "rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("lang") != "en" or c.get("layout") in ("token", "art_series", "emblem", "double_faced_token"):
                continue
            for key in {c["name"], faces(c)[0].get("name", c["name"])}:
                prev = idx.get(key)
                if prev is None or (prev.get("digital") and not c.get("digital")):
                    idx[key] = c
    return idx


def card_row(c: dict, source: str) -> dict:
    f0 = faces(c)[0]
    colors = c.get("colors")
    if colors is None:
        colors = sorted({x for f in faces(c) for x in f.get("colors", [])})
    return {
        "source": source,
        "scry_set": c.get("set"),
        "collector_number": c.get("collector_number"),
        "rarity": c.get("rarity"),
        "mana_cost": f0.get("mana_cost", c.get("mana_cost", "")),
        "cmc": c.get("cmc"),
        "colors": "".join(colors),
        "color_identity": "".join(c.get("color_identity", [])),
        "type_line": c.get("type_line"),
        "power": f0.get("power"),
        "toughness": f0.get("toughness"),
        "layout": c.get("layout"),
        "keywords": ";".join(c.get("keywords", [])),
        "oracle": oracle(c),
        "full_text": full_text(c),
    }


def main() -> None:
    stats = pd.read_csv(f"{OUT}/hist_card_stats.csv")
    bulk = all_printings_by_name()
    rows = []
    missing = []
    for code in SETS:
        own = load_set(code)
        by_name: dict[str, dict] = {}
        for c in sorted(own, key=cn_int):
            by_name.setdefault(c["name"], c)
            by_name.setdefault(faces(c)[0].get("name", c["name"]), c)
            if c.get("printed_name"):
                by_name.setdefault(c["printed_name"], c)
            for f in c.get("card_faces") or []:
                if f.get("printed_name"):
                    by_name.setdefault(f["printed_name"], c)
        sub = stats[stats.set == code]
        for _, r in sub.iterrows():
            name = r["name"]
            c = by_name.get(name)
            src = "set"
            if c is None:
                c = bulk.get(name)
                src = "bonus"
            if c is None and " // " not in name:
                missing.append((code, name))
                continue
            d = r.to_dict()
            d.update(card_row(c, src))
            rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/hist_joined.csv", index=False)
    print("rows", len(df), "missing", len(missing))
    print(missing[:50])
    print(df.groupby(["set", "source"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
