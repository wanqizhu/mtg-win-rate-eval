#!/usr/bin/env python3
"""Build a unified Scryfall + 17lands card catalog for all 10 historical sets and Reality Fracture (FRA)."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

ROOT = Path("/opt/mtg-data")
OUT_DIR = Path("/workspace/analysis")
HISTORICAL_SETS = ["HOB", "MSH", "SOS", "TMT", "ECL", "TLA", "OM1", "EOE", "FIN", "TDM"]


def canonical_sf_card(c: dict, is_bonus: bool = False) -> dict:
    """Extract relevant fields from a Scryfall card object."""
    card_faces = c.get("card_faces") or []
    front_name = c.get("name", "")
    if " // " in front_name:
        front_name = front_name.split(" // ")[0]

    # Also track Arena/printed name (especially for OM1)
    printed_name = c.get("printed_name")
    if not printed_name and card_faces:
        printed_name = card_faces[0].get("printed_name")

    mana_cost = c.get("mana_cost")
    if mana_cost is None and card_faces:
        mana_cost = card_faces[0].get("mana_cost", "")

    oracle_text = c.get("oracle_text")
    if oracle_text is None and card_faces:
        oracle_text = "\n//\n".join(f.get("oracle_text", "") for f in card_faces)

    type_line = c.get("type_line") or ""
    front_type_line = card_faces[0].get("type_line", type_line) if card_faces else type_line
    back_type_line = card_faces[1].get("type_line", "") if len(card_faces) > 1 else ""

    power = c.get("power")
    toughness = c.get("toughness")
    loyalty = c.get("loyalty")
    if power is None and card_faces:
        power = card_faces[0].get("power")
        toughness = card_faces[0].get("toughness")
    if loyalty is None and card_faces:
        loyalty = card_faces[0].get("loyalty")

    colors = c.get("colors")
    if colors is None and card_faces:
        colors = card_faces[0].get("colors") or []

    faces_clean = []
    for f in card_faces:
        faces_clean.append(
            {
                "name": f.get("name"),
                "printed_name": f.get("printed_name"),
                "mana_cost": f.get("mana_cost"),
                "type_line": f.get("type_line"),
                "oracle_text": f.get("oracle_text"),
                "power": f.get("power"),
                "toughness": f.get("toughness"),
                "loyalty": f.get("loyalty"),
                "colors": f.get("colors"),
            }
        )

    return {
        "name": c.get("name"),
        "front_name": front_name,
        "printed_name": printed_name,
        "set": (c.get("set") or "").upper(),
        "collector_number": str(c.get("collector_number") or ""),
        "rarity": c.get("rarity"),
        "layout": c.get("layout"),
        "cmc": float(c.get("cmc") or 0.0),
        "mana_cost": mana_cost or "",
        "type_line": type_line,
        "front_type_line": front_type_line,
        "back_type_line": back_type_line,
        "oracle_text": oracle_text or "",
        "power": power,
        "toughness": toughness,
        "loyalty": loyalty,
        "colors": colors or [],
        "color_identity": c.get("color_identity") or [],
        "keywords": c.get("keywords") or [],
        "produced_mana": c.get("produced_mana") or [],
        "card_faces": faces_clean,
        "is_bonus_sheet": is_bonus,
    }


def cn_int(cn: str) -> int:
    digits = "".join(ch for ch in cn if ch.isdigit())
    return int(digits) if digits else 999999


def load_set_jsonl(set_code: str) -> list[dict]:
    path = ROOT / "scryfall" / "sets" / f"{set_code.lower()}.jsonl.gz"
    cards = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            cards.append(json.loads(line))
    # Sort by collector number so main-set versions come before alt-frame variants
    cards.sort(key=lambda c: (cn_int(str(c.get("collector_number", ""))), str(c.get("collector_number", ""))))
    return cards


def main() -> None:
    game_stats = json.loads((OUT_DIR / "historical_game_stats.json").read_text())

    # 1. Load 17lands cards.csv for rarity/bonus sheet metadata on historical sets
    cards_17_meta: dict[tuple[str, str], dict] = {}
    with open(ROOT / "17lands" / "cards" / "cards.csv", "rt") as f:
        for row in csv.DictReader(f):
            if row["is_booster"] == "True":
                key = (row["expansion"].upper(), row["name"])
                if key not in cards_17_meta:
                    cards_17_meta[key] = row

    # 2. First match historical sets against their set jsonl.gz
    needed_global_names: set[str] = set()
    set_card_maps: dict[str, dict[str, dict]] = {}

    for s in HISTORICAL_SETS:
        primary_key = f"{s}_PremierDraft" if f"{s}_PremierDraft" in game_stats else f"{s}_PickTwoDraft"
        card_names_17 = set(game_stats[primary_key]["cards"].keys())
        raw_sf = load_set_jsonl(s)

        by_17_name: dict[str, dict] = {}
        for c in raw_sf:
            can = canonical_sf_card(c, is_bonus=False)
            candidates = [
                can["name"],
                can["front_name"],
                can["printed_name"],
            ]
            if can["card_faces"]:
                f0 = can["card_faces"][0]
                candidates.extend([f0.get("name"), f0.get("printed_name")])
                if len(can["card_faces"]) >= 2:
                    p0 = can["card_faces"][0].get("printed_name") or can["card_faces"][0].get("name")
                    p1 = can["card_faces"][1].get("printed_name") or can["card_faces"][1].get("name")
                    if p0 and p1:
                        candidates.append(f"{p0} // {p1}")
                        candidates.append(p0)
            for cand in list(candidates):
                if cand:
                    ascii_q = "".join(ch if ord(ch) < 128 else "?" for ch in cand)
                    if ascii_q != cand:
                        candidates.append(ascii_q)
            for cand in candidates:
                if cand and cand in card_names_17 and cand not in by_17_name:
                    by_17_name[cand] = can

        missing = card_names_17 - set(by_17_name.keys())
        needed_global_names |= missing
        set_card_maps[s] = by_17_name
        print(f"{s}: matched {len(by_17_name)}/{len(card_names_17)} in set jsonl, missing {len(missing)}")

    # 3. Scan default-cards.jsonl.gz once for all missing bonus sheet cards
    global_lookup: dict[str, dict] = {}
    if needed_global_names:
        print(f"Scanning default-cards.jsonl.gz for {len(needed_global_names)} bonus sheet cards...")
        with gzip.open(ROOT / "scryfall" / "default-cards.jsonl.gz", "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                if c.get("layout") in {"art_series", "token", "double_faced_token", "emblem"}:
                    continue
                name = c.get("name") or ""
                front = name.split(" // ")[0] if " // " in name else name
                printed = c.get("printed_name") or ""
                ascii_q = "".join(ch if ord(ch) < 128 else "?" for ch in name)
                for cand in (name, front, printed, ascii_q):
                    if cand in needed_global_names and cand not in global_lookup:
                        global_lookup[cand] = canonical_sf_card(c, is_bonus=True)

    for s in HISTORICAL_SETS:
        primary_key = f"{s}_PremierDraft" if f"{s}_PremierDraft" in game_stats else f"{s}_PickTwoDraft"
        card_names_17 = set(game_stats[primary_key]["cards"].keys())
        by_17_name = set_card_maps[s]
        for m in sorted(card_names_17 - set(by_17_name.keys())):
            if m in global_lookup:
                entry = dict(global_lookup[m])
                # Check if 17lands cards.csv has rarity override for this expansion
                meta = cards_17_meta.get((s, m))
                if meta and meta.get("rarity"):
                    entry["rarity"] = meta["rarity"]
                by_17_name[m] = entry
            else:
                print(f"WARNING: Still unmatched in {s}: {m}")
        print(f"{s}: final matched {len(by_17_name)}/{len(card_names_17)}")

    # 4. Build Reality Fracture (FRA) card catalog
    raw_fra = load_set_jsonl("FRA")
    fra_by_name: dict[str, dict] = {}
    for c in raw_fra:
        name = c["name"]
        if name not in fra_by_name:
            can = canonical_sf_card(c, is_bonus=False)
            can["is_legendary_sheet"] = 195 <= cn_int(can["collector_number"]) <= 280
            can["is_basic_land"] = 281 <= cn_int(can["collector_number"]) <= 290
            can["is_special_guest"] = False
            fra_by_name[name] = can

    # Also load Special Guests for FRA
    spg_path = ROOT / "scryfall" / "sets" / "fra-special-guests.jsonl"
    fra_spg_by_name: dict[str, dict] = {}
    if spg_path.exists():
        for line in spg_path.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            can = canonical_sf_card(c, is_bonus=True)
            can["is_legendary_sheet"] = False
            can["is_basic_land"] = False
            can["is_special_guest"] = True
            fra_spg_by_name[c["name"]] = can

    catalog = {
        "historical": set_card_maps,
        "fra_main": fra_by_name,
        "fra_special_guests": fra_spg_by_name,
    }
    out_path = OUT_DIR / "card_catalog.json"
    out_path.write_text(json.dumps(catalog, indent=2) + "\n")
    print(
        f"Saved catalog: FRA main unique cards={len(fra_by_name)}, FRA special guests={len(fra_spg_by_name)}"
    )


if __name__ == "__main__":
    main()
