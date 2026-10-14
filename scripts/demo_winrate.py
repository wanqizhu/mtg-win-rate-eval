#!/usr/bin/env python3
"""Hello-world: historical GIH win rates, Reality Fracture cards, and a leakage check."""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/opt/mtg-data")
GAME = ROOT / "17lands" / "sets" / "HOB" / "game_data_public.HOB.PremierDraft.csv.gz"
FRA = ROOT / "scryfall" / "sets" / "fra.jsonl.gz"
RULES = ROOT / "rules" / "MagicCompRules-20260925.txt"


def gih_win_rates(path: Path) -> dict:
    games = 0
    wins = 0
    seen = defaultdict(lambda: [0, 0])  # games in hand, wins
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        won_index = header.index("won")
        expansion_index = header.index("expansion")
        hand_cols = []
        for index, name in enumerate(header):
            if name.startswith("opening_hand_") or name.startswith("drawn_"):
                card = name.split("_", 2)[-1] if name.startswith("opening_hand_") else name[len("drawn_") :]
                # Prefixes are opening_hand_ and drawn_; the card name is the remainder.
                if name.startswith("opening_hand_"):
                    card = name[len("opening_hand_") :]
                else:
                    card = name[len("drawn_") :]
                hand_cols.append((index, card))
        # Pair opening hand and drawn columns for the same card.
        paired: dict[str, list[int]] = defaultdict(list)
        for index, card in hand_cols:
            paired[card].append(index)
        for row in reader:
            games += 1
            if row[expansion_index] != "HOB":
                raise SystemExit(f"unexpected expansion in HOB file: {row[expansion_index]}")
            won = row[won_index] in {"1", "True", "true", "TRUE"}
            if won:
                wins += 1
            for card, indexes in paired.items():
                in_hand = False
                for index in indexes:
                    value = row[index]
                    if value not in {"", "0"}:
                        in_hand = True
                        break
                if in_hand:
                    seen[card][0] += 1
                    if won:
                        seen[card][1] += 1
    ranked = []
    for card, (gih, gih_wins) in seen.items():
        if gih < 200:
            continue
        ranked.append({"card": card, "gih": gih, "gih_win_rate": round(gih_wins / gih, 4)})
    ranked.sort(key=lambda item: item["gih_win_rate"], reverse=True)
    return {
        "file": str(path),
        "games": games,
        "game_win_rate": round(wins / games, 4) if games else None,
        "cards_with_at_least_200_gih": len(ranked),
        "highest_gih_win_rate": ranked[:8],
        "lowest_gih_win_rate": list(reversed(ranked[-5:])),
    }


def fra_summary(path: Path) -> dict:
    rarities: dict[str, int] = defaultdict(int)
    cards = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            card = json.loads(line)
            rarities[card.get("rarity") or "unknown"] += 1
            if len(cards) < 3:
                cards.append(
                    {
                        "name": card.get("name"),
                        "mana_cost": card.get("mana_cost"),
                        "type_line": card.get("type_line"),
                        "rarity": card.get("rarity"),
                        "oracle_text": card.get("oracle_text"),
                    }
                )
    return {"cards": sum(rarities.values()), "rarities": dict(rarities), "samples": cards}


def main() -> None:
    report_path = ROOT / "manifests" / "leakage_report.json"
    leakage = json.loads(report_path.read_text())
    if leakage.get("reality_fracture_seventeenlands_rows") != 0:
        raise SystemExit("leakage report is not clean")
    if leakage.get("reality_fracture_seventeenlands_files") != 0:
        raise SystemExit("leakage report found Reality Fracture 17lands files")
    rules_head = RULES.read_text(errors="replace").splitlines()[:12]
    report = {
        "leakage": {
            "seventeenlands_files": leakage["seventeenlands_file_count"],
            "reality_fracture_seventeenlands_files": 0,
            "reality_fracture_seventeenlands_rows": 0,
        },
        "historical_premier_draft": gih_win_rates(GAME),
        "reality_fracture_scryfall": fra_summary(FRA),
        "comprehensive_rules_head": rules_head,
    }
    out = ROOT / "manifests" / "demo_winrate.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
