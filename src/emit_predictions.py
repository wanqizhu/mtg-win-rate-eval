"""Write the Premier Draft GIH table from the card-by-card rates."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd

try:
    from hand_rates import RATES
except ImportError:  # python -m src.emit_predictions
    from src.hand_rates import RATES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "predictions"
ANALYSIS = ROOT / "analysis"


def collector_number(card: dict) -> int:
    digits = []
    for char in card["collector_number"]:
        if char.isdigit():
            digits.append(char)
        else:
            break
    return int("".join(digits))


def load_main_set() -> list[dict]:
    path = Path("/opt/mtg-data/scryfall/sets/trk.jsonl.gz")
    cards = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.strip():
                cards.append(json.loads(line))
    english = [card for card in cards if card.get("lang") == "en"]
    english.sort(key=collector_number)
    seen = set()
    main = []
    for card in english:
        number = collector_number(card)
        if number > 311 or card["name"] in seen:
            continue
        seen.add(card["name"])
        main.append(card)
    return main


def notes_by_number() -> dict[int, str]:
    text = Path(__file__).with_name("hand_rates.py").read_text()
    found = {}
    for match in re.finditer(r"^    (\d+): [0-9.]+,\s*# ?(.*)$", text, flags=re.M):
        found[int(match.group(1))] = match.group(2).strip()
    return found


def main() -> None:
    notes = notes_by_number()
    rows = []
    for card in load_main_set():
        number = collector_number(card)
        if number not in RATES:
            raise SystemExit(f"missing rate for {number} {card['name']}")
        rate = RATES[number]
        rows.append({
            "name": card["name"],
            "collector_number": card["collector_number"],
            "rarity": card["rarity"],
            "mana_cost": card.get("mana_cost") or "",
            "type_line": card["type_line"],
            "colors": "".join(card.get("colors") or []),
            "gih_win_rate": f"{rate:.4f}",
            "note": notes.get(number, ""),
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 311:
        raise SystemExit(f"expected 311 cards, got {len(frame)}")
    OUT.mkdir(exist_ok=True)
    ANALYSIS.mkdir(exist_ok=True)
    table_cols = [
        "name", "collector_number", "rarity", "mana_cost", "type_line", "colors", "gih_win_rate",
    ]
    frame[table_cols].to_csv(OUT / "trk_premier_draft_gih.csv", index=False)
    frame.to_csv(ANALYSIS / "prediction_notes.csv", index=False)
    rates = frame["gih_win_rate"].astype(float)
    print(f"cards {len(frame)}  mean {rates.mean():.4f}  std {rates.std():.4f}")
    print(frame.assign(rate=rates).groupby("rarity").rate.agg(["count", "mean", "std", "min", "max"]).round(4))
    print(frame.assign(rate=rates).nlargest(8, "rate")[["gih_win_rate", "rarity", "name"]].to_string(index=False))


if __name__ == "__main__":
    main()
