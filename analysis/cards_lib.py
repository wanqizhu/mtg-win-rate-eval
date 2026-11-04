"""Shared helpers to load Scryfall card objects for historical sets and Star Trek."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

SCRY = Path("/opt/mtg-data/scryfall/sets")


def load_set(code: str) -> list[dict]:
    with gzip.open(SCRY / f"{code.lower()}.jsonl.gz", "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cn_int(card: dict) -> int:
    m = re.match(r"(\d+)", card.get("collector_number", ""))
    return int(m.group(1)) if m else 10**6


def faces(card: dict) -> list[dict]:
    return card.get("card_faces") or [card]


def full_text(card: dict) -> str:
    parts = []
    for f in faces(card):
        bits = [f.get("name", ""), f.get("mana_cost", ""), f.get("type_line", "")]
        if f.get("power") is not None:
            bits.append(f"{f.get('power')}/{f.get('toughness')}")
        if f.get("loyalty") is not None:
            bits.append(f"loyalty {f.get('loyalty')}")
        if f.get("defense") is not None:
            bits.append(f"defense {f.get('defense')}")
        parts.append(" | ".join(b for b in bits if b) + "\n" + (f.get("oracle_text") or ""))
    return "\n//\n".join(parts)


def oracle(card: dict) -> str:
    return "\n".join((f.get("oracle_text") or "") for f in faces(card))


def unique_by_name(cards: list[dict]) -> dict[str, dict]:
    """Prefer the lowest collector number printing of each name."""
    out: dict[str, dict] = {}
    for c in sorted(cards, key=cn_int):
        out.setdefault(c["name"], c)
    return out
