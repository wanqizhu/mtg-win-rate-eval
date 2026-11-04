#!/usr/bin/env python3
"""Stream 17lands Premier Draft game files into per-card GIH win rates.

GIH matches the 17lands definition used by scripts/demo_winrate.py: a card is
in hand when opening_hand or drawn is present and not zero. Win rate is the
fraction of those games the recorded player won. No rank filter.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/opt/mtg-data/17lands/sets")
OUT = Path("/tmp/mtg_analysis/historical_gih")

# OM1 has no Premier Draft file. PickTwoDraft is recorded separately and is
# not mixed into the Premier Draft training label.
PREMIER = {
    "FRA": "game_data_public.FRA.PremierDraft.csv.gz",
    "HOB": "game_data_public.HOB.PremierDraft.csv.gz",
    "MSH": "game_data_public.MSH.PremierDraft.csv.gz",
    "SOS": "game_data_public.SOS.PremierDraft.csv.gz",
    "TMT": "game_data_public.TMT.PremierDraft.csv.gz",
    "ECL": "game_data_public.ECL.PremierDraft.csv.gz",
    "TLA": "game_data_public.TLA.PremierDraft.csv.gz",
    "EOE": "game_data_public.EOE.PremierDraft.csv.gz",
    "FIN": "game_data_public.FIN.PremierDraft.csv.gz",
}


def _truthy_count(value: str) -> int:
    if value in {"", "0"}:
        return 0
    try:
        return int(value)
    except ValueError:
        return 1 if value.lower() in {"true", "1"} else 0


def compute_set(code: str, filename: str) -> dict:
    path = ROOT / code / filename
    games = 0
    wins = 0
    on_play = 0
    on_play_wins = 0
    turns = 0
    color_games: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    # card -> [gih, gih_wins, oh, oh_wins, drawn_only, drawn_only_wins, deck, deck_wins]
    cards: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0])

    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: i for i, name in enumerate(header)}
        won_i = index["won"]
        play_i = index.get("on_play")
        turns_i = index.get("num_turns")
        colors_i = index.get("main_colors")
        names = []
        for name in header:
            if not name.startswith("opening_hand_"):
                continue
            card = name[len("opening_hand_") :]
            names.append(
                (
                    card,
                    index[name],
                    index[f"drawn_{card}"],
                    index.get(f"deck_{card}"),
                )
            )
        for row in reader:
            games += 1
            won = row[won_i] in {"1", "True", "true", "TRUE"}
            won_i_val = 1 if won else 0
            if won:
                wins += 1
            if play_i is not None and row[play_i] in {"1", "True", "true", "TRUE"}:
                on_play += 1
                on_play_wins += won_i_val
            if turns_i is not None and row[turns_i] not in {"", "0"}:
                try:
                    turns += int(float(row[turns_i]))
                except ValueError:
                    pass
            if colors_i is not None:
                bucket = color_games[row[colors_i] or "?"]
                bucket[0] += 1
                bucket[1] += won_i_val
            for card, oh_i, dr_i, deck_i in names:
                oh = _truthy_count(row[oh_i])
                dr = _truthy_count(row[dr_i])
                stats = None
                if oh or dr:
                    stats = cards[card]
                    stats[0] += 1
                    stats[1] += won_i_val
                    if oh:
                        stats[2] += 1
                        stats[3] += won_i_val
                    else:
                        stats[4] += 1
                        stats[5] += won_i_val
                if deck_i is not None and _truthy_count(row[deck_i]):
                    if stats is None:
                        stats = cards[card]
                    stats[6] += 1
                    stats[7] += won_i_val

    rows = []
    for card, (gih, gih_wins, oh, oh_wins, drawn, drawn_wins, deck, deck_wins) in cards.items():
        rows.append(
            {
                "set": code,
                "card": card,
                "gih_n": gih,
                "gih_wins": gih_wins,
                "gih_wr": (gih_wins / gih) if gih else None,
                "oh_n": oh,
                "oh_wr": (oh_wins / oh) if oh else None,
                "drawn_n": drawn,
                "drawn_wr": (drawn_wins / drawn) if drawn else None,
                "deck_n": deck,
                "deck_wr": (deck_wins / deck) if deck else None,
            }
        )
    rows.sort(key=lambda item: (-(item["gih_wr"] or 0), -item["gih_n"]))
    summary = {
        "set": code,
        "games": games,
        "win_rate": wins / games if games else None,
        "on_play_games": on_play,
        "on_play_wr": on_play_wins / on_play if on_play else None,
        "mean_turns": turns / games if games else None,
        "cards": len(rows),
        "colors": {
            color: {"games": n, "wr": w / n if n else None}
            for color, (n, w) in sorted(color_games.items(), key=lambda kv: -kv[1][0])
        },
    }
    return {"summary": summary, "cards": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for code, filename in PREMIER.items():
        print(f"computing {code}...", flush=True)
        result = compute_set(code, filename)
        summaries.append(result["summary"])
        out = OUT / f"{code}.json"
        out.write_text(json.dumps(result["cards"]))
        print(
            f"  games={result['summary']['games']} wr={result['summary']['win_rate']:.4f} "
            f"cards={result['summary']['cards']} turns={result['summary']['mean_turns']:.2f}",
            flush=True,
        )
    (OUT / "summaries.json").write_text(json.dumps(summaries, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
