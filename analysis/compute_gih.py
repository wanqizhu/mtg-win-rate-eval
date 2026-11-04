#!/usr/bin/env python3
"""Compute 17lands-style per-card metrics for each historical set's draft game file.

GIH = game where the card was in the opening hand or drawn (17lands "Games In Hand").
Outputs analysis/out/hist_card_stats.csv with one row per (set, card).
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

ROOT = Path("/opt/mtg-data/17lands/sets")
OUT = Path(__file__).resolve().parent / "out"
SETS = ["FRA", "HOB", "MSH", "SOS", "TMT", "ECL", "TLA", "OM1", "EOE", "FIN"]


def game_file(code: str) -> Path:
    p = ROOT / code / f"game_data_public.{code}.PremierDraft.csv.gz"
    if p.exists():
        return p
    return ROOT / code / f"game_data_public.{code}.PickTwoDraft.csv.gz"


def header(path: Path) -> list[str]:
    with gzip.open(path, "rt", newline="") as fh:
        import csv

        return next(csv.reader(fh))


def process(code: str) -> pd.DataFrame:
    path = game_file(code)
    cols = header(path)
    cards = sorted({c[len("deck_"):] for c in cols if c.startswith("deck_")})
    want = ["won", "user_game_win_rate_bucket", "on_play", "num_turns"]
    for card in cards:
        want += [f"opening_hand_{card}", f"drawn_{card}", f"deck_{card}"]
    types = {c: pa.float32() for c in want if c.startswith(("opening_hand_", "drawn_", "deck_"))}
    types["won"] = pa.bool_()
    types["on_play"] = pa.bool_()
    types["user_game_win_rate_bucket"] = pa.float32()
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(include_columns=want, column_types=types),
    )
    n = len(cards)
    acc = {k: np.zeros(n, dtype=np.float64) for k in [
        "gih", "gih_w", "oh", "oh_w", "gd", "gd_w", "gns", "gns_w", "deck", "deck_w",
        "gih_skill", "deck_copies"]}
    games = wins = 0
    skill_sum = 0.0
    for batch in reader:
        won = batch.column("won").to_numpy(zero_copy_only=False).astype(np.float64)
        sk = batch.column("user_game_win_rate_bucket").to_numpy(zero_copy_only=False).astype(np.float64)
        sk = np.nan_to_num(sk, nan=np.nanmean(sk))
        games += len(won)
        wins += won.sum()
        skill_sum += sk.sum()
        oh = np.stack([batch.column(f"opening_hand_{c}").to_numpy(zero_copy_only=False) for c in cards], 1) > 0
        dr = np.stack([batch.column(f"drawn_{c}").to_numpy(zero_copy_only=False) for c in cards], 1) > 0
        dk_n = np.stack([batch.column(f"deck_{c}").to_numpy(zero_copy_only=False) for c in cards], 1)
        dk = dk_n > 0
        gih = oh | dr
        gd = dr & ~oh
        gns = dk & ~gih
        acc["gih"] += gih.sum(0)
        acc["gih_w"] += won @ gih
        acc["oh"] += oh.sum(0)
        acc["oh_w"] += won @ oh
        acc["gd"] += gd.sum(0)
        acc["gd_w"] += won @ gd
        acc["gns"] += gns.sum(0)
        acc["gns_w"] += won @ gns
        acc["deck"] += dk.sum(0)
        acc["deck_w"] += won @ dk
        acc["gih_skill"] += sk @ gih
        acc["deck_copies"] += dk_n.sum(0)
        print(code, games, file=sys.stderr, flush=True)
    df = pd.DataFrame({"set": code, "name": cards, **acc})
    df["games_total"] = games
    df["set_wr"] = wins / games
    df["set_skill"] = skill_sum / games
    df["gih_wr"] = df.gih_w / df.gih
    df["oh_wr"] = df.oh_w / df.oh
    df["gd_wr"] = df.gd_w / df.gd
    df["gns_wr"] = df.gns_w / df.gns
    df["iwd"] = df.gih_wr - df.gns_wr
    df["gih_skill"] = df.gih_skill / df.gih
    return df


def main() -> None:
    sets = sys.argv[1:] or SETS
    OUT.mkdir(exist_ok=True)
    for code in sets:
        df = process(code)
        df.to_csv(OUT / f"stats_{code}.csv", index=False)
    frames = [pd.read_csv(p) for p in sorted(OUT.glob("stats_*.csv"))]
    pd.concat(frames).to_csv(OUT / "hist_card_stats.csv", index=False)


if __name__ == "__main__":
    main()
