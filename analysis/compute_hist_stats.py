#!/usr/bin/env python3
"""Compute 17lands-style per-card statistics (GIH WR, OH WR, GD WR, GNS WR, GP WR, IWD)
for every historical set from the public game data files."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/mtg-data/17lands/sets")
OUT = Path("/workspace/analysis/out")
OUT.mkdir(parents=True, exist_ok=True)

SETS = {
    "HOB": "PremierDraft", "MSH": "PremierDraft", "SOS": "PremierDraft", "TMT": "PremierDraft",
    "ECL": "PremierDraft", "TLA": "PremierDraft", "OM1": "PickTwoDraft", "EOE": "PremierDraft",
    "FIN": "PremierDraft", "TDM": "PremierDraft",
}


def process(code: str, fmt: str) -> pd.DataFrame:
    path = ROOT / code / f"game_data_public.{code}.{fmt}.csv.gz"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cards = [c[len("deck_"):] for c in header if c.startswith("deck_")]
    usecols = ["won", "on_play", "num_turns", "main_colors", "user_game_win_rate_bucket", "rank"]
    for c in cards:
        usecols += [f"opening_hand_{c}", f"drawn_{c}", f"deck_{c}"]
    n = len(cards)
    agg = {k: np.zeros(n) for k in [
        "gp", "gp_w", "oh", "oh_w", "gd", "gd_w", "gih", "gih_w", "gns", "gns_w",
        "deck_copies"]}
    total = 0
    total_w = 0
    colors = {}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=40000, low_memory=False):
        won = chunk["won"].astype(str).isin(["True", "true", "1"]).to_numpy()
        total += len(chunk)
        total_w += won.sum()
        mc = chunk["main_colors"].fillna("").to_numpy()
        for col, w in zip(mc, won):
            a = colors.setdefault(col, [0, 0])
            a[0] += 1
            a[1] += w
        oh = chunk[[f"opening_hand_{c}" for c in cards]].fillna(0).to_numpy(dtype=np.int16) > 0
        dr = chunk[[f"drawn_{c}" for c in cards]].fillna(0).to_numpy(dtype=np.int16) > 0
        dk = chunk[[f"deck_{c}" for c in cards]].fillna(0).to_numpy(dtype=np.int16)
        gp = dk > 0
        gih = oh | dr
        gd = dr & ~oh
        gns = gp & ~gih
        wcol = won[:, None]
        for name, m in [("gp", gp), ("oh", oh), ("gd", gd), ("gih", gih), ("gns", gns)]:
            agg[name] += m.sum(0)
            agg[name + "_w"] += (m & wcol).sum(0)
        agg["deck_copies"] += dk.sum(0)
        print(code, total, file=sys.stderr, flush=True)
    df = pd.DataFrame({"name": cards, **agg})
    df["set"] = code
    df["format"] = fmt
    df["set_games"] = total
    df["set_wr"] = total_w / total
    for k in ["gp", "oh", "gd", "gih", "gns"]:
        df[k + "_wr"] = df[k + "_w"] / df[k].replace(0, np.nan)
    df["iwd"] = df["gih_wr"] - df["gns_wr"]
    cdf = pd.DataFrame([(k, v[0], v[1] / v[0]) for k, v in colors.items()], columns=["main_colors", "games", "wr"])
    cdf["set"] = code
    cdf.sort_values("games", ascending=False).to_csv(OUT / f"colors_{code}.csv", index=False)
    return df


if __name__ == "__main__":
    which = sys.argv[1:] or list(SETS)
    for code in which:
        df = process(code, SETS[code])
        df.to_csv(OUT / f"stats_{code}.csv", index=False)
        print("done", code, df["set_games"].iloc[0], round(df["set_wr"].iloc[0], 4), flush=True)
