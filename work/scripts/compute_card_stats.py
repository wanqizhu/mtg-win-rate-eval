#!/usr/bin/env python3
"""Compute 17lands-style per-card statistics (GIH WR, OH WR, GD WR, GP WR, IWD, ALSA, ATA)
for each historical set from the public game_data / draft_data files.

Outputs: work/stats/<SET>_<FORMAT>_card_stats.csv and work/stats/set_summary.csv
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/mtg-data/17lands/sets")
OUT = Path("/workspace/work/stats")
OUT.mkdir(parents=True, exist_ok=True)

SETS = ["TDM", "FIN", "EOE", "OM1", "TLA", "ECL", "TMT", "SOS", "MSH", "HOB"]


def game_stats(path: Path) -> tuple[pd.DataFrame, dict]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    prefixes = ("opening_hand_", "drawn_", "tutored_", "deck_", "sideboard_")
    names = sorted({c[len("opening_hand_"):] for c in header if c.startswith("opening_hand_")})
    # column groups
    oh_cols = [f"opening_hand_{n}" for n in names]
    dr_cols = [f"drawn_{n}" for n in names]
    tu_cols = [f"tutored_{n}" for n in names]
    dk_cols = [f"deck_{n}" for n in names]
    keep = ["won", "num_turns", "on_play", "num_mulligans", "main_colors"] + oh_cols + dr_cols + tu_cols + dk_cols
    agg = {
        "n_gih": np.zeros(len(names)), "w_gih": np.zeros(len(names)),
        "n_oh": np.zeros(len(names)), "w_oh": np.zeros(len(names)),
        "n_gd": np.zeros(len(names)), "w_gd": np.zeros(len(names)),
        "n_gp": np.zeros(len(names)), "w_gp": np.zeros(len(names)),
        "n_gns": np.zeros(len(names)), "w_gns": np.zeros(len(names)),
        "copies_deck": np.zeros(len(names)),
    }
    games = 0
    wins = 0
    turns = 0.0
    for chunk in pd.read_csv(path, usecols=keep, chunksize=20000, low_memory=False):
        won = chunk["won"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        games += len(chunk)
        wins += won.sum()
        turns += chunk["num_turns"].fillna(0).sum()
        oh = chunk[oh_cols].fillna(0).to_numpy(dtype=np.float32)
        dr = chunk[dr_cols].fillna(0).to_numpy(dtype=np.float32)
        tu = chunk[tu_cols].fillna(0).to_numpy(dtype=np.float32)
        dk = chunk[dk_cols].fillna(0).to_numpy(dtype=np.float32)
        gih = (oh + dr + tu) > 0
        ohm = oh > 0
        gdm = (dr + tu) > 0
        gpm = dk > 0
        gns = gpm & ~gih
        w = won[:, None]
        agg["n_gih"] += gih.sum(0); agg["w_gih"] += (gih & w).sum(0)
        agg["n_oh"] += ohm.sum(0); agg["w_oh"] += (ohm & w).sum(0)
        agg["n_gd"] += gdm.sum(0); agg["w_gd"] += (gdm & w).sum(0)
        agg["n_gp"] += gpm.sum(0); agg["w_gp"] += (gpm & w).sum(0)
        agg["n_gns"] += gns.sum(0); agg["w_gns"] += (gns & w).sum(0)
        agg["copies_deck"] += dk.sum(0)
    df = pd.DataFrame({"name": names, **agg})
    with np.errstate(invalid="ignore", divide="ignore"):
        df["gih_wr"] = df.w_gih / df.n_gih
        df["oh_wr"] = df.w_oh / df.n_oh
        df["gd_wr"] = df.w_gd / df.n_gd
        df["gp_wr"] = df.w_gp / df.n_gp
        df["gns_wr"] = df.w_gns / df.n_gns
        df["iwd"] = df.gih_wr - df.gns_wr
    summary = {"games": games, "wins": int(wins), "win_rate": wins / games, "avg_turns": turns / games}
    return df, summary


def draft_stats(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    pack_cols = [c for c in header if c.startswith("pack_card_")]
    names = [c[len("pack_card_"):] for c in pack_cols]
    keep = ["pick", "pick_number", "pack_number"] + pack_cols
    seen_sum = np.zeros(len(names))  # sum of (pick_number+1) over appearances in pack
    seen_n = np.zeros(len(names))
    taken_sum = np.zeros(len(names))
    taken_n = np.zeros(len(names))
    idx = {n: i for i, n in enumerate(names)}
    for chunk in pd.read_csv(path, usecols=keep, chunksize=50000, low_memory=False):
        pn = chunk["pick_number"].to_numpy() + 1  # 1-indexed
        pc = chunk[pack_cols].fillna(0).to_numpy(dtype=np.float32) > 0
        seen_n += pc.sum(0)
        seen_sum += (pc * pn[:, None]).sum(0)
        picks = chunk["pick"].astype(str).to_numpy()
        for p, k in zip(picks, pn):
            i = idx.get(p)
            if i is not None:
                taken_n[i] += 1
                taken_sum[i] += k
    with np.errstate(invalid="ignore", divide="ignore"):
        df = pd.DataFrame({"name": names, "n_seen": seen_n, "alsa": seen_sum / seen_n, "n_taken": taken_n, "ata": taken_sum / taken_n})
    return df


def main(sets: list[str]) -> None:
    rows = []
    for s in sets:
        for fmt in ["PremierDraft", "TradDraft", "PickTwoDraft", "Sealed"]:
            gpath = ROOT / s / f"game_data_public.{s}.{fmt}.csv.gz"
            if not gpath.exists():
                continue
            print(f"[{s} {fmt}] computing game stats...", flush=True)
            df, summ = game_stats(gpath)
            dpath = ROOT / s / f"draft_data_public.{s}.{fmt}.csv.gz"
            if dpath.exists():
                print(f"[{s} {fmt}] computing draft stats...", flush=True)
                dd = draft_stats(dpath)
                df = df.merge(dd, on="name", how="left")
            df.insert(0, "set", s)
            df.insert(1, "format", fmt)
            df.to_csv(OUT / f"{s}_{fmt}_card_stats.csv", index=False)
            rows.append({"set": s, "format": fmt, **summ, "n_cards": len(df)})
            print(f"[{s} {fmt}] games={summ['games']} wr={summ['win_rate']:.4f} turns={summ['avg_turns']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "set_summary.csv", index=False, mode="a", header=not (OUT / "set_summary.csv").exists())


if __name__ == "__main__":
    main(sys.argv[1:] or SETS)
