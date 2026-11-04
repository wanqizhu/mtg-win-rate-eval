#!/usr/bin/env python3
"""Compute 17lands-style per-card statistics (GIH WR, OH WR, GD WR, GP WR, GNS WR, IWD,
play rate) from the public game_data files, for one set / event type.

Usage: compute_card_stats.py SET EVENT  (e.g. FIN PremierDraft)
Writes /tmp/trk/hist/<SET>_<EVENT>_cards.csv and a set summary json.
"""
import sys, gzip, json
import numpy as np
import pandas as pd

ROOT = "/opt/mtg-data/17lands/sets"
OUT = "/tmp/trk/hist"

def main(set_code, event):
    path = f"{ROOT}/{set_code}/game_data_public.{set_code}.{event}.csv.gz"
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
    prefixes = ("opening_hand_", "drawn_", "tutored_", "deck_", "sideboard_")
    cards = sorted({h.split("_", 1)[1] for h in header if h.startswith(prefixes)})
    idx = {c: i for i, c in enumerate(cards)}
    n = len(cards)
    # accumulators
    gih = np.zeros(n); gih_w = np.zeros(n)
    oh = np.zeros(n); oh_w = np.zeros(n)
    gd = np.zeros(n); gd_w = np.zeros(n)
    gp = np.zeros(n); gp_w = np.zeros(n)
    gns = np.zeros(n); gns_w = np.zeros(n)
    copies_in_deck = np.zeros(n)
    games_total = 0; wins_total = 0
    turns_sum = 0.0
    color_games = {}
    meta = {"num_turns": [], "won": []}
    cols = [h for h in header if h.startswith(prefixes)]
    usecols = ["won", "num_turns", "main_colors", "on_play", "num_mulligans"] + cols
    dtype = {c: "int16" for c in cols}
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=40000, low_memory=False):
        won = chunk["won"].astype(str).str.lower().isin(["true", "1"]).to_numpy().astype(np.int8)
        games_total += len(chunk); wins_total += int(won.sum())
        turns_sum += float(chunk["num_turns"].fillna(0).sum())
        for mc, grp in chunk.groupby(chunk["main_colors"].fillna("")):
            g = color_games.setdefault(mc, [0, 0])
            g[0] += len(grp); g[1] += int(won[grp.index.get_indexer(grp.index)].sum()) if False else int(won[chunk.index.get_indexer(grp.index)].sum())
        def mat(prefix):
            m = np.zeros((len(chunk), n), dtype=np.int16)
            present = [c for c in cards if prefix + c in chunk.columns]
            arr = chunk[[prefix + c for c in present]].to_numpy(dtype=np.int16)
            m[:, [idx[c] for c in present]] = arr
            return m
        m_oh = mat("opening_hand_"); m_dr = mat("drawn_"); m_tu = mat("tutored_"); m_dk = mat("deck_")
        in_oh = m_oh > 0
        in_dr = (m_dr > 0) | (m_tu > 0)
        in_gih = in_oh | in_dr
        in_deck = m_dk > 0
        in_gns = in_deck & ~in_gih
        wcol = won[:, None]
        gih += in_gih.sum(0); gih_w += (in_gih & (wcol == 1)).sum(0)
        oh += in_oh.sum(0); oh_w += (in_oh & (wcol == 1)).sum(0)
        gd += in_dr.sum(0); gd_w += (in_dr & (wcol == 1)).sum(0)
        gp += in_deck.sum(0); gp_w += (in_deck & (wcol == 1)).sum(0)
        gns += in_gns.sum(0); gns_w += (in_gns & (wcol == 1)).sum(0)
        copies_in_deck += m_dk.sum(0)
    def rate(w, t):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(t > 0, w / t, np.nan)
    df = pd.DataFrame({
        "set": set_code, "event": event, "name": cards,
        "games": games_total,
        "gp": gp, "gp_wr": rate(gp_w, gp),
        "oh": oh, "oh_wr": rate(oh_w, oh),
        "gd": gd, "gd_wr": rate(gd_w, gd),
        "gih": gih, "gih_wr": rate(gih_w, gih),
        "gns": gns, "gns_wr": rate(gns_w, gns),
        "copies_in_deck": copies_in_deck,
    })
    df["iwd"] = df["gih_wr"] - df["gns_wr"]
    df["play_rate"] = df["gp"] / games_total
    df["avg_copies_when_played"] = np.where(df["gp"] > 0, df["copies_in_deck"] / df["gp"], np.nan)
    df.to_csv(f"{OUT}/{set_code}_{event}_cards.csv", index=False)
    summary = {
        "set": set_code, "event": event, "games": games_total,
        "win_rate": wins_total / games_total, "avg_turns": turns_sum / games_total,
        "color_games": {k: {"games": v[0], "wr": v[1] / v[0]} for k, v in sorted(color_games.items(), key=lambda kv: -kv[1][0])[:25]},
    }
    json.dump(summary, open(f"{OUT}/{set_code}_{event}_summary.json", "w"), indent=1)
    print(set_code, event, games_total, round(wins_total / games_total, 4), "cards", n)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
