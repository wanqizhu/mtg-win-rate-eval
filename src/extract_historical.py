#!/usr/bin/env python3
"""Vectorized extraction of 17lands historical game stats and Scryfall card metadata."""

from __future__ import annotations

import csv
import gzip
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/mtg-data")
OUT_DIR = Path("/workspace/analysis")
HISTORICAL_SETS = ["HOB", "MSH", "SOS", "TMT", "ECL", "TLA", "OM1", "EOE", "FIN", "TDM"]
TWO_COLOR_PAIRS = ["WU", "UB", "BR", "RG", "GW", "WB", "UR", "BG", "RW", "GU"]


def normalize_color_pair(mc: str) -> str:
    if not isinstance(mc, str) or not mc:
        return "OTHER"
    s = set(mc)
    if len(s) == 1 and mc in "WUBRG":
        return f"MONO_{mc}"
    if len(s) == 2:
        for pair in TWO_COLOR_PAIRS:
            if set(pair) == s:
                return pair
    if len(s) >= 3:
        return "3PLUS"
    return "OTHER"


def process_game_file(args: tuple[str, str, Path]) -> dict:
    expansion, event_type, path = args
    t0 = time.time()
    with gzip.open(path, "rt", newline="") as f:
        header = next(csv.reader(f))

    cards = sorted(
        col[len("opening_hand_") :]
        for col in header
        if col.startswith("opening_hand_")
    )
    n_cards = len(cards)

    # Accumulators
    oh_games_bin = np.zeros(n_cards, dtype=np.int64)
    oh_wins_bin = np.zeros(n_cards, dtype=np.int64)
    gd_games_bin = np.zeros(n_cards, dtype=np.int64)
    gd_wins_bin = np.zeros(n_cards, dtype=np.int64)
    gih_games_bin = np.zeros(n_cards, dtype=np.int64)
    gih_wins_bin = np.zeros(n_cards, dtype=np.int64)
    gih_count_wt = np.zeros(n_cards, dtype=np.int64)
    gih_wins_wt = np.zeros(n_cards, dtype=np.int64)
    gp_games_bin = np.zeros(n_cards, dtype=np.int64)
    gp_wins_bin = np.zeros(n_cards, dtype=np.int64)
    gns_games_bin = np.zeros(n_cards, dtype=np.int64)
    gns_wins_bin = np.zeros(n_cards, dtype=np.int64)

    # Track card main-color distribution when in deck
    pair_to_idx = {p: i for i, p in enumerate(TWO_COLOR_PAIRS)}
    card_pair_games = np.zeros((n_cards, len(TWO_COLOR_PAIRS)), dtype=np.int64)
    card_pair_wins = np.zeros((n_cards, len(TWO_COLOR_PAIRS)), dtype=np.int64)

    total_games = 0
    total_wins = 0
    on_play_games = 0
    on_play_wins = 0
    total_turns = 0.0
    valid_turn_games = 0

    color_games: dict[str, int] = {}
    color_wins: dict[str, int] = {}

    oh_cols = [f"opening_hand_{c}" for c in cards]
    dr_cols = [f"drawn_{c}" for c in cards]
    dk_cols = [f"deck_{c}" for c in cards]

    for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
        won = chunk["won"].astype(bool).to_numpy()
        on_play = chunk["on_play"].astype(bool).to_numpy()
        turns = pd.to_numeric(chunk["num_turns"], errors="coerce").to_numpy()

        n = len(won)
        total_games += n
        total_wins += int(won.sum())
        on_play_games += int(on_play.sum())
        on_play_wins += int((on_play & won).sum())

        valid_t = ~np.isnan(turns)
        total_turns += float(np.nansum(turns))
        valid_turn_games += int(valid_t.sum())

        mc_raw = chunk["main_colors"].fillna("").astype(str).tolist()
        mc_norm = [normalize_color_pair(x) for x in mc_raw]
        for m, w in zip(mc_norm, won):
            color_games[m] = color_games.get(m, 0) + 1
            if w:
                color_wins[m] = color_wins.get(m, 0) + 1

        oh = chunk[oh_cols].fillna(0).to_numpy(dtype=np.int32)
        dr = chunk[dr_cols].fillna(0).to_numpy(dtype=np.int32)
        dk = chunk[dk_cols].fillna(0).to_numpy(dtype=np.int32)

        in_hand_cnt = oh + dr
        oh_b = oh > 0
        dr_b = dr > 0
        gih_b = in_hand_cnt > 0
        dk_b = dk > 0
        gns_b = dk_b & (~gih_b)

        won_col = won[:, None]
        oh_games_bin += oh_b.sum(axis=0)
        oh_wins_bin += (oh_b & won_col).sum(axis=0)
        gd_games_bin += dr_b.sum(axis=0)
        gd_wins_bin += (dr_b & won_col).sum(axis=0)
        gih_games_bin += gih_b.sum(axis=0)
        gih_wins_bin += (gih_b & won_col).sum(axis=0)

        gih_count_wt += in_hand_cnt.sum(axis=0)
        gih_wins_wt += (in_hand_cnt * won_col).sum(axis=0)

        gp_games_bin += dk_b.sum(axis=0)
        gp_wins_bin += (dk_b & won_col).sum(axis=0)
        gns_games_bin += gns_b.sum(axis=0)
        gns_wins_bin += (gns_b & won_col).sum(axis=0)

        # Pair stats for GIH
        mc_arr = np.array([pair_to_idx.get(m, -1) for m in mc_norm], dtype=np.int32)
        for p_idx in range(len(TWO_COLOR_PAIRS)):
            mask = mc_arr == p_idx
            if mask.any():
                gih_sub = gih_b[mask]
                won_sub = won_col[mask]
                card_pair_games[:, p_idx] += gih_sub.sum(axis=0)
                card_pair_wins[:, p_idx] += (gih_sub & won_sub).sum(axis=0)

    card_stats = {}
    for i, c in enumerate(cards):
        g_bin = int(gih_games_bin[i])
        w_bin = int(gih_wins_bin[i])
        g_wt = int(gih_count_wt[i])
        w_wt = int(gih_wins_wt[i])
        oh_g = int(oh_games_bin[i])
        oh_w = int(oh_wins_bin[i])
        gd_g = int(gd_games_bin[i])
        gd_w = int(gd_wins_bin[i])
        gp_g = int(gp_games_bin[i])
        gp_w = int(gp_wins_bin[i])
        gns_g = int(gns_games_bin[i])
        gns_w = int(gns_wins_bin[i])

        gih_wr_bin = w_bin / g_bin if g_bin > 0 else None
        gih_wr_wt = w_wt / g_wt if g_wt > 0 else None
        oh_wr = oh_w / oh_g if oh_g > 0 else None
        gd_wr = gd_w / gd_g if gd_g > 0 else None
        gp_wr = gp_w / gp_g if gp_g > 0 else None
        gns_wr = gns_w / gns_g if gns_g > 0 else None
        iwd = (gih_wr_bin - gns_wr) if (gih_wr_bin is not None and gns_wr is not None) else None

        pair_breakdown = {}
        for p_idx, pair in enumerate(TWO_COLOR_PAIRS):
            pg = int(card_pair_games[i, p_idx])
            pw = int(card_pair_wins[i, p_idx])
            if pg > 0:
                pair_breakdown[pair] = {"gih": pg, "wins": pw, "wr": pw / pg}

        card_stats[c] = {
            "gih": g_bin,
            "gih_wins": w_bin,
            "gih_wr": gih_wr_bin,
            "gih_wt": g_wt,
            "gih_wins_wt": w_wt,
            "gih_wr_wt": gih_wr_wt,
            "oh": oh_g,
            "oh_wr": oh_wr,
            "gd": gd_g,
            "gd_wr": gd_wr,
            "gp": gp_g,
            "gp_wr": gp_wr,
            "gns": gns_g,
            "gns_wr": gns_wr,
            "iwd": iwd,
            "pairs": pair_breakdown,
        }

    color_summary = {}
    for k, g in sorted(color_games.items(), key=lambda x: -x[1]):
        w = color_wins.get(k, 0)
        color_summary[k] = {
            "games": g,
            "wins": w,
            "wr": w / g if g > 0 else None,
            "share": g / total_games if total_games > 0 else 0.0,
        }

    elapsed = time.time() - t0
    print(
        f"Done {expansion} {event_type}: {total_games} games, {n_cards} cards in {elapsed:.1f}s",
        flush=True,
    )
    return {
        "expansion": expansion,
        "event_type": event_type,
        "total_games": total_games,
        "total_wins": total_wins,
        "game_win_rate": total_wins / total_games if total_games > 0 else None,
        "on_play_win_rate": on_play_wins / on_play_games if on_play_games > 0 else None,
        "mean_num_turns": total_turns / valid_turn_games if valid_turn_games > 0 else None,
        "color_summary": color_summary,
        "cards": card_stats,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for s in HISTORICAL_SETS:
        s_dir = ROOT / "17lands" / "sets" / s
        for event_type in ["PremierDraft", "PickTwoDraft", "TradDraft", "Sealed"]:
            p = s_dir / f"game_data_public.{s}.{event_type}.csv.gz"
            if p.exists():
                tasks.append((s, event_type, p))

    results = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(process_game_file, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            key = f"{res['expansion']}_{res['event_type']}"
            results[key] = res

    out_path = OUT_DIR / "historical_game_stats.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {len(results)} dataset summaries to {out_path}")


if __name__ == "__main__":
    main()
