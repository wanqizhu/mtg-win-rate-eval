#!/usr/bin/env python3
"""Build historical 17Lands Premier Draft card dataset across the 10 preceding sets (FRA, HOB, MSH, SOS, TMT, ECL, TLA, OM1, EOE, FIN) and join 100% of rows with Scryfall metadata."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/opt/mtg-data")
HISTORICAL_SETS = ["FRA", "HOB", "MSH", "SOS", "TMT", "ECL", "TLA", "OM1", "EOE", "FIN"]


def load_scryfall_lookup() -> tuple[dict[tuple[str, str], dict], dict[str, dict]]:
    by_set_name: dict[tuple[str, str], dict] = {}
    by_name: dict[str, dict] = {}

    for scode in [s.lower() for s in HISTORICAL_SETS] + ["trk", "sds"]:
        p = DATA_DIR / "scryfall" / "sets" / f"{scode}.jsonl.gz"
        if not p.exists():
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                name = c["name"]
                front = name.split(" // ")[0].strip()
                key = (scode.upper(), name)
                if key not in by_set_name or c.get("booster", False):
                    by_set_name[key] = c
                    by_set_name[(scode.upper(), front)] = c
                if name not in by_name or c.get("booster", False):
                    by_name[name] = c
                    by_name[front] = c
                if "printed_name" in c and c["printed_name"]:
                    pname = c["printed_name"]
                    pfront = pname.split(" // ")[0].strip()
                    by_set_name[(scode.upper(), pname)] = c
                    by_set_name[(scode.upper(), pfront)] = c
                    by_name[pname] = c
                    by_name[pfront] = c
                if "flavor_name" in c and c["flavor_name"]:
                    fname = c["flavor_name"]
                    ffront = fname.split(" // ")[0].strip()
                    by_set_name[(scode.upper(), fname)] = c
                    by_set_name[(scode.upper(), ffront)] = c
                    by_name[fname] = c
                    by_name[ffront] = c

    default_path = DATA_DIR / "scryfall" / "default-cards.jsonl.gz"
    if default_path.exists():
        with gzip.open(default_path, "rt", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                name = c["name"]
                front = name.split(" // ")[0].strip()
                if name not in by_name:
                    by_name[name] = c
                if front not in by_name:
                    by_name[front] = c
                if c.get("printed_name"):
                    pname = c["printed_name"]
                    pfront = pname.split(" // ")[0].strip()
                    if pname not in by_name:
                        by_name[pname] = c
                    if pfront not in by_name:
                        by_name[pfront] = c

    return by_set_name, by_name


def compute_17l_stats_for_set(scode: str) -> tuple[pd.DataFrame, dict]:
    game_files = list((DATA_DIR / "17lands" / "sets" / scode).glob("game_data_public.*.PremierDraft.csv.gz"))
    if not game_files:
        raise FileNotFoundError(f"No game file for {scode}")
    gf = game_files[0]

    with gzip.open(gf, "rt", encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    meta_cols = ["won", "num_turns", "on_play", "user_game_win_rate_bucket"]
    meta_cols = [c for c in meta_cols if c in header]
    oh_cols = [c for c in header if c.startswith("opening_hand_")]
    dr_cols = [c for c in header if c.startswith("drawn_")]
    dk_cols = [c for c in header if c.startswith("deck_")]

    cards = [c[len("opening_hand_") :] for c in oh_cols]
    n_cards = len(cards)

    gih_games = np.zeros(n_cards, dtype=np.int64)
    gih_wins = np.zeros(n_cards, dtype=np.int64)
    oh_games = np.zeros(n_cards, dtype=np.int64)
    oh_wins = np.zeros(n_cards, dtype=np.int64)
    dr_games = np.zeros(n_cards, dtype=np.int64)
    dr_wins = np.zeros(n_cards, dtype=np.int64)
    gd_games = np.zeros(n_cards, dtype=np.int64)
    gd_wins = np.zeros(n_cards, dtype=np.int64)
    gns_games = np.zeros(n_cards, dtype=np.int64)
    gns_wins = np.zeros(n_cards, dtype=np.int64)
    total_copies_in_decks = np.zeros(n_cards, dtype=np.int64)

    total_games = 0
    total_wins = 0
    total_turns = 0.0
    on_play_wins = 0
    on_play_games = 0

    usecols = meta_cols + oh_cols + dr_cols + dk_cols
    dtype_map = {c: np.int8 for c in oh_cols + dr_cols + dk_cols}
    dtype_map["won"] = bool

    for chunk in pd.read_csv(gf, usecols=usecols, dtype=dtype_map, chunksize=50000):
        won = chunk["won"].to_numpy(dtype=bool)
        n_g = len(won)
        total_games += n_g
        total_wins += int(won.sum())
        if "num_turns" in chunk:
            total_turns += float(chunk["num_turns"].sum())
        if "on_play" in chunk:
            op = chunk["on_play"].to_numpy(dtype=bool)
            on_play_games += int(op.sum())
            on_play_wins += int((op & won).sum())

        oh = chunk[oh_cols].to_numpy(dtype=np.int8) > 0
        dr = chunk[dr_cols].to_numpy(dtype=np.int8) > 0
        dk_raw = chunk[dk_cols].to_numpy(dtype=np.int8)
        dk = dk_raw > 0

        gih = oh | dr
        gns = dk & (~gih)
        won_col = won[:, None]

        gih_games += gih.sum(axis=0)
        gih_wins += (gih & won_col).sum(axis=0)
        oh_games += oh.sum(axis=0)
        oh_wins += (oh & won_col).sum(axis=0)
        dr_games += dr.sum(axis=0)
        dr_wins += (dr & won_col).sum(axis=0)
        gd_games += dk.sum(axis=0)
        gd_wins += (dk & won_col).sum(axis=0)
        gns_games += gns.sum(axis=0)
        gns_wins += (gns & won_col).sum(axis=0)
        total_copies_in_decks += dk_raw.sum(axis=0)

    base_wr = total_wins / max(total_games, 1)
    avg_turns = total_turns / max(total_games, 1)
    play_wr = on_play_wins / max(on_play_games, 1)

    df = pd.DataFrame(
        {
            "set": scode,
            "card_name": cards,
            "gih_games": gih_games,
            "gih_wins": gih_wins,
            "oh_games": oh_games,
            "oh_wins": oh_wins,
            "dr_games": dr_games,
            "dr_wins": dr_wins,
            "gd_games": gd_games,
            "gd_wins": gd_wins,
            "gns_games": gns_games,
            "gns_wins": gns_wins,
            "total_copies": total_copies_in_decks,
            "set_total_games": total_games,
            "set_base_wr": base_wr,
            "set_avg_turns": avg_turns,
            "set_play_wr": play_wr,
        }
    )

    df["gih_wr_raw"] = np.where(df["gih_games"] > 0, df["gih_wins"] / df["gih_games"], np.nan)
    df["oh_wr_raw"] = np.where(df["oh_games"] > 0, df["oh_wins"] / df["oh_games"], np.nan)
    df["dr_wr_raw"] = np.where(df["dr_games"] > 0, df["dr_wins"] / df["dr_games"], np.nan)
    df["gd_wr_raw"] = np.where(df["gd_games"] > 0, df["gd_wins"] / df["gd_games"], np.nan)
    df["gns_wr_raw"] = np.where(df["gns_games"] > 0, df["gns_wins"] / df["gns_games"], np.nan)
    df["iwd_raw"] = df["gih_wr_raw"] - df["gns_wr_raw"]
    df["play_rate"] = df["gd_games"] / max(total_games, 1)

    summary = {
        "set": scode,
        "total_games": total_games,
        "base_wr": round(base_wr, 4),
        "avg_turns": round(avg_turns, 2),
        "play_wr": round(play_wr, 4),
        "n_cards": n_cards,
    }
    return df, summary


def main() -> None:
    by_set_name, by_name = load_scryfall_lookup()
    all_dfs = []
    summaries = []
    for scode in HISTORICAL_SETS:
        df_s, summ = compute_17l_stats_for_set(scode)
        all_dfs.append(df_s)
        summaries.append(summ)

    full_df = pd.concat(all_dfs, ignore_index=True)
    meta_rows = []
    for _, row in full_df.iterrows():
        scode = row["set"]
        cname = row["card_name"]
        c = by_set_name.get((scode, cname)) or by_name.get(cname)
        if c is None:
            meta_rows.append({"matched": False})
            continue
        faces = c.get("card_faces", [])
        oracle_text = c.get("oracle_text", "")
        if not oracle_text and faces:
            oracle_text = "\n//\n".join(f.get("oracle_text", "") for f in faces)
        mana_cost = c.get("mana_cost", "")
        if not mana_cost and faces:
            mana_cost = " // ".join(f.get("mana_cost", "") for f in faces)
        power = c.get("power")
        toughness = c.get("toughness")
        if power is None and faces:
            power = faces[0].get("power")
            toughness = faces[0].get("toughness")
        colors = c.get("colors")
        if colors is None and faces:
            colors = list({col for f in faces for col in f.get("colors", [])})
        meta_rows.append(
            {
                "matched": True,
                "scryfall_set": c.get("set", "").upper(),
                "rarity": c.get("rarity", ""),
                "cmc": c.get("cmc", 0.0),
                "mana_cost": mana_cost,
                "type_line": c.get("type_line", ""),
                "oracle_text": oracle_text,
                "power": power,
                "toughness": toughness,
                "colors": colors or [],
                "color_identity": c.get("color_identity", []),
                "keywords": c.get("keywords", []),
                "layout": c.get("layout", "normal"),
                "booster": c.get("booster", False),
                "collector_number": c.get("collector_number", ""),
            }
        )

    meta_df = pd.DataFrame(meta_rows)
    combined = pd.concat([full_df, meta_df], axis=1)
    combined.to_pickle("/tmp/historical_17l_full.pkl")
    combined.to_csv("/tmp/historical_17l_cards.csv", index=False)
    with open("/tmp/historical_17l_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved {len(combined)} historical card records (matched: {int(combined['matched'].sum())})")


if __name__ == "__main__":
    main()
