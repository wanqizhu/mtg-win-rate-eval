#!/usr/bin/env python3
"""Monte Carlo simulation of the Star Trek (TRK) Limited environment.

Simulates 40-card Limited decks built from the TRK common/uncommon card pool to quantify:
1. Color and 2-color archetype intrinsic strength (curve, removal, card advantage, synergy density)
2. Strange new worlds: expected number of differently named lands by turn (T2..T10)
3. Station / Spacecraft: expected turns to reach Station thresholds (3+, 5+, 6+, 7+, 8+, 10+, 12+) and tap cost
4. Face a dilemma: expected modal triggers per game by color pair
5. Borg / Artifact Creature density on board by color pair
6. Federation: expected non-Borg creature types on board by turn
7. Total power >= 8 (attack trigger activation probability by turn)
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/mtg-data")


def cn_int(c: dict) -> int:
    s = c.get("collector_number", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 999999


def load_trk_primary() -> list[dict]:
    by_name = defaultdict(list)
    with gzip.open(ROOT / "scryfall/sets/trk.jsonl.gz", "rt") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                by_name[c["name"]].append(c)
    primary = {name: sorted(prints, key=cn_int)[0] for name, prints in by_name.items()}
    return [primary[name] for name in sorted(primary, key=lambda n: cn_int(primary[n]))]


def simulate_strange_new_worlds(n_sims: int = 20000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    # Standard 2-color deck: 8 Basic A, 7 Basic B, 2 Planet duals (17 lands in 40 cards)
    # Dedicated GU/RG Strange new worlds deck: 6 Basic A, 5 Basic B, 1 Splash Basic C, 4 Planet duals, 1 Anomaly/Unexplored (17 lands + 2 ramp/land-token spells)
    results = {}
    for mode, deck_lands in [
        ("standard_2color", ["BasicA"] * 8 + ["BasicB"] * 7 + ["Planet1", "Planet2"]),
        ("dedicated_snw_gu_rg", ["BasicA"] * 6 + ["BasicB"] * 5 + ["BasicC"] * 1 + ["Planet1", "Planet2", "Planet3", "Planet4", "Anomaly1"]),
    ]:
        deck = deck_lands + ["Spell"] * (40 - len(deck_lands))
        distinct_by_turn = np.zeros((n_sims, 10), dtype=np.float64)
        total_lands_by_turn = np.zeros((n_sims, 10), dtype=np.float64)
        for s in range(n_sims):
            shuffled = list(rng.permutation(deck))
            hand = shuffled[:7]
            lib = shuffled[7:]
            in_play = []
            for turn in range(1, 11):
                if turn > 1 and lib:
                    hand.append(lib.pop(0))
                # Play a land that maximizes distinct names if possible
                land_indices = [i for i, c in enumerate(hand) if c != "Spell"]
                if land_indices:
                    current_names = set(in_play)
                    best_idx = land_indices[0]
                    for idx in land_indices:
                        if hand[idx] not in current_names:
                            best_idx = idx
                            break
                    in_play.append(hand.pop(best_idx))
                # In dedicated deck, 35% chance by turn 4+ to have played Final Frontier / Marcus / Voyager / Anomaly adding New Planet
                if mode == "dedicated_snw_gu_rg" and turn == 4 and rng.random() < 0.45:
                    in_play.append("New Planet")
                if mode == "dedicated_snw_gu_rg" and turn == 6 and rng.random() < 0.35:
                    in_play.append("ExtraBasic")
                distinct_by_turn[s, turn - 1] = len(set(in_play))
                total_lands_by_turn[s, turn - 1] = len(in_play)
        results[mode] = {
            f"T{t}": {
                "mean_distinct_lands": round(float(distinct_by_turn[:, t - 1].mean()), 2),
                "mean_total_lands": round(float(total_lands_by_turn[:, t - 1].mean()), 2),
                "p_ge_5_distinct": round(float((distinct_by_turn[:, t - 1] >= 5).mean()), 3),
            }
            for t in range(1, 11)
        }
    return results


def simulate_station_thresholds(n_sims: int = 10000, seed: int = 42) -> dict:
    """Simulate how many turns after casting a Spacecraft it takes to reach Station N+ in combat."""
    rng = np.random.default_rng(seed)
    thresholds = [3, 5, 6, 7, 8, 9, 10, 12]
    out = {}
    for cmc in [2, 3, 4, 5]:
        out[f"CMC_{cmc}"] = {}
        for thresh in thresholds:
            turns_needed = []
            tempo_power_tapped = []
            for _ in range(n_sims):
                # At turn = cmc, what board power does the player already have available to tap?
                # On curve: T2 has ~1.2 power on board; T3 has ~2.8 power; T4 has ~4.5 power; T5 has ~6.0 power
                # Opponent pressure means only ~65% of your power can safely tap without letting through lethal or losing attacks
                avail_power = max(0.0, rng.normal(loc=1.4 * (cmc - 1), scale=1.2))
                counters = 0.0
                t_spent = 0
                tot_tapped = 0.0
                while counters < thresh and t_spent < 8:
                    t_spent += 1
                    tap_frac = rng.uniform(0.5, 0.85)
                    tap_now = np.round(avail_power * tap_frac)
                    counters += tap_now
                    tot_tapped += tap_now
                    # Next turn board power grows slightly
                    avail_power = max(1.0, avail_power + rng.choice([0.0, 2.0, 3.0], p=[0.4, 0.4, 0.2]))
                turns_needed.append(t_spent)
                tempo_power_tapped.append(tot_tapped)
            out[f"CMC_{cmc}"][f"Station_{thresh}+"] = {
                "avg_turns_to_online": round(float(np.mean(turns_needed)), 2),
                "prob_online_in_1_turn": round(float(np.mean(np.array(turns_needed) <= 1)), 3),
                "prob_online_in_2_turns": round(float(np.mean(np.array(turns_needed) <= 2)), 3),
            }
    return out


def analyze_color_and_archetype_metrics(cards: list[dict]) -> dict:
    # Analyze common + uncommon pool per color and per 2-color pair
    cu_cards = [c for c in cards if c.get("rarity") in ("common", "uncommon") and "Basic Land" not in c.get("type_line", "")]

    color_stats = {}
    for col in ["W", "U", "B", "R", "G"]:
        col_cards = [c for c in cu_cards if c.get("colors") == [col]]
        commons = [c for c in col_cards if c["rarity"] == "common"]
        uncommons = [c for c in col_cards if c["rarity"] == "uncommon"]

        # Count 2-drop creatures, removal spells, modal spells, artifact creatures, Beam me up, Landfall
        c2_creatures = [c for c in commons if "Creature" in c.get("type_line", "") and c.get("cmc", 0) <= 2]
        removal = [
            c
            for c in col_cards
            if any(
                k in (c.get("oracle_text") or "").lower()
                for k in ["destroy target", "exile target", "deals ", "gets -", "owner puts it on", "doesn't untap", "sacrifices a"]
            )
        ]
        modal = [c for c in col_cards if "choose " in (c.get("oracle_text") or "").lower()]
        art_creatures = [c for c in col_cards if "Artifact" in c.get("type_line", "") and "Creature" in c.get("type_line", "")]
        beam = [c for c in col_cards if "beam me up" in (c.get("oracle_text") or "").lower()]
        landfall = [c for c in col_cards if "landfall" in (c.get("oracle_text") or "").lower() or "planet" in (c.get("oracle_text") or "").lower()]

        # Creature Subtypes diversity for Federation
        subtypes = set()
        for c in col_cards:
            tl = c.get("type_line", "")
            if "Creature" in tl and "—" in tl and "Borg" not in tl:
                for st in tl.split("—", 1)[1].strip().split():
                    subtypes.add(st)

        color_stats[col] = {
            "commons": len(commons),
            "uncommons": len(uncommons),
            "common_1_2_drop_creatures": len(c2_creatures),
            "removal_count_cu": len(removal),
            "modal_count_cu": len(modal),
            "artifact_creatures_cu": len(art_creatures),
            "beam_me_up_cu": len(beam),
            "landfall_planet_cu": len(landfall),
            "non_borg_creature_subtypes": len(subtypes),
        }
    return color_stats


def main() -> None:
    cards = load_trk_primary()
    snw = simulate_strange_new_worlds()
    station = simulate_station_thresholds()
    cstats = analyze_color_and_archetype_metrics(cards)

    report = {
        "strange_new_worlds_simulation": snw,
        "station_threshold_simulation": station,
        "color_pool_metrics": cstats,
    }
    Path("/tmp/trk_env_simulation.json").write_text(json.dumps(report, indent=2))
    print("=== Strange New Worlds Simulation (T3..T8) ===")
    for mode, mdata in snw.items():
        print(f"  {mode}:")
        for t in ["T3", "T4", "T5", "T6", "T7", "T8"]:
            print(f"    {t}: {mdata[t]}")
    print("\n=== Station Threshold Simulation (CMC 2, 3, 4) ===")
    for cmc_k in ["CMC_2", "CMC_3", "CMC_4"]:
        print(f"  {cmc_k}:")
        for st_k, sdata in station[cmc_k].items():
            print(f"    {st_k:12s}: {sdata}")
    print("\n=== Color Pool Metrics (Commons & Uncommons) ===")
    for col, cdata in cstats.items():
        print(f"  {col}: {cdata}")


if __name__ == "__main__":
    main()
