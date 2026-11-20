#!/usr/bin/env python3
"""Feature engineering and Leave-One-Set-Out cross-validation pipeline for MTG Limited GIH win-rate prediction."""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

WORD_NUM = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "ten": 10,
}

# Strip only boilerplate reminder text that does not hide token/counter/card-draw mechanics
BOILERPLATE_REMINDERS = re.compile(
    r"\((?:[^()]*?(?:beam me up|station only|can't be blocked except|reach|flying|vigilance|trample|haste|lifelink|deathtouch|first strike|double strike|flash|ward|equip |kicker|landcycling|cycling|surveil|mill |scry |you face a dilemma|stun counter|finality counter|doesn't untap|treat phased-out|they're every creature type|damage causes loss|can be paid with)[^()]*)\)",
     flags=re.I,
)


def strip_boilerplate_reminders(oracle_text: str) -> str:
    return BOILERPLATE_REMINDERS.sub(" ", str(oracle_text or ""))


def normalize_oracle(card_name: str, type_line: str, oracle_text: str, rarity: str, cmc: float) -> str:
    txt = strip_boilerplate_reminders(oracle_text)
    if card_name:
        front = card_name.split("//")[0].strip()
        txt = txt.replace(front, "CARDNAME")
        short = front.split(",")[0].strip()
        if len(short) >= 3:
            txt = txt.replace(short, "CARDNAME")
    txt = re.sub(
        r"\bthis (?:creature|artifact|spacecraft|vehicle|enchantment|aura|equipment|land|permanent|spell|saga)\b",
        "CARDNAME",
        txt,
        flags=re.I,
    )
    txt = re.sub(r"\{[0-9]+\}", "{N}", txt)
    return f"RARITY_{rarity} CMC_{int(min(cmc, 8))} {type_line} | {txt}".lower()


def parse_etb_bonus_pt(ot: str, is_creature: int, is_veh_or_sc: int, is_equipment: int) -> tuple[float, float, int]:
    bonus_p, bonus_t, n_bodies = 0.0, 0.0, 0
    for m in re.finditer(
        r"create (a|an|one|two|three|four|five|\d+) (?:tapped )?(\d+)/(\d+) [a-z0-9 ,\-']*creature token", ot
    ):
        before = ot[: m.start()]
        if any(
            w in before[-85:]
            for w in [
                "dies",
                "when this creature dies",
                "its controller creates",
                "that player creates",
                "if your life total is 0",
                "sacrifice this land",
                "if x is 6 or more",
                "from anywhere other than your hand",
            ]
        ):
            continue
        cnt_s = m.group(1)
        cnt = WORD_NUM.get(cnt_s, int(cnt_s) if cnt_s.isdigit() else 1)
        tp, tt = float(m.group(2)), float(m.group(3))
        if ("whenever " in before[-85:] or "at the beginning of " in before[-85:]) and "enters" not in before[-85:]:
            bonus_p += 0.75 * cnt * tp
            bonus_t += 0.75 * cnt * tt
        else:
            bonus_p += cnt * tp
            bonus_t += cnt * tt
            n_bodies += cnt

    for m in re.finditer(r"create [a-z0-9 ,\-']+, a legendary (\d+)/(\d+)", ot):
        bonus_p += float(m.group(1))
        bonus_t += float(m.group(2))
        n_bodies += 1

    # Equipment that creates a token and attaches (Job select / For Mirrodin!)
    if is_equipment and n_bodies > 0:
        eq_m = re.search(r"equipped creature gets \+(\d+)/\+(\d+)", ot)
        if eq_m:
            bonus_p += float(eq_m.group(1))
            bonus_t += float(eq_m.group(2))

    if is_creature and re.search(r"enters(?: with|, [a-z ]*put) (a|an|one|two|three|\d+) \+1/\+1 counter", ot):
        m = re.search(r"enters(?: with|, [a-z ]*put) (a|an|one|two|three|\d+) \+1/\+1 counter", ot)
        c_s = m.group(1)
        c_n = WORD_NUM.get(c_s, int(c_s) if c_s.isdigit() else 1)
        bonus_p += c_n
        bonus_t += c_n
    elif is_creature and "when " in ot and "enters" in ot and "put a +1/+1 counter on" in ot:
        bonus_p += 1.0
        bonus_t += 1.0
        if "each creature you control with a counter" in ot:
            bonus_p += 1.5
            bonus_t += 1.5

    if is_creature and "create a" in ot and "equipment" in ot and "+1/+1" in ot and "attach it" in ot:
        bonus_p += 1.0
        bonus_t += 1.0

    if is_creature and "that creature explores" in ot:
        bonus_p += 1.0
        bonus_t += 1.0

    return bonus_p, bonus_t, n_bodies


def extract_card_features(df_in: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    set_stats = {}
    for exp, grp in df_in.groupby("expansion"):
        main_grp = grp[grp["is_main_set"]]
        creatures = main_grp[
            main_grp["type_line"].str.contains("Creature", na=False) & main_grp["rarity"].isin(["common", "uncommon"])
        ]
        weights = np.where(creatures["rarity"] == "common", 3.0, 1.0)
        p_vals = np.clip(pd.to_numeric(creatures["power"], errors="coerce").fillna(2.0).to_numpy(), 0, 10)
        t_vals = np.clip(pd.to_numeric(creatures["toughness"], errors="coerce").fillna(2.5).to_numpy(), 0, 10)
        c_vals = np.clip(creatures["cmc"].fillna(3.0).to_numpy(), 0, 10)
        art_share = float(np.sum(weights[creatures["type_line"].str.contains("Artifact", na=False)]) / np.sum(weights))
        set_stats[exp] = {
            "weights": weights,
            "p": p_vals,
            "t": t_vals,
            "cmc": c_vals,
            "art_share": art_share,
        }

    feats = []
    texts = []
    for idx, r in df_in.iterrows():
        exp = r["expansion"]
        sstat = set_stats[exp]
        tl = str(r.get("type_line") or "")
        ot_raw = str(r.get("oracle_text") or "")
        card_name = str(r.get("card") or "")
        ot_clean = strip_boilerplate_reminders(ot_raw)
        ot = ot_clean.lower()
        ot_full = ot_raw.lower()
        mc = str(r.get("mana_cost") or "")
        cmc = float(np.clip(r.get("cmc") or 0.0, 0.0, 10.0))
        rarity = str(r.get("rarity") or "common")
        colors = str(r.get("colors") or "")
        ci = str(r.get("color_identity") or "")

        texts.append(normalize_oracle(card_name, tl, ot_raw, rarity, cmc))

        is_basic = int(
            "Basic Land" in tl or card_name in ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
        )
        is_land = int("Land" in tl and not is_basic)
        is_creature = int("Creature" in tl)
        is_art_creature = int("Artifact" in tl and "Creature" in tl)
        is_spacecraft = int("Spacecraft" in tl)
        is_vehicle = int("Vehicle" in tl)
        is_veh_or_sc = int(is_spacecraft or is_vehicle)
        is_equipment = int("Equipment" in tl)
        is_aura = int("Aura" in tl)
        is_saga = int("Saga" in tl)
        is_planeswalker = int("Planeswalker" in tl)
        is_instant = int("Instant" in tl)
        is_sorcery = int("Sorcery" in tl)
        is_enchantment = int("Enchantment" in tl and not is_aura and not is_saga and not is_creature)
        is_noncreature_art = int(
            "Artifact" in tl and not is_creature and not is_spacecraft and not is_vehicle and not is_equipment
        )
        is_legendary = int("Legendary" in tl)

        rar_map = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}
        rar_rank = rar_map.get(rarity, 0)

        pips = re.findall(r"\{([^}]+)\}", mc)
        has_x = int("X" in pips)
        is_hybrid = int(any("/" in p for p in pips))
        color_pips = [p for p in pips if any(c in p for c in "WUBRG")]
        num_colored_pips = min(len(color_pips), 6)
        single_color_counts = [sum(1 for p in pips if p == c) for c in "WUBRG"]
        max_single_pips = min(max(single_color_counts) if single_color_counts else 0, 5)
        num_colors = min(len(colors), 5)
        off_color_act = int(len(ci) > len(colors) and not is_land and not is_basic)

        p_raw = r.get("power")
        t_raw = r.get("toughness")
        has_star_pt = int(str(p_raw) == "*" or str(t_raw) == "*")
        try:
            power = float(np.clip(float(p_raw), 0.0, 12.0))
        except (ValueError, TypeError):
            power = 3.0 if has_star_pt else np.nan
        try:
            toughness = float(np.clip(float(t_raw), 0.0, 12.0))
        except (ValueError, TypeError):
            toughness = 3.0 if has_star_pt else np.nan

        has_pt = int(not np.isnan(power) and not np.isnan(toughness))
        p_val = power if (is_creature and has_pt) else 0.0
        t_val = toughness if (is_creature and has_pt) else 0.0
        bonus_p, bonus_t, n_bodies = parse_etb_bonus_pt(ot, is_creature, is_veh_or_sc, is_equipment)
        eff_p = p_val + bonus_p
        eff_t = t_val + bonus_t
        pt_sum = p_val + t_val
        eff_pt_sum = eff_p + eff_t
        pt_per_cmc = np.clip(pt_sum / max(cmc, 1.0), 0.0, 4.0) if is_creature else 0.0
        eff_pt_per_cmc = np.clip(eff_pt_sum / max(cmc, 1.0), 0.0, 5.0) if (is_creature or bonus_p > 0) else 0.0
        stat_surplus = np.clip(pt_sum - (2.0 * cmc + 0.5), -6.0, 6.0) if is_creature else 0.0
        eff_stat_surplus = (
            np.clip(eff_pt_sum - (2.0 * cmc + 0.5), -6.0, 8.0) if (is_creature or bonus_p > 0) else 0.0
        )

        has_flying = int("flying" in ot_full or "flying" in str(r.get("keywords", "")).lower())
        has_reach = int("reach" in ot_full)
        has_vigilance = int("vigilance" in ot_full)
        has_menace = int("menace" in ot_full)
        has_trample = int("trample" in ot_full)
        has_first_strike = int("first strike" in ot_full)
        has_double_strike = int("double strike" in ot_full)
        has_deathtouch = int("deathtouch" in ot_full)
        has_lifelink = int("lifelink" in ot_full)
        has_haste = int("haste" in ot_full)
        has_flash = int("flash" in ot_full)
        has_ward = int("ward" in ot_full or "hexproof" in ot_full)
        has_indestructible = int("indestructible" in ot_full)
        has_defender = int("defender" in ot_full)
        cant_block = int(
            "can't block" in ot
            and "target creature can't block" not in ot
            and "creatures can't block" not in ot
            and "blocking" not in ot
        )
        cant_attack_alone = int("can't attack alone" in ot)

        if (is_creature and has_pt) or (n_bodies > 0 and eff_p > 0):
            sp, st, sw, sc = sstat["p"], sstat["t"], sstat["weights"], sstat["cmc"]
            combat_p = eff_p * (2.0 if has_double_strike else 1.0)
            combat_t = eff_t
            kills = (st <= combat_p) | (bool(has_deathtouch) & (combat_p >= 1.0))
            if has_first_strike or has_double_strike:
                survives = kills | (sp < combat_t)
            else:
                survives = sp < combat_t
            dom = kills & survives
            kill_rate = float(np.sum(sw[kills]) / np.sum(sw))
            surv_rate = float(np.sum(sw[survives]) / np.sum(sw))
            dom_rate = float(np.sum(sw[dom]) / np.sum(sw))
            mask_curve = sc <= max(cmc, 2.0)
            dom_curve = float(np.sum(sw[dom & mask_curve]) / np.maximum(np.sum(sw[mask_curve]), 1.0))
        else:
            kill_rate, surv_rate, dom_rate, dom_curve = 0.0, 0.0, 0.0, 0.0

        station_tiers = [int(x) for x in re.findall(r"(?:^|\n|\|)\s*(\d+)\+", ot_raw)]
        crew_tiers = [int(x) for x in re.findall(r"crew (\d+)", ot_full)]
        tap_power_req = min(station_tiers) if station_tiers else (min(crew_tiers) * 2 if crew_tiers else 0)
        tap_power_req = min(tap_power_req, 15)
        has_etb = int(
            bool(re.search(r"when(?:ever)? (?:this |it |" + re.escape(card_name.split(",")[0].lower()) + r")[^.]*enters", ot))
            or n_bodies > 0
        )
        veh_sc_passive = 0
        if is_veh_or_sc:
            before_st = re.split(r"Station|Crew", ot_raw)[0]
            before_clean = re.sub(r"\([^)]*\)", "", before_st).strip()
            if before_clean and "enters" not in before_clean.lower():
                veh_sc_passive = 1
        veh_sc_no_etb_penalty = int(is_veh_or_sc and not has_etb and not veh_sc_passive and tap_power_req >= 6)

        hard_rem = int(
            bool(
                re.search(
                    r"destroy target (?:creature|nonartifact creature|tapped creature|nonland permanent)|exile target (?:creature|nonland permanent|artifact or creature)|exile up to one target (?:creature|nontoken artifact or creature)|owner puts it on (?:their choice of )?the top or bottom",
                    ot,
                )
            )
        )
        exile_until_leaves = int("until this" in ot and "exile target" in ot)
        dmg_matches = re.findall(r"deals (\d+) damage to (?:any target|target (?:creature|attacking|blocking))", ot)
        dmg_val = min(max([int(x) for x in dmg_matches], default=0), 8)
        if "deals twice x damage to target creature" in ot:
            dmg_val = 5
        elif "deals x damage to target" in ot or "deals x damage to any target" in ot:
            dmg_val = 4
        minus_matches = re.findall(r"target creature[^.]*gets -(\d+)/-(\d+)", ot)
        minus_val = min(max([int(x[1]) for x in minus_matches], default=0), 8)
        rem_reach = max(dmg_val, minus_val)
        rem_kill_frac = (
            float(np.sum(sstat["weights"][sstat["t"] <= rem_reach]) / np.sum(sstat["weights"])) if rem_reach > 0 else 0.0
        )
        rem_eff = (1.0 if hard_rem else rem_kill_frac) / max(cmc, 1.5)

        fight_bite = int(bool(re.search(r"fights |deals? damage equal to (?:its|their) power to target", ot)))
        is_bite = int(bool(re.search(r"deals? damage equal to (?:its|their) power to target", ot)))
        edict = int(
            bool(re.search(r"opponent sacrifices a (?:creature|nontoken creature)|each player sacrifices a creature", ot))
        )
        sweeper = int(
            bool(
                re.search(
                    r"all (?:other )?creatures get -\d+/-\d+|deals \d+ damage to each creature|destroy all (?:creatures|nonartifact creatures)|exile all creatures",
                    ot,
                )
            )
        )
        bounce_opp = int(
            bool(
                re.search(
                    r"return target (?:creature|nonland permanent) an opponent controls to its owner's hand|return up to (?:one|two) target [a-z -]*creatures? to their owners' hands",
                    ot,
                )
            )
        )
        tap_stun = int("stun counter" in ot_full or "doesn't untap during its controller's untap step" in ot_full)
        pacifism = int(
            is_aura and ("can't attack" in ot or "can't block" in ot or "doesn't untap" in ot or "loses all" in ot)
        )
        positive_aura = int(is_aura and not pacifism and ("equipped" in ot or "enchanted creature gets +" in ot))
        bad_aura = int(is_aura and not pacifism and not positive_aura and not hard_rem and not exile_until_leaves)
        counterspell = int(bool(re.search(r"counter target [a-z ]*spell", ot)))
        combat_trick = int(is_instant and bool(re.search(r"target creature [a-z ]*gets \+\d+/\+\d+", ot)))
        art_ench_rem = (
            int(bool(re.search(r"destroy target (?:artifact|enchantment)|sacrifices a nontoken artifact or enchantment", ot)))
            * sstat["art_share"]
        )

        draw_two_plus = int(
            bool(re.search(r"draw (?:two|three|x|that many) cards|draw cards equal to", ot))
            and "each player draws" not in ot
        )
        draw_one = int(
            "draw a card" in ot
            and "may draw a card. if they do, they can't attack" not in ot
            and "each player draws" not in ot
        )
        loot_rummage = int(
            bool(
                re.search(
                    r"draw a card, then discard a card|draw two cards, then discard|discard a card\. if you do, draw|discard a card, then draw",
                    ot,
                )
            )
        )
        pure_draw_one = int(draw_one and not loot_rummage)
        impulse_draw = int(bool(re.search(r"exile (?:the top|cards from the top)[^.]*you may (?:play|cast)", ot)))
        discover_cascade = int("discover " in ot or "without paying its mana cost" in ot)
        look_top_hand = int(
            bool(
                re.search(
                    r"look at the top [a-z0-9]+ cards[^.]*put (?:one|a [a-z ]+card|up to [a-z]+) (?:of them |revealed this way )?into your hand|reveal the top [a-z0-9]+ cards[^.]*put .* into your hand|mill [a-z0-9]+ cards\. you may put a .* into your hand",
                    ot,
                )
            )
        )
        gravedigger = int(
            bool(re.search(r"return (?:up to one |one or two )?target [a-z /]*card[a-z ]* from your graveyard to your hand", ot))
        )
        reanimate = int(
            bool(re.search(r"from (?:a|your) graveyard (?:onto|to) the battlefield|assimilate target creature card from a graveyard", ot))
        )
        beam_me_up = int(
            "beam me up" in ot_full
            or "flashback" in ot_full
            or "harmonize" in ot_full
            or "warp" in ot_full
            or "mayhem" in ot_full
        )
        has_landcycling = int("landcycling" in ot_full or ("search your library for a basic" in ot and cmc <= 3))
        has_cycling = int("cycling" in ot_full or bool(re.search(r"discard this card:", ot)))
        is_modal = int(bool(re.search(r"choose one|choose two|choose up to", ot)))
        dilemma_payoff = int("whenever you face a dilemma" in ot or "if you faced a dilemma" in ot)

        makes_creature_token = int(
            bool(re.search(r"create [a-z0-9 ,]+creature token|create a(?: tapped)? token that's a copy", ot)) or n_bodies > 0
        )
        makes_multi_tokens = int(bool(re.search(r"create (?:two|three|four|x) [a-z0-9 /]+creature tokens", ot)) or n_bodies >= 2)
        repeatable_tokens = int(makes_creature_token and bool(re.search(r"whenever |at the beginning of ", ot)))
        repeatable_counters = int(
            bool(
                re.search(
                    r"(?:whenever |at the beginning of combat|at the beginning of your|other creatures you control enter with)[^.]*put (?:a|two|x|that many) \+1/\+1 counter",
                    ot,
                )
            )
        )
        investigates = int("investigate" in ot_full)
        makes_treasure = int("treasure token" in ot and "can't sacrifice treasure" not in ot)
        makes_food = int("food token" in ot or "Artifact — Food" in tl)
        makes_planet_or_land = int(
            "planet land token" in ot
            or bool(re.search(r"put (?:it|that card|those land cards|a land card[^.]*) onto the battlefield", ot))
        )
        puts_p1p1 = int("+1/+1 counter" in ot)
        team_pump = int(
            bool(
                re.search(
                    r"creatures you control get \+[123x]/\+[0123x]|artifact creatures you control get \+1/\+[01]|put a \+1/\+1 counter on each creature you control|other creatures you control get \+1/\+[01]|heroes you control get \+2/\+2",
                    ot,
                )
            )
        )
        mana_dork_or_rock = int(
            bool(re.search(r"\{t\}: add \{[wubrgc]\}|\{t\}: add one mana", ot)) and not is_land and not is_basic
        )

        # Land sub-features
        land_dual_gainlife = int(is_land and "enters tapped" in ot and "gain 1 life" in ot)
        land_colorless_only = int(is_land and "{t}: add {c}" in ot and "add one mana of any color" not in ot and not any(f"add {{{c}}}" in ot for c in "wubrg"))
        land_rare_slow_or_shock = int(is_land and rarity == "rare" and ("pay 2 life" in ot or "two or more other lands" in ot or "control a basic land" in ot))

        # Equipment quality
        eq_has_etb_effect = int(is_equipment and (n_bodies > 0 or dmg_val > 0 or minus_val > 0 or tap_stun > 0))
        eq_clunky_no_etb = int(is_equipment and not eq_has_etb_effect and cmc >= 2)

        sac_cost_drawback = int(
            bool(
                re.search(
                    r"as an additional cost to cast this spell, sacrifice|at the beginning of your end step, if this creature attacked this turn, sacrifice",
                    ot,
                )
            )
        )
        lose_game_trap = int(
            "you lose the game" in ot
            or "can't sacrifice treasure" in ot
            or "that player may draw a card. if they do, they can't attack" in ot
            or "each player draws a card. this saga deals" in ot
            or "the \"legend rule\" doesn't apply" in ot
        )
        do_nothing_ench_art = int(
            (is_enchantment or is_noncreature_art)
            and not makes_creature_token
            and not hard_rem
            and not exile_until_leaves
            and not draw_two_plus
            and not pure_draw_one
            and not reanimate
            and not sweeper
            and not team_pump
            and not repeatable_counters
            and not mana_dork_or_rock
            and not makes_planet_or_land
            and not has_landcycling
        )
        narrow_hate = int("choose a nonland card name" in ot or "choose a number between" in ot)
        expensive_clunker = int(cmc >= 7 and not has_landcycling and not has_cycling and "costs {" not in ot and not is_land)

        feats.append(
            {
                "rar_rank": rar_rank,
                "is_common": int(rarity == "common" and not is_basic),
                "is_uncommon": int(rarity == "uncommon"),
                "is_rare": int(rarity == "rare"),
                "is_mythic": int(rarity == "mythic"),
                "is_basic": is_basic,
                "is_land": is_land,
                "land_dual_gainlife": land_dual_gainlife,
                "land_colorless_only": land_colorless_only,
                "land_rare_slow_or_shock": land_rare_slow_or_shock,
                "is_main_set": int(r.get("is_main_set", True)),
                "cmc": cmc,
                "cmc_sq": cmc**2,
                "num_colored_pips": num_colored_pips,
                "max_single_pips": max_single_pips,
                "double_pip_early": int(max_single_pips >= 2 and cmc <= 3),
                "triple_pip": int(max_single_pips >= 3),
                "num_colors": num_colors,
                "is_gold": int(num_colors == 2 and not is_hybrid),
                "is_3plus_color": int(num_colors >= 3),
                "is_hybrid": is_hybrid,
                "has_x": has_x,
                "off_color_act": off_color_act,
                "is_creature": is_creature,
                "is_art_creature": is_art_creature,
                "is_veh_or_sc": is_veh_or_sc,
                "is_equipment": is_equipment,
                "eq_has_etb_effect": eq_has_etb_effect,
                "eq_clunky_no_etb": eq_clunky_no_etb,
                "is_aura": is_aura,
                "bad_aura": bad_aura,
                "is_saga": is_saga,
                "is_planeswalker": is_planeswalker,
                "is_instant": is_instant,
                "is_sorcery": is_sorcery,
                "is_enchantment": is_enchantment,
                "is_noncreature_art": is_noncreature_art,
                "is_legendary": is_legendary,
                "p_val": p_val,
                "t_val": t_val,
                "pt_sum": pt_sum,
                "eff_p": eff_p,
                "eff_t": eff_t,
                "eff_pt_sum": eff_pt_sum,
                "n_bodies": n_bodies,
                "pt_per_cmc": pt_per_cmc,
                "eff_pt_per_cmc": eff_pt_per_cmc,
                "stat_surplus": stat_surplus,
                "eff_stat_surplus": eff_stat_surplus,
                "has_flying": has_flying,
                "has_reach": has_reach,
                "has_vigilance": has_vigilance,
                "has_menace": has_menace,
                "has_trample": has_trample,
                "has_first_strike": has_first_strike,
                "has_double_strike": has_double_strike,
                "has_deathtouch": has_deathtouch,
                "has_lifelink": has_lifelink,
                "has_haste": has_haste,
                "has_flash": has_flash,
                "has_ward": has_ward,
                "has_indestructible": has_indestructible,
                "has_defender": has_defender,
                "cant_block": cant_block,
                "cant_attack_alone": cant_attack_alone,
                "kill_rate": kill_rate,
                "surv_rate": surv_rate,
                "dom_rate": dom_rate,
                "dom_curve": dom_curve,
                "tap_power_req": tap_power_req,
                "has_etb": has_etb,
                "veh_sc_passive": veh_sc_passive,
                "veh_sc_no_etb_penalty": veh_sc_no_etb_penalty,
                "hard_rem": hard_rem,
                "exile_until_leaves": exile_until_leaves,
                "dmg_val": dmg_val,
                "minus_val": minus_val,
                "rem_kill_frac": rem_kill_frac,
                "rem_eff": rem_eff,
                "fight_bite": fight_bite,
                "is_bite": is_bite,
                "edict": edict,
                "sweeper": sweeper,
                "bounce_opp": bounce_opp,
                "tap_stun": tap_stun,
                "pacifism": pacifism,
                "counterspell": counterspell,
                "combat_trick": combat_trick,
                "art_ench_rem": art_ench_rem,
                "draw_two_plus": draw_two_plus,
                "pure_draw_one": pure_draw_one,
                "loot_rummage": loot_rummage,
                "impulse_draw": impulse_draw,
                "discover_cascade": discover_cascade,
                "look_top_hand": look_top_hand,
                "gravedigger": gravedigger,
                "reanimate": reanimate,
                "beam_me_up": beam_me_up,
                "has_landcycling": has_landcycling,
                "has_cycling": has_cycling,
                "is_modal": is_modal,
                "dilemma_payoff": dilemma_payoff,
                "makes_creature_token": makes_creature_token,
                "makes_multi_tokens": makes_multi_tokens,
                "repeatable_tokens": repeatable_tokens,
                "repeatable_counters": repeatable_counters,
                "investigates": investigates,
                "makes_treasure": makes_treasure,
                "makes_food": makes_food,
                "makes_planet_or_land": makes_planet_or_land,
                "puts_p1p1": puts_p1p1,
                "team_pump": team_pump,
                "mana_dork_or_rock": mana_dork_or_rock,
                "sac_cost_drawback": sac_cost_drawback,
                "lose_game_trap": lose_game_trap,
                "do_nothing_ench_art": do_nothing_ench_art,
                "narrow_hate": narrow_hate,
                "expensive_clunker": expensive_clunker,
                "ot_len": min(len(ot), 500),
                "rare_mythic_cmc": rar_rank * cmc * (1 - is_land),
                "rare_mythic_land": rar_rank * is_land,
                "rare_mythic_etb": rar_rank * has_etb,
                "rare_mythic_stat": rar_rank * eff_stat_surplus,
                "rare_mythic_do_nothing": rar_rank * (do_nothing_ench_art + eq_clunky_no_etb + bad_aura),
            }
        )
    return pd.DataFrame(feats, index=df_in.index), texts


def run_loso_diagnostics() -> None:
    df = pd.read_pickle("/tmp/historical_17l_full.pkl")
    X_all, texts = extract_card_features(df)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=4, max_features=1200, sublinear_tf=True)
    X_tfidf = vec.fit_transform(texts)

    train_mask = df["bin_gih"] >= 50
    eval_mask = (df["is_main_set"]) & (df["bin_gih"] >= 200)

    models = ["ridge", "tfidf", "hgb", "lgb", "et", "iwd", "oh_dr"]
    oof = {m: np.zeros(len(df)) for m in models}

    for held_out in df["expansion"].unique():
        tr_idx = df.index[train_mask & (df["expansion"] != held_out)]
        te_idx = df.index[df["expansion"] == held_out]

        set_means = df.loc[tr_idx].groupby("expansion")["bin_gih_wr"].transform("mean")
        y_tr = df.loc[tr_idx, "bin_gih_wr"].to_numpy() - set_means.to_numpy() + 0.5575
        w_tr = np.clip(np.sqrt(df.loc[tr_idx, "bin_gih"].to_numpy()), 5.0, 120.0)

        tfidf_reg = Ridge(alpha=15.0)
        tfidf_reg.fit(X_tfidf[tr_idx], y_tr, sample_weight=w_tr)
        oof["tfidf"][te_idx] = tfidf_reg.predict(X_tfidf[te_idx])

        X_tr = X_all.loc[tr_idx].to_numpy(dtype=np.float64)
        X_te = X_all.loc[te_idx].to_numpy(dtype=np.float64)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = np.clip(scaler.transform(X_te), -8.0, 8.0)

        ridge = Ridge(alpha=300.0)
        ridge.fit(X_tr_s, y_tr, sample_weight=w_tr)
        oof["ridge"][te_idx] = ridge.predict(X_te_s)

        hgb = HistGradientBoostingRegressor(
            max_iter=160, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=12.0, random_state=42
        )
        hgb.fit(X_tr, y_tr, sample_weight=w_tr)
        oof["hgb"][te_idx] = hgb.predict(X_te)

        lgbm = lgb.LGBMRegressor(
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=12,
            min_child_samples=25,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=12.0,
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_tr, y_tr, sample_weight=w_tr)
        oof["lgb"][te_idx] = lgbm.predict(X_te)

        et = ExtraTreesRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=10, max_features=0.6, random_state=42, n_jobs=-1
        )
        et.fit(X_tr, y_tr, sample_weight=w_tr)
        oof["et"][te_idx] = et.predict(X_te)

        y_iwd_tr = df.loc[tr_idx, "iwd"].fillna(0.0).to_numpy()
        y_gns_tr = df.loc[tr_idx, "gns_wr"].fillna(0.54).to_numpy() - set_means.to_numpy() + 0.5575
        m_iwd = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=12.0, random_state=42
        )
        m_gns = Ridge(alpha=400.0)
        m_iwd.fit(X_tr, y_iwd_tr, sample_weight=w_tr)
        m_gns.fit(X_tr_s, y_gns_tr, sample_weight=w_tr)
        oof["iwd"][te_idx] = m_gns.predict(X_te_s) + m_iwd.predict(X_te)

        # Also train Opening Hand (OH) + Drawn (DR) decomposed model (37% OH + 63% DR)
        y_oh_tr = df.loc[tr_idx, "oh_wr"].fillna(0.55).to_numpy() - set_means.to_numpy() + 0.5575
        y_dr_tr = df.loc[tr_idx, "dr_wr"].fillna(0.55).to_numpy() - set_means.to_numpy() + 0.5575
        m_oh = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.035, num_leaves=10, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=15.0, random_state=43, verbose=-1
        )
        m_dr = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.035, num_leaves=10, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=15.0, random_state=44, verbose=-1
        )
        m_oh.fit(X_tr, y_oh_tr, sample_weight=w_tr)
        m_dr.fit(X_tr, y_dr_tr, sample_weight=w_tr)
        oof["oh_dr"][te_idx] = 0.37 * m_oh.predict(X_te) + 0.63 * m_dr.predict(X_te)

    M = np.column_stack([oof[m] for m in models])
    stacked_oof = np.zeros(len(df))
    for held_out in df["expansion"].unique():
        for rar in ["common", "uncommon", "rare", "mythic"]:
            if rar in ("rare", "mythic"):
                tr_m = eval_mask & (df["expansion"] != held_out) & (df["rarity"].isin(["rare", "mythic"]))
            else:
                tr_m = eval_mask & (df["expansion"] != held_out) & (df["rarity"] == rar)
            te_m = (df["expansion"] == held_out) & (df["rarity"] == rar)
            meta = Ridge(alpha=0.20, positive=True)
            meta.fit(M[tr_m], df.loc[tr_m, "bin_gih_wr"].to_numpy())
            stacked_oof[te_m] = meta.predict(M[te_m])

    # Exact reprint prior for cards that appeared in other historical sets (e.g. Shock Lands, Basic Lands)
    for held_out in df["expansion"].unique():
        tr_df = df[eval_mask & (df["expansion"] != held_out)]
        card_means = tr_df.groupby("card")["bin_gih_wr"].mean().to_dict()
        for idx in df.index[df["expansion"] == held_out]:
            cname = df.loc[idx, "card"]
            if cname in card_means:
                stacked_oof[idx] = 0.65 * card_means[cname] + 0.35 * stacked_oof[idx]

    y_true = df.loc[eval_mask, "bin_gih_wr"].to_numpy()
    p = stacked_oof[eval_mask]
    print(
        f"Upgraded Rarity-Stacked Ensemble: RMSE={root_mean_squared_error(y_true, p):.4f} | MAE={mean_absolute_error(y_true, p):.4f} | Pearson r={pearsonr(y_true, p)[0]:.4f} | Spearman rho={spearmanr(y_true, p)[0]:.4f}"
    )
    loso_summary = {
        "overall": {
            "n_cards": int(eval_mask.sum()),
            "rmse": round(float(root_mean_squared_error(y_true, p)), 5),
            "mae": round(float(mean_absolute_error(y_true, p)), 5),
            "pearson_r": round(float(pearsonr(y_true, p)[0]), 5),
            "spearman_rho": round(float(spearmanr(y_true, p)[0]), 5),
        },
        "by_rarity": {},
        "by_set": {},
    }
    for rar in ["common", "uncommon", "rare", "mythic"]:
        m = eval_mask & (df["rarity"] == rar)
        yt, yp = df.loc[m, "bin_gih_wr"], stacked_oof[m]
        loso_summary["by_rarity"][rar] = {
            "n_cards": int(m.sum()),
            "rmse": round(float(root_mean_squared_error(yt, yp)), 5),
            "mae": round(float(mean_absolute_error(yt, yp)), 5),
            "pearson_r": round(float(pearsonr(yt, yp)[0]), 5),
            "spearman_rho": round(float(spearmanr(yt, yp)[0]), 5),
            "pred_std": round(float(yp.std()), 5),
        }
        print(
            f"  {rar:8s}: RMSE={root_mean_squared_error(yt, yp):.4f} | MAE={mean_absolute_error(yt, yp):.4f} | r={pearsonr(yt, yp)[0]:.4f} | std={yp.std():.4f}"
        )
    for exp_code in sorted(df["expansion"].unique()):
        m = eval_mask & (df["expansion"] == exp_code)
        yt, yp = df.loc[m, "bin_gih_wr"], stacked_oof[m]
        loso_summary["by_set"][exp_code] = {
            "n_cards": int(m.sum()),
            "rmse": round(float(root_mean_squared_error(yt, yp)), 5),
            "mae": round(float(mean_absolute_error(yt, yp)), 5),
            "pearson_r": round(float(pearsonr(yt, yp)[0]), 5),
            "spearman_rho": round(float(spearmanr(yt, yp)[0]), 5),
        }

    # Now train final models on all 10 historical sets and predict TRK (311 cards) and SDS (30 cards)
    def cn_int(c: dict) -> int:
        s = c.get("collector_number", "")
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else 999999

    def load_set_cards(set_code: str) -> list[dict]:
        by_name = defaultdict(list)
        with gzip.open(Path("/opt/mtg-data") / f"scryfall/sets/{set_code}.jsonl.gz", "rt") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    by_name[c["name"]].append(c)
        primary = {name: sorted(prints, key=cn_int)[0] for name, prints in by_name.items()}
        return [primary[name] for name in sorted(primary, key=lambda n: cn_int(primary[n]))]

    trk_cards = load_set_cards("trk")
    sds_cards = load_set_cards("sds")

    target_rows = []
    for scode, clist in [("TRK", trk_cards), ("SDS", sds_cards)]:
        for c in clist:
            is_main = scode == "TRK"
            target_rows.append(
                {
                    "expansion": "TRK",
                    "set": scode,
                    "is_main_set": is_main,
                    "collector_number": c.get("collector_number", ""),
                    "cn_int": cn_int(c),
                    "card": c["name"],
                    "rarity": c.get("rarity", ""),
                    "mana_cost": c.get("mana_cost", ""),
                    "cmc": float(c.get("cmc", 0.0)),
                    "type_line": c.get("type_line", ""),
                    "colors": "".join(c.get("colors", [])),
                    "color_identity": "".join(c.get("color_identity", [])),
                    "keywords": "|".join(c.get("keywords", [])),
                    "oracle_text": c.get("oracle_text") or "",
                    "power": c.get("power"),
                    "toughness": c.get("toughness"),
                }
            )
    df_target = pd.DataFrame(target_rows)
    X_target, target_texts = extract_card_features(df_target)
    X_target = X_target.fillna(0.0)[X_all.columns]
    X_target_tfidf = vec.transform(target_texts)

    tr_idx = df.index[train_mask]
    set_means = df.loc[tr_idx].groupby("expansion")["bin_gih_wr"].transform("mean")
    y_tr = df.loc[tr_idx, "bin_gih_wr"].to_numpy() - set_means.to_numpy() + 0.5575
    w_tr = np.clip(np.sqrt(df.loc[tr_idx, "bin_gih"].to_numpy()), 5.0, 120.0)

    X_tr = X_all.loc[tr_idx].to_numpy(dtype=np.float64)
    X_te = X_target.to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = np.clip(scaler.transform(X_te), -8.0, 8.0)

    pred_sub = {}
    tfidf_reg = Ridge(alpha=15.0)
    tfidf_reg.fit(X_tfidf[tr_idx], y_tr, sample_weight=w_tr)
    pred_sub["tfidf"] = tfidf_reg.predict(X_target_tfidf)

    ridge = Ridge(alpha=300.0)
    ridge.fit(X_tr_s, y_tr, sample_weight=w_tr)
    pred_sub["ridge"] = ridge.predict(X_te_s)

    hgb = HistGradientBoostingRegressor(
        max_iter=160, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=12.0, random_state=42
    )
    hgb.fit(X_tr, y_tr, sample_weight=w_tr)
    pred_sub["hgb"] = hgb.predict(X_te)

    lgbm = lgb.LGBMRegressor(
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=12,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=12.0,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_tr, y_tr, sample_weight=w_tr)
    pred_sub["lgb"] = lgbm.predict(X_te)

    et = ExtraTreesRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=10, max_features=0.6, random_state=42, n_jobs=-1
    )
    et.fit(X_tr, y_tr, sample_weight=w_tr)
    pred_sub["et"] = et.predict(X_te)

    y_iwd_tr = df.loc[tr_idx, "iwd"].fillna(0.0).to_numpy()
    y_gns_tr = df.loc[tr_idx, "gns_wr"].fillna(0.54).to_numpy() - set_means.to_numpy() + 0.5575
    m_iwd = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=12.0, random_state=42
    )
    m_gns = Ridge(alpha=400.0)
    m_iwd.fit(X_tr, y_iwd_tr, sample_weight=w_tr)
    m_gns.fit(X_tr_s, y_gns_tr, sample_weight=w_tr)
    pred_sub["iwd"] = m_gns.predict(X_te_s) + m_iwd.predict(X_te)

    y_oh_tr = df.loc[tr_idx, "oh_wr"].fillna(0.55).to_numpy() - set_means.to_numpy() + 0.5575
    y_dr_tr = df.loc[tr_idx, "dr_wr"].fillna(0.55).to_numpy() - set_means.to_numpy() + 0.5575
    m_oh = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.035, num_leaves=10, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=15.0, random_state=43, verbose=-1
    )
    m_dr = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.035, num_leaves=10, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=15.0, random_state=44, verbose=-1
    )
    m_oh.fit(X_tr, y_oh_tr, sample_weight=w_tr)
    m_dr.fit(X_tr, y_dr_tr, sample_weight=w_tr)
    pred_sub["oh_dr"] = 0.37 * m_oh.predict(X_te) + 0.63 * m_dr.predict(X_te)

    M_target = np.column_stack([pred_sub[m] for m in models])
    stacked_target = np.zeros(len(df_target))
    for rar in ["common", "uncommon", "rare", "mythic"]:
        if rar in ("rare", "mythic"):
            tr_m = eval_mask & (df["rarity"].isin(["rare", "mythic"]))
        else:
            tr_m = eval_mask & (df["rarity"] == rar)
        te_m = (df_target["rarity"] == rar).to_numpy()
        meta = Ridge(alpha=0.20, positive=True)
        meta.fit(M[tr_m], df.loc[tr_m, "bin_gih_wr"].to_numpy())
        stacked_target[te_m] = meta.predict(M_target[te_m])

    # Apply reprint prior where exact card appeared in historical 10 sets
    hist_reprint_means = df[df["bin_gih"] >= 100].groupby("card")["bin_gih_wr"].mean().to_dict()
    for i, row in df_target.iterrows():
        for m in models:
            df_target.loc[i, f"pred_{m}"] = round(float(pred_sub[m][i]), 5)
        raw_ens = float(stacked_target[i])
        df_target.loc[i, "pred_ml_raw"] = round(raw_ens, 5)
        if row["card"] in hist_reprint_means:
            df_target.loc[i, "reprint_hist_wr"] = round(float(hist_reprint_means[row["card"]]), 5)
            df_target.loc[i, "pred_ml_stacked"] = round(0.65 * float(hist_reprint_means[row["card"]]) + 0.35 * raw_ens, 5)
        else:
            df_target.loc[i, "reprint_hist_wr"] = np.nan
            df_target.loc[i, "pred_ml_stacked"] = round(raw_ens, 5)

    # Compute nearest historical analogs (cosine similarity in feature space within same major card category)
    hist_eval_idx = df.index[eval_mask].to_numpy()
    X_hist_eval = X_all.loc[hist_eval_idx].to_numpy(dtype=np.float64)
    X_hist_s = scaler.transform(X_hist_eval)
    # Normalize rows for cosine similarity
    hist_norms = np.linalg.norm(X_hist_s, axis=1, keepdims=True) + 1e-8
    X_hist_norm = X_hist_s / hist_norms
    te_norms = np.linalg.norm(X_te_s, axis=1, keepdims=True) + 1e-8
    X_te_norm = X_te_s / te_norms
    sim_matrix = X_te_norm @ X_hist_norm.T

    analogs_list = []
    for i in range(len(df_target)):
        sims = sim_matrix[i].copy()
        # Require matching major card type (is_creature, is_land, is_vehicle_or_spacecraft)
        for j, h_idx in enumerate(hist_eval_idx):
            if X_all.loc[h_idx, "is_land"] != X_target.loc[i, "is_land"]:
                sims[j] -= 2.0
            if X_all.loc[h_idx, "is_creature"] != X_target.loc[i, "is_creature"]:
                sims[j] -= 1.0
            if X_all.loc[h_idx, "is_veh_or_sc"] != X_target.loc[i, "is_veh_or_sc"]:
                sims[j] -= 1.0
        top_k = np.argsort(sims)[::-1][:3]
        top_analogs = []
        for j in top_k:
            h_idx = hist_eval_idx[j]
            h_row = df.loc[h_idx]
            top_analogs.append(
                f"{h_row['card']} ({h_row['expansion']} {h_row['rarity'][0].upper()}, {h_row['bin_gih_wr']*100:.1f}%)"
            )
        analogs_list.append(" | ".join(top_analogs))
    df_target["nearest_historical_analogs"] = analogs_list

    # Save feature importances
    fi = pd.Series(lgbm.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    loso_summary["top_30_features"] = {k: int(v) for k, v in fi.head(30).items()}

    Path("/tmp/trk_loso_summary.json").write_text(json.dumps(loso_summary, indent=2))
    df_target.to_csv("/tmp/trk_ml_predictions.csv", index=False)
    print(f"Saved {len(df_target)} target card predictions to /tmp/trk_ml_predictions.csv")


if __name__ == "__main__":
    run_loso_diagnostics()
