#!/usr/bin/env python3
"""Enhanced feature extraction and multi-task stacked Leave-One-Set-Out CV ensemble for 17lands GiH WR."""

from __future__ import annotations

import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from env_simulation import compute_set_environment_features

OUT_DIR = Path("/workspace/analysis")
HISTORICAL_SETS = ["HOB", "MSH", "SOS", "TMT", "ECL", "TLA", "OM1", "EOE", "FIN", "TDM"]
IN_UNIVERSE_SETS = {"TDM", "EOE", "OM1", "ECL", "SOS", "FRA"}
TWO_COLOR_PAIRS = ["WU", "UB", "BR", "RG", "GW", "WB", "UR", "BG", "RW", "GU"]

WORD_TO_NUM = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def parse_num(val: str | None, default: float = 0.0) -> float:
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in WORD_TO_NUM:
        return float(WORD_TO_NUM[s])
    try:
        return float(s)
    except ValueError:
        return default


def strip_reminder_text(text: str) -> str:
    return re.sub(r"\([^()]*\)", "", text)


def parse_mana_cost(mc: str) -> dict[str, float]:
    symbols = re.findall(r"\{([^}]+)\}", mc or "")
    colored_pips = 0.0
    hybrid_pips = 0.0
    twobrid_pips = 0.0
    generic_mana = 0.0
    x_pips = 0.0
    color_counts = {c: 0.0 for c in "WUBRG"}

    for sym in symbols:
        sym_u = sym.upper()
        if sym_u.isdigit():
            generic_mana += float(sym_u)
        elif sym_u == "X":
            x_pips += 1.0
        elif "/" in sym_u:
            if sym_u[0].isdigit():
                twobrid_pips += 1.0
                colored_pips += 0.25
            else:
                hybrid_pips += 1.0
                colored_pips += 0.5
            for c in "WUBRG":
                if c in sym_u:
                    color_counts[c] += 0.5
        elif sym_u in color_counts:
            colored_pips += 1.0
            color_counts[sym_u] += 1.0

    max_single_color_pips = max(color_counts.values()) if color_counts else 0.0
    return {
        "colored_pips": colored_pips,
        "hybrid_pips": hybrid_pips,
        "twobrid_pips": twobrid_pips,
        "generic_mana": generic_mana,
        "x_pips": x_pips,
        "max_single_color_pips": max_single_color_pips,
    }


def extract_card_features(card: dict) -> dict[str, float]:
    """Extract ~150 domain-specific MTG Limited features from a canonical Scryfall card dict."""
    tl = card.get("type_line") or ""
    front_tl = card.get("front_type_line") or tl
    raw_text = card.get("oracle_text") or ""
    clean_text = strip_reminder_text(raw_text).lower()
    full_lower = raw_text.lower()

    cmc = float(card.get("cmc") or 0.0)
    mc_info = parse_mana_cost(card.get("mana_cost") or "")

    # Effective CMC accounting for built-in cost reduction and twobrid mana
    cost_reduction = 0.0
    if "costs {x} less to cast" in clean_text:
        if "greatest power" in clean_text or "greatest toughness" in clean_text:
            cost_reduction = 3.5
        elif "total mana value" in clean_text:
            cost_reduction = 3.0
        elif "card types among cards in your graveyard" in clean_text:
            cost_reduction = 3.0
        else:
            cost_reduction = 2.5
    elif "costs {1} less to cast for each" in clean_text:
        if "instant and sorcery" in clean_text or "land card" in clean_text or "creature" in clean_text:
            cost_reduction = 2.5
        else:
            cost_reduction = 1.5
    elif re.search(r"costs \{(\d+)\} less to cast", clean_text):
        m_cr = re.search(r"costs \{(\d+)\} less to cast", clean_text)
        cost_reduction = 0.65 * float(m_cr.group(1)) if m_cr else 1.5

    eff_cmc = max(1.0 if cmc > 0 else 0.0, cmc - cost_reduction)
    if mc_info["twobrid_pips"] >= 3:
        eff_cmc = 7.5

    rarity_str = (card.get("rarity") or "common").lower()
    rarity_map = {"common": 0.0, "uncommon": 1.0, "rare": 2.0, "mythic": 3.0, "special": 2.5, "bonus": 2.0}
    rarity_num = rarity_map.get(rarity_str, 0.0)

    colors = card.get("colors") or []
    ci = card.get("color_identity") or []
    eff_colors = colors if colors else ci
    n_colors = len(eff_colors)
    if mc_info["twobrid_pips"] >= 3:
        n_colors = 0

    is_creature = 1.0 if "Creature" in front_tl else 0.0
    is_instant = 1.0 if "Instant" in front_tl else 0.0
    is_sorcery = 1.0 if "Sorcery" in front_tl else 0.0
    is_enchantment = 1.0 if "Enchantment" in front_tl else 0.0
    is_artifact = 1.0 if "Artifact" in front_tl else 0.0
    is_planeswalker = 1.0 if "Planeswalker" in front_tl else 0.0
    is_land = 1.0 if ("Land" in front_tl and "Creature" not in front_tl) else 0.0
    is_basic_land = 1.0 if "Basic Land" in front_tl else 0.0
    is_equipment = 1.0 if "Equipment" in front_tl else 0.0
    is_vehicle = 1.0 if ("Vehicle" in front_tl or "Spacecraft" in front_tl) else 0.0
    is_aura = 1.0 if "Aura" in front_tl else 0.0
    is_saga = 1.0 if "Saga" in front_tl else 0.0
    is_legendary = 1.0 if "Legendary" in front_tl else 0.0
    is_bonus_sheet = 1.0 if card.get("is_bonus_sheet") else 0.0

    # Power / Toughness (accounting for toughness-damage assignment)
    has_body = 1.0 if (is_creature or is_vehicle) else 0.0
    p_raw = card.get("power")
    t_raw = card.get("toughness")
    is_var_pt = 1.0 if (p_raw == "*" or t_raw == "*") else 0.0
    raw_power = min(10.0, parse_num(p_raw, 2.5 if is_var_pt else 0.0)) if has_body else 0.0
    toughness = min(10.0, parse_num(t_raw, 2.5 if is_var_pt else 0.0)) if has_body else 0.0

    deals_toughness_dmg = 1.0 if (
        "assigns combat damage equal to its toughness" in clean_text
        or "deal combat damage equal to their toughness" in clean_text
    ) else 0.0
    power = max(raw_power, toughness) if deals_toughness_dmg else raw_power
    pt_sum = power + toughness
    pt_diff = power - toughness
    stats_vs_cmc = float(np.clip((pt_sum - 2.0 * eff_cmc) if (has_body and eff_cmc > 0) else 0.0, -6.0, 6.0))
    power_vs_cmc = float(np.clip((power - eff_cmc) if (has_body and eff_cmc > 0) else 0.0, -5.0, 5.0))
    toughness_vs_cmc = float(np.clip((toughness - eff_cmc) if (has_body and eff_cmc > 0) else 0.0, -5.0, 5.0))

    # Static vs Conditional Keywords
    kws = {k.lower() for k in (card.get("keywords") or [])}
    front_text = clean_text.split("//")[0] if "//" in clean_text else clean_text

    def has_static_kw(k: str) -> float:
        if k in kws:
            return 1.0
        for line in front_text.split("\n"):
            line_s = line.strip()
            if re.match(rf"^([a-z ]+,\s*)*{re.escape(k)}(,\s*[a-z ]+)*$", line_s):
                return 1.0
        if re.search(rf"\b{re.escape(k)}\b", front_text):
            return 0.45
        return 0.0

    kw_flying = has_static_kw("flying")
    kw_reach = has_static_kw("reach")
    kw_trample = has_static_kw("trample")
    kw_menace = has_static_kw("menace")
    kw_first_strike = has_static_kw("first strike")
    kw_double_strike = has_static_kw("double strike")
    kw_deathtouch = has_static_kw("deathtouch")
    kw_lifelink = has_static_kw("lifelink")
    kw_vigilance = has_static_kw("vigilance")
    kw_haste = has_static_kw("haste")
    kw_flash = has_static_kw("flash")
    kw_ward = 1.0 if ("ward" in kws or "ward" in front_text) else 0.0
    kw_hexproof = has_static_kw("hexproof")
    kw_indestructible = has_static_kw("indestructible")
    kw_defender = 1.0 if ("defender" in kws and "can attack as though" not in clean_text) else 0.0
    kw_prowess = has_static_kw("prowess")
    kw_flashback = 1.0 if ("flashback" in kws or "flashback" in front_text) else 0.0
    kw_cycling = 1.0 if any("cycling" in k for k in kws) or "cycling" in front_text else 0.0
    kw_landcycling = 1.0 if ("landcycling" in front_text or "typecycling" in kws) else 0.0

    evasion_flag = max(kw_flying, 0.7 * kw_menace, 0.5 * kw_trample, 1.0 if "can't be blocked" in front_text else 0.0)
    evasion_power = power * evasion_flag if has_body else 0.0
    combat_kw_count = (
        kw_flying
        + 0.5 * kw_reach
        + 0.6 * kw_trample
        + 0.7 * kw_menace
        + 0.7 * kw_first_strike
        + 1.6 * kw_double_strike
        + 0.8 * kw_deathtouch
        + 0.8 * kw_lifelink
        + 0.5 * kw_vigilance
        + 0.5 * kw_haste
        + 0.6 * kw_flash
        + 0.6 * kw_ward
        + 0.5 * kw_prowess
        - 1.2 * kw_defender
    )
    effective_pt_vs_cmc = stats_vs_cmc + 1.2 * combat_kw_count if has_body else 0.0

    # --- Removal & Interaction Detection ---
    uncond_destroy_exile = 1.0 if re.search(
        r"(destroy|exile) (up to (one|two|three) )?target (creature|nonland permanent)s?(?! (that's|with|you control))",
        clean_text,
    ) else 0.0

    cond_destroy_exile = 1.0 if (
        not uncond_destroy_exile
        and re.search(
            r"((destroy|exile) target (tapped |attacking |blocking |attacking or blocking )?(creature|planeswalker|nonland permanent)|target (opponent|player) sacrifices a creature)",
            clean_text,
        )
    ) else 0.0

    is_color_hose = 1.0 if re.search(
        r"target [a-z ]*(that's|that is) (white|blue|black|red|green) or (white|blue|black|red|green)|target (white|blue|black|red|green) or (white|blue|black|red|green) (creature|permanent|planeswalker)",
        clean_text,
    ) else 0.0
    if is_color_hose:
        uncond_destroy_exile = 0.0

    burn_dmg = 0.0
    burn_matches = re.findall(
        r"deals (\d+|x) damage (divided as you choose )?to (any target|(up to (one|two|three) )?target (creature|planeswalker|attacking|blocking))",
        clean_text,
    )
    for amt_s, _, _, _, _, _ in burn_matches:
        amt = 4.0 if amt_s == "x" else float(amt_s)
        burn_dmg = max(burn_dmg, amt)
    if "deals damage equal to the number of" in clean_text and "target creature" in clean_text:
        burn_dmg = max(burn_dmg, 3.5)
    burn_dmg = min(burn_dmg, 7.0)

    minus_tou = 0.0
    for _, m_t in re.findall(r"target creature [a-z ]*gets -(\d+|x)/-(\d+|x)", clean_text):
        amt = 3.5 if m_t == "x" else float(m_t)
        minus_tou = max(minus_tou, amt)
    minus_tou = min(minus_tou, 6.0)

    is_bite = 1.0 if re.search(
        r"deals damage equal to its (power|loyalty) to (up to one )?target (creature|planeswalker)",
        clean_text,
    ) else 0.0
    is_fight = 1.0 if re.search(r"fights (up to one )?target creature", clean_text) else 0.0

    is_pacifism = 1.0 if (
        ("enchanted creature can't attack" in clean_text)
        or ("enchanted creature loses all abilities" in clean_text)
        or ("stun counter" in clean_text and "tap " in clean_text)
    ) else 0.0

    is_bounce = 1.0 if re.search(
        r"(return (up to (one|two) )?(other )?target [a-z ]*(creature|nonland permanent)s?[a-z ]* to (its|their) owner('s|s') hand|owner of (up to one other )?target (creature|nonland permanent) puts it on .* (top|bottom) of their library|shuffles it into their library)",
        clean_text,
    ) else 0.0

    is_sweeper = 1.0 if re.search(
        r"(destroy all creatures|exile all creatures|deals ([2-9]|x) damage to each creature|all creatures get -([2-9]|x)/-([2-9]|x)|each player sacrifices all)",
        clean_text,
    ) else 0.0
    # Penalize narrow low-MV sweepers ("mana value 2 or less") that wipe your own 2-drops
    if "mana value 2 or less" in clean_text and is_sweeper:
        is_sweeper = 0.2

    is_counterspell = 1.0 if re.search(r"counter target [a-z ]*spell", clean_text) else 0.0

    removal_quality = 0.0
    if uncond_destroy_exile:
        removal_quality = max(removal_quality, 3.2 - 0.35 * max(0.0, cmc - 2.0))
    if cond_destroy_exile:
        removal_quality = max(removal_quality, 1.8 - 0.25 * max(0.0, cmc - 2.0))
    if burn_dmg > 0:
        dmg_cap = min(burn_dmg, 6.0)
        removal_quality = max(removal_quality, 0.55 * dmg_cap - 0.30 * max(0.0, cmc - 1.0))
    if minus_tou > 0:
        t_cap = min(minus_tou, 5.0)
        removal_quality = max(removal_quality, 0.65 * t_cap - 0.30 * max(0.0, cmc - 1.0))
    if is_bite:
        removal_quality = max(removal_quality, 1.6 - 0.25 * max(0.0, cmc - 1.0))
    if is_fight:
        removal_quality = max(removal_quality, 1.2 - 0.20 * max(0.0, cmc - 1.0))
    if is_pacifism:
        removal_quality = max(removal_quality, 1.7 - 0.30 * max(0.0, cmc - 2.0))
    if is_sweeper >= 0.8:
        removal_quality = max(removal_quality, 2.8)
    if is_color_hose:
        removal_quality *= 0.35

    is_any_removal = 1.0 if removal_quality >= 0.8 else 0.0
    is_creature_with_removal = 1.0 if (has_body and (removal_quality >= 0.8 or is_bounce > 0)) else 0.0

    # Specific "when target creature dies this turn" combat trick detector
    when_dies_this_turn_trick = 1.0 if (
        not is_any_removal
        and re.search(r"(when that creature dies this turn|gains [\"']when this creature dies)", clean_text)
    ) else 0.0

    is_combat_trick = 1.0 if (
        (is_instant or kw_flash)
        and not is_any_removal
        and re.search(r"target creature [a-z ]*gets \+\d+/\+\d+ until end of turn", clean_text)
    ) else 0.0

    is_narrow_disenchant = 1.0 if (
        not is_any_removal
        and not has_body
        and not kw_cycling
        and re.search(r"(destroy|exile) target (artifact|enchantment|nonbasic land)", clean_text)
    ) else 0.0

    # --- Card Draw, Selection & Empower Jace Decomposition ---
    draw_amt = 0.0
    for m in re.findall(r"draws? (a|an|one|two|three|four|five|x|\d+) cards?", clean_text):
        val = 2.0 if m == "x" else parse_num(m, 1.0)
        draw_amt += val
    if "draw a card for each" in clean_text:
        draw_amt += 2.0
    if "each player" in clean_text and "draw" in clean_text:
        draw_amt *= 0.25
    is_loot_rummage = 1.0 if re.search(
        r"(draw (a|two) cards?, then discard|discard a card[:,] (then )?draw a card|discard a card\. if you do, draw a card)",
        clean_text,
    ) else 0.0
    if is_loot_rummage and draw_amt > 0:
        draw_amt = max(0.35, draw_amt - 0.65)
    if re.search(r"exile the top (card|two cards|three cards) of your library.*you may (play|cast)", clean_text):
        draw_amt += 1.2 if "two" in clean_text else 0.75
    if re.search(r"look at the top .* cards of your library.*put (one|two|a .* card|up to two) .* into your hand", clean_text):
        draw_amt += 1.0

    surveil_scry = 0.0
    for _, amt_s in re.findall(r"\b(surveil|scry) (\d+|one|two|three)\b", full_lower):
        surveil_scry += parse_num(amt_s, 1.0)

    empower_jace_n = 0.0
    for amt_s in re.findall(r"empower jace (\d+|x|one|two|three|four|five|six|seven|eight)", full_lower):
        val = 3.0 if amt_s == "x" else parse_num(amt_s, 1.0)
        empower_jace_n = max(empower_jace_n, val)

    empower_jace_immediate = empower_jace_n
    if "exile this card from your graveyard: empower jace" in clean_text:
        empower_jace_immediate = 0.5 * empower_jace_n
    elif re.search(r"\{[4-9]\}.*empower jace", clean_text):
        empower_jace_immediate = 0.4 * empower_jace_n

    jace_equiv_draw = 0.0
    jace_equiv_surveil = 0.0
    jace_equiv_token_body = 0.0
    if empower_jace_immediate > 0:
        if empower_jace_immediate >= 6:
            jace_equiv_draw = 1.75
            jace_equiv_surveil = 1.0
        elif empower_jace_immediate >= 5:
            jace_equiv_draw = 1.35
            jace_equiv_surveil = 1.0
        elif empower_jace_immediate >= 4:
            jace_equiv_draw = 1.15
            jace_equiv_surveil = 1.0
        elif empower_jace_immediate >= 3:
            jace_equiv_draw = 0.95
            jace_equiv_surveil = 0.5
        elif empower_jace_immediate >= 2:
            jace_equiv_draw = 0.45
            jace_equiv_surveil = 1.2
        else:
            jace_equiv_draw = 0.20
            jace_equiv_surveil = 1.0

        if "planeswalkers you control have" in clean_text:
            if "create a 4/4 green beast" in clean_text and "sacrifice a creature" not in clean_text:
                jace_equiv_token_body += 4.0
            elif "create a 2/2" in clean_text:
                jace_equiv_token_body += 2.5
            elif "deals 2 damage to up to one target creature" in clean_text:
                removal_quality = max(removal_quality, 1.8)
            elif "add {r}" in clean_text:
                jace_equiv_draw += 0.35

    total_effective_draw = min(4.0, draw_amt + jace_equiv_draw)
    total_surveil_scry = min(5.0, surveil_scry + jace_equiv_surveil)

    # --- Token Creation & +1/+1 Counters ---
    token_score = jace_equiv_token_body
    for cnt_s, p_s, t_s in re.findall(
        r"create (a|an|one|two|three|four|x|\d+) (\d+|x)/(\d+|x) ", clean_text
    ):
        cnt = 2.5 if cnt_s == "x" else parse_num(cnt_s, 1.0)
        tp = 2.0 if p_s == "x" else float(p_s)
        tt = 2.0 if t_s == "x" else float(t_s)
        token_score += cnt * (0.5 * (tp + tt))
    for amt_s in re.findall(r"earthbend (\d+)", full_lower):
        token_score += float(amt_s)
    if "job select" in full_lower or "for mirrodin" in full_lower:
        token_score += 1.5
    token_score = min(token_score, 8.0)

    plus1_counters = 0.0
    for cnt_s in re.findall(r"put (a|an|one|two|three|four|x|\d+) \+1/\+1 counters? on", clean_text):
        cnt = 2.0 if cnt_s == "x" else parse_num(cnt_s, 1.0)
        plus1_counters += cnt
    if "enters with" in clean_text and "+1/+1 counter" in clean_text:
        plus1_counters += 2.0
    if "whenever " in clean_text and "+1/+1 counter" in clean_text:
        plus1_counters += 1.0
    plus1_counters = min(plus1_counters, 5.0)

    # Team-wide pump / overrun / counters on each creature you control
    team_pump_or_counters_all = 1.0 if re.search(
        r"(put .* \+1/\+1 counters? on each creature (you|target player) controls?|creatures you control get \+[1-9x]/\+[1-9x]|other creatures you control get \+[1-9]/\+[1-9])",
        clean_text,
    ) else 0.0

    # --- Structural Triggers & Archetype Value Engines ---
    has_etb_trigger = 1.0 if re.search(r"when (this |\w+ )?enters", clean_text) else 0.0
    has_attack_trigger = 1.0 if re.search(r"whenever (this |\w+ |you )?attacks?", clean_text) else 0.0
    has_dies_trigger = 1.0 if re.search(r"when(ever)? (this |\w+ |another creature )?dies", clean_text) else 0.0

    etb_removal_or_bounce_on_creature = 1.0 if (
        has_body and has_etb_trigger and (is_any_removal or is_bounce)
    ) else 0.0
    etb_token_on_permanent = 1.0 if (
        (has_body or is_equipment or is_enchantment or is_saga) and token_score >= 1.5
    ) else 0.0

    # Equipment without ETB token
    equip_cost = 0.0
    eq_m = re.search(r"equip \{(\d+)\}", full_lower)
    if eq_m:
        equip_cost = min(6.0, float(eq_m.group(1)))
    equipment_no_etb_token = 1.0 if (is_equipment and token_score == 0.0 and not is_any_removal) else 0.0

    # Saga quality
    saga_creates_body_or_removal = 1.0 if (is_saga and (token_score >= 1.5 or is_any_removal or is_bounce)) else 0.0
    saga_no_board_impact = 1.0 if (is_saga and not saga_creates_body_or_removal) else 0.0

    # Multi-effect spell count (bombs that do 2-3 distinct things)
    effect_axes = (
        (1.0 if (is_any_removal or is_bounce) else 0.0)
        + (1.0 if total_effective_draw >= 1.0 else 0.0)
        + (1.0 if token_score >= 1.5 else 0.0)
        + (1.0 if plus1_counters >= 1.0 else 0.0)
        + (1.0 if "gain" in clean_text and "life" in clean_text else 0.0)
    )

    # --- Graveyard Recursion & Mana Ramp ---
    recursion_hand = 1.0 if re.search(
        r"return target [a-z ]*card [a-z ]*from your graveyard to your hand", clean_text
    ) else 0.0
    recursion_bf = 1.0 if re.search(
        r"(return|put) target [a-z ]*card [a-z ]*from (your|a) graveyard (onto|to) the battlefield",
        clean_text,
    ) else 0.0

    is_mana_producer = 1.0 if (
        not is_land
        and (
            re.search(r"\{t\}: add \{", clean_text)
            or "create a treasure token" in clean_text
            or "create a heartwood token" in clean_text
            or "search your library for a basic land" in clean_text
        )
    ) else 0.0

    # --- Multi-Face / Prepare / Adventure Analysis ---
    layout = card.get("layout") or "normal"
    is_prepare = 1.0 if layout == "prepare" else 0.0
    is_adventure = 1.0 if layout == "adventure" else 0.0
    is_transform = 1.0 if layout in {"transform", "modal_dfc"} else 0.0

    enters_prepared = 1.0 if (is_prepare and "enters prepared" in full_lower) else 0.0
    upkeep_prepared = 1.0 if (is_prepare and "beginning of your upkeep" in full_lower and "becomes prepared" in full_lower) else 0.0
    cond_prepared = 1.0 if (is_prepare and not enters_prepared and not upkeep_prepared) else 0.0

    prepare_spell_value = 0.0
    if is_prepare and len(card.get("card_faces") or []) > 1:
        f1 = card["card_faces"][1]
        f1_txt = (f1.get("oracle_text") or "").lower()
        if "create" in f1_txt and "token" in f1_txt and ("2/2" in f1_txt or "3/3" in f1_txt or "fractal" in f1_txt):
            prepare_spell_value = 2.2
        elif "search your library for a basic land" in f1_txt and "battlefield" in f1_txt:
            prepare_spell_value = 2.0
        elif "draw" in f1_txt and ("three" in f1_txt or "two" in f1_txt or "a card" in f1_txt):
            prepare_spell_value = 1.8 if "discard" not in f1_txt else 0.7
        elif "exile target creature" in f1_txt or "deals 3 damage" in f1_txt:
            prepare_spell_value = 2.2
        elif "return target creature" in f1_txt and "hand" in f1_txt:
            prepare_spell_value = 1.5
        elif "+1/+1 counter" in f1_txt and "gain 1 life" in f1_txt:
            prepare_spell_value = 1.3
        elif "heartwood token" in f1_txt or "treasure token" in f1_txt:
            prepare_spell_value = 1.2
        elif "mill three cards" in f1_txt:
            prepare_spell_value = 0.8
        elif "deals 1 damage to target opponent" in f1_txt:
            prepare_spell_value = 0.8
        else:
            prepare_spell_value = 0.7

        if cond_prepared:
            prepare_spell_value *= 0.35

    # --- Constructed Trap / Do-Nothing Detector ---
    affects_board = (
        has_body
        or is_any_removal
        or is_bounce
        or (token_score >= 1.5)
        or is_planeswalker
        or (is_equipment and not equipment_no_etb_token)
        or is_pacifism
        or team_pump_or_counters_all
    )
    is_do_nothing_noncreature = 1.0 if (
        not affects_board
        and not is_land
        and total_effective_draw < 1.0
        and not is_combat_trick
    ) else 0.0

    is_pw_card_tutor_trap = 1.0 if "search your library for a planeswalker card" in clean_text else 0.0
    is_pure_lifegain = 1.0 if (
        not affects_board
        and total_effective_draw < 0.5
        and re.search(r"gains? (\d+|x) life", clean_text)
    ) else 0.0

    word_count = float(min(100, len(clean_text.split())))
    creature_word_count = word_count if (has_body or is_planeswalker) else 0.0
    noncreature_word_count = word_count if not (has_body or is_planeswalker) else 0.0

    feats = {
        "cmc": cmc,
        "eff_cmc": eff_cmc,
        "cost_reduction": cost_reduction,
        "cmc_sq": min(81.0, eff_cmc * eff_cmc),
        "rarity_num": rarity_num,
        "is_common": 1.0 if rarity_str == "common" else 0.0,
        "is_uncommon": 1.0 if rarity_str == "uncommon" else 0.0,
        "is_rare": 1.0 if rarity_str == "rare" else 0.0,
        "is_mythic": 1.0 if rarity_str == "mythic" else 0.0,
        "is_bonus_sheet": is_bonus_sheet,
        "is_legendary": is_legendary,
        "n_colors": float(n_colors),
        "is_colorless": 1.0 if (n_colors == 0 and not is_land) else 0.0,
        "is_mono": 1.0 if n_colors == 1 else 0.0,
        "is_two_color": 1.0 if n_colors == 2 else 0.0,
        "is_three_plus": 1.0 if n_colors >= 3 else 0.0,
        "colored_pips": mc_info["colored_pips"],
        "hybrid_pips": mc_info["hybrid_pips"],
        "twobrid_pips": mc_info["twobrid_pips"],
        "x_pips": mc_info["x_pips"],
        "max_single_color_pips": mc_info["max_single_color_pips"],
        "pip_intensity": mc_info["colored_pips"] / max(1.0, eff_cmc) if not is_land else 0.0,
        "color_W": 1.0 if "W" in eff_colors else 0.0,
        "color_U": 1.0 if "U" in eff_colors else 0.0,
        "color_B": 1.0 if "B" in eff_colors else 0.0,
        "color_R": 1.0 if "R" in eff_colors else 0.0,
        "color_G": 1.0 if "G" in eff_colors else 0.0,
        "is_creature": is_creature,
        "is_instant": is_instant,
        "is_sorcery": is_sorcery,
        "is_enchantment": is_enchantment,
        "is_artifact": is_artifact,
        "is_planeswalker": is_planeswalker,
        "is_land": is_land,
        "is_basic_land": is_basic_land,
        "is_equipment": is_equipment,
        "is_vehicle": is_vehicle,
        "is_aura": is_aura,
        "is_saga": is_saga,
        "has_body": has_body,
        "power": power,
        "toughness": toughness,
        "deals_toughness_dmg": deals_toughness_dmg,
        "pt_sum": pt_sum,
        "pt_diff": pt_diff,
        "stats_vs_cmc": stats_vs_cmc,
        "power_vs_cmc": power_vs_cmc,
        "toughness_vs_cmc": toughness_vs_cmc,
        "is_var_pt": is_var_pt,
        "creat_cmc_1": 1.0 if (has_body and eff_cmc == 1) else 0.0,
        "creat_cmc_2": 1.0 if (has_body and eff_cmc == 2) else 0.0,
        "creat_cmc_3": 1.0 if (has_body and eff_cmc == 3) else 0.0,
        "creat_cmc_4": 1.0 if (has_body and eff_cmc == 4) else 0.0,
        "creat_cmc_5": 1.0 if (has_body and eff_cmc == 5) else 0.0,
        "creat_cmc_6plus": 1.0 if (has_body and eff_cmc >= 6) else 0.0,
        "kw_flying": kw_flying,
        "kw_reach": kw_reach,
        "kw_trample": kw_trample,
        "kw_menace": kw_menace,
        "kw_first_strike": kw_first_strike,
        "kw_double_strike": kw_double_strike,
        "kw_deathtouch": kw_deathtouch,
        "kw_lifelink": kw_lifelink,
        "kw_vigilance": kw_vigilance,
        "kw_haste": kw_haste,
        "kw_flash": kw_flash,
        "kw_ward": kw_ward,
        "kw_hexproof": kw_hexproof,
        "kw_indestructible": kw_indestructible,
        "kw_defender": kw_defender,
        "kw_prowess": kw_prowess,
        "kw_flashback": kw_flashback,
        "kw_cycling": kw_cycling,
        "kw_landcycling": kw_landcycling,
        "evasion_power": evasion_power,
        "combat_kw_count": combat_kw_count,
        "effective_pt_vs_cmc": effective_pt_vs_cmc,
        "uncond_destroy_exile": uncond_destroy_exile,
        "cond_destroy_exile": cond_destroy_exile,
        "is_color_hose": is_color_hose,
        "burn_dmg": burn_dmg,
        "minus_tou": minus_tou,
        "is_bite": is_bite,
        "is_fight": is_fight,
        "is_pacifism": is_pacifism,
        "is_bounce": is_bounce,
        "is_sweeper": is_sweeper,
        "is_counterspell": is_counterspell,
        "removal_quality": removal_quality,
        "is_any_removal": is_any_removal,
        "is_creature_with_removal": is_creature_with_removal,
        "when_dies_this_turn_trick": when_dies_this_turn_trick,
        "is_combat_trick": is_combat_trick,
        "is_narrow_disenchant": is_narrow_disenchant,
        "draw_amt": draw_amt,
        "is_loot_rummage": is_loot_rummage,
        "surveil_scry": surveil_scry,
        "empower_jace_n": empower_jace_n,
        "total_effective_draw": total_effective_draw,
        "total_surveil_scry": total_surveil_scry,
        "token_score": token_score,
        "plus1_counters": plus1_counters,
        "team_pump_or_counters_all": team_pump_or_counters_all,
        "has_etb_trigger": has_etb_trigger,
        "has_attack_trigger": has_attack_trigger,
        "has_dies_trigger": has_dies_trigger,
        "etb_removal_or_bounce_on_creature": etb_removal_or_bounce_on_creature,
        "etb_token_on_permanent": etb_token_on_permanent,
        "equip_cost": equip_cost,
        "equipment_no_etb_token": equipment_no_etb_token,
        "saga_creates_body_or_removal": saga_creates_body_or_removal,
        "saga_no_board_impact": saga_no_board_impact,
        "effect_axes": effect_axes,
        "recursion_hand": recursion_hand,
        "recursion_bf": recursion_bf,
        "is_mana_producer": is_mana_producer,
        "is_prepare": is_prepare,
        "is_adventure": is_adventure,
        "is_transform": is_transform,
        "enters_prepared": enters_prepared,
        "upkeep_prepared": upkeep_prepared,
        "cond_prepared": cond_prepared,
        "prepare_spell_value": prepare_spell_value,
        "is_do_nothing_noncreature": is_do_nothing_noncreature,
        "is_pw_card_tutor_trap": is_pw_card_tutor_trap,
        "is_pure_lifegain": is_pure_lifegain,
        "oracle_word_count": word_count,
        "creature_word_count": creature_word_count,
        "noncreature_word_count": noncreature_word_count,
        "rare_mythic_creature": 1.0 if (has_body and rarity_num >= 2.0) else 0.0,
        "rare_mythic_noncreature_do_nothing": 1.0 if (is_do_nothing_noncreature and rarity_num >= 2.0) else 0.0,
        "common_removal": removal_quality * (1.0 if rarity_num == 0.0 else 0.0),
    }
    return feats


def extract_set_features(cards_dict: dict[str, dict]) -> dict[str, dict[str, float]]:
    base_feats = {cname: extract_card_features(c) for cname, c in cards_dict.items()}
    env_feats = compute_set_environment_features(cards_dict, base_feats)
    for cname in base_feats:
        base_feats[cname].update(env_feats[cname])
    return base_feats


def build_historical_dataframe() -> tuple[pd.DataFrame, list[str]]:
    stats = json.loads((OUT_DIR / "historical_game_stats.json").read_text())
    cat = json.loads((OUT_DIR / "card_catalog.json").read_text())

    rows = []
    for s in HISTORICAL_SETS:
        k = f"{s}_PremierDraft" if f"{s}_PremierDraft" in stats else f"{s}_PickTwoDraft"
        set_data = stats[k]
        set_base_wr = set_data["game_win_rate"]
        total_games = set_data["total_games"]
        set_feats = extract_set_features(cat["historical"][s])
        for cname, cinfo in cat["historical"][s].items():
            st = set_data["cards"].get(cname, {})
            gih = st.get("gih", 0)
            gih_wr = st.get("gih_wr")
            if gih < 200 or gih_wr is None:
                continue
            feats = dict(set_feats[cname])
            feats["set"] = s
            feats["card_name"] = cname
            feats["gih"] = gih
            feats["gih_wr"] = gih_wr
            feats["set_base_wr"] = set_base_wr
            feats["delta_wr"] = gih_wr - set_base_wr
            feats["iwd"] = st.get("iwd") if st.get("iwd") is not None else 0.0
            feats["gp_rate"] = np.log1p(st.get("gp", 0) / max(1.0, total_games / 1000.0))
            rows.append(feats)

    df = pd.DataFrame(rows)
    meta_cols = {"set", "card_name", "gih", "gih_wr", "set_base_wr", "delta_wr", "iwd", "gp_rate"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return df, feature_cols


def get_predicted_set_baseline(set_code: str, n_cards_in_set: int) -> float:
    if set_code in IN_UNIVERSE_SETS:
        return 0.5525
    return 0.5594 if n_cards_in_set >= 240 else 0.5635


def run_loso_cv() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    df, base_feature_cols = build_historical_dataframe()
    print(f"Dataset built: {len(df)} cards across {df['set'].nunique()} sets, {len(base_feature_cols)} base features.")

    # Stage 1: Predict out-of-fold auxiliary targets (IWD and GP_rate) using LightGBM + Ridge
    df["pred_iwd"] = 0.0
    df["pred_gp_rate"] = 0.0

    for test_set in HISTORICAL_SETS:
        val_mask = (df["set"] == test_set).to_numpy()
        tr_mask = ~val_mask
        X_tr = df.loc[tr_mask, base_feature_cols].to_numpy(dtype=np.float64)
        X_va = df.loc[val_mask, base_feature_cols].to_numpy(dtype=np.float64)
        w_tr = np.sqrt(np.clip(df.loc[tr_mask, "gih"].to_numpy(), 200, 25000))

        # Predict IWD
        m_iwd_lgb = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.03, num_leaves=12, min_child_samples=25,
            subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
        ).fit(X_tr, df.loc[tr_mask, "iwd"].to_numpy(), sample_weight=w_tr)
        m_iwd_ridge = make_pipeline(StandardScaler(), Ridge(alpha=500.0)).fit(
            X_tr, df.loc[tr_mask, "iwd"].to_numpy(), ridge__sample_weight=w_tr
        )
        df.loc[val_mask, "pred_iwd"] = 0.65 * m_iwd_lgb.predict(X_va) + 0.35 * m_iwd_ridge.predict(X_va)

        # Predict GP_rate (playability / draftability)
        m_gp_lgb = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.03, num_leaves=12, min_child_samples=25,
            subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
        ).fit(X_tr, df.loc[tr_mask, "gp_rate"].to_numpy(), sample_weight=w_tr)
        df.loc[val_mask, "pred_gp_rate"] = m_gp_lgb.predict(X_va)

    feature_cols = base_feature_cols + ["pred_iwd", "pred_gp_rate"]

    oof_heuristic = np.zeros(len(df))
    model_names = ["lgb_d15", "lgb_d8", "hgb", "gbr", "rf", "et", "ridge", "enet"]
    oof_by_model = {m: np.zeros(len(df)) for m in model_names}

    for test_set in HISTORICAL_SETS:
        val_mask = (df["set"] == test_set).to_numpy()
        tr_mask = ~val_mask

        X_tr = df.loc[tr_mask, feature_cols].to_numpy(dtype=np.float64)
        X_va = df.loc[val_mask, feature_cols].to_numpy(dtype=np.float64)
        w_tr = np.sqrt(np.clip(df.loc[tr_mask, "gih"].to_numpy(), 200, 25000))

        n_cards_in_set = int(val_mask.sum())
        pred_set_base = get_predicted_set_baseline(test_set, n_cards_in_set)
        y_tr = df.loc[tr_mask, "delta_wr"].to_numpy(dtype=np.float64)

        # Heuristic baseline
        r_tr = df.loc[tr_mask, ["rarity_num", "cmc", "is_creature", "is_basic_land"]].to_numpy()
        r_va = df.loc[val_mask, ["rarity_num", "cmc", "is_creature", "is_basic_land"]].to_numpy()
        heur_reg = Ridge(alpha=10.0).fit(r_tr, y_tr, sample_weight=w_tr)
        oof_heuristic[val_mask] = pred_set_base + heur_reg.predict(r_va)

        # 1. LightGBM (leaves=15)
        m1 = lgb.LGBMRegressor(
            n_estimators=260, learning_rate=0.025, num_leaves=15, min_child_samples=25,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.2, reg_lambda=6.0, random_state=42, verbose=-1
        ).fit(X_tr, y_tr, sample_weight=w_tr)
        oof_by_model["lgb_d15"][val_mask] = pred_set_base + m1.predict(X_va)

        # 2. LightGBM shallow (leaves=8)
        m2 = lgb.LGBMRegressor(
            n_estimators=320, learning_rate=0.02, num_leaves=8, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.65, reg_alpha=1.5, reg_lambda=8.0, random_state=101, verbose=-1
        ).fit(X_tr, y_tr, sample_weight=w_tr)
        oof_by_model["lgb_d8"][val_mask] = pred_set_base + m2.predict(X_va)

        # 3. HistGradientBoosting
        m3 = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_leaf_nodes=12, min_samples_leaf=25,
            l2_regularization=5.0, random_state=42
        ).fit(X_tr, y_tr, sample_weight=w_tr)
        oof_by_model["hgb"][val_mask] = pred_set_base + m3.predict(X_va)

        # 4. GradientBoosting
        m4 = GradientBoostingRegressor(
            n_estimators=240, learning_rate=0.025, max_depth=4, min_samples_leaf=22,
            subsample=0.8, max_features=0.65, random_state=42
        ).fit(X_tr, y_tr, sample_weight=w_tr)
        oof_by_model["gbr"][val_mask] = pred_set_base + m4.predict(X_va)

        # 5. RandomForest
        m5 = RandomForestRegressor(
            n_estimators=300, max_depth=11, min_samples_leaf=10, max_features=0.45,
            random_state=42, n_jobs=-1
        ).fit(X_tr, y_tr, sample_weight=w_tr)
        oof_by_model["rf"][val_mask] = pred_set_base + m5.predict(X_va)

        # 6. ExtraTrees
        m6 = ExtraTreesRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=8, max_features=0.55,
            random_state=42, n_jobs=-1
        ).fit(X_tr, y_tr, sample_weight=w_tr)
        oof_by_model["et"][val_mask] = pred_set_base + m6.predict(X_va)

        # 7. Ridge
        m7 = make_pipeline(StandardScaler(), Ridge(alpha=600.0)).fit(
            X_tr, y_tr, ridge__sample_weight=w_tr
        )
        oof_by_model["ridge"][val_mask] = pred_set_base + m7.predict(X_va)

        # 8. ElasticNet
        m8 = make_pipeline(
            StandardScaler(), ElasticNet(alpha=0.0012, l1_ratio=0.25, max_iter=2000, random_state=42)
        ).fit(X_tr, y_tr, elasticnet__sample_weight=w_tr)
        oof_by_model["enet"][val_mask] = pred_set_base + m8.predict(X_va)

    y_true = df["gih_wr"].to_numpy()
    is_main = (df["is_bonus_sheet"] == 0.0).to_numpy()

    # Optimize non-negative ensemble weights on OOF
    P = np.column_stack([oof_by_model[m] for m in model_names])

    def loss_fn(w: np.ndarray) -> float:
        w = w / np.sum(w)
        blend = P @ w
        return float(root_mean_squared_error(y_true[is_main], blend[is_main]))

    res = minimize(
        loss_fn,
        x0=np.ones(len(model_names)) / len(model_names),
        bounds=[(0.0, 1.0)] * len(model_names),
    )
    opt_w = res.x / np.sum(res.x)
    oof_preds = P @ opt_w
    df["oof_pred"] = oof_preds
    df["oof_heuristic"] = oof_heuristic

    print("\nOptimal Ensemble Weights:")
    for m, w in zip(model_names, opt_w):
        print(f"  {m:10s}: {w:.3f}")

    print("\n=== LEAVE-ONE-SET-OUT CV RESULTS (ALL 2,756 CARDS) ===")
    print(
        f"Heuristic (Rarity+CMC) : MAE={mean_absolute_error(y_true, oof_heuristic):.4f} | RMSE={root_mean_squared_error(y_true, oof_heuristic):.4f} | r={pearsonr(y_true, oof_heuristic)[0]:.4f} | rho={spearmanr(y_true, oof_heuristic)[0]:.4f}"
    )
    for m in model_names:
        p = oof_by_model[m]
        print(
            f"Model {m:16s} : MAE={mean_absolute_error(y_true, p):.4f} | RMSE={root_mean_squared_error(y_true, p):.4f} | r={pearsonr(y_true, p)[0]:.4f} | rho={spearmanr(y_true, p)[0]:.4f}"
        )
    print(
        f"Stacked Ensemble Blend : MAE={mean_absolute_error(y_true, oof_preds):.4f} | RMSE={root_mean_squared_error(y_true, oof_preds):.4f} | r={pearsonr(y_true, oof_preds)[0]:.4f} | rho={spearmanr(y_true, oof_preds)[0]:.4f}"
    )

    print("\n=== LEAVE-ONE-SET-OUT CV RESULTS (MAIN-SET CARDS ONLY, n=2,512) ===")
    print(
        f"Heuristic (Rarity+CMC) : MAE={mean_absolute_error(y_true[is_main], oof_heuristic[is_main]):.4f} | RMSE={root_mean_squared_error(y_true[is_main], oof_heuristic[is_main]):.4f} | r={pearsonr(y_true[is_main], oof_heuristic[is_main])[0]:.4f} | rho={spearmanr(y_true[is_main], oof_heuristic[is_main])[0]:.4f}"
    )
    print(
        f"Stacked Ensemble Blend : MAE={mean_absolute_error(y_true[is_main], oof_preds[is_main]):.4f} | RMSE={root_mean_squared_error(y_true[is_main], oof_preds[is_main]):.4f} | r={pearsonr(y_true[is_main], oof_preds[is_main])[0]:.4f} | rho={spearmanr(y_true[is_main], oof_preds[is_main])[0]:.4f}"
    )

    return df, base_feature_cols, opt_w


def train_and_predict_fra() -> None:
    df, base_feature_cols, opt_w = run_loso_cv()
    cat = json.loads((OUT_DIR / "card_catalog.json").read_text())

    # Combine FRA main set + special guests for environment feature extraction
    fra_all_cards: dict[str, dict] = {}
    for cname, c in cat["fra_main"].items():
        fra_all_cards[cname] = c
    for cname, c in cat["fra_special_guests"].items():
        fra_all_cards[cname] = c

    fra_feats_dict = extract_set_features(fra_all_cards)
    fra_rows = []
    for cname, c in fra_all_cards.items():
        f = dict(fra_feats_dict[cname])
        f["card_name"] = cname
        f["collector_number"] = c.get("collector_number", "")
        f["set_group"] = "SPG" if c.get("is_bonus_sheet") else "FRA"
        fra_rows.append(f)
    fra_df = pd.DataFrame(fra_rows)

    X_tr_base = df[base_feature_cols].to_numpy(dtype=np.float64)
    X_fra_base = fra_df[base_feature_cols].to_numpy(dtype=np.float64)
    w_tr = np.sqrt(np.clip(df["gih"].to_numpy(), 200, 25000))

    # Stage 1: Predict IWD and GP_rate on FRA
    m_iwd_lgb = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.03, num_leaves=12, min_child_samples=25,
        subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
    ).fit(X_tr_base, df["iwd"].to_numpy(), sample_weight=w_tr)
    m_iwd_ridge = make_pipeline(StandardScaler(), Ridge(alpha=500.0)).fit(
        X_tr_base, df["iwd"].to_numpy(), ridge__sample_weight=w_tr
    )
    fra_df["pred_iwd"] = 0.65 * m_iwd_lgb.predict(X_fra_base) + 0.35 * m_iwd_ridge.predict(X_fra_base)

    m_gp_lgb = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.03, num_leaves=12, min_child_samples=25,
        subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
    ).fit(X_tr_base, df["gp_rate"].to_numpy(), sample_weight=w_tr)
    fra_df["pred_gp_rate"] = m_gp_lgb.predict(X_fra_base)

    feature_cols = base_feature_cols + ["pred_iwd", "pred_gp_rate"]
    X_tr = df[feature_cols].to_numpy(dtype=np.float64)
    X_fra = fra_df[feature_cols].to_numpy(dtype=np.float64)
    y_tr = df["delta_wr"].to_numpy(dtype=np.float64)

    fra_base_wr = 0.5535  # In-Universe standard set baseline (SOS=0.5538, EOE=0.5532, OM1=0.5551)

    m1 = lgb.LGBMRegressor(
        n_estimators=260, learning_rate=0.025, num_leaves=15, min_child_samples=25,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.2, reg_lambda=6.0, random_state=42, verbose=-1
    ).fit(X_tr, y_tr, sample_weight=w_tr)
    m2 = lgb.LGBMRegressor(
        n_estimators=320, learning_rate=0.02, num_leaves=8, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.65, reg_alpha=1.5, reg_lambda=8.0, random_state=101, verbose=-1
    ).fit(X_tr, y_tr, sample_weight=w_tr)
    m3 = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.03, max_leaf_nodes=12, min_samples_leaf=25,
        l2_regularization=5.0, random_state=42
    ).fit(X_tr, y_tr, sample_weight=w_tr)
    m4 = GradientBoostingRegressor(
        n_estimators=240, learning_rate=0.025, max_depth=4, min_samples_leaf=22,
        subsample=0.8, max_features=0.65, random_state=42
    ).fit(X_tr, y_tr, sample_weight=w_tr)
    m5 = RandomForestRegressor(
        n_estimators=300, max_depth=11, min_samples_leaf=10, max_features=0.45,
        random_state=42, n_jobs=-1
    ).fit(X_tr, y_tr, sample_weight=w_tr)
    m6 = ExtraTreesRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=8, max_features=0.55,
        random_state=42, n_jobs=-1
    ).fit(X_tr, y_tr, sample_weight=w_tr)
    m7 = make_pipeline(StandardScaler(), Ridge(alpha=600.0)).fit(
        X_tr, y_tr, ridge__sample_weight=w_tr
    )
    m8 = make_pipeline(
        StandardScaler(), ElasticNet(alpha=0.0012, l1_ratio=0.25, max_iter=2000, random_state=42)
    ).fit(X_tr, y_tr, elasticnet__sample_weight=w_tr)

    models = [m1, m2, m3, m4, m5, m6, m7, m8]
    model_names = ["lgb_d15", "lgb_d8", "hgb", "gbr", "rf", "et", "ridge", "enet"]
    P_fra = np.column_stack([fra_base_wr + m.predict(X_fra) for m in models])

    # Use blend with diversity across all model families
    blend_w = 0.70 * opt_w + 0.30 * (np.ones(len(models)) / len(models))
    blend_w = blend_w / np.sum(blend_w)
    raw_pred = P_fra @ blend_w

    # Calibrate mean across FRA main-set non-basic cards to match In-Universe SOS/TDM/ECL mean (~0.5580)
    main_mask = (fra_df["set_group"] == "FRA") & (fra_df["is_basic_land"] == 0.0)
    shift = raw_pred[main_mask].mean() - 0.5580
    cal_pred = raw_pred - shift

    fra_df["ml_pred"] = cal_pred
    for idx, mname in enumerate(model_names):
        fra_df[f"pred_{mname}"] = P_fra[:, idx] - shift

    # Save OOF historical predictions and FRA ML predictions
    df.to_csv(OUT_DIR / "historical_oof_predictions.csv", index=False)
    fra_df.to_csv(OUT_DIR / "fra_ml_predictions.csv", index=False)
    print(f"\nSaved FRA ML predictions for {len(fra_df)} cards to {OUT_DIR / 'fra_ml_predictions.csv'}")
    print(f"FRA Main-Set Mean ML Pred: {fra_df.loc[main_mask, 'ml_pred'].mean():.4f}, Std: {fra_df.loc[main_mask, 'ml_pred'].std():.4f}")
    print(f"FRA Min: {fra_df['ml_pred'].min():.4f}, Max: {fra_df['ml_pred'].max():.4f}")


if __name__ == "__main__":
    train_and_predict_fra()
