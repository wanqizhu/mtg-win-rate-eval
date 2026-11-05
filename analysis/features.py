"""Hand-engineered card features shared by the historical training set and Star Trek."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

KEYWORDS = [
    "flying", "deathtouch", "lifelink", "trample", "haste", "flash", "vigilance", "menace",
    "first strike", "double strike", "reach", "ward", "hexproof", "indestructible", "defender",
    "prowess", "landcycling", "kicker", "flashback",
]

PATTERNS = {
    "rm_destroy": r"destroy target (?:\w+ )?(?:creature|permanent|nonland)",
    "rm_exile": r"exile target (?:\w+ )?(?:creature|permanent|nonland)|exile .{0,40}until .{0,30}leaves the battlefield",
    "rm_damage": r"deals? (?:\d+|x|twice x) damage to (?:target|any target|each creature)",
    "rm_minus": r"gets? -\d+/-\d+|gets? -x/-x",
    "rm_fight": r"fights?|deals damage equal to (?:its|their) power",
    "rm_sac_opp": r"(?:each|target) opponent sacrifices|each player sacrifices",
    "rm_bounce": r"return target (?:\w+ )?(?:creature|permanent|nonland).{0,40}to its owner's hand",
    "rm_tap_stun": r"stun counter|doesn't untap",
    "wrath": r"destroy all|all creatures get -|deals \d+ damage to each creature",
    "counter_spell": r"counter target",
    "draw1": r"draw a card",
    "draw2": r"draw (?:two|three|four|x) cards",
    "loot": r"draw a card, then discard|discard a card.{0,15}draw",
    "surveil": r"surveil|scry",
    "token": r"create (?:a|an|one|two|three|x|\w+) .{0,60}token",
    "counters_p1": r"\+1/\+1 counter",
    "pump_eot": r"gets? \+\d+/\+\d+ until end of turn",
    "anthem": r"creatures you control get \+|other creatures you control get \+",
    "etb": r"when (?:this|\w+).{0,30} enters",
    "attack_trig": r"whenever .{0,30} attacks",
    "upkeep": r"at the beginning of (?:your|each) (?:upkeep|end step|first main|combat)",
    "lifegain": r"gain \d+ life|gain life|you gain",
    "drain": r"loses? \d+ life",
    "ramp": r"search your library for (?:a|up to \w+) basic land.{0,40}onto the battlefield|add (?:one|two) mana|add \{",
    "fetch_hand": r"search your library for a basic land card, reveal it, put it into your hand",
    "recursion": r"return target .{0,40}card from your graveyard",
    "cast_from_gy": r"cast (?:this card|.{0,30}) from your graveyard",
    "cant_block": r"can't block",
    "cant_attack": r"can't attack",
    "tutor": r"search your library for (?:a|an) (?!basic)",
    "modal": r"choose one|choose two|choose one or more",
    "activated": r"\{[0-9x]\}.{0,20}: |\{t\}: ",
    "equip": r"equip \{",
    "mill": r"\bmill",
    "discard_opp": r"opponent discards|target player discards",
    "copy": r"token that's a copy|copy target|copy it",
    "station": r"\bstation\b",
    "treasure": r"treasure",
    "clue": r"investigate|clue",
    "food": r"\bfood\b",
    "extra_cost_less": r"costs? \{\d\} less|costs? \{x\} less",
    "self_drawback": r"sacrifice a land|you lose the game|lose \d+ life",
    "unblockable": r"can't be blocked",
    "landfall": r"landfall|whenever a land you control enters",
}

COLORS = list("WUBRG")


def parse_pt(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    text = df.oracle.fillna("").str.lower()
    tl = df.type_line.fillna("")
    f = pd.DataFrame(index=df.index)
    for r in ["common", "uncommon", "rare", "mythic"]:
        f[f"rar_{r}"] = (df.rarity == r).astype(float)
    f["rar_special"] = (~df.rarity.isin(["common", "uncommon", "rare", "mythic"])).astype(float)
    f["cmc"] = df.cmc.fillna(0).clip(0, 10)
    cols = df.colors.fillna("")
    f["ncol"] = cols.str.len()
    for c in COLORS:
        f[f"col_{c}"] = cols.str.contains(c).astype(float)
    f["colorless"] = (cols.str.len() == 0).astype(float)
    mc = df.mana_cost.fillna("")
    f["pips"] = mc.str.count(r"\{[WUBRG]\}")
    f["hybrid"] = mc.str.count(r"\{[WUBRG]/[WUBRG]\}")
    f["x_cost"] = mc.str.contains(r"\{X\}").astype(float)
    for t in ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Land", "Planeswalker",
              "Legendary", "Equipment", "Aura", "Saga", "Vehicle", "Spacecraft", "Battle"]:
        f[f"type_{t.lower()}"] = tl.str.contains(t).astype(float)
    f["type_dfc"] = df.layout.isin(["transform", "modal_dfc", "adventure", "split", "flip", "prepare"]).astype(float)
    p = df.power.map(parse_pt)
    t = df.toughness.map(parse_pt)
    is_c = f.type_creature > 0
    f["power"] = p.where(is_c, 0).fillna(0)
    f["tough"] = t.where(is_c, 0).fillna(0)
    f["pt_star"] = (df.power.astype(str).str.contains(r"\*") & is_c).astype(float)
    f["stat_eff"] = np.where(is_c, (f.power + f.tough) / np.maximum(f.cmc, 1), 0)
    f["pt_minus_cmc"] = np.where(is_c, f.power + f.tough - 2 * f.cmc, 0)
    kw = df.keywords.fillna("").str.lower()
    for k in KEYWORDS:
        f[f"kw_{k.replace(' ', '_')}"] = (kw.str.contains(k) | text.str.contains(rf"\b{k}\b")).astype(float)
    for k, pat in PATTERNS.items():
        f[f"t_{k}"] = text.str.contains(pat, regex=True).astype(float)
    f["text_len"] = text.str.len().clip(0, 800) / 100
    f["n_lines"] = text.str.count("\n") + 1
    f["removal_any"] = f[[c for c in f.columns if c.startswith("t_rm_")]].max(axis=1)
    f["removal_x_cmc"] = f.removal_any * f.cmc
    f["creature_x_cmc"] = f.type_creature * f.cmc
    f["evasive_creature"] = f.type_creature * f[["kw_flying", "kw_menace", "t_unblockable"]].max(axis=1)
    return f
