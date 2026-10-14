#!/usr/bin/env python3
"""Card feature extraction from Scryfall card objects, shared by the historical
training pipeline and the FRA prediction pipeline."""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRY = Path("/opt/mtg-data/scryfall/sets")

KEYWORDS = [
    "Flying", "Trample", "Haste", "Lifelink", "Deathtouch", "Vigilance", "Menace", "First strike",
    "Double strike", "Reach", "Ward", "Flash", "Prowess", "Defender", "Hexproof", "Indestructible",
    "Flashback", "Cycling", "Landcycling", "Surveil", "Scry", "Mill", "Convoke", "Kicker", "Landfall",
    "Threshold", "Exhaust", "Equip", "Prepared", "Behold", "Domain", "Adventure", "Offspring",
]


def load_set(code: str) -> list[dict]:
    p = SCRY / f"{code.lower()}.jsonl.gz"
    return [json.loads(l) for l in gzip.open(p, "rt") if l.strip()]


def full_text(c: dict) -> str:
    if "card_faces" in c and c["card_faces"]:
        return "\n".join(f.get("oracle_text", "") for f in c["card_faces"])
    return c.get("oracle_text", "") or ""


def front(c: dict) -> dict:
    if "card_faces" in c and c["card_faces"] and "power" not in c and "mana_cost" not in c:
        return c["card_faces"][0]
    return c


def norm_text(c: dict) -> str:
    t = full_text(c)
    names = [c["name"]] + [f["name"] for f in c.get("card_faces", [])]
    for n in names:
        t = t.replace(n, "~")
        short = n.split(",")[0]
        if len(short) > 3:
            t = t.replace(short, "~")
    t = re.sub(r"\([^)]*\)", "", t)  # drop reminder text
    return t.lower()


def to_int(v) -> float:
    try:
        return float(v)
    except Exception:
        if isinstance(v, str) and "*" in v:
            return 2.0
        return np.nan


def extract(c: dict) -> dict:
    fr = front(c)
    text = norm_text(c)
    tl = (c.get("type_line") or "").lower()
    cost = c.get("mana_cost") or fr.get("mana_cost") or ""
    colors = c.get("colors")
    if colors is None:
        colors = fr.get("colors", [])
    ci = c.get("color_identity", [])
    kw = set(c.get("keywords", []))
    f = {
        "name": c["name"],
        "rarity": c["rarity"],
        "cmc": float(c.get("cmc", 0) or 0),
        "n_colors": len(colors),
        "n_color_identity": len(ci),
        "is_colorless": int(len(colors) == 0 and "land" not in tl),
        "is_hybrid": int("/" in cost and "P" not in cost),
        "n_pips": len(re.findall(r"\{[WUBRG]\}", cost)) + len(re.findall(r"\{[WUBRG]/[WUBRG]\}", cost)),
        "has_x": int("{X}" in cost),
    }
    for col in "WUBRG":
        f[f"c_{col}"] = int(col in colors)
        f[f"ci_{col}"] = int(col in ci)
    for t in ["creature", "instant", "sorcery", "enchantment", "artifact", "land", "planeswalker", "legendary", "aura", "equipment", "token"]:
        f[f"t_{t}"] = int(t in tl)
    f["is_dfc_or_prepare"] = int("card_faces" in c and bool(c["card_faces"]))
    f["layout_prepare"] = int(c.get("layout") == "prepare")
    f["power"] = to_int(fr.get("power", c.get("power")))
    f["toughness"] = to_int(fr.get("toughness", c.get("toughness")))
    f["loyalty"] = to_int(c.get("loyalty", np.nan))
    pt_sum = (f["power"] if not np.isnan(f["power"]) else 0) + (f["toughness"] if not np.isnan(f["toughness"]) else 0)
    f["pt_sum"] = pt_sum
    f["pt_per_cmc"] = pt_sum / max(f["cmc"], 1) if f["t_creature"] else 0
    f["stats_minus_cmc"] = (pt_sum - 2 * f["cmc"]) if f["t_creature"] else 0
    for k in KEYWORDS:
        f[f"kw_{k.lower().replace(' ', '_')}"] = int(k in kw or k.lower() in text)
    # text-derived features
    pats = {
        "x_destroy_creature": r"destroy target (creature|nonland permanent|permanent|artifact, creature|creature or planeswalker|creature or enchantment)",
        "x_destroy_all": r"destroy all|exile all|each creature|all creatures get",
        "x_exile_target": r"exile target (creature|nonland permanent|permanent|artifact|creature or planeswalker)",
        "x_damage_target": r"deals \d+ damage to (target|any target|up to one target)",
        "x_damage_any": r"damage to any target",
        "x_damage_creature": r"damage to target creature",
        "x_fight_bite": r"fights|deals damage equal to its power to",
        "x_minus_x": r"gets -\d+/-\d+|-x/-x|base power and toughness 0/0|base power and toughness 1/1",
        "x_bounce": r"return target (creature|nonland permanent|permanent).*to its owner's hand|return target creature an opponent controls",
        "x_counterspell": r"counter target (spell|noncreature spell|creature spell|creature or legendary spell|white or black spell|activated)",
        "x_draw": r"draw a card|draw two|draw three|draws? \w+ cards?|draw cards",
        "x_draw_two_plus": r"draw two|draw three|draw cards equal",
        "x_etb": r"when ~ enters|when this (creature|artifact|enchantment|permanent) enters|enters, ",
        "x_dies_trigger": r"when ~ dies|when this creature dies|whenever a creature you control dies",
        "x_attack_trigger": r"whenever ~ attacks|whenever this creature attacks",
        "x_counters": r"\+1/\+1 counter",
        "x_token": r"create (a|an|two|three|x) .*token",
        "x_token_multi": r"create (two|three|x) .*token",
        "x_token_big": r"create a [4-9]/[4-9]",
        "x_lifegain": r"gain \d+ life|you gain life|lifelink|gain life",
        "x_discard_opp": r"opponent discards|each opponent discards|reveals their hand",
        "x_mill_self": r"mill (three|four|two|a card|\d+)|surveil",
        "x_tutor": r"search your library for a (card|planeswalker|creature|basic land|land)",
        "x_ramp": r"search your library for a basic land|add \{[cwubrg]\}|add one mana|put .*land card.*onto the battlefield",
        "x_tap_stun": r"tap target|stun counter|doesn't untap",
        "x_pump": r"gets \+\d+/\+\d+ until end of turn|get \+\d+/\+\d+ until end of turn",
        "x_team_pump": r"creatures you control get \+",
        "x_aura": r"enchant creature",
        "x_equipment": r"equipped creature",
        "x_graveyard_recursion": r"return target .*card from your graveyard|return .* from your graveyard to the battlefield|from your graveyard to your hand",
        "x_sacrifice_cost": r"as an additional cost to cast this spell, sacrifice|sacrifice a creature:|sacrifice another",
        "x_opp_sacrifice": r"opponent sacrifices",
        "x_cant_be_blocked": r"can't be blocked",
        "x_conditional": r"if you control|as long as you control|only if|unless",
        "x_color_hate": r"that's (green|blue|red|white|black) or|(white|black|blue|red|green) or (white|black|blue|red|green) (spell|creature|permanent)",
        "x_mv_restriction": r"mana value \d or (less|greater)|toughness \d or greater|with flying\.",
        "x_choose_one": r"choose one|choose two",
        "x_empower": r"empower jace",
        "x_wrath_small": r"deals? \d+ damage to each creature",
        "x_each_opponent_damage": r"damage to each opponent|damage to target opponent|loses \d+ life",
        "x_haste_pump_combat": r"first strike|double strike",
        "x_enters_tapped": r"enters tapped|enters the battlefield tapped",
        "x_mana_fixing": r"add one mana of any color|add \{[wubrg]\} or \{[wubrg]\}",
        "x_board_wipe": r"destroy all creatures|exile all creatures|all creatures get -",
        "x_steal": r"gain control",
        "x_copy": r"copy",
        "x_vehicle": r"crew",
        "x_opponent_control": r"an opponent controls",
        "x_you_control_target": r"target creature you control",
    }
    for k, p in pats.items():
        f[k] = int(re.search(p, text) is not None)
    nums = re.findall(r"deals (\d+) damage", text)
    f["dmg_amount"] = max([int(n) for n in nums], default=0)
    f["text_len"] = len(text)
    f["n_abilities"] = text.count("\n") + 1 if text else 0
    f["empower_n"] = sum(int(n) for n in re.findall(r"empower jace (\d+)", text))
    f["text"] = text
    f["type_line"] = c.get("type_line", "")
    f["mana_cost"] = cost
    f["oracle_text_raw"] = full_text(c)
    return f


def set_features(code: str, booster_only: bool = True) -> pd.DataFrame:
    cards = load_set(code)
    if booster_only and any(c.get("booster") for c in cards):
        cards = [c for c in cards if c.get("booster")]
    seen = set()
    rows = []
    for c in cards:
        if c["name"] in seen:
            continue
        if c.get("layout") in ("art_series", "token", "emblem", "double_faced_token"):
            continue
        seen.add(c["name"])
        rows.append(extract(c))
    df = pd.DataFrame(rows)
    df.insert(0, "set", code.upper())
    return df


if __name__ == "__main__":
    import sys
    for s in sys.argv[1:]:
        d = set_features(s)
        print(s, d.shape)
        print(d.head())
