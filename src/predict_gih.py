#!/usr/bin/env python3
"""Predict Premier Draft games-in-hand win rates for Star Trek.

Training labels are 17lands Premier Draft GIH win rates for the nine sets that
published that format. Features come only from Scryfall oracle text, mana cost,
types, and rarity. Set identity is not a feature. Leave-one-set-out scores the
model before it is fit on every historical set and applied to TRK.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/opt/mtg-data")
GIH_DIR = Path("/tmp/mtg_analysis/historical_gih")
OUT = Path("/agent/repos/mtg-win-rate-eval/analysis")
PRED = Path("/agent/repos/mtg-win-rate-eval/predictions")

HIST_SETS = ["FRA", "HOB", "MSH", "SOS", "TMT", "ECL", "TLA", "EOE", "FIN"]
# Median-centered expectation for a new 17lands Premier Draft population.
# Computed from the nine historical set win rates (mean 0.5577). Using the
# mean keeps TRK on the same absolute scale as the training labels.
TRK_BASELINE = 0.5577

EVERGREEN = {
    "flying", "trample", "lifelink", "deathtouch", "haste", "vigilance", "menace",
    "reach", "ward", "flash", "defender", "first strike", "double strike",
    "hexproof", "indestructible", "prowess", "convoke", "delve", "cascade",
    "kicker", "cycling", "equip", "enchant", "station", "surveil", "landfall",
    "investigate", "explore", "discover", "mill", "affinity", "sunburst",
    "flashback", "unearth", "escape", "disturb", "miracle", "overload",
    "aftermath", "embalm", "eternalize", "jump-start", "retrace", "encore",
    "blitz", "offspring", "gift", "harmonize", "plot", "exploit", "emerge",
    "improvise", "crew", "prototype", "craft", "adapt", "amass", "mentor",
    "afterlife", "riot", "backup", "incubate", "endure", "mobilize", "toxic",
    "warden", "compleated", "phasing", "shadow", "fear", "intimidate",
    "skulk", "persist", "undying", "modular", "living weapon", "for mirrodin",
    "descend", "fathomless descent",
}

WORD_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "x": 3,
}


def cn_num(card: dict) -> int:
    raw = str(card.get("collector_number") or "")
    digits = ""
    for ch in raw:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else 10**9


def word_to_n(token: str, default: float = 1.0) -> float:
    token = token.lower().strip()
    if token in WORD_NUM:
        return float(WORD_NUM[token])
    try:
        return float(token)
    except ValueError:
        return default


def parse_pt(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "*", "X", "?", "*+1", "1+*", "2+*"}:
        return None
    try:
        return float(text)
    except ValueError:
        match = re.match(r"(-?\d+)", text)
        return float(match.group(1)) if match else None


def normalize_card(card: dict, set_code: str, bonus: bool = False) -> dict | None:
    if card.get("lang") not in (None, "en"):
        return None
    faces = card.get("card_faces") or []
    if faces:
        front = faces[0]
        back = faces[1] if len(faces) > 1 else {}
        name = front.get("name") or card.get("name")
        oracle_parts = [front.get("oracle_text") or "", back.get("oracle_text") or ""]
        type_line = front.get("type_line") or card.get("type_line") or ""
        if back.get("type_line"):
            type_line = f"{type_line} // {back['type_line']}"
        mana = front.get("mana_cost") or card.get("mana_cost") or ""
        power = front.get("power") if front.get("power") is not None else card.get("power")
        toughness = front.get("toughness") if front.get("toughness") is not None else card.get("toughness")
        keywords = list(card.get("keywords") or [])
        for face in faces:
            for kw in face.get("keywords") or []:
                if kw not in keywords:
                    keywords.append(kw)
        oracle = "\n".join(part for part in oracle_parts if part)
        cmc = card.get("cmc")
        if cmc is None:
            cmc = front.get("cmc") or 0
        colors = card.get("colors")
        if not colors:
            colors = front.get("colors") or []
        color_identity = card.get("color_identity") or front.get("color_identity") or []
    else:
        name = card.get("name")
        oracle = card.get("oracle_text") or ""
        type_line = card.get("type_line") or ""
        mana = card.get("mana_cost") or ""
        power = card.get("power")
        toughness = card.get("toughness")
        keywords = list(card.get("keywords") or [])
        cmc = card.get("cmc") or 0
        colors = card.get("colors") or []
        color_identity = card.get("color_identity") or []
    if not name:
        return None
    return {
        "name": name,
        "full_name": card.get("name") or name,
        "set": set_code,
        "bonus": bonus,
        "rarity": "bonus" if bonus else (card.get("rarity") or "common"),
        "mana_cost": mana,
        "cmc": float(cmc or 0),
        "type_line": type_line,
        "oracle_text": oracle,
        "keywords": keywords,
        "power": power,
        "toughness": toughness,
        "colors": colors or [],
        "color_identity": color_identity or [],
        "collector_number": str(card.get("collector_number") or ""),
        "cn": cn_num(card),
        "layout": card.get("layout") or "",
        "oracle_id": card.get("oracle_id") or "",
    }


def load_set_cards(set_code: str, main_only_max_cn: int | None = None) -> list[dict]:
    path = ROOT / "scryfall" / "sets" / f"{set_code.lower()}.jsonl.gz"
    raw = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.strip():
                raw.append(json.loads(line))
    raw.sort(key=cn_num)
    seen = set()
    cards = []
    for card in raw:
        if card.get("lang") not in (None, "en"):
            continue
        if main_only_max_cn is not None and cn_num(card) > main_only_max_cn:
            continue
        norm = normalize_card(card, set_code)
        if norm is None or norm["name"] in seen:
            continue
        # Skip later alternate-language or duplicate names. For historical sets,
        # the lowest collector number is the primary printing.
        seen.add(norm["name"])
        cards.append(norm)
    return cards


def load_bonus_cards(needed: set[str]) -> dict[str, dict]:
    """Oracle text for 17lands names that are not in the set extract.

    Bonus-sheet and special-guest cards are in the draft files but live in
    other Scryfall sets. Rarity is marked 'bonus' so the model does not treat
    a reprinted common as a main-set common.
    """
    if not needed:
        return {}
    found: dict[str, dict] = {}
    path = ROOT / "scryfall" / "default-cards.jsonl.gz"
    name_re = re.compile(r'"name":"((?:[^"\\]|\\.)*)"')
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if '"lang": "en"' not in line and '"lang":"en"' not in line:
                continue
            names_here = name_re.findall(line)
            if not any(name in needed for name in names_here):
                continue
            card = json.loads(line)
            if card.get("lang") != "en":
                continue
            names = {card.get("name")}
            faces = card.get("card_faces") or []
            if faces:
                names.add(faces[0].get("name"))
            for name in names:
                if name in needed and name not in found:
                    norm = normalize_card(card, card.get("set", "bonus").upper(), bonus=True)
                    if norm:
                        norm["name"] = name
                        found[name] = norm
            if len(found) == len(needed):
                break
    return found


def _bonus_probe(line: str, needed: set[str]) -> list[str]:
    # Checking thousands of names per line is expensive. Use a rolling cache of
    # names that actually occur as substrings by probing a hashed bucket.
    # Implemented below as a module-level filter set once names are known.
    return _PROBE_HITS(line)


_PROBE_NAMES: list[str] = []


def _PROBE_HITS(line: str) -> list[str]:
    hits = []
    for name in _PROBE_NAMES:
        if name in line:
            hits.append(name)
            if len(hits) >= 3:
                break
    return hits


def load_gih() -> tuple[pd.DataFrame, dict]:
    frames = []
    summaries = json.loads((GIH_DIR / "summaries.json").read_text())
    baseline = {row["set"]: row["win_rate"] for row in summaries}
    for code in HIST_SETS:
        rows = json.loads((GIH_DIR / f"{code}.json").read_text())
        frame = pd.DataFrame(rows)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["set_wr"] = data["set"].map(baseline)
    return data, {"summaries": summaries, "baseline": baseline}


def attach_cards(gih: pd.DataFrame) -> pd.DataFrame:
    by_set = {code: {card["name"]: card for card in load_set_cards(code)} for code in HIST_SETS}
    records = []
    missing = defaultdict(set)
    for row in gih.itertuples(index=False):
        card = by_set[row.set].get(row.card)
        if card is None:
            missing[row.set].add(row.card)
            continue
        records.append((row, card, False))
    needed = set()
    for names in missing.values():
        needed |= names
    global _PROBE_NAMES
    _PROBE_NAMES = sorted(needed, key=len, reverse=True)
    print(f"main-set matches {len(records)}; bonus names to resolve {len(needed)}", flush=True)
    bonus = load_bonus_cards(needed)
    print(f"resolved bonus cards {len(bonus)}", flush=True)
    for row in gih.itertuples(index=False):
        if row.card in by_set[row.set]:
            continue
        card = bonus.get(row.card)
        if card is None:
            continue
        # Copy so per-set rarity/bonus flags stay independent.
        card = dict(card)
        card["set"] = row.set
        card["bonus"] = True
        card["rarity"] = "bonus"
        records.append((row, card, True))
    out_rows = []
    for row, card, is_bonus in records:
        merged = dict(card)
        merged["gih_n"] = int(row.gih_n)
        merged["gih_wr"] = float(row.gih_wr) if row.gih_n else None
        merged["deck_n"] = int(row.deck_n)
        merged["deck_wr"] = float(row.deck_wr) if row.deck_wr is not None else None
        merged["set_wr"] = float(row.set_wr)
        merged["is_bonus"] = is_bonus
        out_rows.append(merged)
    still = needed - set(bonus)
    print(f"unresolved names {len(still)}: {sorted(still)[:12]}", flush=True)
    return pd.DataFrame(out_rows)


def count_draws(text: str) -> float:
    total = 0.0
    for match in re.finditer(
        r"draw (?:(a|an|one|two|three|four|five|six|seven|x|\d+) cards|a card)",
        text,
    ):
        token = match.group(1) or "a"
        total += word_to_n(token, 1)
    return total


def first_number(pattern: str, text: str, default: float = 0.0) -> float:
    match = re.search(pattern, text)
    if not match:
        return default
    return word_to_n(match.group(1), default)


def featurize(card: dict) -> dict:
    text = card.get("oracle_text") or ""
    low = text.lower()
    type_line = card.get("type_line") or ""
    tl = type_line.lower()
    mana = card.get("mana_cost") or ""
    keywords = {kw.lower() for kw in card.get("keywords") or []}
    cmc = float(card.get("cmc") or 0)
    colors = card.get("colors") or []
    ci = card.get("color_identity") or []
    if "land" in tl and not colors:
        colors = []
    power = parse_pt(card.get("power"))
    toughness = parse_pt(card.get("toughness"))
    variable_pt = str(card.get("power") or "") in {"*", "X", "*+1", "1+*"} or str(card.get("toughness") or "") in {"*", "X"}

    def has_kw(*names: str) -> int:
        return int(any(name in keywords or name in low for name in names))

    is_land = int("land" in tl)
    is_creature = int("creature" in tl)
    is_instant = int(tl.startswith("instant") or " instant" in tl or tl.split(" // ")[0].strip() == "instant")
    is_sorcery = int("sorcery" in tl)
    is_enchant = int("enchantment" in tl)
    is_artifact = int("artifact" in tl)
    is_aura = int("aura" in tl)
    is_equipment = int("equipment" in tl)
    is_saga = int("saga" in tl)
    is_spacecraft = int("spacecraft" in tl)
    is_vehicle = int("vehicle" in tl)
    is_legendary = int("legendary" in tl)
    is_basic = int("basic" in tl)
    is_battle = int("battle" in tl)
    is_planeswalker = int("planeswalker" in tl)

    w = mana.count("{W}") + low.count("{w}")
    u = mana.count("{U}")
    b = mana.count("{B}")
    r = mana.count("{R}")
    g = mana.count("{G}")
    # Hybrid symbols look like {U/R}. Count them as gold, not as a full pip of each.
    hybrid = len(re.findall(r"\{[WUBRG]/[WUBRG]\}", mana))
    generic = 0
    for match in re.finditer(r"\{(\d+)\}", mana):
        generic += int(match.group(1))
    x_cost = int("{X}" in mana or "{x}" in mana.lower())
    n_colors = len(colors) if colors else (0 if is_land else len(ci))
    if is_land:
        n_colors = len(ci)

    # Effects.
    draw_n = count_draws(low)
    loot = len(re.findall(r"draw a card, then discard|discard a card, then draw|draw two cards, then discard", low))
    impulse = len(re.findall(r"exile (?:the top|cards from the top|target).{0,80}you may (?:play|cast)", low))
    impulse += len(re.findall(r"you may play (?:that|those|the exiled|it|them)", low))
    free_cast = len(re.findall(r"without paying (?:its|their) mana cost", low))
    discover = first_number(r"discover (\d+|x)", low, 0)
    surveil_n = 0.0
    for match in re.finditer(r"surveil (\d+|x)", low):
        surveil_n += word_to_n(match.group(1), 1)
    if "surveil" in keywords and surveil_n == 0:
        surveil_n = 1.0

    destroy_creature = len(re.findall(r"destroy target (?:creature|artifact creature|nonartifact creature|tapped creature|attacking creature|blocking creature)", low))
    destroy_any = len(re.findall(r"destroy (?:target|all|each)", low))
    exile_perm = len(re.findall(r"exile target (?:creature|nonland|artifact|permanent|enchantment)", low))
    exile_until = int("exile target" in low and "until" in low and "leaves the battlefield" in low)
    damage_nums = [word_to_n(n, 0) for n in re.findall(r"deals (\d+|x) damage", low)]
    max_damage = max(damage_nums) if damage_nums else 0.0
    damage_any = int("damage to any target" in low)
    damage_each = int("damage to each creature" in low or "damage to each opponent" in low)
    max_minus = 0.0
    if re.search(r"-(\d+)/-\1", low):
        max_minus = max(word_to_n(n, 0) for n in re.findall(r"-(\d+)/-\1", low))
    fight = int("fights" in low or "fight " in low)
    bite = int("deals damage equal to its power" in low or "deal damage equal to their power" in low)
    bounce = int("return target" in low and "to its owner's hand" in low)
    tuck = int("top or bottom of their library" in low or "on top of its owner" in low or "top of their library" in low)
    counterspell = int("counter target spell" in low or "counter target noncreature" in low or "counter target activated" in low)
    counter_unless = int("unless its controller pays" in low or "unless that player pays" in low)
    wrath = int(
        "destroy all creatures" in low
        or "all creatures get -" in low
        or "damage to each creature" in low
        or "destroy all nonartifact" in low
    )
    minus_x = int("-x/-x" in low)

    tokens = 0.0
    token_power = 0.0
    for match in re.finditer(
        r"create (?:(a|an|one|two|three|four|five|six|seven|eight|ten|\d+) )?(\d+)/(\d+)",
        low,
    ):
        n = word_to_n(match.group(1) or "a", 1)
        tokens += n
        token_power += n * word_to_n(match.group(2), 0)
    token_x = int(re.search(r"create x \d+/\d+", low) is not None)
    treasure = int("treasure" in low)
    clue = int("clue" in low or "investigate" in low or "investigate" in keywords)
    food = int("food" in low or "food" in keywords)
    lander = int("lander" in low)

    etb = int(
        "when this creature enters" in low
        or "when this artifact enters" in low
        or "when this enchantment enters" in low
        or "when this spacecraft enters" in low
        or "when this equipment enters" in low
        or "when this aura enters" in low
        or re.search(r"when [a-z0-9'., \-]+ enters", low) is not None
    )
    dies = int("when this creature dies" in low or "when this permanent dies" in low or re.search(r"when [a-z0-9'., \-]+ dies", low) is not None)
    attacks = int("whenever this creature attacks" in low or "whenever you attack" in low or re.search(r"whenever [a-z0-9'., \-]+ attacks", low) is not None)
    upkeep = int("at the beginning of your upkeep" in low or "at the beginning of each upkeep" in low)
    end_step = int("end step" in low)
    activated = int("{t}:" in low or "{t}," in low or re.search(r"\{[wubrg\d/]+\}(?:, \{t\})?:", low) is not None)
    static_anthem = int(re.search(r"(creatures you control|other creatures you control) get \+1/\+1", low) is not None)
    team_pump = int("creatures you control get +" in low)
    single_pump = int(re.search(r"gets \+\d/\+\d", low) is not None)

    modal = int("choose one" in low or "choose two" in low or "choose one or more" in low or "choose one or both" in low)
    choose_two = int("choose two" in low)
    choose_more = int("choose one or more" in low or "choose one or both" in low)
    grave_cast = int(
        "cast this card from your graveyard" in low
        or "flashback" in low
        or "unearth" in low
        or "escape" in keywords
        or "disturb" in keywords
        or "jump-start" in low
        or "beam me up" in low
    )
    beam = int("beam me up" in low)
    reanimate = int("return target creature card from" in low and "graveyard" in low and "battlefield" in low)
    reanimate += int("put target creature card from a graveyard onto the battlefield" in low)
    reanimate += int("assimilate" in low)
    gy_to_hand = int("from your graveyard to your hand" in low or "from a graveyard to your hand" in low)
    blink = int("exile" in low and "return" in low and "battlefield" in low and "end step" in low)
    copy_effect = int("copy" in low)
    extra_turn = int("extra turn" in low)
    you_win = int("you win the game" in low)
    cant_lose = int("you don't lose the game" in low or "can't lose the game" in low)
    gain_control = int("gain control" in low)
    mill_n = 0.0
    for match in re.finditer(r"mill (\d+|x|a|two|three|four|five)", low):
        mill_n += word_to_n(match.group(1), 1)
    discard_opp = int("discard a card" in low or "discard two cards" in low) and int("opponent" in low or "each player discards" in low or "target player discards" in low)
    life_gain = int("you gain" in low or "gain life" in low)
    drain = int("loses" in low and "life" in low and "you gain" in low)
    plus_counter = int("+1/+1 counter" in low)
    double_counters = int("twice that many" in low or "double the number" in low or "put twice" in low)
    draw_double = int("if you would draw a card, draw two" in low)
    proliferate = int("proliferate" in low)
    sac_outlet = int("sacrifice a creature" in low or "sacrifice another" in low or "sacrifice an artifact" in low)
    search_land = int("search your library for a basic land" in low or "search your library for a basic land card or planet" in low or "search your library for up to two basic" in low)
    ramp_battlefield = int("put" in low and "onto the battlefield" in low and "land" in low)
    any_color_mana = int("mana of any color" in low or "add one mana of any color" in low)
    stun = int("stun counter" in low)
    charge = int("charge counter" in low)
    tap_down = int("tap target creature" in low or "tap target artifact or creature" in low or "tap up to one target" in low)
    hexproof_g = int("gains hexproof" in low or "have hexproof" in low)
    indestructible_g = int("gains indestructible" in low or "have indestructible" in low)
    cant_block = int("can't be blocked" in low)
    cost_reduction = int("costs {" in low and "less to cast" in low) or int("this spell costs" in low)
    enters_tapped = int("enters tapped" in low)
    untapped_option = int("you may pay 2 life" in low and "enters tapped" in low)
    shock_land = int(untapped_option and is_land and "add" not in low and ("swamp" in tl or "island" in tl or "plains" in tl or "mountain" in tl or "forest" in tl) and tl.count("—") >= 0 and len(ci) == 2 and "basic" not in tl and "you may pay 2 life" in low)
    # Shock lands have the pay-2-life line and two basic land types.
    shock_land = int(is_land and "you may pay 2 life" in low and not is_basic)
    gain_land = int(is_land and enters_tapped and "you gain 1 life" in low and not is_basic)
    fetchish = int(is_land and "search your library for a basic land" in low)
    taps_two_colors = int(is_land and len(ci) >= 2)
    colorless_source = int(is_land and len(ci) == 0)

    station_threshold = 0.0
    station_match = re.findall(r"at (\d+)\+", low)
    if station_match:
        station_threshold = min(float(x) for x in station_match)
    early_station = 0.0
    early = re.findall(r"(\d+)\+ \|", text)
    # The printed station abilities use "N+ |" in oracle. The smallest N is the
    # first ability you can turn on.
    if early:
        early_station = min(float(x) for x in early)

    beam_cost_cmc = 0.0
    beam_match = re.search(r"beam me up (\{[^)]+?) \(", text, flags=re.I)
    if beam_match:
        beam_cost_cmc = float(len(re.findall(r"\{[^}]+\}", beam_match.group(1))))

    strange = int("strange new worlds" in low)
    federation = int("creature types among non-borg" in low or low.strip().startswith("federation"))
    dilemma_payoff = int("face a dilemma" in low or "faced a dilemma" in low)
    power8 = int("total power 8 or greater" in low)
    unique_lands = int("differently named lands" in low)
    noncreature_payoff = int("noncreature spell" in low)
    artifact_payoff = int("artifact creatures you control" in low or "artifacts you control" in low)
    officer = int("officer" in low)
    borg = int("borg" in tl or "borg" in low)
    klingon = int("klingon" in tl or "klingon" in low)
    landfall = int("landfall" in low or "landfall" in keywords)
    second_land = int("second time this ability" in low)

    flying = has_kw("flying")
    trample = has_kw("trample")
    lifelink = has_kw("lifelink")
    deathtouch = has_kw("deathtouch")
    haste = has_kw("haste")
    vigilance = has_kw("vigilance")
    menace = has_kw("menace")
    reach = has_kw("reach")
    flash = has_kw("flash")
    defender = has_kw("defender")
    first_strike = has_kw("first strike")
    double_strike = has_kw("double strike")
    ward = int("ward" in keywords or "ward {" in low)
    ward_2 = int("ward {2}" in low or "ward {3}" in low)
    cycling = int("cycling" in keywords or "landcycling" in low or "cycling {" in low)

    body = 0.0
    if is_creature and power is not None and toughness is not None and not is_spacecraft:
        body = (power + toughness) - (2.0 * max(cmc, 0.5))
    evasion = flying + trample + menace + deathtouch + first_strike + double_strike + cant_block + lifelink * 0.5
    removal_raw = (
        3.2 * exile_perm
        + 2.6 * exile_until
        + 3.0 * destroy_creature
        + 1.4 * max(0, destroy_any - destroy_creature)
        + 0.55 * max_damage
        + 0.45 * max_minus
        + 2.4 * counterspell
        + 1.6 * fight
        + 1.8 * bite
        + 1.5 * bounce
        + 1.2 * tap_down
        + 2.2 * wrath
        + 1.5 * minus_x
        + 0.8 * stun
    )
    if is_instant:
        removal_raw *= 1.12
    conditional = int(any(phrase in low for phrase in [
        "power 4 or greater", "with power", "tapped creature", "attacking creature",
        "blocking creature", "nonartifact creature", "mana value 3 or less",
        "mana value 2 or less", "unless its controller",
    ]))
    if conditional and removal_raw > 0:
        removal_raw *= 0.82
    removal_eff = removal_raw / math.sqrt(max(cmc, 0.7))
    ca_raw = (
        1.6 * draw_n
        + 1.3 * discover
        + 1.1 * free_cast
        + 0.7 * impulse
        + 0.45 * surveil_n
        + 0.8 * loot
        + 0.9 * clue
        + 0.35 * mill_n * 0.25
        + 1.4 * draw_double
        + 0.8 * gy_to_hand
        + 1.6 * reanimate
    )
    token_value = 0.55 * tokens + 0.15 * token_power + 0.4 * treasure + 0.25 * food
    if is_land:
        land_score = (
            1.6 * shock_land
            + 0.55 * gain_land
            + 0.7 * taps_two_colors * (1 - enters_tapped)
            + 0.35 * taps_two_colors * enters_tapped
            + 0.45 * fetchish
            + 0.15 * (not enters_tapped)
            - 0.35 * enters_tapped * (not gain_land) * (not shock_land)
            - 0.45 * colorless_source * (not fetchish)
            + 0.2 * any_color_mana
        )
    else:
        land_score = 0.0

    # A single index the linear model can lean on. Trees also see the parts.
    power_index = (
        0.85 * removal_eff
        + 0.9 * ca_raw
        + 0.45 * token_value
        + 0.35 * evasion
        + 0.25 * body
        + 0.55 * etb
        + 0.35 * modal
        + 0.4 * static_anthem
        + 0.7 * grave_cast
        + 0.5 * search_land
        + 0.6 * ramp_battlefield
        + 0.8 * extra_turn
        + 0.4 * hexproof_g
        + 0.4 * ward
        + 0.3 * flash * is_creature
        - 0.55 * max(0.0, cmc - 5.0) * (0 if (search_land or ramp_battlefield or cost_reduction or removal_raw > 2 or free_cast) else 1)
        - 0.8 * defender * (0 if "can attack" in low else 1)
        - 0.7 * is_aura * (0.4 if removal_raw > 0 else 1)
        + land_score
        + 0.5 * plus_counter * is_creature * etb
        + 0.4 * dies
    )

    augmented = low
    if beam:
        augmented += " flashback you may cast this card from your graveyard by paying a cost and returning a creature you control to its owner's hand"
    if strange or unique_lands:
        augmented += " domain for each differently named land you control"
    if dilemma_payoff:
        augmented += " whenever you cast a modal spell or choose modes draw a card or drain"
    if federation:
        augmented += " party creature types among creatures you control"
    if "assimilate" in low:
        augmented += " return target creature card from a graveyard to the battlefield under your control"
    if is_spacecraft:
        augmented += " crew vehicle station charge counters artifact creature"
    if explore := ("explore" in low):
        augmented += " explore reveal the top card of your library if it's a land put it into your hand otherwise put a +1/+1 counter"

    features = {
        "cmc": cmc,
        "generic": generic,
        "x_cost": x_cost,
        "hybrid": hybrid,
        "n_colors": n_colors,
        "c_w": int("W" in colors or (is_land and "W" in ci)),
        "c_u": int("U" in colors or (is_land and "U" in ci)),
        "c_b": int("B" in colors or (is_land and "B" in ci)),
        "c_r": int("R" in colors or (is_land and "R" in ci)),
        "c_g": int("G" in colors or (is_land and "G" in ci)),
        "is_land": is_land,
        "is_creature": is_creature,
        "is_instant": is_instant,
        "is_sorcery": is_sorcery,
        "is_enchant": is_enchant,
        "is_artifact": is_artifact,
        "is_aura": is_aura,
        "is_equipment": is_equipment,
        "is_saga": is_saga,
        "is_spacecraft": is_spacecraft,
        "is_vehicle": is_vehicle,
        "is_legendary": is_legendary,
        "is_basic": is_basic,
        "is_battle": is_battle,
        "is_planeswalker": is_planeswalker,
        "is_bonus": int(card.get("is_bonus") or card.get("bonus") or False),
        "power": power if power is not None else 0.0,
        "toughness": toughness if toughness is not None else 0.0,
        "has_pt": int(power is not None and toughness is not None),
        "variable_pt": int(variable_pt),
        "body": body,
        "flying": flying,
        "trample": trample,
        "lifelink": lifelink,
        "deathtouch": deathtouch,
        "haste": haste,
        "vigilance": vigilance,
        "menace": menace,
        "reach": reach,
        "flash": flash,
        "defender": defender,
        "first_strike": first_strike,
        "double_strike": double_strike,
        "ward": ward,
        "ward_2": ward_2,
        "cycling": cycling,
        "evasion": evasion,
        "draw_n": draw_n,
        "loot": loot,
        "impulse": impulse,
        "free_cast": free_cast,
        "discover": discover,
        "surveil_n": surveil_n,
        "destroy_creature": destroy_creature,
        "exile_perm": exile_perm,
        "exile_until": exile_until,
        "max_damage": max_damage,
        "damage_any": damage_any,
        "damage_each": damage_each,
        "max_minus": max_minus,
        "fight": fight,
        "bite": bite,
        "bounce": bounce,
        "tuck": tuck,
        "counterspell": counterspell,
        "counter_unless": counter_unless,
        "wrath": wrath,
        "minus_x": minus_x,
        "removal_eff": removal_eff,
        "ca_raw": ca_raw,
        "tokens": tokens,
        "token_power": token_power,
        "token_x": token_x,
        "token_value": token_value,
        "treasure": treasure,
        "clue": clue,
        "food": food,
        "etb": etb,
        "dies": dies,
        "attacks": attacks,
        "upkeep": upkeep,
        "end_step": end_step,
        "activated": int(activated),
        "static_anthem": static_anthem,
        "team_pump": team_pump,
        "single_pump": single_pump,
        "modal": modal,
        "choose_two": choose_two,
        "choose_more": choose_more,
        "grave_cast": grave_cast,
        "beam": beam,
        "beam_cost_cmc": beam_cost_cmc,
        "reanimate": reanimate,
        "gy_to_hand": gy_to_hand,
        "blink": blink,
        "copy_effect": copy_effect,
        "extra_turn": extra_turn,
        "you_win": you_win,
        "cant_lose": cant_lose,
        "gain_control": gain_control,
        "mill_n": mill_n,
        "discard_opp": int(discard_opp),
        "life_gain": life_gain,
        "drain": drain,
        "plus_counter": plus_counter,
        "double_counters": double_counters,
        "draw_double": draw_double,
        "sac_outlet": int(sac_outlet),
        "search_land": search_land,
        "ramp_battlefield": ramp_battlefield,
        "any_color_mana": any_color_mana,
        "stun": stun,
        "charge": charge,
        "tap_down": tap_down,
        "hexproof_g": hexproof_g,
        "indestructible_g": indestructible_g,
        "cant_block": cant_block,
        "cost_reduction": cost_reduction,
        "enters_tapped": enters_tapped,
        "shock_land": shock_land,
        "gain_land": gain_land,
        "fetchish": fetchish,
        "taps_two_colors": taps_two_colors,
        "colorless_land": colorless_source,
        "land_score": land_score,
        "station_threshold": station_threshold if station_threshold else early_station,
        "early_station": early_station,
        "strange": strange,
        "federation": int(federation),
        "dilemma_payoff": dilemma_payoff,
        "power8": power8,
        "unique_lands": unique_lands,
        "noncreature_payoff": noncreature_payoff,
        "artifact_payoff": artifact_payoff,
        "landfall": landfall,
        "second_land": second_land,
        "officer": officer,
        "borg": borg,
        "klingon": klingon,
        "power_index": power_index,
        "oracle_len": len(text),
        "text": augmented,
        "rarity": card.get("rarity") or "common",
        "pips_w": w,
        "pips_u": u,
        "pips_b": b,
        "pips_r": r,
        "pips_g": g,
    }
    return features


STRUCTURAL = [
    "cmc", "generic", "x_cost", "hybrid", "n_colors",
    "c_w", "c_u", "c_b", "c_r", "c_g",
    "is_land", "is_creature", "is_instant", "is_sorcery", "is_enchant", "is_artifact",
    "is_aura", "is_equipment", "is_saga", "is_spacecraft", "is_vehicle", "is_legendary",
    "is_basic", "is_battle", "is_planeswalker", "is_bonus",
    "power", "toughness", "has_pt", "variable_pt", "body",
    "flying", "trample", "lifelink", "deathtouch", "haste", "vigilance", "menace",
    "reach", "flash", "defender", "first_strike", "double_strike", "ward", "ward_2", "cycling",
    "evasion", "draw_n", "loot", "impulse", "free_cast", "discover", "surveil_n",
    "destroy_creature", "exile_perm", "exile_until", "max_damage", "damage_any", "damage_each",
    "max_minus", "fight", "bite", "bounce", "tuck", "counterspell", "counter_unless",
    "wrath", "minus_x", "removal_eff", "ca_raw", "tokens", "token_power", "token_x",
    "token_value", "treasure", "clue", "food", "etb", "dies", "attacks", "upkeep",
    "end_step", "activated", "static_anthem", "team_pump", "single_pump", "modal",
    "choose_two", "choose_more", "grave_cast", "beam", "beam_cost_cmc", "reanimate",
    "gy_to_hand", "blink", "copy_effect", "extra_turn", "you_win", "cant_lose",
    "gain_control", "mill_n", "discard_opp", "life_gain", "drain", "plus_counter",
    "double_counters", "draw_double", "sac_outlet", "search_land", "ramp_battlefield",
    "any_color_mana", "stun", "charge", "tap_down", "hexproof_g", "indestructible_g",
    "cant_block", "cost_reduction", "enters_tapped", "shock_land", "gain_land", "fetchish",
    "taps_two_colors", "colorless_land", "land_score", "station_threshold", "early_station",
    "strange", "federation", "dilemma_payoff", "power8", "unique_lands", "noncreature_payoff",
    "artifact_payoff", "landfall", "second_land", "officer", "borg", "klingon",
    "power_index", "oracle_len", "pips_w", "pips_u", "pips_b", "pips_r", "pips_g",
    "r_common", "r_uncommon", "r_rare", "r_mythic", "r_bonus",
]


def rarity_dummies(rarity: str) -> dict:
    rarity = rarity or "common"
    return {
        "r_common": int(rarity == "common"),
        "r_uncommon": int(rarity == "uncommon"),
        "r_rare": int(rarity == "rare"),
        "r_mythic": int(rarity == "mythic"),
        "r_bonus": int(rarity == "bonus"),
    }


def add_pool_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Color-environment features computed from the card pool, not from win rates.

    A card is scored relative to the other cards of its colors in the same set,
    and each color gets a raw 'materials' rating (removal, bodies, card advantage)
    that the model can use as format context.
    """
    frame = frame.copy()
    # Monocolor commons/uncommons describe a color. Gold and lands are excluded
    # from the color aggregate so signposts do not define their own environment.
    mono = frame[(frame["n_colors"] == 1) & (frame["is_land"] == 0) & (frame["is_bonus"] == 0)].copy()
    color_cols = ["c_w", "c_u", "c_b", "c_r", "c_g"]
    rows = []
    for (set_code, color), group in _iter_colors(mono, color_cols):
        rows.append({
            "set": set_code,
            "color": color,
            "col_removal": group["removal_eff"].mean(),
            "col_ca": group["ca_raw"].mean(),
            "col_body2": group.loc[group["cmc"].between(2, 3) & (group["is_creature"] == 1), "body"].mean() if ((group["cmc"].between(2, 3)) & (group["is_creature"] == 1)).any() else 0.0,
            "col_cheap": ((group["cmc"] <= 2) & (group["is_creature"] == 1)).mean(),
            "col_interaction": ((group["removal_eff"] > 0.8) | (group["counterspell"] == 1)).mean(),
            "col_modal": group["modal"].mean(),
            "n": len(group),
        })
    env = pd.DataFrame(rows)
    env_map = {(r.set, r.color): r for r in env.itertuples(index=False)}
    col_removal = []
    col_ca = []
    col_body2 = []
    col_cheap = []
    col_interaction = []
    col_modal = []
    for rec in frame.itertuples(index=False):
        present = [name for name, flag in zip("WUBRG", [rec.c_w, rec.c_u, rec.c_b, rec.c_r, rec.c_g]) if flag]
        if not present:
            present = ["W", "U", "B", "R", "G"]  # colorless sees the whole format
        stats = [env_map.get((rec.set, color)) for color in present]
        stats = [s for s in stats if s is not None]
        if not stats:
            col_removal.append(0)
            col_ca.append(0)
            col_body2.append(0)
            col_cheap.append(0)
            col_interaction.append(0)
            col_modal.append(0)
            continue
        col_removal.append(float(np.mean([s.col_removal for s in stats])))
        col_ca.append(float(np.mean([s.col_ca for s in stats])))
        col_body2.append(float(np.mean([s.col_body2 for s in stats])))
        col_cheap.append(float(np.mean([s.col_cheap for s in stats])))
        col_interaction.append(float(np.mean([s.col_interaction for s in stats])))
        col_modal.append(float(np.mean([s.col_modal for s in stats])))
    frame["col_removal"] = col_removal
    frame["col_ca"] = col_ca
    frame["col_body2"] = col_body2
    frame["col_cheap"] = col_cheap
    frame["col_interaction"] = col_interaction
    frame["col_modal"] = col_modal
    # Set-level fixing, available for TRK from the card file alone.
    fixing = frame.groupby("set")["taps_two_colors"].mean().to_dict()
    speed = frame.groupby("set").apply(
        lambda g: float(((g["cmc"] <= 2) & (g["is_creature"] == 1) & (g["is_bonus"] == 0)).mean()),
        include_groups=False,
    ).to_dict()
    frame["set_fixing"] = frame["set"].map(fixing)
    frame["set_cheap_creatures"] = frame["set"].map(speed)
    return frame


def _iter_colors(frame: pd.DataFrame, color_cols: list[str]):
    names = "WUBRG"
    for set_code, group in frame.groupby("set"):
        for name, col in zip(names, color_cols):
            sub = group[group[col] == 1]
            if len(sub):
                yield (set_code, name), sub


CONTEXT_COLS = [
    "col_removal", "col_ca", "col_body2", "col_cheap", "col_interaction", "col_modal",
    "set_fixing", "set_cheap_creatures",
]


def design_matrix(frame: pd.DataFrame, svd_components: np.ndarray | None = None) -> np.ndarray:
    cols = STRUCTURAL + CONTEXT_COLS
    base = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    if svd_components is not None:
        return np.hstack([base, np.nan_to_num(svd_components, nan=0.0)])
    return base


def fit_svd(train_text: pd.Series, test_text: pd.Series, n_components: int = 48):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        max_features=6000,
        strip_accents="unicode",
    )
    train_tf = vectorizer.fit_transform(train_text)
    # TruncatedSVD needs enough features. Fall back if a tiny fold is weird.
    n_comp = min(n_components, train_tf.shape[1] - 1, train_tf.shape[0] - 1)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    train_z = svd.fit_transform(train_tf)
    test_z = svd.transform(vectorizer.transform(test_text))
    return train_z, test_z, vectorizer, svd


def _lgb(train_x, train_y, train_w, valid_x=None, valid_y=None):
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=1.5,
        reg_alpha=0.1,
        random_state=0,
        verbosity=-1,
    )
    fit_kwargs = {}
    if valid_x is not None:
        fit_kwargs["eval_X"] = valid_x
        fit_kwargs["eval_y"] = valid_y
        fit_kwargs["callbacks"] = [lgb.early_stopping(40, verbose=False)]
    model.fit(train_x, train_y, sample_weight=train_w, **fit_kwargs)
    return model


def _hgb():
    return HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.1,
        random_state=0,
    )


def loo_and_fit(hist: pd.DataFrame):
    hist = hist[hist["gih_n"] >= 200].reset_index(drop=True).copy()
    hist["weight"] = np.sqrt(np.clip(hist["gih_n"], 200, 8000))
    y = hist["gih_wr"].to_numpy()
    sets = list(hist["set"].unique())
    oof = np.full(len(hist), np.nan)
    oof_parts = {name: np.full(len(hist), np.nan) for name in ["lgb", "hgb", "ridge", "knn"]}

    for hold in sets:
        test_idx = np.where(hist["set"].to_numpy() == hold)[0]
        train_idx = np.where(hist["set"].to_numpy() != hold)[0]
        # Inner early-stopping split: last remaining set alphabetically, stable.
        inner_sets = sorted(set(sets) - {hold})
        inner_hold = inner_sets[-1]
        inner_valid = np.where(hist["set"].to_numpy() == inner_hold)[0]
        inner_train = np.array([i for i in train_idx if i not in set(inner_valid)])
        train_z, test_z, _, _ = fit_svd(hist.loc[train_idx, "text"], hist.loc[test_idx, "text"])
        # Refit svd pieces for the inner split by slicing the train_z rows.
        # train_z is aligned to train_idx order, not original index. Rebuild simply.
        train_z, valid_and_test_z, _, _ = fit_svd(
            hist.loc[inner_train, "text"],
            pd.concat([hist.loc[inner_valid, "text"], hist.loc[test_idx, "text"]], ignore_index=True),
        )
        valid_z = valid_and_test_z[: len(inner_valid)]
        test_only_z = valid_and_test_z[len(inner_valid) :]
        x_inner = design_matrix(hist.loc[inner_train])
        x_valid = design_matrix(hist.loc[inner_valid])
        x_test = design_matrix(hist.loc[test_idx])
        x_inner = np.hstack([x_inner, train_z])
        x_valid = np.hstack([x_valid, valid_z])
        x_test = np.hstack([x_test, test_only_z])
        y_inner = y[inner_train]
        y_valid = y[inner_valid]
        w_inner = hist.loc[inner_train, "weight"].to_numpy()
        lgb_model = _lgb(x_inner, y_inner, w_inner, x_valid, y_valid)
        # Refit trees on all non-hold rows with the iteration count selected above.
        full_train_z, full_test_z, _, _ = fit_svd(hist.loc[train_idx, "text"], hist.loc[test_idx, "text"])
        x_train = np.hstack([design_matrix(hist.loc[train_idx]), full_train_z])
        x_test = np.hstack([design_matrix(hist.loc[test_idx]), full_test_z])
        best_iter = getattr(lgb_model, "best_iteration_", None) or lgb_model.n_estimators
        lgb_full = lgb.LGBMRegressor(
            n_estimators=max(int(best_iter), 80),
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_lambda=1.5,
            reg_alpha=0.1,
            random_state=0,
            verbosity=-1,
        )
        lgb_full.fit(x_train, y[train_idx], sample_weight=hist.loc[train_idx, "weight"].to_numpy())
        hgb = _hgb()
        hgb.fit(x_train, y[train_idx], sample_weight=hist.loc[train_idx, "weight"].to_numpy())
        ridge = Ridge(alpha=4.0)
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)
        ridge.fit(x_train_s, y[train_idx], sample_weight=hist.loc[train_idx, "weight"].to_numpy())
        train_sets = hist.loc[train_idx].groupby("set")["set_wr"].first()
        center = float(train_sets.mean())
        knn_pred = knn_predict(
            x_train_s, y[train_idx], hist.loc[train_idx, "set_wr"].to_numpy(),
            hist.loc[train_idx, "weight"].to_numpy(), x_test_s, center=center, k=25,
        )
        p_lgb = lgb_full.predict(x_test)
        p_hgb = hgb.predict(x_test)
        p_ridge = ridge.predict(x_test_s)
        oof_parts["lgb"][test_idx] = p_lgb
        oof_parts["hgb"][test_idx] = p_hgb
        oof_parts["ridge"][test_idx] = p_ridge
        oof_parts["knn"][test_idx] = knn_pred
        print(f"  holdout {hold} n={len(test_idx)} lgb_trees={max(int(best_iter), 80)}", flush=True)

    # Stack on OOF predictions. Non-negative weights via a clipped least squares.
    stack_x = np.column_stack([oof_parts[name] for name in ["lgb", "hgb", "ridge", "knn"]])
    meta = Ridge(alpha=1.0, positive=True)
    meta.fit(stack_x, y, sample_weight=hist["weight"].to_numpy())
    oof = meta.predict(stack_x)
    oof = np.clip(oof, 0.40, 0.75)
    weights = {"lgb": float(meta.coef_[0]), "hgb": float(meta.coef_[1]), "ridge": float(meta.coef_[2]), "knn": float(meta.coef_[3]), "intercept": float(meta.intercept_)}
    print("stack weights", weights, flush=True)

    # Spread correction. GBM stacks compress tails. Choose a global alpha on OOF
    # that minimizes weighted MAE, searched on a small grid. Alpha is fit on OOF
    # predictions, which is slightly optimistic but the grid is one parameter.
    rarity = hist["rarity"].to_numpy()
    alpha, oof_cal = calibrate_spread(y, oof, rarity, hist["weight"].to_numpy())
    print(f"spread alpha {alpha:.3f}", flush=True)

    metrics = evaluate(hist, y, oof_cal)
    metrics["stack_weights"] = weights
    metrics["spread_alpha"] = alpha
    metrics["oof_mae_raw"] = float(mean_absolute_error(y, oof, sample_weight=hist["weight"]))
    return hist, oof_cal, metrics, alpha, weights


def knn_predict(train_x, train_y, train_base, train_w, test_x, center: float, k=25):
    nn = NearestNeighbors(n_neighbors=min(k, len(train_x)), metric="euclidean")
    nn.fit(train_x)
    dist, idx = nn.kneighbors(test_x)
    pred = np.zeros(len(test_x))
    for i in range(len(test_x)):
        d = dist[i]
        j = idx[i]
        # Distance weights. Neighbors are recentered on their own set's win
        # rate so what transfers is card quality, then placed on `center`,
        # the best guess of the target format's baseline.
        w = 1.0 / np.maximum(d, 1e-3)
        w = w * train_w[j]
        residual = train_y[j] - train_base[j]
        pred[i] = center + np.sum(w * residual) / np.sum(w)
    return pred


def calibrate_spread(y, pred, rarity, weight):
    best_alpha = 1.0
    best_mae = 1e9
    best = pred
    # Per-rarity mean of predictions, then spread around that mean.
    for alpha in np.linspace(1.0, 1.6, 13):
        adjusted = pred.copy()
        for rarity_name in np.unique(rarity):
            mask = rarity == rarity_name
            center = np.average(pred[mask], weights=weight[mask])
            adjusted[mask] = center + alpha * (pred[mask] - center)
        adjusted = np.clip(adjusted, 0.40, 0.75)
        mae = mean_absolute_error(y, adjusted, sample_weight=weight)
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
            best = adjusted
    return best_alpha, best


def evaluate(hist: pd.DataFrame, y, pred) -> dict:
    weight = hist["weight"].to_numpy()
    mae = float(mean_absolute_error(y, pred, sample_weight=weight))
    rmse = float(math.sqrt(mean_squared_error(y, pred, sample_weight=weight)))
    null = float(np.average(y, weights=weight))
    null_mae = float(np.average(np.abs(y - null), weights=weight))
    by_rarity = {}
    for rarity, group in hist.groupby("rarity"):
        idx = group.index.to_numpy()
        # groupby index may not be positional if filtered. Use positions.
        mask = (hist["rarity"] == rarity).to_numpy()
        by_rarity[rarity] = {
            "n": int(mask.sum()),
            "mae": float(mean_absolute_error(y[mask], pred[mask], sample_weight=weight[mask])),
            "mean_actual": float(np.average(y[mask], weights=weight[mask])),
            "mean_pred": float(np.average(pred[mask], weights=weight[mask])),
            "std_actual": float(np.sqrt(np.average((y[mask] - np.average(y[mask], weights=weight[mask])) ** 2, weights=weight[mask]))),
            "std_pred": float(np.sqrt(np.average((pred[mask] - np.average(pred[mask], weights=weight[mask])) ** 2, weights=weight[mask]))),
        }
    by_set = {}
    for set_code, group in hist.groupby("set"):
        mask = (hist["set"] == set_code).to_numpy()
        by_set[set_code] = {
            "n": int(mask.sum()),
            "mae": float(mean_absolute_error(y[mask], pred[mask], sample_weight=weight[mask])),
            "bias": float(np.average(pred[mask] - y[mask], weights=weight[mask])),
        }
    # Spearman via rank correlation.
    spearman = float(pd.Series(y).corr(pd.Series(pred), method="spearman"))
    return {
        "n": int(len(y)),
        "mae": mae,
        "rmse": rmse,
        "null_predict_mean_mae": null_mae,
        "spearman": spearman,
        "by_rarity": by_rarity,
        "by_set": by_set,
    }


def fit_full_and_predict(hist: pd.DataFrame, trk: pd.DataFrame, alpha: float, weights: dict):
    y = hist["gih_wr"].to_numpy()
    w = hist["weight"].to_numpy()
    train_z, test_z, vectorizer, svd = fit_svd(hist["text"], trk["text"])
    x_train = np.hstack([design_matrix(hist), train_z])
    x_test = np.hstack([design_matrix(trk), test_z])
    # Use a modest tree count close to what early stopping picked on average.
    lgb_full = lgb.LGBMRegressor(
        n_estimators=280,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=1.5,
        reg_alpha=0.1,
        random_state=0,
        verbosity=-1,
    )
    lgb_full.fit(x_train, y, sample_weight=w)
    hgb = _hgb()
    hgb.fit(x_train, y, sample_weight=w)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    ridge = Ridge(alpha=4.0)
    ridge.fit(x_train_s, y, sample_weight=w)
    knn = knn_predict(
        x_train_s, y, hist["set_wr"].to_numpy(), w, x_test_s, center=TRK_BASELINE, k=25,
    )
    parts = {
        "lgb": lgb_full.predict(x_test),
        "hgb": hgb.predict(x_test),
        "ridge": ridge.predict(x_test_s),
        "knn": knn,
    }
    coef = np.array([weights["lgb"], weights["hgb"], weights["ridge"], weights["knn"]])
    raw = weights["intercept"] + np.column_stack([parts[name] for name in ["lgb", "hgb", "ridge", "knn"]]) @ coef
    # Spread around per-rarity prediction centers, using the OOF-chosen alpha.
    pred = raw.copy()
    for rarity in trk["rarity"].unique():
        mask = (trk["rarity"] == rarity).to_numpy()
        if mask.sum() == 0:
            continue
        center = pred[mask].mean()
        pred[mask] = center + alpha * (pred[mask] - center)
    pred = np.clip(pred, 0.42, 0.72)
    # Neighbor table for review, from the scaled structural+text space.
    nn = NearestNeighbors(n_neighbors=8, metric="euclidean")
    nn.fit(x_train_s)
    dist, idx = nn.kneighbors(x_test_s)
    neighbors = []
    hist_names = hist["name"].tolist()
    hist_set = hist["set"].tolist()
    hist_y = hist["gih_wr"].tolist()
    hist_n = hist["gih_n"].tolist()
    for i in range(len(trk)):
        entry = []
        for d, j in zip(dist[i], idx[i]):
            entry.append({
                "name": hist_names[j],
                "set": hist_set[j],
                "gih_wr": hist_y[j],
                "gih_n": hist_n[j],
                "distance": float(d),
            })
        neighbors.append(entry)
    importances = sorted(
        zip(STRUCTURAL + CONTEXT_COLS, lgb_full.feature_importances_[: len(STRUCTURAL) + len(CONTEXT_COLS)]),
        key=lambda item: -item[1],
    )
    return pred, parts, neighbors, importances


def build_frame(cards: list[dict], set_wr: float | None = None) -> pd.DataFrame:
    rows = []
    for card in cards:
        try:
            feat = featurize(card)
        except Exception:
            print("featurize failed on", card.get("set"), card.get("name"), flush=True)
            raise
        feat.update(rarity_dummies(feat.pop("rarity")))
        feat["name"] = card["name"]
        feat["set"] = card["set"]
        feat["rarity"] = card.get("rarity") or "common"
        feat["mana_cost"] = card.get("mana_cost") or ""
        feat["type_line"] = card.get("type_line") or ""
        feat["oracle_text"] = card.get("oracle_text") or ""
        feat["collector_number"] = card.get("collector_number") or ""
        feat["cn"] = card.get("cn") or 0
        feat["colors"] = "".join(card.get("colors") or []) or ("C" if "land" not in (card.get("type_line") or "").lower() else "")
        if "land" in (card.get("type_line") or "").lower():
            feat["colors"] = "".join(card.get("color_identity") or []) or "C"
        feat["gih_n"] = card.get("gih_n")
        feat["gih_wr"] = card.get("gih_wr")
        feat["set_wr"] = card.get("set_wr", set_wr if set_wr is not None else TRK_BASELINE)
        feat["is_bonus"] = int(card.get("is_bonus") or card.get("bonus") or False)
        rows.append(feat)
    frame = pd.DataFrame(rows)
    frame = add_pool_context(frame)
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    print("loading GIH", flush=True)
    gih, meta = load_gih()
    cache = Path("/tmp/mtg_analysis/joined.pkl")
    if cache.exists():
        print("loading cached card join", flush=True)
        joined = pd.read_pickle(cache)
    else:
        print("joining scryfall", flush=True)
        joined = attach_cards(gih)
        joined.to_pickle(cache)
    print(f"joined rows {len(joined)}", flush=True)
    hist = build_frame(joined.to_dict(orient="records"))
    # Restore fields build_frame already copied via card dicts. joined records
    # include gih fields because attach_cards merged them into the card dict.
    print("historical feature frame", hist.shape, flush=True)
    print("leave-one-set-out", flush=True)
    hist_fit, oof, metrics, alpha, weights = loo_and_fit(hist)
    print(json.dumps({k: metrics[k] for k in ["n", "mae", "rmse", "null_predict_mean_mae", "spearman", "by_rarity", "by_set"]}, indent=2))
    (OUT / "cv_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("loading TRK", flush=True)
    trk_cards = load_set_cards("TRK", main_only_max_cn=311)
    print(f"trk cards {len(trk_cards)}", flush=True)
    trk = build_frame(trk_cards, set_wr=TRK_BASELINE)
    # TRK pool context must be computed on TRK alone, which build_frame does
    # because every row has set TRK. Historical context was computed including
    # TRK if we concatenated first. build_frame(trk) is independent. Good.
    # But wait: build_frame on hist already ran. trk is separate. Good.
    pred, parts, neighbors, importances = fit_full_and_predict(hist_fit, trk, alpha, weights)
    trk = trk.copy()
    trk["gih_win_rate"] = pred
    for name, values in parts.items():
        trk[f"part_{name}"] = values
    trk["neighbors"] = neighbors
    # Sanity: feature spot checks
    spot = trk[trk["name"].isin([
        "Malfunctioning Holodeck", "Transporter Malfunction", "Mr. Spock, Science Officer",
        "Blood Crypt", "Vulcan", "Tea, Earl Grey, Hot", "U.S.S. Discovery, at Black Alert",
        "Peace in Our Time", "Command Subadjunct",
    ])][["name", "rarity", "gih_win_rate", "removal_eff", "ca_raw", "power_index", "part_lgb", "part_knn"]]
    print(spot.to_string(index=False))
    print("rarity means")
    print(trk.groupby("rarity")["gih_win_rate"].agg(["mean", "std", "min", "max"]))

    keep = [
        "name", "collector_number", "cn", "rarity", "mana_cost", "type_line", "colors",
        "gih_win_rate", "part_lgb", "part_hgb", "part_ridge", "part_knn",
        "power_index", "removal_eff", "ca_raw", "body", "cmc",
    ]
    table = trk[keep].sort_values("cn")
    table["gih_win_rate"] = table["gih_win_rate"].round(4)
    table["gih_win_rate_pct"] = (table["gih_win_rate"] * 100).round(2)
    table.to_csv(OUT / "trk_raw_predictions.csv", index=False)
    # Review sheet
    lines = []
    order = trk.sort_values(["colors", "cn"])
    for i, row in order.iterrows():
        neigh = row["neighbors"]
        neigh_s = "; ".join(
            f"{n['set']} {n['name']} {n['gih_wr']*100:.1f}% (n={n['gih_n']})" for n in neigh[:5]
        )
        oracle = (row["oracle_text"] or "").replace("\n", " | ")
        lines.append(
            f"#{row['collector_number']:>4} [{row['rarity'][:1].upper()}] {row['gih_win_rate']*100:5.2f}%  "
            f"{row['mana_cost']} {row['name']}  pi={row['power_index']:.2f} rem={row['removal_eff']:.2f} ca={row['ca_raw']:.2f}\n"
            f"    {row['type_line']}\n"
            f"    {oracle}\n"
            f"    neighbors: {neigh_s}\n"
        )
    (OUT / "trk_review.txt").write_text("\n".join(lines))
    (OUT / "feature_importance.csv").write_text("feature,importance\n" + "\n".join(f"{name},{score}" for name, score in importances))
    # OOF error sample: worst absolute errors
    err = hist_fit.copy()
    err["oof"] = oof
    err["abs_err"] = (err["oof"] - err["gih_wr"]).abs()
    worst = err.nlargest(40, "abs_err")[["set", "name", "rarity", "gih_n", "gih_wr", "oof", "abs_err"]]
    worst.to_csv(OUT / "loo_worst_errors.csv", index=False)
    # Color residual of commons, to see systematic misses.
    commons = err[(err["rarity"] == "common") & (err["is_land"] == 0) & (err["n_colors"] == 1)]
    rows = []
    for (set_code, color), group in _color_groups(commons):
        rows.append({
            "set": set_code,
            "color": color,
            "n": len(group),
            "actual": group["gih_wr"].mean(),
            "pred": group["oof"].mean(),
            "bias": (group["oof"] - group["gih_wr"]).mean(),
        })
    pd.DataFrame(rows).to_csv(OUT / "loo_color_bias.csv", index=False)
    meta_out = {
        "historical_set_win_rates": meta["baseline"],
        "trk_baseline_assumption": TRK_BASELINE,
        "min_gih_n": 200,
        "cv": metrics,
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta_out, indent=2))
    print("wrote", OUT)


def _color_groups(frame: pd.DataFrame):
    names = [("W", "c_w"), ("U", "c_u"), ("B", "c_b"), ("R", "c_r"), ("G", "c_g")]
    for set_code, group in frame.groupby("set"):
        for name, col in names:
            sub = group[group[col] == 1]
            if len(sub):
                yield (set_code, name), sub


if __name__ == "__main__":
    main()
