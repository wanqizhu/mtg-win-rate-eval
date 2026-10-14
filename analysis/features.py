#!/usr/bin/env python3
"""Card text/attribute features shared by historical sets and Reality Fracture."""
import gzip
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SCRY = Path("/opt/mtg-data/scryfall")
OUT = Path("/workspace/analysis/out")
HIST = ["HOB", "MSH", "SOS", "TMT", "ECL", "TLA", "OM1", "EOE", "FIN", "TDM"]


def _cn(c):
    m = re.match(r"(\d+)", c["collector_number"])
    return int(m.group(1)) if m else 99999


def load_set(code):
    if code == "fra-special-guests":
        lines = (SCRY / "sets" / "fra-special-guests.jsonl").read_text().splitlines()
    else:
        lines = gzip.open(SCRY / "sets" / f"{code.lower()}.jsonl.gz", "rt").read().splitlines()
    cards = [json.loads(l) for l in lines if l.strip()]
    return sorted(cards, key=_cn)


@lru_cache(maxsize=1)
def default_index():
    idx = {}
    with gzip.open(SCRY / "default-cards.jsonl.gz", "rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("lang") != "en":
                continue
            for k in name_keys(c):
                idx.setdefault(k, c)
    return idx


def name_keys(c):
    keys = {c["name"]}
    if "card_faces" in c:
        keys.add(c["card_faces"][0]["name"])
    return keys


def faces(c):
    return c["card_faces"] if "card_faces" in c else [c]


def full_text(c):
    return "\n".join((f.get("oracle_text") or "") for f in faces(c))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


WORDNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "x": 3}

KEYWORDS = ["flying", "trample", "deathtouch", "lifelink", "vigilance", "haste", "first strike", "double strike",
            "menace", "reach", "flash", "ward", "hexproof", "indestructible", "defender", "prowess"]


def featurize(c):
    """Return a dict of numeric features for one Scryfall card object."""
    f0 = faces(c)[0]
    text = full_text(c)
    t = text.lower()
    name = c["name"]
    # replace self-references so regexes are simpler
    for nm in {name, f0.get("name", "")} | {fc.get("name", "") for fc in faces(c)}:
        if nm:
            t = t.replace(nm.lower(), "~")
            short = nm.split(",")[0].lower()
            if len(short) > 3:
                t = t.replace(short, "~")
    t = t.replace("this creature", "~").replace("this enchantment", "~").replace("this artifact", "~")
    type_line = (c.get("type_line") or f0.get("type_line") or "")
    front_type = (f0.get("type_line") or type_line)
    tl = front_type.lower()
    colors = c.get("colors") if c.get("colors") is not None else f0.get("colors", [])
    ci = c.get("color_identity", [])
    mana_cost = f0.get("mana_cost") or c.get("mana_cost") or ""
    d = {}
    d["rarity"] = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}.get(c["rarity"], 1)
    for r in ["common", "uncommon", "rare", "mythic"]:
        d[f"r_{r}"] = int(c["rarity"] == r)
    d["cmc"] = c.get("cmc", 0) or 0
    d["n_colors"] = len(ci)
    d["n_cost_colors"] = len(set(re.findall(r"\{([WUBRG])\}", mana_cost)))
    d["hybrid"] = int(bool(re.search(r"\{[WUBRG2]/[WUBRGP]\}", mana_cost)))
    d["pips"] = len(re.findall(r"\{[WUBRG]\}", mana_cost))
    d["max_pip_one_color"] = max([mana_cost.count("{%s}" % x) for x in "WUBRG"] + [0])
    d["has_x"] = int("{X}" in mana_cost)
    for col in "WUBRG":
        d[f"c_{col}"] = int(col in ci)
    d["colorless"] = int(len(ci) == 0)
    for ty in ["creature", "instant", "sorcery", "enchantment", "artifact", "planeswalker", "land", "legendary",
               "aura", "equipment", "saga", "vehicle", "battle", "basic"]:
        d[f"t_{ty}"] = int(ty in tl)
    d["is_dfc"] = int("card_faces" in c and c.get("layout") in ("transform", "modal_dfc", "meld"))
    d["is_split_like"] = int(c.get("layout") in ("adventure", "prepare", "split", "omen"))
    d["layout_prepare"] = int(c.get("layout") == "prepare")
    p = num(f0.get("power"))
    tg = num(f0.get("toughness"))
    d["power"] = p if not np.isnan(p) else (0 if "*" in str(f0.get("power", "")) else np.nan)
    d["toughness"] = tg if not np.isnan(tg) else (1 if "*" in str(f0.get("toughness", "")) else np.nan)
    if d["t_creature"] and not np.isnan(d["power"]):
        d["pt_sum"] = d["power"] + d["toughness"]
        d["pt_minus_cmc"] = d["pt_sum"] - 2 * d["cmc"]
        d["power_minus_cmc"] = d["power"] - d["cmc"]
    else:
        d["pt_sum"] = np.nan
        d["pt_minus_cmc"] = np.nan
        d["power_minus_cmc"] = np.nan
    kws = [k.lower() for k in c.get("keywords", [])]
    for k in KEYWORDS:
        d[f"kw_{k.replace(' ', '_')}"] = int(k in kws or bool(re.search(r"(^|\n|, )" + k + r"($|\n|,| \()", t)))
    d["n_keywords"] = len(kws)
    d["text_len"] = len(t)
    d["n_lines"] = t.count("\n") + 1 if t else 0
    # removal-ish
    d["destroy_target"] = int(bool(re.search(r"destroy (up to one )?target (creature|nonland|permanent|artifact or creature)", t)))
    d["exile_target"] = int(bool(re.search(r"exile (up to one )?target (creature|nonland|permanent|artifact or creature)", t)))
    dmg = re.findall(r"deals? (\d+|x) damage to (?:any target|target creature|target attacking|up to one target|each creature|target player or planeswalker|target opponent)", t)
    d["damage_spell"] = max([int(x) if x.isdigit() else 3 for x in dmg] + [0])
    d["minus_pt"] = int(bool(re.search(r"gets? -\d+/-\d+|gets? -x/-x|-1/-1 counter", t)))
    d["fight"] = int(bool(re.search(r"fights?|deals damage equal to its power to", t)))
    d["edict"] = int(bool(re.search(r"(each opponent|target opponent|target player) sacrifices", t)))
    d["counterspell"] = int("counter target" in t)
    d["bounce"] = int(bool(re.search(r"return (up to one )?target (creature|nonland permanent|permanent|artifact or creature)[^.]*to its owner's hand", t)))
    d["tap_down"] = int(bool(re.search(r"tap (up to one )?target|stun counter|doesn't untap", t)))
    d["cant_block"] = int("can't block" in t)
    d["wrath"] = int(bool(re.search(r"(destroy|exile) all creatures|all creatures get -|damage to each creature", t)))
    d["removal_any"] = int(any([d["destroy_target"], d["exile_target"], d["damage_spell"] >= 2, d["minus_pt"],
                                d["fight"], d["edict"]]))
    # card advantage
    draws = re.findall(r"draws? (a|one|two|three|four|x|\d+) cards?", t)
    d["draw_n"] = sum(WORDNUM.get(x, int(x) if x.isdigit() else 1) for x in draws)
    d["loot"] = int(bool(re.search(r"draw a card, then discard a card|discard a card, then draw|discard a card. if you do, draw", t)))
    d["create_token"] = len(re.findall(r"create", t))
    d["etb"] = int(bool(re.search(r"when ~ enters|when .{0,40} enters", t)))
    d["dies_trigger"] = int(bool(re.search(r"when ~ dies|when .{0,30} dies", t)))
    d["attack_trigger"] = int(bool(re.search(r"whenever ~ attacks|whenever .{0,30} attacks", t)))
    d["pump_eot"] = int(bool(re.search(r"gets? \+\d+/\+\d+ until end of turn", t)))
    d["anthem"] = int(bool(re.search(r"creatures you control get \+", t)))
    d["counters_p1"] = int("+1/+1 counter" in t)
    d["lifegain"] = int(bool(re.search(r"gain (\d+|x) life|gain life|lifelink", t)))
    d["landcycling"] = int("landcycling" in t or "cycling" in t)
    d["ramp"] = int(bool(re.search(r"add \{|search your library for (a|up to \w+) basic land", t)))
    d["tutor"] = int(bool(re.search(r"search your library for a (card|creature|planeswalker)", t)))
    d["discard_opp"] = int(bool(re.search(r"(each opponent|target opponent|target player) discards|discards a card", t)))
    d["mill"] = int("mill" in t)
    d["surveil_scry"] = int(bool(re.search(r"surveil|scry", t)))
    d["recursion"] = int(bool(re.search(r"from your graveyard to (your hand|the battlefield)|return this card from your graveyard", t)))
    d["flashback"] = int("flashback" in t)
    d["activated"] = len(re.findall(r"\{[^}]+\}[^:\n]{0,40}: ", t))
    d["equip_cost"] = num((re.findall(r"equip \{(\d+)\}", t) or [np.nan])[0])
    d["conditional"] = len(re.findall(r"\bif\b|as long as|only if", t))
    d["empower"] = sum(int(x) if x.isdigit() else 3 for x in re.findall(r"empower jace (\d+|x)", t))
    d["prepared_text"] = int("prepared" in t)
    d["planeswalker_synergy"] = int("planeswalker" in t and not d["t_planeswalker"])
    d["loyalty"] = num(f0.get("loyalty"))
    d["self_mill_synergy"] = int(bool(re.search(r"threshold|seven or more cards in your graveyard|cards in your graveyard", t)))
    d["legend_synergy"] = int("legendary" in t)
    d["noncreature_synergy"] = int("noncreature spell" in t or "instant or sorcery" in t)
    d["sacrifice_cost"] = int(bool(re.search(r"sacrifice (a|another) (creature|permanent|artifact)", t)))
    d["enters_tapped"] = int("enters tapped" in t and d["t_land"])
    d["dual_land"] = int(d["t_land"] and len(re.findall(r"\{[wubrg]\}", t)) >= 2)
    d["vanilla"] = int(d["t_creature"] and len(t.strip()) == 0)
    return d


def build_set_frame(code):
    """Unique-name frame for a set with features. Returns DataFrame keyed by 17lands-style name."""
    cards = load_set(code)
    rows = []
    seen = set()
    for c in cards:
        nm = c["name"]
        if nm in seen:
            continue
        seen.add(nm)
        f = featurize(c)
        f["name_full"] = nm
        f["name"] = faces(c)[0]["name"] if "card_faces" in c else nm
        f["printed_name"] = c.get("printed_name") or (faces(c)[0].get("printed_name") if "card_faces" in c else None)
        f["collector_number"] = c["collector_number"]
        f["cn_int"] = _cn(c)
        f["type_line"] = c.get("type_line") or faces(c)[0].get("type_line")
        f["mana_cost"] = faces(c)[0].get("mana_cost") or c.get("mana_cost") or ""
        f["color_identity"] = "".join(c.get("color_identity", []))
        f["oracle"] = full_text(c)
        f["rarity_str"] = c["rarity"]
        f["layout"] = c.get("layout")
        f["scry_set"] = c["set"]
        rows.append(f)
    return pd.DataFrame(rows)


def historical_frame():
    frames = []
    idx = None
    for code in HIST:
        st = pd.read_csv(OUT / f"stats_{code}.csv")
        sf = build_set_frame(code)
        by_front = {r["name"]: r for _, r in sf.iterrows()}
        by_full = {r["name_full"]: r for _, r in sf.iterrows()}
        for _, r in sf.iterrows():
            if isinstance(r.get("printed_name"), str):
                by_full.setdefault(r["printed_name"], r)
        rows = []
        for _, s in st.iterrows():
            nm = s["name"]
            r = by_front.get(nm)
            if r is None:
                r = by_full.get(nm)
            in_set = r is not None
            if r is None:
                if idx is None:
                    idx = default_index()
                c = idx.get(nm)
                if c is None:
                    continue
                f = featurize(c)
                f["name_full"] = c["name"]
                f["collector_number"] = c["collector_number"]
                f["cn_int"] = 99999
                f["type_line"] = c.get("type_line")
                f["mana_cost"] = faces(c)[0].get("mana_cost") or c.get("mana_cost") or ""
                f["color_identity"] = "".join(c.get("color_identity", []))
                f["oracle"] = full_text(c)
                f["rarity_str"] = c["rarity"]
                f["layout"] = c.get("layout")
                f["scry_set"] = c["set"]
                r = pd.Series(f)
            row = dict(r)
            row.update({k: s[k] for k in s.index if k != "name"})
            row["name"] = nm
            row["in_main_set"] = int(in_set)
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    h = historical_frame()
    h.to_csv(OUT / "historical_features.csv", index=False)
    print(h.shape)
    print(h.groupby("set").agg(n=("name", "size"), in_main=("in_main_set", "sum")))
    fra = pd.concat([build_set_frame("fra"), build_set_frame("fra-special-guests")], ignore_index=True)
    fra["is_guest"] = (fra["scry_set"] != "fra").astype(int)
    fra = fra[(fra["is_guest"] == 1) | (fra["cn_int"] <= 280)]
    fra.to_csv(OUT / "fra_features.csv", index=False)
    print(fra.shape)
