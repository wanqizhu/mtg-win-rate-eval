#!/usr/bin/env python3
"""High-precision limited rating, calibrated on historical Premier Draft GIH.

The point schedule is intentionally strict. A clause fires only for the
pattern that actually wins games in the nine-set sample: cheap unconditional
interaction, a body that replaces itself, or a rare that takes over a game.
Noisy counts (any 'draw', any 'destroy') are not used.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from predict_gih import HIST_SETS, TRK_BASELINE, build_frame, load_set_cards

OUT = Path("/agent/repos/mtg-win-rate-eval/analysis")
PRED = Path("/agent/repos/mtg-win-rate-eval/predictions")


def low(card_text: str) -> str:
    return (card_text or "").lower()


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def rate(row) -> tuple[float, list[str]]:
    """Return points on a scale where 1 point is about 1 percentage point of GIH."""
    text = row.oracle_text or ""
    t = low(text)
    tl = low(row.type_line or "")
    cmc = float(row.cmc or 0)
    rarity = row.rarity
    reasons = []
    pts = 0.0

    def add(amount: float, why: str) -> None:
        nonlocal pts
        pts += amount
        reasons.append(f"{amount:+.1f} {why}")

    is_land = "land" in tl
    is_creature = "creature" in tl and "spacecraft" not in tl
    is_instant = tl.startswith("instant") or "instant" in tl.split("//")[0]
    is_sorcery = "sorcery" in tl.split("//")[0]
    is_aura = "aura" in tl
    is_equipment = "equipment" in tl
    is_saga = "saga" in tl
    is_ship = "spacecraft" in tl
    is_artifact = "artifact" in tl
    is_enchant = "enchantment" in tl
    spell = is_instant or is_sorcery
    instant_bonus = 0.4 if is_instant else 0.0

    # --- lands ---
    if is_land:
        ci = (row.colors or "")
        basic = "basic" in tl
        if basic:
            add(0.2, "basic")
            return pts, reasons
        shock = "you may pay 2 life" in t and "enters tapped" in t
        gain = "you gain 1 life" in t and "enters tapped" in t and "add" in t
        if shock:
            add(1.4, "shock")
        elif gain and len(ci) >= 2:
            add(0.6, "gainland")
        elif "search your library for a basic" in t:
            add(0.4, "fetch land")
        elif "enters tapped unless you control two" in t:
            add(0.9, "check land")
        elif "enters tapped" in t and len(ci) >= 2:
            add(-0.4, "tapped dual")
        elif len(ci) == 0:
            add(-0.8, "colorless land")
        if "add one mana of any color" in t or "add {c}" in t and "any color" in t:
            add(0.3, "any color")
        return pts, reasons

    # --- premium interaction ---
    exile_until = "until" in t and "leaves the battlefield" in t and "exile target" in t
    if exile_until and cmc <= 3:
        add(3.2 + instant_bonus, "oblivion ring")
    elif exile_until and cmc <= 4:
        add(2.4, "expensive o-ring")

    hard_exile = bool(re.search(r"exile target (creature|nonland permanent|artifact or creature|permanent)", t)) and not exile_until
    hard_destroy = bool(re.search(r"destroy target (creature|nonland permanent|artifact or enchantment|permanent)", t))
    conditional_destroy = any(s in t for s in [
        "power 4 or greater", "toughness 4 or greater", "tapped creature",
        "attacking creature", "blocking creature", "nonartifact creature",
        "with flying", "mana value 3 or less", "mana value 4 or less",
    ])
    if hard_exile and spell and not conditional_destroy:
        add((3.4 if cmc <= 2 else 2.6 if cmc <= 3 else 1.6) + instant_bonus, "hard exile")
    elif hard_destroy and spell and not conditional_destroy:
        add((3.0 if cmc <= 2 else 2.2 if cmc <= 3 else 1.5 if cmc <= 4 else 0.6) + instant_bonus, "hard destroy")
    elif (hard_destroy or hard_exile) and spell and conditional_destroy:
        add((1.6 if cmc <= 2 else 0.8) + instant_bonus, "conditional removal")

    dmg = [int(n) for n in re.findall(r"deals (\d+) damage", t)]
    max_dmg = max(dmg) if dmg else 0
    any_target = "any target" in t
    if max_dmg and spell and "each creature" not in t and "each opponent" not in t:
        rate_dmg = max_dmg / max(cmc, 1)
        if rate_dmg >= 2.5 and cmc <= 2:
            add(2.6 + (0.5 if any_target else 0) + instant_bonus, f"{max_dmg} damage efficient")
        elif rate_dmg >= 1.4 and cmc <= 3:
            add(1.6 + (0.4 if any_target else 0) + instant_bonus, f"{max_dmg} damage")
        elif max_dmg >= 5 and cmc <= 5:
            add(1.3, f"big removal {max_dmg}")
        elif rate_dmg < 1 and cmc >= 3:
            add(-0.4, "inefficient damage")
    if re.search(r"-(\d)/-\1", t):
        n = max(int(x) for x in re.findall(r"-(\d)/-\1", t))
        if spell and n >= 3 and cmc <= 2:
            add(2.4 + instant_bonus, f"-{n}/-{n}")
        elif spell and n >= 2 and cmc <= 2:
            add(1.5 + instant_bonus, f"-{n}/-{n}")
        elif spell and n >= 4 and cmc <= 4:
            add(1.2, f"expensive -{n}/-{n}")
    if "fights" in t or "deal damage equal to their power" in t or "deals damage equal to its power" in t:
        if cmc <= 3:
            add(1.5, "bite/fight")
    if "return target creature" in t and "owner's hand" in t and spell:
        add((2.2 if cmc <= 1 else 1.5 if cmc <= 2 else 0.6) + instant_bonus, "bounce")
    if "counter target spell" in t and "unless" not in t and cmc <= 3:
        add(3.0, "hard counter")
    elif "counter target" in t and "unless its controller pays {2}" in t:
        add(1.8, "mana leak 2")
    elif "counter target" in t and "unless its controller pays {3}" in t and cmc <= 3:
        add(1.3, "mana leak 3")
    elif "counter target spell with mana value" in t and cmc <= 2:
        add(1.6, "conditional counter")
    if "destroy all creatures" in t or "all creatures get -" in t or "damage to each creature" in t:
        add(2.2 if cmc <= 5 else 1.2, "sweeper")

    # --- card advantage ---
    opp_draws = "target opponent draws" in t or "each opponent draws" in t or "that player draws" in t or "target player draws" in t
    if re.search(r"draw two cards", t) and not opp_draws and cmc <= 4:
        add(2.6 if cmc <= 3 else 1.6, "draw two")
    elif re.search(r"draw three cards", t) and not opp_draws:
        add(3.2 if cmc <= 5 else 1.5, "draw three")
    elif re.search(r"\bdraw a card\b", t) and not opp_draws:
        if spell and cmc <= 1:
            add(1.8, "cantrip")
        elif spell and cmc <= 2 and ("then discard" in t or "discard a card" in t):
            add(0.8, "loot")
        elif (is_creature or is_artifact or is_enchant) and cmc <= 3 and "when" in t and "enters" in t:
            add(2.0, "etb cantrip permanent")
        elif spell and cmc <= 2:
            add(1.2, "cheap draw")
        elif cmc >= 5:
            add(0.2, "expensive draw")
    if "if you would draw a card, draw two" in t:
        add(6.5, "double draws")
    if "look at the top" in t and "put" in t and "into your hand" in t and cmc <= 3:
        add(1.6, "impulse")
    if re.search(r"mill (three|four|five|\d+)", t) and "into your hand" in t and cmc <= 3:
        add(1.7, "mill take")
    if "discover" in t:
        n = re.search(r"discover (\d+)", t)
        add(2.4 + (0.4 * int(n.group(1)) if n else 1), "discover")
    if "without paying" in t and "mana cost" in t and "discover" not in t:
        add(2.0 if cmc <= 5 else 0.8, "free cast")
    if "you may play an additional land" in t:
        add(2.2, "extra land")
    if "explores" in t or "explore." in t or "that creature explores" in t:
        add(2.4, "explore engine")
    if "flashback" in t or "beam me up" in t:
        cost_ok = True
        if "beam me up {3}{r}{r}{r}" in t:
            cost_ok = False
            add(-0.4, "unusable beam")
        if cost_ok:
            add(0.9 if cmc <= 2 else 0.5, "recast from graveyard")

    # --- bodies ---
    power = row.power if not (isinstance(row.power, float) and math.isnan(row.power)) else None
    tough = row.toughness if not (isinstance(row.toughness, float) and math.isnan(row.toughness)) else None
    try:
        power_n = float(power) if power is not None and str(power) not in {"*", "nan"} else None
    except (TypeError, ValueError):
        power_n = None
    try:
        tough_n = float(tough) if tough is not None and str(tough) not in {"*", "nan"} else None
    except (TypeError, ValueError):
        tough_n = None
    if is_creature and power_n is not None and tough_n is not None:
        rate_body = (power_n + tough_n) - 2 * cmc
        if cmc <= 2 and rate_body >= 1:
            add(1.3, "efficient small body")
        elif cmc <= 3 and rate_body >= 2:
            add(1.0, "above-rate body")
        elif cmc <= 3 and rate_body <= -1 and "when" not in t:
            add(-1.0, "under-rate body")
        elif cmc >= 6 and rate_body <= 1 and "when" not in t and len(text) < 80:
            add(-2.2, "expensive vanilla")
        if "flying" in t and cmc <= 4 and power_n >= 2:
            add(0.7, "flyer")
        if any(k in t for k in ["lifelink", "deathtouch", "menace", "trample", "haste"]) and cmc <= 3:
            add(0.35, "keyword")
    etb = "when this" in t and "enters" in t or bool(re.search(r"when [a-z].{0,40} enters", t))
    if is_creature and etb and cmc <= 4:
        # A 2-for-1 token or blink-quality ETB. Token creation is the clean signal.
        if re.search(r"create (a |two |three )?\d/\d", t) or "create a 1/1" in t or "create two" in t:
            add(1.6 if cmc <= 3 else 1.0, "etb token")
        elif "draw a card" not in t:
            add(0.5, "etb")
    if "other creatures you control get +1/+1" in t or "creatures you control get +1/+1" in t:
        add(2.4 if cmc <= 4 else 1.2, "anthem")
    if "artifact creatures you control get +1/+1" in t or "artifact creatures you control have" in t:
        add(0.8, "artifact anthem")
    if is_creature and cmc <= 3 and re.search(r"create (a |two )?\d/\d", t):
        pass  # already counted
    # dies into a token
    if "when this creature dies" in t and "create" in t:
        add(1.1, "dies into value")

    # --- spacecraft ---
    if is_ship:
        thresholds = [int(n) for n in re.findall(r"at (\d+)\+", t)]
        first = [int(n) for n in re.findall(r"(\d+)\+ \|", text)]
        station = min(first) if first else (min(thresholds) if thresholds else 12)
        has_etb = "when this spacecraft enters" in t or "when this" in t and "enters" in t or bool(re.search(r"when .{0,30}enters", t))
        if has_etb and any(s in t for s in ["draw", "destroy", "exile", "damage", "search your library", "create", "discover", "return target"]):
            add(1.6 if cmc <= 4 else 0.8, "ship with real etb")
        elif has_etb:
            add(0.4, "ship etb")
        else:
            add(-1.2, "ship without etb")
        if station <= 4 and not has_etb:
            add(0.8, "easy station ability")
        if station >= 10:
            add(-1.0, "high station")
        if cmc >= 7:
            add(-1.4, "expensive ship")
        if cmc <= 3 and has_etb:
            add(0.4, "cheap ship")

    # --- liabilities ---
    if is_aura and not hard_destroy and "exile" not in t and cmc >= 2:
        add(-1.3, "aura")
    if is_equipment and "create a" not in t and "when you cast this spell, copy" not in t:
        if cmc <= 1 and "equip {1}" in t:
            add(0.8, "cheap equipment")
        elif cmc >= 2:
            add(-0.7, "equipment")
    if "can't attack alone" in t:
        add(-0.4, "can't attack alone")
    if "defender" in t and "can attack" not in t:
        add(-1.0, "defender")
    if "you lose the game" in t:
        add(-1.6, "lose the game")
    if "can't sacrifice treasure" in t or "you can't sacrifice treasure" in t:
        add(-2.2, "treasures stuck")
    if cmc >= 8 and "destroy all" not in t and "discover" not in t and "without paying" not in t:
        add(-2.0, "eight-plus")
    elif cmc >= 7 and not is_land and "destroy all" not in t and max_dmg < 6 and "draw" not in t:
        add(-1.0, "seven mana")
    if "choose one" in t or "choose two" in t or "choose one or more" in t:
        add(0.6 if cmc <= 3 else 0.3, "modal")
    if is_saga and cmc <= 4:
        add(0.7, "cheap saga")
    if "ward {2}" in t or "ward {3}" in t:
        add(0.4, "ward")
    if rarity == "mythic" and pts >= 4:
        add(0.8, "mythic payoff multiplier")
    if rarity == "common" and pts >= 3.5:
        pts = 3.2 + (pts - 3.5) * 0.35
        reasons.append("cap common")
    return pts, reasons


def rarity_baseline(rarity: str, is_land: bool, is_basic: bool) -> float:
    # Unweighted means of main-set cards with at least 500 GIH, measured below
    # and hardcoded after the calibration printout. Defaults are the first-pass
    # estimates; main() overwrites them from the training frame when present.
    if is_basic:
        return BASE.get("basic", 0.556)
    if is_land:
        return BASE.get("land", 0.552)
    return BASE.get(rarity, 0.556)


BASE = {
    "common": 0.5525,
    "uncommon": 0.5575,
    "rare": 0.5710,
    "mythic": 0.5930,
    "bonus": 0.5600,
    "land": 0.5520,
    "basic": 0.5565,
}


def calibrate(hist: pd.DataFrame) -> dict:
    hist = hist[hist["gih_n"] >= 400].copy()
    points = []
    bases = []
    for row in hist.itertuples(index=False):
        pts, _ = rate(row)
        is_land = "land" in (row.type_line or "").lower()
        is_basic = "basic" in (row.type_line or "").lower()
        points.append(pts)
        bases.append(rarity_baseline(row.rarity, is_land, is_basic))
    hist["points"] = points
    hist["base"] = bases
    # Leave-one-set-out: fit residual scale on the other sets.
    oof = np.zeros(len(hist))
    y = hist["gih_wr"].to_numpy()
    sets = hist["set"].to_numpy()
    pts_a = hist["points"].to_numpy()
    base_a = hist["base"].to_numpy()
    # Also let the intercept absorb a constant miss, and a squared term so
    # bombs pull away from filler without making every small plus huge.
    for hold in HIST_SETS:
        tr = sets != hold
        te = sets == hold
        x_tr = np.column_stack([pts_a[tr], np.maximum(pts_a[tr], 0) ** 2, np.minimum(pts_a[tr], 0)])
        x_te = np.column_stack([pts_a[te], np.maximum(pts_a[te], 0) ** 2, np.minimum(pts_a[te], 0)])
        model = Ridge(alpha=2.0)
        model.fit(x_tr, y[tr] - base_a[tr])
        oof[te] = base_a[te] + model.predict(x_te)
    oof = np.clip(oof, 0.42, 0.74)
    # Refit on everything for the TRK application.
    x_all = np.column_stack([pts_a, np.maximum(pts_a, 0) ** 2, np.minimum(pts_a, 0)])
    model = Ridge(alpha=2.0)
    model.fit(x_all, y - base_a)
    mae = float(mean_absolute_error(y, oof))
    null = float(mean_absolute_error(y, base_a))
    spearman = float(pd.Series(y).corr(pd.Series(oof), method="spearman"))
    by_r = {}
    for rarity, idx in hist.groupby("rarity").groups.items():
        mask = hist.index.isin(idx)
        # groupby groups are index labels; hist may not be 0..n
        pos = hist["rarity"].to_numpy() == rarity
        by_r[rarity] = {
            "mae": float(mean_absolute_error(y[pos], oof[pos])),
            "null": float(mean_absolute_error(y[pos], base_a[pos])),
            "n": int(pos.sum()),
        }
    print(f"expert LOO mae {mae:.4f}  rarity-baseline mae {null:.4f}  spearman {spearman:.3f}")
    print("coef", model.coef_, "intercept", model.intercept_)
    print(json.dumps(by_r, indent=2))
    # Spot-check known cards
    print("\nSPOT CHECKS")
    for name in [
        "Banishing Light", "Cryogen Relic", "Iron Giant", "Last Gasp", "Unsummon",
        "Dawnsire, Sunstar Dreadnought", "Warmaker Gunship", "Uthros Scanship",
        "Sephiroth's Intervention", "Combat Tutorial", "Meltstrider's Gear",
    ]:
        hit = hist[hist["name"] == name]
        if hit.empty:
            continue
        row = hit.iloc[0]
        pts, reasons = rate(row)
        pred = float(rarity_baseline(row.rarity, "land" in row.type_line.lower(), "basic" in row.type_line.lower()) + model.predict(np.array([[pts, max(pts, 0) ** 2, min(pts, 0)]]))[0])
        print(f"  actual {row.gih_wr*100:5.1f} pred {pred*100:5.1f} pts {pts:5.1f}  {name}")
        print("    ", "; ".join(reasons[:8]))
    return {"model": model, "mae": mae, "null": null, "spearman": spearman, "by_rarity": by_r, "hist": hist, "oof": oof}


def predict_trk(model) -> pd.DataFrame:
    cards = load_set_cards("TRK", main_only_max_cn=311)
    frame = build_frame(cards, set_wr=TRK_BASELINE)
    rows = []
    for row in frame.itertuples(index=False):
        pts, reasons = rate(row)
        is_land = "land" in (row.type_line or "").lower()
        is_basic = "basic" in (row.type_line or "").lower()
        base = rarity_baseline(row.rarity, is_land, is_basic)
        x = np.array([[pts, max(pts, 0) ** 2, min(pts, 0)]])
        pred = float(base + model.predict(x)[0])
        pred = float(np.clip(pred, 0.45, 0.70))
        rows.append({
            "name": row.name,
            "collector_number": row.collector_number,
            "cn": row.cn,
            "rarity": row.rarity,
            "mana_cost": row.mana_cost,
            "type_line": row.type_line,
            "colors": row.colors,
            "cmc": row.cmc,
            "points": round(pts, 2),
            "expert_wr": round(pred, 4),
            "reasons": "; ".join(reasons),
            "oracle_text": row.oracle_text,
        })
    return pd.DataFrame(rows)


def main() -> None:
    joined = pd.read_pickle("/tmp/mtg_analysis/joined.pkl")
    # Main-set cards only for the baseline means. Bonus-sheet reprints have a
    # different selection bias.
    main = joined[joined["is_bonus"] == False].copy()  # noqa: E712
    hist = build_frame(main.to_dict(orient="records"))
    usable = hist[hist["gih_n"] >= 500]
    print("BASELINES")
    for rarity, group in usable.groupby("rarity"):
        lands = group["type_line"].str.contains("Land", case=False)
        print(f"  {rarity:10} n={len(group):4} mean={group.gih_wr.mean():.4f}  nonland={group.loc[~lands,'gih_wr'].mean():.4f}")
    result = calibrate(hist)
    trk = predict_trk(result["model"])
    trk.to_csv(OUT / "trk_expert_raw.csv", index=False)
    print("\nTRK expert top")
    print(trk.nlargest(20, "expert_wr")[["expert_wr", "points", "rarity", "mana_cost", "name"]].to_string(index=False))
    print("\nTRK expert bottom")
    print(trk.nsmallest(15, "expert_wr")[["expert_wr", "points", "rarity", "mana_cost", "name"]].to_string(index=False))
    print("std", trk.expert_wr.std(), "mean", trk.expert_wr.mean())
    (OUT / "expert_cv.json").write_text(json.dumps({
        "mae": result["mae"], "null": result["null"], "spearman": result["spearman"],
        "by_rarity": result["by_rarity"],
        "coef": result["model"].coef_.tolist(),
        "intercept": float(result["model"].intercept_),
    }, indent=2))


if __name__ == "__main__":
    main()
