"""Feature engineering shared by the historical model and the TRK scoring step."""
import re
import numpy as np
import pandas as pd

KEYWORDS = ["Flying", "Trample", "Menace", "Lifelink", "Deathtouch", "Haste", "First strike", "Double strike",
            "Vigilance", "Reach", "Ward", "Flash", "Hexproof", "Indestructible", "Defender", "Landfall",
            "Kicker", "Cycling", "Investigate", "Surveil", "Scry", "Mill", "Treasure", "Food", "Station", "Equip",
            "Enchant", "Fight", "Explore", "Landcycling", "Prowess", "Exhaust", "Offspring", "Flashback",
            "Convoke", "Affinity", "Changeling", "Warp", "Void", "Mobilize", "Harmonize", "Job select", "Tiered"]

TEXT_FLAGS = {
    "destroy_creature": r"destroy target (creature|nonartifact creature|artifact or creature|creature or planeswalker|creature or spacecraft|creature or vehicle|creature or enchantment|creature or artifact|nonland permanent|permanent|attacking creature|tapped creature|creature with|creature an opponent)",
    "exile_creature": r"exile target (creature|nonland permanent|permanent|artifact or creature|creature or spacecraft|creature or enchantment|creature or planeswalker|creature an opponent|nontoken)",
    "exile_until": r"exile target .* until",
    "dmg_any": r"deals? (\d+|x|twice x|damage equal|that much) damage to (any target|target creature or planeswalker|target creature or player)",
    "dmg_creature": r"deals? (\d+|x|twice x|damage equal|that much) damage to (target creature|each creature|target attacking|target blocking|target creature or spacecraft|up to one target creature|target attacking or blocking)",
    "dmg_each_opp": r"damage to each opponent",
    "minus_x": r"gets? -\d+/-\d+|gets -x/-x",
    "all_minus": r"all (other )?creatures get -",
    "fight": r"fights?\b",
    "bite": r"deals? damage equal to (its|their) power to target creature",
    "counter_spell": r"counter target (spell|noncreature spell|creature spell|activated|instant)",
    "bounce_opp": r"return target (creature|nonland permanent|permanent|artifact or creature) (an opponent controls )?to its owner's hand",
    "tap_stun": r"stun counter",
    "tap_target": r"tap target (creature|permanent|artifact)",
    "cant_block": r"can't block",
    "draw_card": r"draw a card|draws a card",
    "draw_two": r"draw two cards|draw (three|x) cards|draws two cards",
    "loot": r"draw a card, then discard|draw two cards, then discard|discard a card, then draw|draw .* then discard",
    "card_adv_ability": r"\{t\}: draw|: draw a card|: you draw",
    "impulse": r"exile the top (card|two cards|three cards|x cards) of your library.*(you may play|may cast)",
    "look_top": r"look at the top (\w+) cards? of your library",
    "tutor": r"search your library for",
    "ramp_land": r"land card.*put (it|them|that card) onto the battlefield|onto the battlefield tapped",
    "mana_ability": r"\{t\}: add|add \{[wubrgc]\}|add (one|two) mana|add \{g\}\{g\}",
    "token_creature": r"create (a|an|two|three|x|that many|ten) [^.]*creature token",
    "token_small": r"create (a|an|two|three|x) 1/1",
    "token_big": r"create (a|an|two|three|x) [3-9]/[3-9]|create (a|an|two) 2/2",
    "plus_counter": r"\+1/\+1 counter",
    "counters_each": r"\+1/\+1 counter on each creature",
    "anthem": r"creatures you control get \+\d/\+\d(?! until)|other creatures you control get \+",
    "pump_eot": r"gets? \+\d+/\+\d+ until end of turn",
    "team_pump_eot": r"creatures you control get \+\d/\+\d until end of turn",
    "etb": r"when (this creature|this artifact|this enchantment|this spacecraft|this permanent|[\w'\-,. ]+) enters",
    "attack_trigger": r"whenever (this creature|[\w'\-,. ]+) attacks",
    "dies_trigger": r"when (this creature|[\w'\-,. ]+) dies",
    "each_opp_lose": r"each opponent loses (\d+|x|life)",
    "gain_life": r"gain (\d+|x|life equal|that much) life",
    "sacrifice_cost": r"sacrifice (a|an|another) (creature|artifact|artifact or creature|creature or artifact|permanent|nonland)",
    "opp_sacrifice": r"(each opponent|target opponent|each player) sacrifices",
    "discard_opp": r"(each opponent|target opponent|that player) discards",
    "reanimate": r"return target creature card from your graveyard to the battlefield|put (it|that card) onto the battlefield",
    "regrowth": r"return target .* card from your graveyard to your hand",
    "cast_from_gy": r"cast (this card|it|.*) from your graveyard|beam me up|flashback|you may cast .* from your graveyard",
    "cast_from_exile": r"you may (cast|play) (it|that card|those cards|the chosen|them)",
    "copy": r"create a token that's a copy|copy (it|that|target)",
    "wrath": r"destroy all|each creature|all creatures|exile all",
    "hexproof_grant": r"gains? hexproof|have hexproof|gain hexproof and indestructible",
    "protection_trick": r"gains? indestructible|prevent all damage",
    "unblockable": r"can't be blocked(?! by)|can't be blocked this turn",
    "evasion_cond": r"can't be blocked by",
    "extra_turn": r"extra turn",
    "win_game": r"you win the game",
    "lose_game": r"you lose the game",
    "you_lose_life": r"you lose (\d+|x) life|lose 1 life\b",
    "additional_cost": r"as an additional cost",
    "untap": r"untap (target|it|this|that|another)",
    "enters_tapped": r"enters (the battlefield )?tapped",
    "aura_debuff": r"enchanted (creature|permanent) (gets -|can't|doesn't untap)",
    "aura_buff": r"enchanted creature gets \+",
    "equipped_buff": r"equipped creature gets \+",
    "pay_life": r"pay \d life|pay 2 life",
    "cost_reduction": r"costs? \{?\d?\}? ?less to cast|costs \{x\} less",
    "x_cost": r"\{x\}",
    "choose_one": r"choose one|choose two|choose one or more",
    "opponent_choice": r"(target opponent|that player|its controller) may|opponent (may|chooses)",
    "symmetric": r"each player (draws|loses|sacrifices)|each opponent and you",
    "legendary": r"^legendary",
    "activated_mana": r"\{\d\}[^:]*: ",
    "lord_tribal": r"(other )?\w+s you control get \+1/\+1|\w+ you control have",
    "conditional_big": r"if you control (five|six|seven)|seven or more|eight or more|as long as you control (five|six|seven)",
    "spacecraft_station": r"station",
    "artifact_matters": r"artifact creatures? you control|artifact spell|noncreature spell|nontoken artifact",
    "doubling": r"twice that many|an additional time|triggers an additional",
    "alt_cost_cheap": r"\{1\}, \{t\}|\{2\}, \{t\}",
}

def parse_pt(v):
    try:
        return float(v)
    except Exception:
        if isinstance(v, str) and "*" in v:
            return np.nan
        return np.nan

def pip_count(mana):
    if not isinstance(mana, str):
        return 0, 0
    pips = re.findall(r"\{([^}]+)\}", mana)
    colored = sum(1 for p in pips if any(ch in p for ch in "WUBRG"))
    hybrid = sum(1 for p in pips if "/" in p and any(ch in p for ch in "WUBRG"))
    return colored, hybrid

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    text = df["oracle_text"].fillna("").str.lower()
    tl = df["type_line"].fillna("")
    for r in ["common", "uncommon", "rare", "mythic"]:
        f[f"rar_{r}"] = (df["rarity"] == r).astype(int)
    f["cmc"] = df["cmc"].fillna(0).astype(float)
    f["cmc_sq"] = f["cmc"] ** 2
    colors = df["colors"].fillna("").astype(str)
    f["ncolors"] = colors.str.len()
    for c in "WUBRG":
        f[f"col_{c}"] = colors.str.contains(c).astype(int)
    f["colorless"] = (f["ncolors"] == 0).astype(int)
    f["multicolor"] = (f["ncolors"] >= 2).astype(int)
    pc = df["mana_cost"].map(pip_count)
    f["colored_pips"] = [p[0] for p in pc]
    f["hybrid_pips"] = [p[1] for p in pc]
    f["heavy_pips"] = (f["colored_pips"] >= 2).astype(int)
    for t in ["Creature", "Instant", "Sorcery", "Enchantment", "Artifact", "Land", "Aura", "Saga", "Equipment",
              "Spacecraft", "Vehicle", "Legendary", "Planeswalker", "Basic", "Token"]:
        f[f"type_{t}"] = tl.str.contains(t).astype(int)
    f["is_dfc"] = df["layout"].fillna("").isin(["transform", "modal_dfc", "adventure", "split"]).astype(int)
    p = df["power"].map(parse_pt); t = df["toughness"].map(parse_pt)
    is_cre = f["type_Creature"] == 1
    f["power"] = p.fillna(0); f["toughness"] = t.fillna(0)
    f["pt_sum"] = f["power"] + f["toughness"]
    f["pt_per_cmc"] = np.where(f["cmc"] > 0, f["pt_sum"] / f["cmc"], 0) * (is_cre | (f["type_Spacecraft"] == 1) | (f["type_Vehicle"] == 1))
    f["pt_minus_cmc"] = (f["pt_sum"] - 2 * f["cmc"]) * is_cre
    f["power_ge_cmc"] = ((f["power"] >= f["cmc"]) & is_cre).astype(int)
    f["star_pt"] = df["power"].astype(str).str.contains(r"\*").astype(int)
    kws = df["keywords"].fillna("").astype(str)
    for k in KEYWORDS:
        f[f"kw_{k.replace(' ', '_')}"] = kws.str.contains(re.escape(k), case=False).astype(int) | text.str.contains(r"\b" + re.escape(k.lower()) + r"\b").astype(int)
    for name, pat in TEXT_FLAGS.items():
        f[f"tx_{name}"] = text.str.contains(pat, regex=True).astype(int)
    f["tx_legendary"] = tl.str.startswith("Legendary").astype(int)
    f["text_len"] = text.str.len()
    f["n_lines"] = text.str.count("\n") + (text.str.len() > 0)
    f["n_abilities_activated"] = text.str.count(r"\n\{|^\{")
    f["removal_any"] = (f[["tx_destroy_creature", "tx_exile_creature", "tx_dmg_any", "tx_dmg_creature", "tx_minus_x", "tx_fight", "tx_bite"]].sum(axis=1) > 0).astype(int)
    f["removal_cheap"] = f["removal_any"] * (f["cmc"] <= 3)
    f["card_adv"] = (f[["tx_draw_card", "tx_draw_two", "tx_impulse", "tx_card_adv_ability"]].sum(axis=1) > 0).astype(int)
    f["creature_removal_etb"] = f["type_Creature"] * f["removal_any"]
    f["creature_card_adv"] = f["type_Creature"] * f["card_adv"]
    f["cheap_creature"] = ((f["cmc"] <= 2) & is_cre).astype(int)
    f["big_creature"] = ((f["cmc"] >= 6) & is_cre).astype(int)
    f["instant_speed"] = f["type_Instant"] | f["kw_Flash"]
    f["evasive_creature"] = is_cre.astype(int) * (f["kw_Flying"] | f["kw_Menace"] | f["tx_unblockable"] | f["kw_Trample"])
    f["flying_power"] = f["kw_Flying"] * f["power"]
    f["nonbasic_land"] = (f["type_Land"] == 1) & (f["type_Basic"] == 0)
    f["nonbasic_land"] = f["nonbasic_land"].astype(int)
    return f.fillna(0)
