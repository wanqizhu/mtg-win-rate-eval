"""Shared feature engineering for MTG GIH win-rate prediction."""
import re, math
import numpy as np

KEYWORDS = ["flying","vigilance","trample","haste","menace","lifelink","deathtouch","first strike",
 "double strike","flash","reach","ward","hexproof","indestructible","defender","prowler","prowess"]

def parse_pt(v):
    if v is None: return (np.nan, 0)
    s=str(v)
    if "*" in s or "X" in s: return (np.nan, 1)
    try: return (float(s), 0)
    except Exception: return (np.nan, 0)

NUM_WORDS={"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"x":2}

def structured_features(rec):
    t=(rec.get("oracle_text") or "").lower()
    tl=rec.get("type_line") or ""
    mc=rec.get("mana_cost") or ""
    cmc=float(rec.get("cmc") or 0)
    power, pvar = parse_pt(rec.get("power")); tough, tvar = parse_pt(rec.get("toughness"))
    is_creature = "creature" in tl.lower()
    f={}
    f["cmc"]=cmc; f["cmc2"]=cmc*cmc
    f["power"]= 0 if np.isnan(power) else power
    f["tough"]= 0 if np.isnan(tough) else tough
    f["pt_missing"]= 1 if np.isnan(power) else 0
    f["pt_var"]= pvar
    f["power_per_cmc"]= f["power"]/(cmc+1)
    f["tough_per_cmc"]= f["tough"]/(cmc+1)
    f["pt_sum"]= f["power"]+f["tough"]
    f["is_creature"]=int(is_creature)
    f["is_land"]=int("land" in tl.lower())
    f["is_instant"]=int("instant" in tl.lower()); f["is_sorcery"]=int("sorcery" in tl.lower())
    f["is_artifact"]=int("artifact" in tl.lower()); f["is_enchantment"]=int("enchantment" in tl.lower())
    f["is_planeswalker"]=int("planeswalker" in tl.lower())
    f["is_saga"]=int("saga" in tl.lower()); f["is_equipment"]=int("equipment" in tl.lower())
    f["is_aura"]=int("aura" in tl.lower()); f["is_spacecraft"]=int("spacecraft" in tl.lower())
    f["is_vehicle"]=int("vehicle" in tl.lower()); f["is_battle"]=int("battle" in tl.lower())
    f["is_legendary"]=int("legendary" in tl.lower())
    f["n_colors"]=len(rec.get("colors") or "")
    f["colorless"]=int(len(rec.get("colors") or "")==0)
    f["gold"]=int(len(rec.get("colors") or "")>=2)
    # colored pips / devotion-ish & color requirements (hybrid counts partial)
    for col in "WUBRG":
        f[f"pips_{col}"]= mc.count("{"+col+"}")
    f["generic_cost"]= sum(int(x) for x in re.findall(r"\{(\d+)\}", mc))
    f["has_x"]= int("{x}" in mc.lower())
    f["hybrid"]= int("/" in mc)
    rar={"common":0,"uncommon":1,"rare":2,"mythic":3}.get(rec.get("rarity"),1)
    f["rarity"]=rar
    f["is_common"]=int(rar==0); f["is_uncommon"]=int(rar==1); f["is_rare"]=int(rar==2); f["is_mythic"]=int(rar==3)
    for kw in KEYWORDS:
        f[f"kw_{kw.replace(' ','_')}"]= int(kw in t or kw in (rec.get("keywords") or "").lower())
    f["text_len"]= len(t)/100.0
    f["n_lines"]= t.count("\n")+1
    # --- hand-crafted semantic features ---
    def has(pat): return int(bool(re.search(pat,t)))
    def count(pat): return len(re.findall(pat,t))
    f["destroy_creature"]= has(r"destroy target (creature|.*creature)")
    f["destroy_any"]= has(r"destroy target")
    f["exile_perm"]= has(r"exile target (creature|permanent|.*nonland|.*artifact)")
    f["damage_to_creature"]= has(r"deals? \d+ damage to (target creature|any target|target attacking|each creature)")
    dmg = re.findall(r"deals? (\d+) damage", t)
    f["max_damage"]= max([int(x) for x in dmg], default=0)
    f["draw_cards"]= has(r"draw (a|two|three|four|\d+|x) cards?")
    draw_n = 0
    for m in re.finditer(r"draw (\w+) cards?", t):
        draw_n += NUM_WORDS.get(m.group(1), 1 if m.group(1)=="a" else 0)
    f["draw_n"]=draw_n
    f["cantrip"]= int(draw_n>=1 and cmc<=3)
    f["create_token"]= has(r"create .*token")
    tok = re.findall(r"create (?:a|an|one|two|three|four|\d+|x) .*?(\d+)/(\d+)[^.]*token", t)
    f["token_pt"]= sum(int(a)+int(b) for a,b in tok)
    f["n_tokens_text"]= count(r"token")
    f["plus_counters"]= has(r"put .*\+1/\+1 counter")
    f["minus_effect"]= has(r"gets? -\d+/-\d+|get -\d+/-\d+|all creatures get -")
    f["bounce"]= has(r"return target .* to its owner's hand|return target .* to their owner's hand")
    f["counter_spell"]= has(r"counter target .*spell")
    f["fight"]= has(r"fight")
    f["search_land"]= has(r"search your library for a .*land")
    f["ramp_extra_land"]= has(r"additional land|put .*land card .* onto the battlefield")
    f["life_gain"]= has(r"gain .* life")
    f["life_drain"]= has(r"(each )?opponent loses .* life")
    f["opp_discard"]= has(r"(opponent|player) discards|target opponent .* discard|reveals their hand. you choose")
    f["mill"]= has(r"mill")
    f["surveil_scry"]= count(r"surveil|scry")
    f["sacrifice_outlet"]= has(r"sacrifice (a|another|any number)")
    f["etb_trigger"]= has(r"when .* enters|whenever .* enters")
    f["attack_trigger"]= has(r"whenever .* attacks|when .* attacks")
    f["death_trigger"]= has(r"when .* dies|whenever .* dies")
    f["cast_trigger"]= has(r"whenever you cast")
    f["upkeep_trigger"]= has(r"at the beginning of your upkeep")
    f["endstep_trigger"]= has(r"at the beginning of .* end step")
    f["graveyard_recursion"]= has(r"from your graveyard to (the battlefield|your hand)|cast .* from your graveyard|return .* from your graveyard")
    f["mana_ability"]= has(r"\{t\}: add")
    f["tap_down"]= has(r"tap target .*opponent controls|doesn't untap|stun counter")
    f["go_wide_anthem"]= has(r"creatures you control get \+")
    f["pump_spell"]= has(r"target creature gets \+\d+/\+\d+")
    f["extra_turn"]= has(r"extra turn")
    f["copy_spell_perm"]= has(r"copy of (target|that|this|it)|create a token that's a copy")
    f["steal"]= has(r"gain control of")
    f["tutor"]= has(r"search your library for a(?! basic land|n? ?land)")
    f["landcycling"]= has(r"landcycling|cycling \{")
    f["cycling"]= has(r"cycling")
    f["kicker"]= has(r"kicker")
    f["flashback_like"]= has(r"flashback|beam me up|escape|aftermath|jump-start|disturb")
    f["conditional_bad"]= has(r"can't block|can't attack alone|can't be blocked|sacrifice .* at the beginning|you lose the game|skip your")
    f["station_threshold"]= 0
    m=re.search(r"at (\d+)\+", t); 
    if m: f["station_threshold"]=int(m.group(1))
    f["is_station"]= int("station" in t)
    f["differently_named_lands"]= count(r"differently named lands")
    f["beam_me_up"]= has(r"beam me up")
    f["face_dilemma_payoff"]= has(r"face a dilemma|faced a dilemma")
    f["modal"]= int(t.count("choose one") + t.count("choose two") + (1 if "choose one —" in t else 0))
    f["n_modes_bullets"]= t.count("•")
    f["landfall"]= has(r"landfall")
    f["investigate"]= has(r"investigate|clue")
    f["treasure"]= has(r"treasure")
    f["food"]= has(r"food token|create a food")
    f["enters_tapped_land"]= int("land" in tl.lower() and "enters tapped" in t)
    f["dual_land"]= int("land" in tl.lower() and len(re.findall(r"add \{[wubrg]\}", t))>=1 and "add {" in t)
    f["variable_power_text"]= has(r"power is equal to|power and toughness are each equal")
    # --- v2 features ---
    f["vanilla_text"]= int(is_creature and len(t.strip())<40)
    f["etb_effect"]= int(is_creature and bool(re.search(r"when .* enters", t)))
    f["creature_no_etb"]= int(is_creature and not re.search(r"when .* enters", t))
    f["n_whenever"]= count(r"whenever")
    f["n_when"]= count(r"when ")
    f["n_activated"]= count(r"\{[^}]*\}(, \{t\})?:|\{t\}:")
    f["token_total_power"]=0; f["token_bodies"]=0
    for m in re.finditer(r"create (a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+|x) .*?(\d+)/(\d+) [a-z ]*?token", t):
        n=NUM_WORDS.get(m.group(1),1); f["token_bodies"]+=n; f["token_total_power"]+= n*int(m.group(2))
    _removal = (f["destroy_creature"] or f["exile_perm"] or (f["max_damage"]>=3) or f["minus_effect"])
    f["removal_conditional"]= int(_removal and bool(re.search(r"attacking|blocking|tapped|power \d|mana value|nonland|with flying|you don't control", t)))
    f["removal_unconditional"]= int((f["destroy_creature"] or f["exile_perm"]) and not f["removal_conditional"])
    f["wrath"]= int(bool(re.search(r"(destroy|exile) all|all creatures get -|each player sacrifices|deals \d+ damage to each creature", t)))
    f["card_advantage_est"]= f["draw_n"] + 0.7*f["token_bodies"] + f["graveyard_recursion"] + f["tutor"]
    f["mana_colors_produced"]=0
    if "land" in tl.lower():
        cols=set(re.findall(r"add \{([wubrg])\}", t)); f["mana_colors_produced"]=len(cols) if cols else 1
        f["any_mana"]= int("any color" in t)
    f["enters_tapped_uncond"]= int("land" in tl.lower() and "enters tapped" in t and "unless" not in t and "you may pay" not in t)
    f["shockland"]= int("land" in tl.lower() and "pay 2 life" in t)
    f["sac_land_utility"]= int("land" in tl.lower() and "sacrifice this land" in t)
    f["cost_reduction"]= has(r"costs? \{?\d*\}? ?(less|more)|affinity")
    f["discount_amount"]= 0
    f["spell"]= int(f["is_instant"] or f["is_sorcery"])
    f["permanent"]= int(is_creature or f["is_artifact"] or f["is_enchantment"] or f["is_planeswalker"])
    f["cmc_le1"]=int(cmc<=1); f["cmc_2"]=int(cmc==2); f["cmc_3"]=int(cmc==3); f["cmc_4"]=int(cmc==4); f["cmc_5"]=int(cmc==5); f["cmc_ge6"]=int(cmc>=6)
    f["cheap_creature"]=int(is_creature and cmc<=2); f["big_creature"]=int(is_creature and f["pt_sum"]>=9)
    f["evasion"]= int(f["kw_flying"] or f["kw_menace"] or "can't be blocked" in t)
    f["body_efficiency"]= (f["pt_sum"]-(2*cmc+1)) if is_creature else 0
    f["power_minus_cmc"]= (f["power"]-cmc) if is_creature else 0
    f["legendary_creature"]= int(f["is_legendary"] and is_creature)
    f["text_sentences"]= t.count(".")+1
    f["opp_interaction"]= count(r"opponent|player controls")
    f["self_harm"]= has(r"you lose \d+ life|lose \d+ life|discard a card,|sacrifice a creature")
    # interactions
    f["creature_x_cmc"]= f["is_creature"]*cmc
    f["efficient_creature"]= int(is_creature and cmc<=3 and f["pt_sum"]>=cmc+2)
    f["removal"]= int(f["destroy_creature"] or f["exile_perm"] or (f["max_damage"]>=3) or f["minus_effect"])
    f["hard_removal"]= int(f["destroy_creature"] or f["exile_perm"])
    f["cheap_removal"]= int(f["removal"] and cmc<=3)
    return f

FEATURE_NAMES=None
def feature_vector(rec, names=None):
    global FEATURE_NAMES
    d=structured_features(rec)
    if names is None:
        names=sorted(d.keys()); FEATURE_NAMES=names
    return np.array([d.get(k,0) for k in names], dtype=float), names

def clean_text(rec):
    t=(rec.get("oracle_text") or "").lower()
    # normalize card self-name tokens are removed by tfidf idf anyway; strip reminder text in parens partially kept
    return t
