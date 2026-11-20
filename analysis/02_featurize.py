"""Featurize Scryfall cards (historical + TRK) into structured numeric features + raw text."""
import gzip, json, re
from pathlib import Path
import pandas as pd

ROOT = Path("/opt/mtg-data")
CODES = ["fra","hob","msh","sos","tmt","ecl","tla","om1","eoe","fin","trk"]

def collector_key(c):
    m = re.match(r"(\d+)", c.get("collector_number","9999"))
    return int(m.group(1)) if m else 9999

def load_dedup(code):
    with gzip.open(ROOT/"scryfall"/"sets"/f"{code}.jsonl.gz","rt") as f:
        cards=[json.loads(l) for l in f if l.strip()]
    # dedup by front-face name (split //), keep lowest collector number
    uniq={}
    for c in sorted(cards, key=collector_key):
        full=c.get("name","")
        front=full.split(" // ")[0].strip()
        if front not in uniq:
            # store with normalized name
            c2=dict(c)
            c2["_front_name"]=front
            uniq[front]=c2
    return uniq

def parse_pt(v):
    out={}
    for stat in ["power","toughness"]:
        s=v.get(stat)
        try:
            out[stat]=float(s) if s is not None and re.fullmatch(r"-?\d+(\.\d+)?", str(s).strip()) else (0.0 if s in ("*","X") else 0.0)
            out[stat+"_is_var"]=1.0 if str(s).strip() in ("*","X") else 0.0
        except: out[stat]=0.0; out[stat+"_is_var"]=0.0
    return out

def featurize(name, c):
    t=(c.get("type_line") or "")
    o=(c.get("oracle_text") or "")
    ol=o.lower()
    mana=c.get("mana_cost") or ""
    cmc=float(c.get("cmc") or 0)
    colors=c.get("colors") or []
    ci=c.get("color_identity") or colors
    rarity=c.get("rarity") or "common"
    pt=parse_pt(c)
    # pips
    pips=re.findall(r"\{([WUBRG])\}", mana)
    generic=re.findall(r"\{(\d+)\}", mana)
    gen=int(generic[0]) if generic else 0
    ncolored=len(pips)
    # hybrid/phyrexian rough
    feat={}
    feat["name"]=name
    feat["cmc"]=cmc
    feat["gen"]=gen
    feat["ncolored"]=ncolored
    feat["npips_W"]=pips.count("W")
    feat["npips_U"]=pips.count("U")
    feat["npips_B"]=pips.count("B")
    feat["npips_R"]=pips.count("R")
    feat["npips_G"]=pips.count("G")
    feat["ncolors"]=len(colors)
    feat["is_W"]="W" in colors
    feat["is_U"]="U" in colors
    feat["is_B"]="B" in colors
    feat["is_R"]="R" in colors
    feat["is_G"]="G" in colors
    feat["is_multicolor"]=len(colors)>=2
    feat["is_colorless"]=len(colors)==0 and "Land" not in t
    feat["is_land"]="Land" in t
    feat["is_basic"]="Basic Land" in t
    feat["is_creature"]="Creature" in t
    feat["is_artifact"]=("Artifact" in t)
    feat["is_enchantment"]="Enchantment" in t
    feat["is_instant"]="Instant" in t
    feat["is_sorcery"]="Sorcery" in t
    feat["is_spacecraft"]="Spacecraft" in t
    feat["is_planet"]="Planet" in t
    feat["is_legendary"]="Legendary" in t
    feat["is_equipment"]="Equipment" in t
    feat["is_aura"]="Aura" in t
    feat["is_saga"]="Saga" in t
    feat["rarity"]=rarity
    feat["rarity_ord"]={"common":0,"uncommon":1,"rare":2,"mythic":3,"special":1,"bonus":2}.get(rarity,1)
    feat["power"]=pt["power"]
    feat["toughness"]=pt["toughness"]
    feat["pt_sum"]=pt["power"]+pt["toughness"]
    feat["stat_eff"]=(pt["power"]+pt["toughness"])/(cmc+1) if not feat["is_land"] else 0
    # keywords
    kws=["flying","vigilance","trample","haste","menace","deathtouch","lifelink","reach","flash","ward","hexproof","indestructible","defender","first strike","double strike","landfall","station","affinity","kicker","surveil","investigate"]
    for k in kws:
        feat[f"kw_{k.replace(' ','_')}"]=1.0 if k in ol or k in t.lower() else 0.0
    # ward value
    m=re.search(r"ward \{(\d+)\}", ol)
    feat["ward_n"]=float(m.group(1)) if m else 0.0
    # station threshold
    ms=re.findall(r"(\d+)\+", o)
    feat["station_thresh"]=float(min(int(x) for x in ms)) if ("station" in ol and ms) else 0.0
    # text stats
    feat["text_len"]=len(o)
    feat["text_lines"]=o.count("\n")+1 if o else 0
    feat["has_choose_one"]="choose one" in ol
    feat["has_dilemma"]="dilemma" in ol
    feat["has_beam"]="beam me up" in ol
    feat["has_landcycling"]="landcycling" in ol
    feat["has_cycling"]=("cycling" in ol)
    feat["has_strange"]="strange new worlds" in ol
    feat["has_kicker"]=("kicker" in ol)
    feat["has_quest"]="quest counter" in ol
    feat["has_counter"]="counter" in ol
    feat["n_draw_trig"]=len(re.findall(r"draw a card|draw two cards|draw x cards|draw cards", ol))
    feat["makes_token"]="create a" in ol and "token" in ol
    feat["gains_life"]="gain" in ol and "life" in ol
    feat["destroys"]="destroy target" in ol
    feat["exiles"]="exile target" in ol
    feat["bounces"]="return target" in ol and ("hand" in ol or "owner" in ol)
    feat["deals_damage"]="deals " in ol and "damage" in ol
    feat["taps"]="tap target" in ol
    feat["stun"]="stun" in ol
    feat["counterspell"]="counter target" in ol
    feat["discard_opp"]="discard" in ol and ("opponent" in ol or "each opponent" in ol)
    feat["ramp"]="search your library for" in ol and ("land" in ol)
    feat["treasure"]="treasure" in ol
    feat["clue"]="clue" in ol or "investigate" in ol
    feat["mill"]="mill" in ol
    feat["fight"]="fight" in ol
    feat["an extra turn"]="extra turn" in ol
    feat["win the game"]="win the game" in ol or "lose the game" in ol
    feat["etb"]="when this" in ol and "enters" in ol
    feat["attack_trig"]="whenever you attack" in ol or "whenever this creature attacks" in ol or "whenever you cast" in ol
    feat["sac_outlet"]="sacrifice" in ol
    feat["x_spell"]="x" in mana.lower().replace("{x}","X") or "{X}" in mana
    feat["oracle_text"]=o
    feat["type_line"]=t
    feat["mana_cost"]=mana
    feat["colors"]=colors
    return feat

rows=[]
for code in CODES:
    uniq=load_dedup(code)
    print(code, len(uniq))
    for name,c in uniq.items():
        f=featurize(name,c)
        f["set"]=code.upper()
        rows.append(f)

df=pd.DataFrame(rows)
df.to_pickle("/agent/repos/mtg-win-rate-eval/analysis/features.pkl")
print("wrote features", df.shape)
print(df[df["set"]=="TRK"].groupby("rarity").size())
