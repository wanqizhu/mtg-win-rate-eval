#!/usr/bin/env python3
"""Apply set-context overlays + manual card-by-card adjustments to FRA base predictions."""
import json, gzip, re

base=json.load(open("/workspace/analysis/fra_base.json"))
with gzip.open("/opt/mtg-data/scryfall/sets/fra.jsonl.gz",'rt') as f:
    fra_all=[json.loads(l) for l in f if l.strip()]
by_id={c['id']:c for c in fra_all}
by_oracle={}
for c in fra_all: by_oracle.setdefault(c['oracle_id'],[]).append(c)
def cn(c):
    try: return int(str(c['collector_number']).split('/')[0])
    except: return 9999
canon={sorted(v,key=cn)[0]['oracle_id']:sorted(v,key=cn)[0] for v in by_oracle.values()}
spg={c['oracle_id']:c for c in [json.loads(l) for l in open("/opt/mtg-data/scryfall/sets/fra-special-guests.jsonl") if l.strip()]}

def full_text(card):
    parts=[]
    if card.get('oracle_text'): parts.append(card['oracle_text'])
    for f in (card.get('card_faces') or []):
        if f.get('oracle_text'): parts.append(f['oracle_text'])
    return "\n".join(parts)

# ---------- systematic mechanic overlays ----------
def empower_n(txt):
    m=re.findall(r'[Ee]mpower Jace (\d+|X)', txt)
    vals=[]
    for v in m:
        if v=='X': vals.append(3.0)  # variable; assume ~3
        else: vals.append(float(v))
    return max(vals) if vals else 0

MANUAL={
# White
"Academic Ascent":0.012,"Blossom-Blessed Angel // Seed Suture":0.008,"Campus Crier":0.008,
"Enlightened Confidant":0.005,"Fateshaper Aspirant":-0.003,"Generous Revival":0.005,
"Germinate Recruits":-0.008,"Graft Surgeon":0.004,"Hexhaven Battalion":-0.003,"Loyal Tutor":-0.012,
"Memory Trap":0.008,"Predictive Preparations":0.005,"Prophesied End":0.008,"Refute Destiny":-0.012,
"Repurposed Enforcer":0.002,"Return to the Light Realms":-0.015,"Shatterwing Pegasus":0.004,
"Surgical Precision":0.003,"Unflinching Hortimancer":0.005,"Your Fate Ends Here":-0.004,
# Blue
"Countersculpt":0.004,"Cruel Calculations":-0.006,"Cryotheory Adept":0.003,"Diviner of Victory // Unwind History":0.008,
"Divining Duelist":0.006,"Icy Reception":0.002,"Infinite Coursework":0.002,"Jace's Machinations":0.012,
"Mindseeker Oculus":0.008,"Perfected Theory":0.004,"Plan for All Outcomes":0.004,"Precise Redaction":-0.014,
"Protege's Awakening":0.012,"Seasoned Cryomancer":0.005,"Semester Foreseer // Peer Review":0.014,
"Sphinx of False Conclusions":0.005,"Sphinx's Approach":0.002,"Surveillance Phantasm":0.006,
"The Theorist, Jace Beleren":0.010,"Theorist's Proxy":0.006,"Undulating Witness":0.004,
"Variable Chaser // Arc of Fortune":0.002,
# Black
"Bloodline Recollector // Ancestral Craving":0.004,"Break Under Pressure":0.004,"Cast Away Doubt":0.006,
"Dark Matter Manipulator":0.006,"Darklight Phoenix":0.006,"Extended Absence":0.004,
"Extrapolate the Impossible":-0.008,"Last Gasp":0.008,"Lich's Relic":0.008,"Overwrite the Multiverse":0.008,
"Rampart Hunter":0.002,"Rank Rat":0.002,"Rewrite Regrets":0.008,"Rise of the Deathbringer":0.006,
"Sanctum Lurker":0.004,"Screeching Soulbreaker":0.002,"Silence the Echo":0.004,"Solve for Disappointment":0.006,
"Terminal Criticism":-0.014,"Theoretical Necromancer":0.006,"Void Extrapolator // Omit Variables":0.010,
"Vraska's Final Mercy":0.004,
# Red
"Ajani's Anguish":0.002,"Artifist Acumen":0.020,"Awaken the Inferno":0.004,"Blazing Crescendo":0.004,
"Chandra's Emberling":0.008,"Command the Stage":0.015,"Curse-Marred Demon":0.004,"Draconic Visitor":0.015,
"Essence Burn":-0.012,"Face Yourself":-0.004,"Fulminous Forte":0.002,"Hallway Heckler // Vicious Verse":0.010,
"Heartstring Puller":0.004,"Identity Echo":-0.004,"No Admittance":0.006,"Pompous Battlemage // Improvised Act":0.012,
"Pyre Rhymer // Molten Tide":0.010,"Skilled Battlecarver":0.002,"Stingcaster Mage":0.004,"Violent Echoes":0.006,
"Wrath of the Bloodmane":0.008,
# Green
"Arcane Amphisbaena":0.004,"Bestial Incursion":0.006,"Budding Insurgent":-0.002,"Carnivorous Cultivator // Enroot":0.006,
"Compel Brutality":0.004,"Flourishing Grapple":-0.012,"Gardenize":-0.004,"Heartwood Crafter // Soul Tether":0.008,
"Hexhaven Invigorator":-0.008,"Hungering Puppetbeast":0.002,"Hunter's Axe":0.014,"Inspired Tethermage":0.004,
"Omnipresence":-0.006,"Puppet Crafting":-0.006,"Restore with Empathy":0.002,"Simulacrum Shaper":0.004,
"Something Worth Saving":0.004,"Sureshot Sower":0.002,"Tarmogoyf":0.008,"Tethermage's Advantage":0.002,
"Verdant Kraken":-0.004,"Wrecking Gecko":0.004,"Vinelasher Adept":0.0,
# Multi
"Aerid Konstrari":0.006,"Avatar of Burgeoning Echoes":0.005,"Denzilore Fatehold":0.006,"Null Summoner":0.004,
"Uldaros Theorix":0.005,"Twisted Fates":-0.004,"Solitary Cell":0.002,"Proctor of Potential":0.004,
"Recursive Recruitment":0.002,"Solarium Sentry":-0.002,"Entrust the Spark":-0.006,"Grim Repriser":0.004,
"Samut, Hazoret's Champion":0.004,"Stingerquill Voxmancer // Vicious Verse":0.006,"Theorix Metamage // Omit Variables":0.010,
"Konstrari Improviser // Soul Tether":0.010,"Woodwork Prodigy // Soul Tether":0.008,"Whiplash Wordsmith // Vicious Verse":0.008,
"Fatehold Chronologist // Peer Review":0.010,"Prudent Fateseer // Peer Review":0.008,"Vigorbloom Vanguard // Seed Suture":0.008,
"Emergency Phytomedic // Seed Suture":0.010,"Twinned Vision":0.004,"Tam's Resistance":0.006,"Tenured Tethermage":0.008,
"Vindictive Triumph":0.004,"Warrior's Blades":0.002,"Frostbite Pyromental":0.004,"Ingris Stingerquill":0.004,
"Kwia Vigorbloom":0.006,"Stingerquill Charm":0.004,"Theorix Charm":0.002,"Vigorbloom Charm":0.006,
"Bloombrute":0.002,"Hapatra, the Desert Fang":-0.004,"Primal Witchstalker":0.006,"Paradox Shaper // Omit Variables":0.004,
"Ferocity of the Hunt":0.006,"Karn, Gilded Guardian":0.006,"Vraska, the Cutting Glare":0.002,
# Colorless/lands
"Afterthought Sentry":-0.002,"Archive Arbiter":-0.002,"Codie, Ravenous Codex":0.004,"The Echoverse Fulcrum":0.002,
"Eye of Jace":0.008,"Keeper of the Quiet Hour":0.008,"Living Library":-0.004,"Medic's Kitesail":0.012,
"Murmuring Volume":0.004,"Hall of Echoes":0.008,"Hexhaven Dueling Arena":0.004,"Theorist's Sanctum":0.006,
"Deserted Beach":0.012,"Haunted Ridge":0.012,"Overgrown Farmland":0.012,"Rockfall Vale":0.012,"Shipwreck Marsh":0.012,
"Dedicated Commons":0.014,"Fatehold Annex":0.014,"Formidable Commons":0.014,"Innovative Commons":0.014,
"Konstrari Annex":0.014,"Meticulous Commons":0.014,"Stingerquill Annex":0.014,"Theorix Annex":0.014,
"Transformative Commons":0.014,"Vigorbloom Annex":0.014,
"Emrakul, the Exigent Doom":-0.010,
# Bonus legends
"Ajani Resolute":0.010,"Garruk, Curse Breaker":0.002,"Yargle, Goliath of Otaria":-0.008,"Yargle, Glutton of Urborg":-0.012,
"Ghalta the Immovable":0.0,"Ghalta the Unstoppable":0.0,"Danitha, Spear of Agony":0.002,"Arni, Renowned Champion":0.0,
"Winter, Team Player":0.004,"Gallia, the Merrymaker":0.0,"Way of the Pyromancer":0.010,"Loot, the Nexus":0.004,
"Titanbones, Towering Heart":0.004,"Ruric Thar, Magecrusher":0.004,"Tomik, Izzet Sparkmage":0.0,"Traxos, Academy Guardian":0.002,
"Teyo, Lightshield Expert":0.002,"Thalia, the Survivor":0.002,"Rescue Girl, First Responder":-0.004,
"Saheeli, Consul of Oversight":-0.002,"Yoshimaru, Beloved Companion":0.0,"Yuriko, Blade of the Mighty":-0.004,
"Fblthp, Impossibly Lost":-0.006,"Geist of Saint Thalia":0.004,"Tetsuko Umezawa, Fugitive":-0.004,
"Ruric Thar, Biomagus":0.004,"Gallia, Tragic Host":0.002,"Loot, the Anomaly":-0.004,"Mabel, Bitter Recluse":0.004,
"Proft, Sinister Mastermind":-0.004,"Teyo, Diamondblade Mage":0.002,"Winter, Tormented Loner":0.002,
"Jiang Yanggu, Alone":0.0,"Kiora of Fire and Ashes":0.0,"Koth, the Geomancer":0.004,"Marwyn, the Clearcutter":0.0,
"Pia, Determined Rebuilder":0.0,"Tetsuko Umezawa, Pursuer":0.0,"Edgar, Moonlit Sovereign":-0.002,
"Fblthp, Knows the Way":0.0,"Jiang Yanggu, Never Alone":0.0,"Marwyn, the Preserver":-0.004,"Pia, Aether Ascetic":-0.006,
"Way of the Paradox":0.006,"Way of the Wildspeaker":0.008,"Yoshimaru, Scrappy Stray":0.002,
"Liliana the Faultless":0.002,"Danitha, Sword of Hope":-0.004,"Koth of the Homestead":0.0,
"Tomik, Orzhov Lawmage":0.002,"Way of the Healer":0.008,"Way of the Mentor":0.008,"Arni, Humble Scribe":0.0,
"Chandra, Chill of Compliance":0.004,"Hapatra, the Desert Frost":0.0,"Jace, Reality Sculptor":0.006,
"Lyra, Tolarian Archangel":0.002,"Proft, Consulting Detective":0.002,"Samut, Tyrant of Naktamun":-0.004,
"Way of the Cryomancer":0.006,"Way of the Mind Sculptor":0.006,"Yuriko, Hope from the Shadows":0.0,
"Gideon the Oathless":0.0,"Liliana the Repentant":0.0,"Massacre Girl, Most Wanted":0.0,"Tinybones, Pocket Nuisance":0.002,
"Way of the Deathbringer":0.006,"Way of the Necromancer":0.006,"Ajani Unrelenting":0.004,"Chandra, Torch of Defiance":0.004,
"Way of the Warlord":0.006,"Garruk, Veiled Butcher":0.0,"Karn, Argent Defender":-0.006,"Traxos, Scourge Eternal":-0.004,
"Lyra, Archangel of Dawn":0.0,"Gideon's Memorial":0.0,"Vraska, Soul of Stone":0.0,"Tam, the Possibility":0.0,
"Mabel, Valley Hero":0.0,"Saheeli, Jewel of Avishkar":0.0,"Kiora of Salt and Sand":0.0,"Edgar, Ancient Bloodlord":0.0,
# SPG
"Austere Command":0.010,"Consider":0.015,"Consign to Memory":-0.006,"Eye of Ugin":-0.030,"Flesh Duplicate":0.002,
"Mind Twist":-0.004,"Necrodominance":-0.004,"Root Maze":-0.008,"Splinter Twin":-0.002,"Sublime Epiphany":0.010,
}

out=[]
unmatched=[]
for o in base:
    card=canon.get(o['oracle_id'], spg.get(o['oracle_id']))
    if card is None:
        unmatched.append(o['name_full']); continue
    txt=full_text(card)
    adj=MANUAL.get(o['name_full'], 0.0)
    # safety: small systematic prepared/empower overlay only if not manually covered?
    # Manual dict already covers all mechanic cards; assert coverage:
    final=round(o['base_pred']+adj,4)
    final=max(0.45,min(0.70,final))
    short=o['name_17lands']
    out.append({**o, "adjustment":adj, "predicted_gih_wr":final})
print("unmatched:", unmatched)
# coverage check: mechanic cards missing from MANUAL
missing=[]
for o in base:
    card=canon.get(o['oracle_id'], spg.get(o['oracle_id']))
    txt=full_text(card)
    if ('mpower Jace' in txt or 'Empower Jace' in txt or 'prepared' in txt.lower() or 'Heartwood' in txt) and o['name_full'] not in MANUAL:
        missing.append(o['name_full'])
print("mechanic cards missing manual:", missing)
json.dump(sorted(out,key=lambda x:-x['predicted_gih_wr']), open("/workspace/analysis/fra_final.json","w"), indent=1)
# distribution
import statistics
vals=[x['predicted_gih_wr'] for x in out]
print("n=",len(out),"mean=",round(statistics.mean(vals),4),"sd=",round(statistics.pstdev(vals),4),
      "min=",round(min(vals),4),"max=",round(max(vals),4))
top=sorted(out,key=lambda x:-x['predicted_gih_wr'])[:12]
for t in top: print(t['predicted_gih_wr'], t['rarity'], t['name_full'])
print("...")
for t in sorted(out,key=lambda x:-x['predicted_gih_wr'])[-8:]: print(t['predicted_gih_wr'], t['rarity'], t['name_full'])
