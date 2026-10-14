#!/usr/bin/env python3
"""Generate final calibrated 17lands GiH win rate predictions and diagnostic artifacts for Reality Fracture (FRA)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("/workspace/analysis")
ARTIFACTS_DIR = Path("/opt/cursor/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# Explicit card-level domain & historical-twin targets (blended with the ML + Env Sim ensemble)
# Each entry is (target_wr, weight_of_domain_target, rationale)
DOMAIN_CALIBRATIONS: dict[str, tuple[float, float, str]] = {
    # --- Exact Reprints & Direct SOS/MSH Functional Twins ---
    "Last Gasp": (0.5765, 0.85, "Exact reprint from SOS (0.5770 GiH WR over 54,802 games); premier 2-mana instant removal"),
    "Blazing Crescendo": (0.5305, 0.85, "Exact reprint from MSH (0.5292 GiH WR over 7,417 games); combat trick + impulse draw"),
    "Unsummon": (0.5585, 0.70, "Classic 1-mana bounce; punishes tokens, expensive threats, and upkeep-prepared creatures"),
    "Hexhaven Dueling Arena": (0.5210, 0.80, "Direct functional twin of SOS Skycoach Waypoint (0.5166 GiH WR); colorless prepare utility land"),
    "Deserted Beach": (0.5725, 0.80, "Allied Rare Slowland (WU Fatehold); SOS Rare Slowland cycle averaged 0.5704 GiH WR"),
    "Overgrown Farmland": (0.5705, 0.80, "Allied Rare Slowland (GW Vigorbloom); SOS Rare Slowland cycle averaged 0.5704 GiH WR"),
    "Shipwreck Marsh": (0.5680, 0.80, "Allied Rare Slowland (UB Theorix); SOS Rare Slowland cycle averaged 0.5704 GiH WR"),
    "Rockfall Vale": (0.5650, 0.80, "Allied Rare Slowland (RG Konstrari); SOS Rare Slowland cycle averaged 0.5704 GiH WR"),
    "Haunted Ridge": (0.5635, 0.80, "Allied Rare Slowland (BR Stingerquill); SOS Rare Slowland cycle averaged 0.5704 GiH WR"),

    # --- Basic Lands (tracking FRA color strength, mirroring SOS Basic Land spread) ---
    "Plains": (0.5645, 0.85, "White is the deepest common/uncommon color in FRA (mirroring SOS Plains 0.5733)"),
    "Island": (0.5585, 0.85, "Blue anchors Empower Jace card advantage across WU/GU/UB"),
    "Forest": (0.5555, 0.85, "Green anchors GW Vigorbloom counters and GU/RG ramp"),
    "Swamp": (0.5525, 0.85, "Black has premium removal but slower UB/BR common creatures"),
    "Mountain": (0.5495, 0.85, "Red has strong burn but weakest BR/RG common Prepare creatures"),

    # --- Common Dual Lands (enter untapped if you control a Jace/Planeswalker) ---
    "Fatehold Annex": (0.5565, 0.70, "WU common dual; frequently enters untapped via Jace tokens"),
    "Transformative Commons": (0.5545, 0.70, "GU common dual; frequently enters untapped via Jace tokens"),
    "Vigorbloom Annex": (0.5535, 0.70, "GW common dual; fixes top allied counters/lifegain pair"),
    "Meticulous Commons": (0.5515, 0.70, "WB common dual"),
    "Theorix Annex": (0.5510, 0.70, "UB common dual"),
    "Innovative Commons": (0.5495, 0.70, "UR common dual"),
    "Dedicated Commons": (0.5480, 0.70, "RW common dual"),
    "Formidable Commons": (0.5465, 0.70, "BG common dual"),
    "Konstrari Annex": (0.5445, 0.70, "RG common dual; fewer Jace tokens to enter untapped"),
    "Stingerquill Annex": (0.5430, 0.70, "BR common dual; fewer Jace tokens to enter untapped"),
    "Room of Refuge": (0.5440, 0.70, "Common thriving-style tapland with late-game +1/+1 counter sac"),

    # --- White Commons & Uncommons ---
    "Memory Trap": (0.5710, 0.75, "Unconditional 3-mana Banishing Light at Common; answers any nonland permanent including PWs"),
    "Surgical Precision": (0.5660, 0.65, "Modal 2-mana destroy toughness 4+ OR cantrip + gain 2 life; never dead"),
    "Graft Surgeon": (0.5670, 0.80, "Enters as a 3-mana 3/3 (via +1/+1 counter) that transfers all counters on death"),
    "Blossom-Blessed Angel // Seed Suture": (0.5645, 0.80, "4-mana 2/4 Flying Vigilance enters prepared with 1-mana +1/+1 counter + 1 life (becomes 3/5 Flying Vigilance)"),
    "Unflinching Hortimancer": (0.5595, 0.65, "2-mana 2/1 Ward {1} Ajani's Pridemate in GW/WB lifegain"),
    "Campus Crier": (0.5585, 0.65, "2-mana 3/1 that exiles from graveyard for {1} to Empower Jace 2"),
    "Hexhaven Battalion": (0.5680, 0.75, "Three 2/2 Cadets + Empower Jace 2 + Basic landcycling {2}"),
    "Fateshaper Aspirant": (0.5535, 0.65, "5-mana 3/4 modal: return legendary from GY or +1/+1 counter + indestructible"),
    "Prophesied End": (0.5645, 0.65, "2-mana instant unconditional creature removal (gives card if not attacking)"),
    "Refute Destiny": (0.5220, 0.85, "Narrow color-hoser (hits only green or blue) in Bo1 Premier Draft"),
    "Teyo, Lightshield Expert": (0.5690, 0.65, "2-mana 1/1 Flash protection + permanent +1/+1 or loyalty counter"),
    "Generous Revival": (0.5665, 0.65, "3-mana reanimate MV<=3 with +1/+1 counter + Flashback {4}{W}"),
    "Way of the Healer": (0.5795, 0.80, "4-mana Empower Jace 5 + [-2] create 2/2 Cadet & Surveil 1; produces two 2/2s + PW"),
    "Thalia, the Survivor": (0.5670, 0.70, "4-mana 3/4 Lifelink that taxes opponent noncreature spells by {1}"),
    "Saheeli, Consul of Oversight": (0.5750, 0.75, "5-mana 4/4 Flyer that creates a 1/1 flying Thopter whenever you scry/surveil"),
    "Ghalta the Immovable": (0.5650, 0.75, "Costs X less for greatest toughness; 4-5 mana 7/7 that makes high-toughness creatures deal toughness damage"),
    "Your Fate Ends Here": (0.5625, 0.65, "3-mana instant destroy MV>=3 creature or PW + Surveil 1"),

    # --- Blue Commons & Uncommons ---
    "Mindseeker Oculus": (0.5740, 0.80, "3-mana 2/1 Common with ETB Empower Jace 4 (immediately draws a card via [-3] and leaves a 1-loyalty Jace)"),
    "Protege's Awakening": (0.5715, 0.75, "4-mana Common draw 1 + Empower Jace 6 (draws 3 total cards across two turns)"),
    "Semester Foreseer // Peer Review": (0.5635, 0.80, "4-mana 3/4 ETB Surveil 1 enters prepared with Peer Review (3-mana 2/2 Cadet + Surveil 1)"),
    "Sphinx's Approach": (0.5580, 0.65, "3-mana instant-speed Divination"),
    "Infinite Coursework": (0.5560, 0.65, "3-mana tap/freeze Aura that also unprepares and strips abilities"),
    "Precise Redaction": (0.5050, 0.85, "Narrow color-hoser counterspell (white or black only) in Bo1 Limited"),
    "Plan for All Outcomes": (0.5710, 0.70, "4-mana nonland top/bottom tuck + repeatable Empower Jace 1 on first noncreature spell each turn"),
    "Hapatra, the Desert Frost": (0.5665, 0.70, "4-mana 4/3 that taps and stuns an opponent's creature on ETB"),
    "Way of the Mind Sculptor": (0.5685, 0.75, "5-mana Empower Jace 5 + draws an extra card whenever you activate a [-2] or [-3] loyalty ability"),
    "Ruric Thar, Biomagus": (0.5695, 0.75, "6-mana 4/6 Flyer with double prowess (+2/+2 per spell) and ward—draw a card"),
    "Yargle, Goliath of Otaria": (0.5250, 0.75, "5-mana 3/9 vanilla creature"),

    # --- Black Commons & Uncommons ---
    "Extended Absence": (0.5735, 0.75, "4-mana instant unconditional exile creature or planeswalker + drain 1"),
    "Silence the Echo": (0.5565, 0.65, "Bone Splinters / Spark Harvest with {3} kicker option"),
    "Solve for Disappointment": (0.5380, 0.65, "Targeted discard + Empower Jace 1"),
    "Cast Away Doubt": (0.5520, 0.70, "3-mana draw 2 + deals 2 damage to each player (triggers BR noncombat damage payoffs)"),
    "Multiply by Zero": (0.5695, 0.85, "2-mana instant sets target creature's base P/T to 0/0, killing any creature without counters/anthems"),
    "Terminal Criticism": (0.5260, 0.85, "Narrow color-hoser removal (blue or red only) in Bo1 Limited"),
    "Way of the Necromancer": (0.5240, 0.70, "Only Empower Jace 2 with no new loyalty ability; requires creatures dying"),
    "Break Under Pressure": (0.5680, 0.70, "3-mana instant sacrifice opponent's highest-MV creature/PW + gain 2 life"),
    "Loot, the Anomaly": (0.5515, 0.75, "3-mana -2/4 that deals combat damage as positive (effective 2/4) and grows to -4/4 (4 dmg) with Threshold"),
    "Proft, Sinister Mastermind": (0.5625, 0.75, "Instant-speed {B} discard mode gives -3/-1 (early removal + self-mill) or 3-mana 5/5 Menace with Threshold"),
    "Way of the Deathbringer": (0.5540, 0.65, "3-mana Empower Jace 5 + [-2] sac a creature to make a 4/4 Trample Beast"),
    "Yargle, Glutton of Urborg": (0.5320, 0.75, "5-mana 9/3 vanilla glass cannon"),

    # --- Red Commons & Uncommons ---
    "No Admittance": (0.5740, 0.80, "2-mana 3 damage to any target + Empower Jace 1; premium Red common removal"),
    "Wrath of the Bloodmane": (0.5685, 0.75, "3-mana (2-mana with legend) instant 4 damage to creature or planeswalker"),
    "Heartstring Puller": (0.5565, 0.70, "4-mana 3/1 Trample + 2/2 Cadet token (5/3 across two bodies at Common)"),
    "Awaken the Inferno": (0.5585, 0.70, "5-mana 6 damage + +1/+1 counter + Basic landcycling {2}"),
    "Essence Burn": (0.5240, 0.85, "Narrow color-hoser removal (black or green only) in Bo1 Limited"),
    "Way of the Pyromancer": (0.5180, 0.75, "Only Empower Jace 2 and [+1]: Add {R}; low board impact"),
    "Fulminous Forte": (0.5745, 0.75, "3-mana instant modal 5 damage to creature/PW OR 1-damage one-sided sweeper"),
    "Pia, Determined Rebuilder": (0.5610, 0.70, "3-mana 2/2 + 1/1 flying Thopter token with late-game pump"),
    "Way of the Warlord": (0.5635, 0.75, "3-mana Empower Jace 5 + [-4] deals 2 damage to creature/PW and 2 to player"),
    "Violent Echoes": (0.5715, 0.75, "4-mana instant 6 damage to creature/PW + Empower Jace for excess damage (often Empower Jace 2-4!)"),
    "Kiora of Fire and Ashes": (0.5790, 0.80, "6-mana 2/2 that creates a 5/5 Flying Dragon on ETB (7/7 across 2 bodies) + {8} repeatable 5/5 Dragon"),

    # --- Green Commons & Uncommons ---
    "Arcane Amphisbaena": (0.5615, 0.70, "2-mana 1/1 Deathtouch + ETB Empower Jace 2; elite defensive speed + Jace setup"),
    "Compel Brutality": (0.5645, 0.70, "2-mana instant bite using either creature power OR planeswalker loyalty!"),
    "Sureshot Sower": (0.5580, 0.65, "2-mana 3/1 Reach + uncounterable {3}{G} discard to destroy a flyer"),
    "Budding Insurgent": (0.5525, 0.65, "3-mana 3/3 Vigilance + sacrifice disenchant (draws a card if destroying a Way of the... enchantment!)"),
    "Greenhouse Propagator": (0.5535, 0.65, "3-mana 2/3 mana dork that gains 1 life whenever a creature enters"),
    "Bestial Incursion": (0.5665, 0.75, "4-mana 4/4 Trample Beast token + Flashback {5}{G} for a second 4/4 Beast"),
    "Wrecking Gecko": (0.5545, 0.70, "5-mana 5/5 Ward {2} with late-game +4/+4 trample pump"),
    "Vinelasher Adept": (0.5620, 0.80, "Enters with three +1/+1 counters on target creature (5/7 Reach or +3/+3 haste boost) + Basic landcycling {2}"),
    "Flourishing Grapple": (0.5170, 0.85, "Narrow color-hoser bite (red or white target only) in Bo1 Limited"),
    "Fblthp, Knows the Way": (0.5520, 0.75, "Domain capped at 2 in normal 2-color Limited decks; X-spell basic land tutor to hand"),
    "Yoshimaru, Scrappy Stray": (0.5680, 0.65, "2-mana 1/1 with ETB fight + late-game counter sink"),
    "Pia, Aether Ascetic": (0.5420, 0.70, "3-mana 2/2 rummage for an enchantment"),
    "Jiang Yanggu, Never Alone": (0.5795, 0.85, "4-mana 2/2 that creates a 3/3 Mowu token (5/5 across 2 bodies for 4 mana) and untaps all tokens each end step"),
    "Edgar, Moonlit Sovereign": (0.5670, 0.70, "5-mana 4/4 Flash that grows by +2/+2 if flashed in and proliferates counters on team"),
    "Way of the Wildspeaker": (0.5790, 0.80, "5-mana Empower Jace 7 + [-4] create a 4/4 Trample Beast (makes a 4/4 immediately + draws a card or makes a 2nd 4/4!)"),
    "Ruric Thar, Magecrusher": (0.5695, 0.70, "7-mana 7/7 Reach, Vigilance, Trample, Hexproof until it deals combat damage"),
    "Ghalta the Unstoppable": (0.5700, 0.75, "Costs X less for greatest power; 4-5 mana 8/8 Trample that grants team-wide Trample"),

    # --- Multicolor / Hybrid Commons & Uncommons ---
    "Fatehold Chronologist // Peer Review": (0.5715, 0.75, "2-mana 1/2 Flyer enters prepared with Peer Review (3-mana 2/2 Cadet + Surveil 1); built-in 2-for-1 curve out"),
    "Tam's Resistance": (0.5665, 0.75, "2-mana sorcery: +1/+1 counter, vigilance, and Empower Jace 4 (immediately draws a card via Jace [-3]!)"),
    "Fatehold Charm": (0.5795, 0.70, "2-mana instant modal: Draw 1 + Empower Jace 2 OR Remand/Unsummon OR team +1/+2"),
    "Theorix Charm": (0.5780, 0.70, "2-mana instant modal: -2/-2 removal OR Quench OR mill 3 + draw 1"),
    "Stingerquill Charm": (0.5735, 0.70, "2-mana instant modal: 3 damage any target OR first strike+deathtouch trick OR 2/2 haste flash Cadet"),
    "Vigorbloom Charm": (0.5745, 0.70, "2-mana instant modal: +1/+1 counter & fight OR draw 1 + 3 life OR hexproof+indestructible"),
    "Prudent Fateseer // Peer Review": (0.5765, 0.75, "3-mana 1/4 enters prepared with Peer Review (2/2 Cadet + Surveil 1) and pumps team +1/+0 on scry/surveil"),
    "Mabel, Valley Hero": (0.5785, 0.85, "3-mana Good-Fortune Unicorn / Grumgully that enters as a 2/3 and puts a +1/+1 counter on every creature/token entering"),
    "Saheeli, Jewel of Avishkar": (0.5735, 0.80, "4-mana 2/4 that creates a 1/1 flying haste Thopter on every noncreature spell cast"),
    "Recursive Recruitment": (0.5795, 0.75, "4-mana two 2/2 Cadets + Flashback {6}{U}{B} making two huge 5/5+ Cadets"),
    "Twisted Fates": (0.5785, 0.75, "5-mana destroy target nonland permanent + put a +1/+1 counter on each creature you control"),
    "Craftwork Crusher": (0.5815, 0.75, "7-mana 7/5 Trample that deals 4 damage + creates a 2/2 Cadet or draws a card on ETB"),

    # --- Rares & Mythics: Constructed Traps / Do-Nothings ---
    "Loyal Tutor": (0.4550, 0.90, "1-mana top-of-library tutor for a planeswalker card; FRA has zero common/uncommon planeswalker cards (Jace tokens are tokens)"),
    "Entrust the Spark": (0.4620, 0.90, "5-mana sacrifice a PW to search library for a PW card; 90%+ of Limited decks have zero PW cards in library"),
    "Extrapolate the Impossible": (0.5010, 0.85, "2-mana sorcery wish where opponent chooses the worse of two draft sideboard cards"),
    "Gardenize": (0.4820, 0.90, "3-mana enchantment that does nothing until creatures die and only produces excess {G}"),
    "Puppet Crafting": (0.5110, 0.85, "Ensoul Artifact aura requiring an existing artifact or non-Aura enchantment"),
    "Karn, Argent Defender": (0.5080, 0.85, "2-mana 1/3 Torpor Orb that shuts off your own Empower Jace and creature ETB triggers"),
    "Omnipresence": (0.5020, 0.85, "8-mana {5}{G}{G}{G} build-around enchantment with no immediate board impact"),
    "Return to the Light Realms": (0.5050, 0.85, "9-mana {7}{W}{W} sorcery mass reanimation; uncastable in standard Limited games"),
    "Cruel Calculations": (0.5180, 0.80, "3-mana sorcery that only draws cards equal to cards milled this turn; requires 2-card combo"),
    "Face Yourself": (0.5180, 0.80, "7-mana sorcery win-more clone effect requiring an existing board and planeswalker"),
    "Codie, Ravenous Codex": (0.5340, 0.75, "3-mana 1/4 build-around that only copies prepared spells"),
    "Identity Echo": (0.5220, 0.80, "3-mana enchantment requiring {3}{R} activation to polymorph a creature/PW"),
    "Roiling Canopy": (0.5420, 0.75, "Mono-green tapland requiring 6+ Forests to trigger"),

    # --- Rares & Mythics: Format-Defining Bombs & High-Synergy Engines ---
    "The Theorist, Jace Beleren": (0.6520, 0.85, "#1 Mythic bomb in FRA: draws extra card every opponent draw step, +1 makes 1/1 token, -2 bounces, and can be directly Empowered by all Empower Jace cards!"),
    "Ajani Unrelenting": (0.6420, 0.85, "Mythic PW that creates a 2/2 Cadet on EVERY loyalty activation (including Jace tokens) and -3 wipes all non-token creatures"),
    "Garruk, Curse Breaker": (0.6380, 0.85, "Mythic PW that creates 4/4 Trample Beasts while drawing a card from its own passive, ramps, and overruns"),
    "Kwia Vigorbloom": (0.6360, 0.85, "6-mana 6/6 Flying, Vigilance, Lifelink, Ward {2} that creates Black Lotus tokens on lifegain"),
    "Denzilore Fatehold": (0.6290, 0.85, "4-mana 3/4 Flash Flyer that puts a +1/+1 counter on your ENTIRE TEAM whenever you scry or surveil (triggers on every Jace [-1]!)"),
    "Garruk, Veiled Butcher": (0.6280, 0.85, "5-mana Mythic PW that shrinks threats, edicts opponent while making a 4/4 Beast, and shreds hand"),
    "Chandra, Torch of Defiance": (0.6220, 0.85, "4-mana 4-ability Mythic Planeswalker: removal, card advantage, ramp, and game-winning emblem"),
    "Jace, Reality Sculptor": (0.6180, 0.85, "5-mana Rare Jace PW whose +1 adds loyalty equal to Islands, -3 blanks attackers (-5/-0), and 0 wins the game via mill"),
    "Verdant Kraken": (0.6160, 0.85, "7-mana 6/6 that creates a 3/3 Forest Tentacle token on EACH player's upkeep (two 3/3s per turn cycle)"),
    "Aerid Konstrari": (0.6150, 0.80, "4-mana 5/4 Flying Mythic Sphinx that creates Heartwood tokens on ETB/death and pumps itself"),
    "Sphinx of False Conclusions": (0.6140, 0.85, "4-mana 4/2 Flash Flyer that loots on attack and creates a non-token-copy 4/2 Flyer when it dies"),
    "Overwrite the Multiverse": (0.6120, 0.80, "6-mana exile all creatures + Empower Jace X (wipes board and leaves a 5-8 loyalty Jace drawing cards on an empty board)"),
    "Ajani Resolute": (0.6110, 0.80, "2-mana Mythic PW that makes Ajani's Pridemate tokens and grows with every lifegain trigger"),
    "Ingris Stingerquill": (0.6090, 0.80, "3-mana 1/4 Flyer that pings on every attacker and creates 2/2 haste Cadets for {4}"),
    "Sanctum Lurker": (0.6080, 0.85, "3-mana 3/2 that makes your Planeswalkers immortal at 0 loyalty and gives them [+2]: Drain 1 life + Empower Jace 1"),
    "Rise of the Deathbringer": (0.6080, 0.80, "5-mana Instant modal -3/-3 sweeper OR draw X cards equal to greatest power"),
    "Hungering Puppetbeast": (0.6080, 0.80, "5-mana 5/5 that enters with a Heartwood token to immediately become a 6/6 with Hexproof/Haste/Trample"),
    "Draconic Visitor": (0.6060, 0.80, "5-mana 5/5 Flying Dragon that turns every artifact token created into a 5/5 Flying Dragon"),
    "Lich's Relic": (0.6020, 0.85, "1-mana +2/+1 Equipment with {2} ETB unconditional destroy target creature or planeswalker"),
    "Uldaros Theorix": (0.6010, 0.80, "6-mana 5/5 Flying Mythic Sphinx that casts up to 6 MV of spells from your graveyard for free on ETB"),
    "Emrakul, the Exigent Doom": (0.5980, 0.80, "Exiles from hand for {3} to turn a land into Ancient Tomb, then casts from exile as a 12/12 Flying Trample that untaps all lands"),
    "Jace's Machinations": (0.5960, 0.80, "3-mana Instant Empower Jace 8 + instant-speed Jace activations (draws 3 cards across 3 turns starting immediately)"),
    "Chandra, Chill of Compliance": (0.5960, 0.75, "3-mana Mythic PW that taps/stuns creatures, ramps, and draws noncreature spells"),
    "Seasoned Cryomancer": (0.5950, 0.75, "3-mana 2/2 loot 2 + tap/stun up to 2 creatures + graveyard draw 2"),
    "Lyra, Tolarian Archangel": (0.5960, 0.75, "3-mana 3/3 Flyer with card draw sink and 3/3 Angel token generation"),
    "Variable Chaser // Arc of Fortune": (0.5920, 0.75, "3-mana 2/3 Flying Prowess enters prepared with optional Wheel of Fortune"),
    "Solitary Cell": (0.5910, 0.75, "2-mana Portable Hole / Glass Casket for MV<=3 that also loots legendary cards"),
    "Guiding Hydra": (0.5940, 0.75, "Scalable X-cost Hydra that transfers a counter to become a team-wide +1/+1 counter anthem every combat"),
    "Kindred Judgment": (0.5910, 0.75, "7-mana plague-wind wrath sparing your most common creature type (e.g. Wizard/Human)"),
    "The Echoverse Fulcrum": (0.5880, 0.80, "2-mana colorless artifact that loots on ETB and exiles for {5} to wrath all creatures"),
    "Vraska's Final Mercy": (0.5870, 0.75, "2-mana {B}{B} modal unconditional creature/PW removal OR Empower Jace 6"),
    "Enlightened Confidant": (0.5930, 0.75, "2-mana 2/1 Lifelink that surveils and draws cards on lifegain"),
    "Tarmogoyf": (0.5860, 0.75, "2-mana 4/5 or 5/6 efficient beater in a set with creatures, instants, sorceries, enchantments, artifacts, lands, and PWs"),

    # --- Special Guests (SPG) ---
    "Sublime Epiphany": (0.6310, 0.85, "Iconic 6-mana 5-mode instant blowout (counter + bounce + clone + draw)"),
    "Austere Command": (0.6080, 0.85, "Flexible 6-mana modal sweeper that crafts one-sided board wipes"),
    "Flesh Duplicate": (0.5760, 0.80, "2-mana Phantasmal Image / clone of best creature on board for 3 turns (re-triggers ETBs!)"),
    "Consider": (0.5680, 0.80, "1-mana instant Surveil 1 + Draw 1; triggers WU/UB surveil and UR prowess"),
    "Mind Twist": (0.5620, 0.80, "Scalable random discard X-spell; devastating at X=3-4"),
    "Splinter Twin": (0.5380, 0.80, "4-mana Aura value engine on ETB creatures, though vulnerable to instant removal"),
    "Consign to Memory": (0.5120, 0.85, "Narrow sideboard counterspell for triggered abilities / colorless spells"),
    "Eye of Ugin": (0.4680, 0.90, "Produces no mana by itself; only tutors colorless creatures for {7}"),
    "Root Maze": (0.4620, 0.90, "1-mana symmetrical stax enchantment with zero Limited board value"),
    "Necrodominance": (0.4780, 0.85, "Triple-black {B}{B}{B} enchantment that skips draw step and costs life in Limited"),
}


def compute_color_adjustment(card: dict) -> float:
    """Apply environment-level color balance adjustment reflecting FRA's common/uncommon depth."""
    if card.get("is_basic_land") or card.get("is_special_guest"):
        return 0.0
    cols = card.get("colors") or card.get("color_identity") or []
    if not cols:
        return 0.0
    col_weights = {"W": 0.0030, "U": 0.0025, "G": 0.0010, "B": -0.0010, "R": -0.0025}
    return float(np.mean([col_weights.get(c, 0.0) for c in cols]))


def main() -> None:
    cat = json.loads((OUT_DIR / "card_catalog.json").read_text())
    ml_df = pd.read_csv(OUT_DIR / "fra_ml_predictions.csv").set_index("card_name")

    all_cards: list[tuple[str, dict]] = []
    for cname, c in cat["fra_main"].items():
        all_cards.append((cname, c))
    for cname, c in cat["fra_special_guests"].items():
        all_cards.append((cname, c))

    # Mean of FRA main set for variance calibration
    fra_mean = float(ml_df.loc[ml_df["set_group"] == "FRA", "ml_pred"].mean())

    rows = []
    for cname, c in all_cards:
        ml_row = ml_df.loc[cname]
        raw_ml = float(ml_row["ml_pred"])

        # Step 1: Variance expansion around set mean (compensating for L2 ensemble shrinkage) + color balance
        col_adj = compute_color_adjustment(c)
        var_expanded = fra_mean + 1.15 * (raw_ml - fra_mean) + col_adj

        # Step 2: Blend with card-specific domain & historical twin calibration when applicable
        if cname in DOMAIN_CALIBRATIONS:
            dom_target, dom_w, note = DOMAIN_CALIBRATIONS[cname]
            final_pred = dom_w * dom_target + (1.0 - dom_w) * var_expanded
        else:
            final_pred = var_expanded
            note = "Ensemble ML + Environment Combat/Removal Simulation + Color/Archetype Calibration"

        final_pred = float(np.clip(final_pred, 0.4400, 0.6650))

        cn_str = str(c.get("collector_number", ""))
        try:
            cn_num = int("".join(ch for ch in cn_str if ch.isdigit()) or "999")
        except ValueError:
            cn_num = 999

        front_name = cname.split(" // ")[0] if " // " in cname else cname
        raw_cols = set(c.get("colors") or c.get("color_identity") or [])
        wubrg_order = [col for col in "WUBRG" if col in raw_cols]
        pair_map = {
            frozenset(["W", "U"]): "WU",
            frozenset(["U", "B"]): "UB",
            frozenset(["B", "R"]): "BR",
            frozenset(["R", "G"]): "RG",
            frozenset(["G", "W"]): "GW",
            frozenset(["W", "B"]): "WB",
            frozenset(["U", "R"]): "UR",
            frozenset(["B", "G"]): "BG",
            frozenset(["R", "W"]): "RW",
            frozenset(["G", "U"]): "GU",
        }
        if len(raw_cols) == 2 and frozenset(raw_cols) in pair_map:
            cols = pair_map[frozenset(raw_cols)]
        elif wubrg_order:
            cols = "".join(wubrg_order)
        else:
            cols = "Land" if "Land" in (c.get("type_line") or "") else "C"

        rows.append({
            "card_name": cname,
            "front_name": front_name,
            "set_code": "SPG" if c.get("is_special_guest") else "FRA",
            "collector_number": cn_str,
            "cn_num": cn_num,
            "rarity": (c.get("rarity") or "common").capitalize(),
            "colors": cols,
            "mana_cost": c.get("mana_cost") or "",
            "cmc": float(c.get("cmc") or 0.0),
            "type_line": c.get("type_line") or "",
            "ml_ensemble_pred": round(raw_ml, 4),
            "env_var_calibrated": round(var_expanded, 4),
            "predicted_gih_wr": round(final_pred, 4),
            "predicted_gih_pct": f"{final_pred * 100:.2f}%",
            "rationale": note,
        })

    out_df = pd.DataFrame(rows)
    out_df.sort_values(by=["set_code", "cn_num", "card_name"], inplace=True)

    # Save CSV and JSON outputs
    out_df.to_csv(OUT_DIR / "fra_final_predictions.csv", index=False)
    out_df.to_csv(Path("/workspace/predictions.csv"), index=False)
    Path("/workspace/predictions.json").write_text(
        json.dumps(out_df.to_dict(orient="records"), indent=2) + "\n"
    )

    main_sub = out_df[out_df["set_code"] == "FRA"]
    print(f"Generated final predictions for {len(out_df)} total cards ({len(main_sub)} FRA main-set + {len(out_df) - len(main_sub)} SPG).")
    print(f"FRA Main-Set Summary: mean={main_sub['predicted_gih_wr'].mean():.4f}, std={main_sub['predicted_gih_wr'].std():.4f}, min={main_sub['predicted_gih_wr'].min():.4f}, max={main_sub['predicted_gih_wr'].max():.4f}")

    print("\n=== BY RARITY (FRA MAIN SET) ===")
    for rar in ["Common", "Uncommon", "Rare", "Mythic"]:
        sub = main_sub[main_sub["rarity"] == rar]
        print(f"  {rar:10s} (n={len(sub):3d}): mean={sub['predicted_gih_wr'].mean():.4f}, std={sub['predicted_gih_wr'].std():.4f}, min={sub['predicted_gih_wr'].min():.4f}, max={sub['predicted_gih_wr'].max():.4f}")

    print("\n=== TOP 20 PREDICTED CARDS IN FRA ===")
    for _, r in out_df.sort_values("predicted_gih_wr", ascending=False).head(20).iterrows():
        print(f"  {r['predicted_gih_wr']:.4f} ({r['predicted_gih_pct']}) | [{r['set_code']} {r['rarity'][0]}] {r['card_name']:38s} | {r['mana_cost']}")

    print("\n=== BOTTOM 15 PREDICTED CARDS IN FRA ===")
    for _, r in out_df.sort_values("predicted_gih_wr", ascending=True).head(15).iterrows():
        print(f"  {r['predicted_gih_wr']:.4f} ({r['predicted_gih_pct']}) | [{r['set_code']} {r['rarity'][0]}] {r['card_name']:38s} | {r['mana_cost']}")


if __name__ == "__main__":
    main()
