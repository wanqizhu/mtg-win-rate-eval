#!/usr/bin/env python3
"""Stage 3: Contextual evaluation, variance calibration, and final GIH win-rate prediction generator for Star Trek (TRK + SDS)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Mono-color and two-color environment adjustments from TRK simulation & pool analysis
MONO_COLOR_ENV_ADJ = {
    "W": 0.0025,
    "B": 0.0020,
    "G": 0.0005,
    "U": -0.0005,
    "R": -0.0030,
    "": 0.0005,
}

PAIR_ENV_ADJ = {
    frozenset(["W", "B"]): 0.0045,  # Tier 1: Borg / Artifact Creatures Matter
    frozenset(["B", "G"]): 0.0035,  # Tier 1: Graveyard Leaves / Attrition
    frozenset(["U", "B"]): 0.0030,  # Tier 1: Face a Dilemma / Modal Control
    frozenset(["G", "U"]): 0.0025,  # Tier 1.5: Strange New Worlds / Planet Ramp
    frozenset(["W", "U"]): 0.0015,  # Tier 1.5: Spacecraft & Tapping
    frozenset(["G", "W"]): 0.0010,  # Tier 2: +1/+1 Counters & Federation
    frozenset(["B", "R"]): 0.0005,  # Tier 2: Klingon Sacrifice
    frozenset(["R", "G"]): 0.0000,  # Tier 2: Landfall Midrange
    frozenset(["U", "R"]): -0.0015,  # Tier 2.5: Noncreature Spells
    frozenset(["R", "W"]): -0.0035,  # Tier 3: Total Power 8+
}

# Empirical 10-set 17Lands target mean and standard deviation per rarity
EMPIRICAL_RARITY_TARGETS = {
    "common": (0.5520, 0.0230),
    "uncommon": (0.5555, 0.0330),
    "rare": (0.5655, 0.0510),
    "mythic": (0.5835, 0.0575),
}

# Card-specific contextual, rules, and simulation adjustments (in win-rate probability units before rarity standardization)
TRK_CARD_ADJUSTMENTS: dict[str, tuple[float, str]] = {
    # WHITE (TRK #1..45)
    "70,000 Light-Years from Home": (0.011, "2-mana exile removal that answers both creatures and Spacecraft"),
    "Andromedan Automatons": (0.009, "Squadron Hawk artifact creature that tutors all copies from library on ETB"),
    "Antagonist Incorporator": (0.008, "Efficient 2-mana 2/3 Borg that turns artifact creature hits into 1/1 Borg tokens"),
    "Assembly Specialist": (0.016, "Premium 2-mana 2-body Borg producer (analog: S.H.I.E.L.D. Deployment Drone 60.0%)"),
    "Brad Boimler, Eager Ensign": (0.012, "2/2 lifelink that boosts counter placement whenever tapped (including via Station)"),
    "Captain Benjamin Sisko": (0.008, "2-drop token engine that triggers on every modal dilemma"),
    "Captain Jean-Luc Picard": (0.068, "Top-tier Mythic bomb: 3-mana unconditional +1/+1 team anthem that creates a 1/1 Officer on every creature/Spacecraft cast"),
    "Command Decision": (0.008, "Modal 2-mana removal/cycling spell that triggers Face a dilemma and is never dead"),
    "Command Subadjunct": (0.010, "1-mana 2/1 Borg that grants artifact creatures vigilance to attack and Station"),
    "Conquest Mission": (0.010, "Creates a 1/1 Borg token immediately on every nontoken artifact ETB"),
    "Deanna Troi, Psychic Counsel": (0.050, "2-mana 2/2 Grumgully/Unicorn that puts a +1/+1 counter on every other creature entering"),
    "DOT-7 Repair Squad": (0.012, "Common 4-mana 3/3 flying artifact creature that untaps creatures"),
    "Dr. Beverly Crusher": (0.020, "3-mana Pridemate that doubles lifegain triggers in WB/GW"),
    "Dr. McCoy, Surly Surgeon": (0.014, "Puts a +1/+1 counter on ETB and grants counter creatures regeneration-like protection"),
    "Emergency Medical Hologram": (0.008, "1-mana effective 1/2 artifact creature + 1 life + premier Beam me up bounce target"),
    "Excalbian Gamemaster": (0.014, "6-mana 6/6 that creates 3-4 Officer tokens on ETB and distributes +1/+1 counters on death"),
    "Federation Field Medic": (0.012, "Common 4-drop that is either a 4/3 vigilance or a 3/2 vigilance that draws a card"),
    "Federation Resupply": (-0.022, "Narrow 2-mana sorcery tutor that costs a card without affecting the board"),
    "Free Borg Revolutionaries": (0.004, "5-mana 4/5 artifact creature with team-wide +1/+1 ETB pump and Basic landcycling {2}"),
    "Ghost of Illyria": (0.012, "4-mana 3/3 flyer that flickers any nonland permanent for ETB value"),
    "Hologram Janeway": (0.008, "Solid 3-mana 3/3 artifact creature base rate despite win-more power 8+ trigger"),
    "Malfunctioning Holodeck": (0.015, "Premium common 3-mana Oblivion Ring artifact with Ward {2}"),
    "The Measure of a Man": (-0.025, "Temporary 2-turn exile Saga that returns the threat on Chapter III"),
    "Miles O'Brien, Transporter Ops": (0.008, "3-mana 3/3 that draws a card on ETB self-bounce and on every Beam me up cast"),
    "The Needs of the Many": (-0.018, "Situational defensive trick requiring a creature sacrifice"),
    "Oberth-Class Starship": (0.010, "Common 2-mana Spacecraft that tutors a basic Plains on ETB and becomes a 2/2 flyer"),
    "Peace in Our Time": (-0.068, "Major Limited trap: 4-mana symmetrical fog that lets opponent draw cards first"),
    "Pet Sehlat": (0.002, "3-mana 4/3 that taps to Station Spacecraft for 4 charge counters immediately"),
    "Picard, Leading by Example": (0.004, "2-mana 2/2 that taps for {1} to pump another attacker by +2/+2 (or more with counters)"),
    "Primary Unimatrix Adjunct": (0.042, "4-mana 2/3 Borg lord that grants all artifact creatures (including itself) double strike"),
    "Pure Energy": (-0.010, "Sorcery-speed counter distribution risks tempo loss without creature-fallbacks"),
    "Rutherford, Upgraded Ensign": (0.014, "Digs 6 cards deep for an artifact in an artifact-dense format and buffs on Spacecraft cast"),
    "Saavik, Stoic Student": (0.006, "5-mana flash removal-on-a-stick scaling with non-Borg creature types"),
    "Set Phasers to . . .": (0.016, "Premium 1-mana modal instant that stuns a blocker or deals 3 damage to an attacker"),
    "Shields Up!": (0.003, "2-mana protection trick that leaves a permanent +1/+1 counter and protects creatures or Spacecraft"),
    "Shuttle Ace": (0.004, "5-mana 4/4 flyer on offense that gains 2 life whenever it attacks or taps to Station"),
    "Sulu, Space Swashbuckler": (0.012, "2-mana 2/2 first strike that draws a card when tapped on turns an Equipment/Spacecraft enters"),
    "Tactical Officer": (-0.004, "Aggressive 2-mana 2/1 Officer with conditional power 8+ upside"),
    "Test of Conviction": (0.024, "2-mana team-wide +1/+1 counter anthem with a card-draw fallback mode"),
    "There's Coffee in that Nebula": (-0.004, "2-mana selection spell that finds lands or early drops and recasts via Beam me up"),
    "Troubling Tribbles": (-0.032, "0/1 tokens cannot block and require external anthems or sacrifice outlets to impact board"),
    "Una Chin-Riley, Number One": (-0.008, "5-mana 4/5 vanilla body unless attacking with total power 8+"),
    "U.S.S. Enterprise-A, Free Spirit": (0.022, "Mythic bomb: creates three 1/1s on ETB with Ward {2} and becomes a 5/5 flying lifelink"),
    "U.S.S. Excelsior, Cutting Edge": (0.020, "Uncommon 4-mana Spacecraft that destroys a tapped creature on ETB and becomes a 4/4 flyer"),
    "Vexing Drone": (0.012, "Top white common 2-drop: 2/2 Borg artifact creature that taps a blocker on ETB"),

    # BLUE (TRK #46..90)
    "Anthropological Mission": (-0.018, "Slow 3-mana enchantment with no board impact"),
    "The Caretaker": (0.016, "6-mana 5/6 vigilance Ward {2} that casts free spells from opponent's deck on dilemma turns"),
    "Cetacean Ops": (0.002, "4-mana 3/3 Ward {1} scaler that also satisfies H.M.S. Bounty's Whale condition"),
    "Cloak and Phaser": (0.006, "Costs 5-6 mana in practice for two 2/2 bodies plus Draw 2 discard 1"),
    "Cloistered Telepath": (0.004, "2-mana 2/1 flash creature that loots on ETB and ambushes 1-toughness attackers"),
    "Consider the Prime Directive": (0.007, "Modal 3-mana Quench or tap+stun+draw cantrip that triggers Face a dilemma"),
    "Data, New Life": (0.022, "2-mana 0/1 artifact creature that grows +1/+1 on every draw step and card draw spell"),
    "Deep Space Scan": (-0.022, "Clunky 6-mana sorcery draw spell"),
    "Dominion Saboteur": (0.028, "5-mana non-legendary Clone for any artifact or creature that doubles its counters"),
    "Emergency Holographic Crew": (0.005, "4-mana 3/4 artifact creature that triggers Face a dilemma and recurs Spacecraft"),
    "Energize": (-0.004, "Sorcery-speed Unsummon that doubles as a repeatable tempo play via Beam me up"),
    "Ensign Pavel Chekov": (-0.012, "Low-impact 4-mana 1/5 that usually only nets a single land"),
    "Enterprise NX-01": (0.010, "4-mana Spacecraft that taps and stuns on ETB and becomes a 4/3 flying vigilance"),
    "Federation Probe": (0.005, "1-mana 1/1 flying artifact creature that triggers Face a dilemma on ETB and bounces cleanly to Beam me up"),
    "Guidance Failure": (0.012, "Premium 3-mana (vs attackers) instant tuck removal for creatures or Spacecraft"),
    "Gwyndala, the Progeny": (0.004, "Effective 2-mana 2/2 that leaves behind a +1/+1 Equipment"),
    "Highly Illogical": (-0.004, "Flexible 2-mana/4-mana counterspell, though reactive in Limited"),
    "H.M.S. Bounty, Unlikely Timeship": (0.004, "Tutors common Cetacean Ops or Humpback Whales on ETB and threatens extra turn"),
    "Hoshi Sato, Exolinguist": (0.003, "5-mana 3/3 that digs 3-5 cards deep for the best card on ETB"),
    "Humpback Whales": (0.005, "5/6 vigilance Ward {2} finisher with Basic landcycling {2} and tutor synergy with H.M.S. Bounty"),
    "Interstellar Effigy": (0.014, "Common 5-mana 4/3 Man-o'-War artifact creature"),
    "Jett Reno, Problem Solver": (0.006, "2-drop engine that grows on every modal dilemma and surveils/draws whenever it attacks or Stations"),
    "Kobayashi Maru Scenario": (-0.095, "Demonic Pact trap: 4th upkeep forces 'You lose the game' unless sacrificed or blinked"),
    "Lieutenant Commander Data": (0.024, "3-mana 2/2 that grows whenever EITHER player casts a noncreature spell and converts counters to cards"),
    "Lt. Cmdr. Geordi La Forge": (0.006, "3-mana 3/2 that doubles charge counters placed on Spacecraft"),
    "Make It So": (0.010, "Common 4-mana Divination that also creates a 1/1 Officer token and triggers a dilemma"),
    "Marooned": (0.002, "1-mana lockdown Aura that answers both creatures and stationed Spacecraft and prevents Beam me up"),
    "Melkotian Adjudicator": (0.008, "3-mana flying vigilance threat that rapidly grows to 3/2 or 4/3 in UB/WU modal decks"),
    "Mr. Spock, Science Officer": (0.030, "4-mana 2/4 that doubles all card draws including normal draw step draws"),
    "Nyota Uhura, Communications": (0.020, "3-mana 2/4 that draws cards whenever opponent casts a 2-drop or 3-drop"),
    "Open Communications": (-0.008, "1-mana sorcery cantrip with expensive Beam me up; mainly filler for UR spell-count decks"),
    "Romulan Bird-of-Prey": (0.006, "2-mana hexproof Spacecraft that self-charges on noncreature spells into a 4/4 flyer"),
    "Rules of Acquisition": (0.005, "Can donate a useless Food/Clue/1/1 token to draw 3 cards or steal an opponent's bomb"),
    "Scotty, Miracle Worker": (0.016, "3-mana 2/3 that grants all graveyard instants and sorceries Beam me up"),
    "Seven of Nine, Unassimilated": (0.018, "2-mana 0/4 artifact creature that loots every turn for free and draws pure cards at 5+ lands"),
    "Starfleet Spanner Jockey": (-0.004, "Defensive 2-mana 3/3 blocker that attacks in UR/UB spell-heavy decks"),
    "Subcommander T'Pol": (0.006, "3-mana 3/1 flash creature that stuns a blocker or stifles an ability and triggers a dilemma"),
    "Suurok-Class Science Vessel": (0.011, "Common 3-mana Spacecraft that draws a card on ETB and Stations at 5+ into a 3/2 flyer"),
    "Suus Mahna Practitioner": (-0.004, "3-mana 3/2 with a mana-intensive tap ability unless you control 3+ differently named lands"),
    "Test of Priorities": (0.002, "2-mana instant modal cantrip or protection/ETB flicker that triggers Face a dilemma"),
    "Those Old Scientists": (-0.004, "3-mana value Saga that loots 2 and recurs an artifact or spell on Chapter II"),
    "V'Ger, the Intruder": (0.002, "7-mana 5/7 flying artifact creature that draws a card on ETB and every attack while triggering Face a dilemma"),
    "Vulcan Mind-Meld": (0.004, "1-mana net +1 card advantage theft spell"),
    "Wesley Crusher, Starfleet Cadet": (0.008, "4-mana 2/2 flash Cryptic Command that chooses two modes on ETB"),
    "Zefram Cochrane, Warp Pioneer": (0.006, "3-mana 2/3 Mystic Forge for artifacts/artifact creatures that accelerates Spacecraft by +2 charge counters"),

    # BLACK (TRK #91..136)
    "Armus, Skin of Evil": (0.032, "Uncounterable instant-speed {2}{B} -4/-4 removal spell via cycling that is also a 7-mana 6/6 menace finisher"),
    "Assault Drone": (0.014, "Common 4-mana 3/4 Borg that gives all other artifact creatures +1/+0"),
    "Automated Warfare System": (0.012, "Common 3-mana 2/2 flying artifact creature with attack sacrifice-to-draw engine"),
    "Bat'leth": (0.014, "Uncommon 3-mana Equipment that gives -2/-2 on ETB and leaves a +2/+0 weapon"),
    "Battle-Scarred Survivalist": (0.003, "2-mana menace threat in BR/WB sacrifice decks that grows into a 3/3+ menace"),
    "Blastoneuron": (0.008, "4-mana 2/2 flyer that unearth-clones a MV<=3 creature on ETB and drains when cards leave graveyard"),
    "Cha'DIch Investigator": (0.010, "Common 5-drop providing 6/5 of stats across two bodies with haste"),
    "Collation Trimatrix": (0.018, "1-mana 1/1 Borg that taps 3 artifact creatures at instant speed to draw cards"),
    "Collective Drone": (0.014, "2-mana */2 Borg that is routinely a 2/2 or 3/2 in WB/UB artifact decks"),
    "Commander Raffi Musiker": (0.006, "3-mana 3/2 Thraben Inspector variant that Investigates and triggers Face a dilemma on ETB"),
    "D'Ghor Bloodletter": (0.022, "3-mana 2/2 modal -1/-1 sweeper (in a 26.3% 1-toughness set) or mutual edict"),
    "Death Is Irrelevant": (-0.012, "Powerful end-step reanimation but very expensive at 8 mana ({6}{B}{B})"),
    "The Destruction of Praxis": (0.025, "5-mana -X/-X board wipe in Black scaling with differently named lands"),
    "De-evolve": (-0.005, "2-mana debuff Aura that eventually exiles the creature once you reach Threshold"),
    "Eject the Warp Core": (0.018, "Top black common: 2-mana instant -2/-2 or kicked -6/-6"),
    "Elim Garak, Simple Tailor": (0.024, "3-mana 3/3 menace that drains 2 life on every modal dilemma"),
    "Fugitive Listener": (0.008, "Common 3-mana 3/1 that forces a discard on ETB and Stations for 3"),
    "Gabriel Lorca, Mirror Menace": (0.012, "5-mana 5/4 menace threat that recurs from your graveyard to hand for {B} whenever you face a dilemma"),
    "General Chang, Cold Warrior": (0.008, "1-mana 1/2 sacrifice outlet that puts +1/+1 counters on all Klingons"),
    "Georgiou, Terran Emperor": (-0.032, "Self-drains 3 life per turn and causes instant game loss if removed while at <=0 life"),
    "A Good Day to Die": (-0.008, "Situational 1-mana combat trick that replaces itself when trading up"),
    "Grand Nagus Zek": (0.016, "3-mana Treasure generator on creature deaths that grows with Treasures"),
    "Gul Dukat, Colonial Prefect": (0.008, "Heavy {4}{B}{B}{B} cost, but 6/6 menace that doubles life loss and drains"),
    "He's Dead, Jim": (0.015, "Unconditional 4-mana instant Murder + Surveil 1 at Common"),
    "Hive Mind Coprocessor": (0.004, "5-mana 4/4 Borg that turns every unblocked 1/1 Borg or flyer into a card draw"),
    "Impossible Choice": (0.012, "Modal 3-mana instant Night's Whisper or emergency creature destruction"),
    "In the Pale Moonlight": (0.010, "3-for-1 Saga: nontoken edict, Clue, and two 2/2 Romulans"),
    "Kai Winn, Faithless Leader": (-0.018, "Blindly naming a card in Limited is low-impact without prior hand knowledge"),
    "Khan, Vengeful Heart": (0.008, "4-mana 5/3 that sacrifices a token/artifact to draw 2 and can be cast from exile"),
    "Obsidian Order Operative": (-0.008, "Defensive 4-mana 2/5 body that requires heavy modal density to pressure opponent"),
    "Organic Avulsion Unit": (0.002, "3-mana 2/3 Borg that grants your artifact creature team menace"),
    "Overflow Submatrix": (0.032, "3-mana 3/3 Borg that replays an artifact creature from your graveyard every turn"),
    "Pattern Buffer Preservation": (-0.002, "2-mana card selection that fuels the graveyard and recasts cheaply via Beam me up"),
    "Relentless Drednok": (0.008, "Common 1-mana 1/1 lifelink artifact creature that exiles from graveyard to draw a card"),
    "Resistance Is Futile": (0.018, "2-mana modal Doom Blade for nonartifacts or Unearth for MV<=3 artifact creatures"),
    "Rogue Artificial Intelligence": (0.004, "Common 6-drop artifact creature with modal ETB (triggering Face a dilemma) and Basic landcycling {2}"),
    "Salt Vampire": (0.005, "2-mana 2/2 lifelink with Vendilion Clique hand disruption"),
    "Sela, Tal Shiar Paradox": (0.022, "4-mana 3/3 that exiles a card from your own graveyard on every attack to create a 2/2 Romulan"),
    "Storied Veteran": (0.002, "1-mana 2/1 recursive attacker that triggers BG graveyard-leaves and provides sacrifice fodder twice"),
    "Stray from History": (0.006, "Modal Soul Salvage (2-for-1 recursion) or targeted discard that triggers Face a dilemma and BG graveyard-leaves"),
    "Tal Shiar Operative": (0.009, "Common 2-drop 2/2 that triggers Face a dilemma on every single attack"),
    "Tell Me What You See": (-0.004, "2-mana repeatable punisher enchantment that bounces back to hand whenever you face a dilemma"),
    "Tenacious Drone": (0.010, "Common 2-drop 2/1 Borg that dies into a 1/1 Borg artifact token"),
    "Test of Loyalty": (-0.002, "1-mana modal cantrip or Raise Dead that triggers Face a dilemma and BG graveyard-leaves"),
    "U.S.S. Reliant, From Hell's Heart": (0.018, "4-mana Spacecraft that immediately reanimates a MV<=4 creature on ETB"),
    "Vengeance Mission": (-0.025, "3-mana do-nothing enchantment"),

    # RED (TRK #137..181)
    "Academy Instructor": (0.012, "3-mana Young Pyromancer that creates 1/1 Officers and scales with Officer tokens"),
    "Beckett Mariner, Impetuous Ensign": (0.002, "2-mana 2/1 first strike attacker that builds toward 2 face damage when attacking or Stationing"),
    "Boarding Action": (-0.006, "2-mana 2/1 haste token sorcery; playable in UR/BR spells and sacrifice decks via Beam me up"),
    "Brazen Bluff": (-0.035, "Narrow reactive punisher counterspell in Red"),
    "Cantankerous Captain": (-0.008, "Below-rate 3-mana 2/3 filler relying on total power 8+"),
    "Captain James T. Kirk": (0.012, "3-mana Hanweir Garrison/rummager that triggers on both ETB and attack"),
    "Captain Michael Burnham": (0.004, "Solid 3-mana 3/3 trample with Landfall reach"),
    "Captain's Tutelage": (-0.012, "Combat-trick Aura that still risks a 2-for-1 against instant-speed removal"),
    "D-7 Battle Cruiser": (0.024, "2-mana Spacecraft that needs only Station 3+ to create a 2/1 haste Klingon every combat"),
    "Dahar Master": (0.008, "Common 4-mana 4/3 that sacrifices a token/artifact to deal 2 damage to any target"),
    "Dominion Infiltration": (-0.028, "Triple-red temporary clone sorcery"),
    "Encounter the Unknown": (0.028, "3-mana instant Pyroclasm (in a 48.8% <=2 toughness set) or impulse draw 2"),
    "Erica Ortegas, Plucky Pilot": (-0.008, "4-mana 3/3 haste with minor temporary Landfall/mana trick"),
    "Exploratory Mission": (-0.015, "4-mana enchantment with no immediate board impact"),
    "Genesis Effect": (0.008, "Common 5-mana instant 6-damage removal that converts a basic land into a rainbow New Planet"),
    "Horta": (0.002, "6-mana 7/6 trample with Basic landcycling {2} floor"),
    "Infinite Diversity": (-0.125, "Uncastable 9-mana ({7}{R}{R}) build-around sorcery requiring graveyard spell + board types"),
    "Khaaaaaaaaaaaannn!": (0.022, "Premium scalable instant burn spell dealing 2*X damage with exile"),
    "Khan, Despot by Design": (0.024, "5-mana 5/4 trample with one-sided 1-damage sweeper in a set with 26.3% 1-toughness creatures"),
    "La'An Noonien-Singh, Security": (-0.010, "Fragile 4-mana 4/3 with conditional power 8+ trigger"),
    "Lwaxana Troi, Charming Guest": (0.005, "2-mana 2/2 that doubles triggered abilities of another creature"),
    "Munitions Enthusiast": (0.006, "Uncommon 2-mana 3/2 trample that pings any target for 1 on death"),
    "The Naked Time": (-0.006, "Temporary 2-mana ramp/haste Saga that sacrifices the land back on Chapter III"),
    "Pah-Wraith": (0.008, "7-mana 4/4 first strike haste that pumps team by Strange new worlds count"),
    "Perils of the Past": (0.010, "Modal 2-mana instant combat trick or Murder for any Artifact Creature/Spacecraft"),
    "Planetary Bombardment": (0.010, "1-mana 2-damage (or 4-damage morbid) removal spell"),
    "Planetary Patrol": (-0.002, "Standard 2-mana 2/2 aggro bear that attacks as a 3/3 on landfall turns"),
    "Plasma Cascade": (0.012, "3-mana instant 3 damage to any target + optional rummage"),
    "Q, Stacking the Deck": (0.008, "4-mana 4/4 that exiles a spell and lets you pitch extra lands to cast copies of it"),
    "Quark, Humble Proprietor": (-0.055, "3-mana 1/2 that locks your Treasure tokens while on the battlefield"),
    "Ravenous Cosmozoan": (0.004, "3/3 flying haste threat that routinely costs 3-4 mana in UR/BR decks"),
    "Reckless Impulse": (0.014, "Proven 2-mana impulse-draw-2 staple"),
    "Red Alert!": (-0.012, "Situational team-wide +2/+0 Trumpet Blast"),
    "Sarcophagus, Ship of the Dead": (0.012, "3-mana unconditional +1/+0 team anthem that charges on creature deaths into a 4/4 flyer"),
    "Savvy Profiteer": (0.002, "Red 2-drop mana dork that accelerates into 4-drops and creates Treasures late"),
    "Silicate Surveyor": (0.004, "5-mana 4/4 reach that replaces itself via 2-turn impulse draw on ETB"),
    "Starfleet Security": (0.002, "1-mana 1/1 haste Officer that generates a Clue token when trading in combat or sacrificed on your turn"),
    "Tal Shiar Analyst": (-0.003, "3-mana 3/2 repeatable face-pinger in UR spells"),
    "Targ Wrangler": (-0.015, "Forces you to sacrifice a nonland permanent at end step whenever it attacks"),
    "Tasha Yar, Out of Time": (0.002, "2-mana 2/1 haste that impulse-draws on death and unearths for a second impulse draw"),
    "Test of Identity": (-0.006, "2-mana instant Tormenting Voice with Klingon double-strike upside"),
    "Transporter Malfunction": (0.008, "2-mana 3 damage to any target with late-game Beam me up"),
    "Will Riker, First in the Field": (0.022, "4-mana 4/3 that puts two untapped lands onto the battlefield every combat, triggering Landfall twice"),
    "Worf, Intelligence Contractor": (0.005, "3-mana 3/2 that threatens an opponent's MV<=3 creature on ETB"),
    "The Wrath of Khan": (0.025, "3-mana Saga that deals 4 damage + threatens on Ch I, rummages on Ch II, and edicts on Ch III"),

    # GREEN (TRK #182..225)
    "Adversary by Algorithm": (0.008, "3-mana 3/3 artifact creature token with Beam me up {3}{G} for a second 3/3"),
    "Amok Time": (0.020, "3-mana 2-for-1 Saga: Fight on Ch I, gain 3 life on Ch II, Regrowth creature on Ch III"),
    "Astute Terraformer": (0.001, "Solid 2-mana 2/2 curve filler with a cheap late-game mana sink in Planet decks"),
    "Branching Timeline": (-0.055, "8-mana sorcery that only looks at 4-5 cards in Limited"),
    "Captain Jonathan Archer": (0.025, "3-mana 3/3 + 1/1 Dog (3 creature types alone) that taps for {G} to give +3/+3 to +5/+5 every turn"),
    "Captain Kathryn Janeway": (0.038, "4-mana Mythic that lets every creature you play explore on ETB"),
    "Captain Pike, Fullness of Time": (0.032, "3-mana Courser of Kruphix that plays lands off the top of your library"),
    "Cold-Blooded Crew": (0.008, "Common 6-drop providing 6/6 of stats (with two +1/+1 counters) plus Basic landcycling {2}"),
    "Common Goal": (0.012, "2-mana one-sided bite spell that can use one or two creatures"),
    "Culber and Stamets, Star-Crossed": (0.008, "4-mana 4/5 (or 2-mana 3/4 with mana rebate) that also triggers Face a dilemma on ETB"),
    "Dathon and Picard at El-Adrel": (-0.010, "3-mana combat trick / edict that triggers Face a dilemma; edict mode can hit minor artifacts"),
    "David and Carol Marcus, Creators": (0.012, "3-mana 1/3 Farhaven Elf that creates a permanent New Planet rainbow land token on ETB"),
    "Dominion Assault Force": (0.008, "Common 4-mana 4/4 vigilance that exiles a graveyard card or surveils 1 and triggers a dilemma"),
    "Dominion Supervisor": (0.008, "Common 3-drop providing 4/2 of stats and granting trample to all counter creatures"),
    "Dr. Joseph M'Benga": (0.028, "4-mana 3/3 that doubles counters placed on creatures and gives counter creatures hexproof"),
    "Dr. Leonard \"Bones\" McCoy": (0.030, "3-mana 3/2 flash Eternal Witness for any non-Doctor permanent or saves a dying creature"),
    "Final Frontier": (0.010, "Common 3-mana ramp that tutors any basic or Planet dual land and has Beam me up {1}{G}"),
    "First Contact": (0.004, "Instant-speed permanent selection that fuels the graveyard and finds creatures, Spacecraft, or Planet lands"),
    "Genesis Device": (0.005, "3-mana ramp artifact that tutors a basic or Planet land on ETB"),
    "Holo-Engineer": (0.035, "Top-5 Rare bomb: 4-mana for two 3/3s (6/6 total) that turns all future tokens into 3/3s"),
    "Hologram Stalker": (0.012, "6-mana 2/2 Eternal Witness that creates a 3/3 token on ETB and whenever cards leave graveyard"),
    "I'm a Doctor, Not a . . .": (0.002, "Efficient 1-mana ambush combat trick providing +2/+2 on the turn cast and a permanent +1/+1 counter"),
    "Jadzia Dax, Living Legacy": (0.035, "2-mana scaling mana dork that grows +1/+1 every turn and adds {G} equal to her counters"),
    "Jake Sisko, War Correspondent": (0.004, "2-mana 2/2 that gains life on every land drop and Investigates with Final Frontier/Voyager"),
    "MACO Strike Team": (0.002, "1-mana 2/1 Savannah Lions in Green that creates a 1/1 Officer whenever your creatures are targeted"),
    "Moopsy": (0.010, "Common 1-mana 1/1 deathtouch with on-demand reach"),
    "Mugato": (0.028, "5-mana 6/6 with reach, vigilance, and deathtouch at Uncommon"),
    "Neelix, Moral Support": (0.010, "2-mana 1/1 that creates a Food on ETB and grows whenever you gain life (including Planet lands)"),
    "Photosynthetic Physician": (0.005, "5-mana 5/5 reach stabilizer that exiles from graveyard for {G} to create a 1/1 token and trigger BG payoffs"),
    "Rok-Tahk, Budding Scientist": (0.005, "6-mana 5/5 + 0/1 indestructible token + team mana ability"),
    "Saru, Unthreatened": (0.004, "3-mana 3/2 flash creature that blanks opponent removal spells for a 2-for-1"),
    "Saurian Explorer": (0.006, "4-mana 4/3 that attacks as a 6/5 on landfall turns and creates New Planet tokens in the late game"),
    "Species 8472": (0.004, "6-mana 7/5 that gains 3-5 life on ETB and is unblockable against WB/WU/UB artifact creatures"),
    "Stable Wormhole": (0.035, "3-mana enchantment that creates a permanent New Planet rainbow land token whenever you attack"),
    "Support Mission": (0.015, "3-mana enchantment that puts a +1/+1 counter on every creature entering under your control"),
    "Sylvia Tilly, Joy of Discovery": (0.008, "Surveils 2 on ETB and exiles a card from graveyard every combat for {1} to add a +1/+1 counter"),
    "Syndicate Liquidators": (0.025, "5-mana 4/4 trample that puts TWO +1/+1 counters on a creature on every Landfall"),
    "Temporal Conqueror": (0.015, "Top green common 3-drop: 2/3 that mills 3 and either draws a land (triggering BG) or makes a Treasure"),
    "Test of Courage": (-0.005, "1-mana modal Disenchant for MV<=3 artifacts/enchantments or graveyard hate that triggers Face a dilemma"),
    "U.S.S. Discovery, at Black Alert": (0.055, "Top-5 Rare bomb: 5-mana 5/5 Spacecraft that Discovers 5 on ETB and on every attack"),
    "U.S.S. Voyager, Lone Wanderer": (0.025, "4-mana Spacecraft that ramps TWO lands from library onto the battlefield on ETB"),
    "Warp Speed": (-0.002, "3-mana Civic Wayfinder enchantment that lets Planet duals enter untapped and cashes in for 2 cards late"),
    "Xenobotanist": (0.014, "Common 2-mana 1/3 mana dork that taps for any color of mana"),
    "Xindi Surveyors": (0.012, "3-mana 3/2 that gains a permanent +1/+1 counter on every land drop (4/3 on Turn 4, 5/4 on Turn 5)"),

    # MULTICOLOR, COLORLESS & LANDS (TRK #226..311)
    "All Good Things...": (0.005, "5-mana GU Saga that draws 6 cards and exiles 3 across 3 turns"),
    "Borg Cube": (0.030, "Mythic WB Spacecraft with Affinity for artifact creatures that exiles and assimilates creatures on ETB/attack"),
    "The City on the Edge of Forever": (0.005, "5-mana WB Saga that phases out the board behind a 0/1 lock token while drawing 2 and gaining 5 life"),
    "Cleveland Booker, Empath": (0.012, "Hybrid 4-drop that creates a 2/2 Worm token on every Landfall"),
    "Constable Odo": (0.012, "Hybrid 5-mana 4/3 that either stuns a blocker for 2 turns or Investigates twice"),
    "Dal R'El, Augment Orphan": (0.008, "3-mana 3/3 with Landfall mana and land-sacrifice card draw"),
    "Daniels, Temporal Agent": (0.008, "Hybrid 4-mana Gravedigger that triggers BG graveyard-leaves on ETB and when tucking itself"),
    "Dathon, His Heart Open": (-0.022, "Narrow 3-color enchantment build-around"),
    "The Doctor, EMH": (0.025, "Hybrid 5-drop that enters as a 5/5 and proliferates +1/+1 counters across your team"),
    "The Duras Sisters": (0.012, "Hybrid 2-drop 3/2 that leaves behind a 2/1 haste Klingon when it dies"),
    "D'Vana Tendi, Optimistic Ensign": (0.004, "Hybrid 3-mana 2/4 ramp/untap engine that untaps lands or creatures that just Stationed"),
    "Gowron, Eager for Battle": (0.018, "4-mana 3/3 that grows on creature deaths and sacrifices creatures/Klingons for trample + card draw"),
    "Gray and Adira, Forever Joined": (0.006, "Hybrid 4-drop that clones your best ETB/attack creature whenever you cast a noncreature spell precombat"),
    "Hemmer, Fixing What Is Broken": (0.010, "2-mana 2/3 that protects Spacecraft and loots whenever tapped (including via Station)"),
    "Hit It!": (-0.002, "3-mana enchantment that ramps a land from top 5 and grants team-wide haste"),
    "The Inner Light": (0.005, "5-mana GW Saga that reanimates a green/white creature on Chapter I"),
    "Kira Nerys, Firebrand": (0.024, "3-mana 4/3 trample that gives entering creatures +1/+1 and haste during combat"),
    "Kirk, Commanding Officer": (0.008, "3-mana 3/3 with impulse draw on power 8+ attacks and built-in artifact/enchantment destruction"),
    "Live Long and Prosper": (-0.032, "Slow 7-mana sorcery with zero board impact"),
    "Locutus, the One Who Speaks": (0.028, "2-mana 2/2 Borg lord that gives all artifact creatures +1/+1 OR cycles for {W}{B} to deal 2 and gain 2"),
    "Lore, Positronic Provocateur": (0.004, "Hybrid 3-mana 2/3 artifact creature that triggers Face a dilemma on ETB"),
    "L'Rell, Fierce Visionary": (0.022, "3-mana 1/4 that creates a free 2/1 haste Klingon every combat and provides a sacrifice outlet"),
    "Mirror Universe Intrusion": (0.006, "3-mana Sorcery that creates a menace token copy of an opponent's creature (or their whole board kicked)"),
    "Nog, Ambitious Cadet": (0.032, "2-mana 2/2 menace that levels up for {1} into a 3/4 menace that surveils 2 whenever tapped"),
    "Phlox, Unorthodox Doctor": (0.022, "4-mana 3/4 trample that draws a card whenever a counter creature deals combat damage"),
    "Sarek, Federation Ambassador": (-0.055, "3-color 0/5 that requires 5 non-Borg creature types just to draw 1 card"),
    "Seven of Nine": (0.038, "4-mana 2/4 artifact creature in UB that draws a card on every modal dilemma (including Tal Shiar Operative)"),
    "Spock, Always Your Friend": (0.008, "2-mana 2/2 that digs 4 cards deep for a noncreature spell"),
    "The Traveler": (0.022, "2-mana ({G}{U}) */3 with reach and vigilance that is a 3/3 or 4/3 by Turn 3-4"),
    "Vash, Freelance Archaeologist": (0.018, "3-mana 2/4 that surveils every upkeep and draws a card whenever a card leaves your graveyard"),
    "Worf, Honor-Bound Warrior": (-0.008, "Slow 2-mana 1/1 requiring tapping two other permanents at sorcery speed to grow"),
    "Year of Hell": (-0.048, "Symmetrical draw/burn Saga where opponent gets the first mana to use the extra card"),
    "Zero, Incorporeal Sensate": (0.010, "Hybrid 3-mana 2/2 lifelink Elite Spellbinder"),
    "Class-F Shuttlecraft": (-0.004, "3-mana Manalith that can eventually Station into a 2/4 flying vigilance threat"),
    "Combadge": (-0.015, "Clunky Equipment requiring two equipped creatures to grant unblockable"),
    "Crystalline Entity": (-0.012, "8-mana cost tempers an otherwise devastating one-sided nonartifact board wipe on a 7/7 flyer"),
    "Cybernetic Specialist": (0.002, "Colorless 2-drop artifact creature that surveils on ETB and repeatedly triggers BG graveyard-leaves"),
    "Deep Space 9": (-0.115, "9-mana Spacecraft clunker that is uncastable in most Limited games"),
    "Engineered Explosives": (0.015, "Flexible ratchet bomb that wipes opponent tokens at X=0 or 2/3-drops"),
    "The Holodeck": (-0.045, "7-mana do-nothing artifact"),
    "Isolinear Processor": (-0.010, "2-mana artifact mana rock/looter with zero board presence"),
    "Replicator": (0.008, "2-mana artifact that makes a Food and copies any token (including New Planet lands and 3/3 Illusions) for {2}"),
    "Sickbay": (-0.014, "Slow 5-mana noncreature artifact that requires existing board presence"),
    "Tea, Earl Grey, Hot": (0.001, "2-mana cantrip Food artifact that triggers Conquest Mission, lifegain, and modal dilemma payoffs"),
    "Tricorder": (-0.008, "Cheap +1/+1 Equipment that surveils on attack but does not create a body"),
    "Type-2 Phaser": (0.006, "2-mana Equipment that taps and stuns a blocker on ETB"),
    "U.S.S. Cerritos, Following Up": (0.012, "Colorless 5-mana Spacecraft that creates two 1/1 Officers on ETB and becomes a 4/4 flyer"),
    "U.S.S. Enterprise-D, Galaxy-Class": (0.055, "Colorless 3-mana Spacecraft that impulse-draws a card every turn it is stationed and becomes a 4/5 flying vigilance"),
    "Warp Core": (-0.005, "2-mana Traveler's Amulet that fixes mana and helps Station high-threshold Spacecraft"),
    "Artificial Anomaly": (0.002, "Colorless utility land that drains 2 life late while replacing itself with a rainbow New Planet land"),
    "Federation Outpost": (0.002, "Mono-white rare land that enters untapped on Turn 3+ and triggers Face a dilemma (gain 2 life or surveil 1)"),
    "Gaseous Anomaly": (0.002, "Colorless utility land that draws 2 cards late while replacing itself with a rainbow New Planet land"),
    "Ni'Var Science Academy": (0.002, "Mono-blue rare land that enters untapped on Turn 3+ and triggers Face a dilemma (grant flying or surveil 1)"),
    "Q Continuum": (0.003, "Mono-green rare land that enters untapped on Turn 3+ and triggers Face a dilemma (+1/+1 vigilance or surveil 1)"),
    "Quark's Bar": (0.001, "Mono-red rare land that enters untapped on Turn 3+ and triggers Face a dilemma (1 damage or surveil 1)"),
    "Spatial Anomaly": (0.003, "Colorless utility land that deals 2 damage to a creature late while replacing itself with a rainbow New Planet land"),
    "Starfleet Academy": (-0.032, "Colorless-only land with a prohibitive 6-mana activation"),
    "Subspace Anomaly": (0.004, "Colorless utility land that creates two 1/1 Officers late while replacing itself with a rainbow New Planet land"),
    "Temporal Anomaly": (0.002, "Colorless utility land that mills 4 and draws a permanent late while replacing itself with a rainbow New Planet land"),
    "Unexplored Territory": (-0.003, "Evolving Wilds variant that taps for {C} immediately or fetches a basic land for Strange new worlds and Landfall"),
    "The Unicomplex": (0.003, "Mono-black rare land that enters untapped on Turn 3+ and triggers Face a dilemma (topdeck creature from graveyard or surveil 1)"),
}

# Explicit evaluated win rates & rationales for the 30 Stardates (SDS) bonus sheet cards
SDS_EVALUATIONS: dict[str, tuple[float, str]] = {
    "Mikaeus, the Lunarch": (0.6180, "Scalable white +1/+1 counter lord with massive synergy alongside TRK counter doublers"),
    "Flare of Fortitude": (0.5640, "Free or 4-mana team hexproof + indestructible combat blowout"),
    "Kenrith, the Returned King": (0.6015, "5-mana 5/5 5-color mana sink; realized 60.02% GIH WR across 1,656 games in FIN"),
    "Sisay, Weatherlight Captain": (0.5240, "3-mana 2/2 requiring WUBRG activation; mostly a vanilla/3/3 in 2-color Limited"),
    "Thada Adel, Acquisitor": (0.5680, "3-mana 2/2 Islandwalk that steals Spacecraft and artifact creatures from opponent's deck"),
    "Empress Galina": (0.5860, "Steals any legendary permanent repeatedly in a set with 104 legendary creatures and Spacecraft"),
    "Arcum Dagsson": (0.5120, "4-mana 2/2 build-around that sacrifices artifact creatures only for noncreature artifacts"),
    "Bruvac the Grandiloquent": (0.4780, "3-mana 1/4 opponent-mill doubler with zero mill win-condition support in TRK"),
    "Decree of Pain": (0.6150, "Uncounterable 5-mana -2/-2 cycling sweeper + draw a card (or 8-mana wrath + draw all)"),
    "Drivnod, Carnage Dominus": (0.5780, "5-mana 8/3 that doubles death triggers and gains an indestructible counter for 4 life"),
    "Sheoldred, the Apocalypse": (0.6460, "Premier 4-mana 4/5 deathtouch bomb that swings 4 life per turn cycle from normal draws alone"),
    "Torbran, Thane of Red Fell": (0.5520, "Heavy {1}{R}{R}{R} mana cost restricts it to mono-red/heavy-red decks"),
    "Loot, Exuberant Explorer": (0.5720, "3-mana 1/4 extra-land ramp + 6-mana creature-to-battlefield engine"),
    "Rishkar, Peema Renegade": (0.6040, "3-mana 4/4 across bodies that turns all counter creatures in TRK into mana dorks"),
    "Bristly Bill, Spine Sower": (0.6380, "Elite 2-mana Landfall +1/+1 counter engine with Planet/Voyager/Final Frontier support"),
    "Sen Triplets": (0.5480, "3-color 5-mana 3/3 artifact creature; strong upkeep Mindslaver-lite if it survives"),
    "Tatyova, Benthic Druid": (0.5940, "5-mana 3/3 Landfall draw+lifegain engine in GU Strange new worlds ramp"),
    "Hull Breach": (0.5640, "2-mana sorcery that destroys an artifact (26.8% of creatures + 17 Spacecraft) and/or enchantment"),
    "Isshin, Two Heavens as One": (0.5410, "3-color 3/4 attack-trigger doubler; realized 53.92% GIH WR in FIN"),
    "Ertai Resurrected": (0.6080, "4-mana 3/2 flash Murder/Counterspell that also triggers UB Face a dilemma payoffs"),
    "Jhoira, Weatherlight Captain": (0.5920, "4-mana 3/3 that draws a card on every historic spell in a set with 104 Legends, 17 Spacecraft, and 50+ Artifact Creatures"),
    "Captain Sisay": (0.5620, "4-mana 2/2 that tutors a legendary card every turn in a 104-legend set; realized 54.95% in TLA"),
    "Riku of Many Paths": (0.5960, "3-color 3/3 that generates cards, counters, and flyers on every modal spell (58 modal cards in TRK)"),
    "Prosper, Tome-Bound": (0.5880, "4-mana 1/4 deathtouch that impulse-draws a card every end step and makes Treasures"),
    "Stella Lee, Wild Card": (0.5580, "3-mana 2/4 that impulse-draws on your second spell each turn"),
    "Mogis, God of Slaughter": (0.5820, "4-mana indestructible God that deals 2 damage or forces a creature sacrifice every upkeep"),
    "Minamo, School at Water's Edge": (0.5620, "Untapped blue source that untaps legendary creatures and stationed Legendary Spacecraft"),
    "Talon Gates of Madara": (0.5680, "Mana-fixing land that phases out a creature on ETB"),
    "Phyrexian Tower": (0.5420, "Colorless land that sacrifices a creature for {B}{B}"),
    "Shizo, Death's Storehouse": (0.5580, "Untapped black source that grants fear to your legendary creatures"),
}


def compute_env_adjustment(row: pd.Series) -> float:
    colors = str(row["colors"] or "") if pd.notna(row["colors"]) else ""
    tl = str(row["type_line"] or "")
    if "Land" in tl:
        ot = str(row["oracle_text"] or "")
        produced = {c for c in "WUBRG" if f"{{{c}}}" in ot}
        if len(produced) == 2:
            return 0.5 * PAIR_ENV_ADJ.get(frozenset(produced), 0.0)
        if len(produced) == 1:
            c = next(iter(produced))
            return 0.5 * MONO_COLOR_ENV_ADJ.get(c, 0.0)
        return 0.0
    if len(colors) == 1:
        return MONO_COLOR_ENV_ADJ.get(colors, 0.0)
    if len(colors) == 2:
        return PAIR_ENV_ADJ.get(frozenset(colors), 0.0)
    if len(colors) >= 3:
        return -0.0040
    return MONO_COLOR_ENV_ADJ[""]


def main() -> None:
    df = pd.read_csv("/tmp/trk_ml_predictions.csv")

    raw_combined = np.zeros(len(df), dtype=np.float64)
    env_adjs = []
    ctx_adjs = []
    rationales = []
    is_special_land = np.zeros(len(df), dtype=bool)

    for i, row in df.iterrows():
        card = str(row["card"])
        scode = str(row["set"])
        ml_raw = float(row["pred_ml_stacked"])
        tl = str(row["type_line"] or "")
        env_adj = compute_env_adjustment(row)

        if scode == "SDS" and card in SDS_EVALUATIONS:
            sds_target, reason = SDS_EVALUATIONS[card]
            ml_bs_cal = ml_raw - 0.0220
            pred = 0.72 * sds_target + 0.28 * ml_bs_cal
            ctx_adj = pred - ml_raw - env_adj
            raw_combined[i] = pred
            env_adjs.append(round(env_adj, 4))
            ctx_adjs.append(round(ctx_adj, 4))
            rationales.append(reason)
            continue

        # Special handling for the 10 Common Planet duals
        if "Land — Planet" in tl:
            is_special_land[i] = True
            ctx_adj = 0.0025
            raw_combined[i] = 0.5610 + env_adj + ctx_adj
            reason = "Common Planet dual land: gains 1 life on ETB (MSH cycle avg 56.5%) and adds a distinct land name for Strange new worlds"
            env_adjs.append(round(env_adj, 4))
            ctx_adjs.append(round(ctx_adj, 4))
            rationales.append(reason)
            continue

        # Special handling for the 10 Rare Shock Lands
        if pd.notna(row["reprint_hist_wr"]) and "Basic Land" not in tl and scode == "TRK":
            is_special_land[i] = True
            hist_wr = float(row["reprint_hist_wr"])
            ctx_adj = 0.0020
            raw_combined[i] = 0.55 * hist_wr + 0.45 * 0.5520 + env_adj + ctx_adj
            reason = f"Shock land reprint (historical 17Lands WR {hist_wr*100:.1f}%); enters untapped for 2 life and adds a distinct land name"
            env_adjs.append(round(env_adj, 4))
            ctx_adjs.append(round(ctx_adj, 4))
            rationales.append(reason)
            continue

        # Basic lands
        if "Basic Land" in tl:
            is_special_land[i] = True
            col_map = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}
            c = col_map.get(card, "")
            env_adj = MONO_COLOR_ENV_ADJ.get(c, 0.0)
            ctx_adj = 0.0
            raw_combined[i] = 0.5580 + env_adj
            reason = f"Basic land; reflects {c} base color win rate in TRK and 10-set 17Lands historical baseline"
            env_adjs.append(round(env_adj, 4))
            ctx_adjs.append(round(ctx_adj, 4))
            rationales.append(reason)
            continue

        ctx_adj, reason = TRK_CARD_ADJUSTMENTS.get(
            card, (0.0, "Standard rate aligned with multi-model ensemble and archetype context")
        )
        raw_combined[i] = ml_raw + env_adj + ctx_adj
        env_adjs.append(round(env_adj, 4))
        ctx_adjs.append(round(ctx_adj, 4))
        rationales.append(reason)

    # Now calibrate each non-special-land rarity stratum in TRK to match the 10-set empirical mean & spread
    final_preds = raw_combined.copy()
    for rar, (target_mean, target_std) in EMPIRICAL_RARITY_TARGETS.items():
        mask = (df["set"].to_numpy() == "TRK") & (df["rarity"].to_numpy() == rar) & (~is_special_land)
        vals = raw_combined[mask]
        cur_mean = float(vals.mean())
        cur_std = float(vals.std()) + 1e-8
        # Blend 75% toward empirical target_std so card-specific magnitudes remain faithful
        eff_scale = 0.75 * (target_std / cur_std) + 0.25 * 1.0
        cal_vals = target_mean + eff_scale * (vals - cur_mean)
        final_preds[mask] = np.clip(cal_vals, 0.410, 0.680)

    final_preds = np.round(final_preds, 4)
    df["env_adjustment"] = env_adjs
    df["context_adjustment"] = ctx_adjs
    df["predicted_gih_wr"] = final_preds
    df["predicted_gih_pct"] = [f"{p*100:.2f}%" for p in final_preds]
    df["evaluation_rationale"] = rationales

    df.to_csv("/tmp/trk_final_predictions.csv", index=False)
    trk_mask = df["set"] == "TRK"
    print(f"Generated final predictions for {len(df)} cards (TRK={int(trk_mask.sum())}, SDS={int((~trk_mask).sum())})")

    print("\n=== Final Predictions Summary by Set & Rarity ===")
    for scode in ["TRK", "SDS"]:
        sub = df[df["set"] == scode]
        print(
            f"Set {scode} (n={len(sub)}): mean={sub['predicted_gih_wr'].mean()*100:.2f}% | std={sub['predicted_gih_wr'].std()*100:.2f}% | min={sub['predicted_gih_wr'].min()*100:.2f}% | max={sub['predicted_gih_wr'].max()*100:.2f}%"
        )
        if scode == "TRK":
            for rar in ["common", "uncommon", "rare", "mythic"]:
                rsub = sub[sub["rarity"] == rar]
                print(
                    f"  {rar:8s} (n={len(rsub):3d}): mean={rsub['predicted_gih_wr'].mean()*100:.2f}% | std={rsub['predicted_gih_wr'].std()*100:.2f}% | min={rsub['predicted_gih_wr'].min()*100:.2f}% | max={rsub['predicted_gih_wr'].max()*100:.2f}%"
                )


if __name__ == "__main__":
    main()
