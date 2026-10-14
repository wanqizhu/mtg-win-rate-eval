# Reality Fracture (`FRA`) 17Lands Games-in-Hand (GiH) Win Rate Predictions & Methodology Report

## 1. Executive Summary & Environment Overview

This report presents the complete 17Lands **Games in Hand (`GiH`) Win Rate** predictions for **Reality Fracture (`FRA`)**, covering all **285 unique main-set cards** (`CN 1–194` core Play Booster slots, `CN 195–280` the 86-card alternate-reality Legendary Sheet, and `CN 281–290` the 5 Basic Lands) as well as all **10 Reality Fracture Special Guests (`SPG`)** (`295` total cards).

To avoid evaluating cards in a vacuum or relying on a naive rarity/mana-value heuristic, we built a four-stage empirical pipeline trained and validated across all **10 historical 17Lands sets** (`HOB`, `MSH`, `SOS`, `TMT`, `ECL`, `TLA`, `OM1`, `EOE`, `FIN`, `TDM`; `2,935` total card-set records; `2,756` cards with `GiH >= 200`, including `2,512` main-set cards):

1. **Vectorized Historical 17Lands & Scryfall Feature Pipeline (`src/extract_historical.py`, `src/build_card_catalog.py`)**:
   - Extracted exact binary `GiH` win rates (`opening_hand_<Card> > 0 OR drawn_<Card> > 0`, matching `scripts/demo_winrate.py`), copy-weighted `GiH` win rates, Opening Hand (`OH`) win rates, Games Drawn (`GD`) win rates, Games Played (`GP`) rates, Games Not Seen (`GNS`) win rates, Improvement When Drawn (`IWD = GiH_WR - GNS_WR`), and 10 two-color archetype breakdowns across all 10 historical sets.
   - Achieved a **100% (`2,935 / 2,935`)** join between 17Lands card names and Scryfall card objects across all 10 sets by resolving `OM1` (`Through the Omenpaths`) `printed_name` mappings, `SOS` `layout: "prepare"` front-face names, and bonus-sheet reprints (`SPG`, `EOS`, `STA`, `MAR`, `FCA`, `PZA`, `TLE`).
2. **Environment-Contextual Combat & Removal Simulation (`src/env_simulation.py`)**:
   - Simulated rarity-weighted (`Common=6.0`, `Uncommon=2.2`, `Rare=0.6`, `Mythic=0.2`) 1v1 combat exchanges for every creature in `FRA` against the entire `FRA` creature pool (and same/adjacent-CMC curve peers), accounting for flying, reach, first/double strike, deathtouch, lifelink, defender, toughness-damage assignment, and built-in cost reduction.
   - Computed each creature's **Removal Survival Score** against `FRA`'s weighted removal suite (`Last Gasp` `-3/-3`, `No Admittance` `3` damage, `Wrath of the Bloodmane` `4` damage, `Fulminous Forte` `5` damage / `1` damage sweeper, `Memory Trap` exile, `Extended Absence` exile, `Multiply by Zero` `0/0`, `Compel Brutality` bite) and each removal spell's **Creature Kill Share** across the `FRA` toughness distribution.
3. **Multi-Task Stacked Leave-One-Set-Out Cross-Validation Ensemble (`src/featurize_and_model.py`)**:
   - Engineered **139 domain-specific features** plus **2 out-of-fold auxiliary Stage-1 predictions** (`pred_iwd` and `pred_gp_rate`) and stacked **8 diverse regressors** (`LightGBM d=15`, `LightGBM d=8`, `HistGradientBoosting`, `GradientBoosting`, `RandomForest`, `ExtraTrees`, `Ridge`, `ElasticNet`).
   - Across strict **Leave-One-Set-Out Cross-Validation (LOSO-CV)** on all 10 historical sets (`n = 2,512` main-set cards), our stacked ensemble more than doubles the correlation of a Rarity + CMC + Type baseline (`Pearson r: 0.2160 -> 0.5056`, `Spearman rho: 0.2401 -> 0.4765`, `MAE: 0.0290 -> 0.0259`, `RMSE: 0.0398 -> 0.0344`).
4. **Format-Specific Mechanics & Historical-Twin Calibration (`src/generate_final_predictions.py`)**:
   - Anchored exact reprints (`Last Gasp` `0.5770` in `SOS`; `Blazing Crescendo` `0.5292` in `MSH`) and direct `SOS` functional cycles (Allied Rare Slowlands `0.5704` mean in `SOS`; `Hexhaven Dueling Arena` matching `SOS`'s `Skycoach Waypoint` `0.5166`).
   - Performed a card-by-card audit of all 295 cards to capture non-linear mechanics (`Empower Jace N` + `Way of the ...` loyalty abilities, `Prepare` spell quality tiers, `+1/+1` counter ETB stats on `Graft Surgeon` / `Vinelasher Adept` / `Mabel, Valley Hero`, token generation on `Jiang Yanggu, Never Alone`, `0/0` base-stat removal on `Multiply by Zero`, planeswalker-card tutor traps on `Loyal Tutor` / `Entrust the Spark`, and Bo1 narrow color hosers).

---

## 2. Key Empirical & Domain Findings

### 2.1 In-Universe vs. Universes Beyond 17Lands Baseline Win Rate
Across the 10 historical sets, 17Lands overall user game win rates split cleanly by release type:
- **In-Universe Sets (5 sets)**: `ECL` (`54.99%`), `TDM` (`55.02%`), `EOE` (`55.32%`), `SOS` (`55.38%`), `OM1` (`55.51%`) — Mean game win rate = **`55.24% ± 0.20%`**, Mean main-set card `GiH` WR = **`55.69%`** (`SOS` = **`55.93%`**).
- **Universes Beyond Sets (5 sets)**: `FIN` (`55.73%`), `TLA` (`56.01%`), `MSH` (`56.08%`), `HOB` (`56.16%`), `TMT` (`56.80%`) — Mean game win rate = **`56.16%`**.
- Because **Reality Fracture (`FRA`)** is a standard 280-card **In-Universe direct sequel to `Secrets of Strixhaven (SOS)`**, calibrating `FRA` to the In-Universe `SOS` baseline (`mean main-set card GiH WR = 55.97%`, compared to `SOS`'s realized `55.93%`) eliminates systematic baseline bias.

### 2.2 `Empower Jace N` (Comprehensive Rules 701.71a) & `Way of the ...` Cycle
Per Rule 701.71a, *"Empower Jace N"* puts $N$ loyalty counters on a Jace planeswalker token you control, first creating a blue Jace planeswalker token with **`[−1]: Surveil 1`** and **`[−3]: Draw a card`** if you don't control one:
- **`Empower Jace 4+` is immediate card advantage**: `Mindseeker Oculus` (`{2}{U}` 2/1 Common, `Empower Jace 4`) and `Tam's Resistance` (`{1}{G/U}` Common, `+1/+1` counter, vigilance, `Empower Jace 4`) allow you to immediately activate `[−3]: Draw a card` on your main phase while leaving a 1-loyalty Jace token on the battlefield (`GiH WR ~56.6%–57.4%`). `Protege's Awakening` (`{3}{U}` Common, Draw 1 + `Empower Jace 6`) draws 2 cards immediately and leaves a 3-loyalty Jace to draw a 3rd card next turn (`57.12%`).
- **The 10 Uncommon `Way of the ...` Legendary Enchantments**:
  - Top-tier entries combine `Empower Jace 5–7` with board-affecting loyalty abilities: `Way of the Healer` (`{3}{W}`, `Empower Jace 5` + `[−2]: Create a 2/2 Cadet, Surveil 1`, `57.82%`), `Way of the Wildspeaker` (`{4}{G}`, `Empower Jace 7` + `[−4]: Create a 4/4 Trample Beast`, `57.76%`), `Way of the Mind Sculptor` (`{4}{U}`, `Empower Jace 5` + draw an extra card whenever removing 2+ loyalty counters, `56.71%`), and `Way of the Warlord` (`{2}{R}`, `Empower Jace 5` + `[−4]: 2 damage to creature/PW and 2 to player`, `56.08%`).
  - Conversely, `Way of the Pyromancer` (`{1}{R}`, `Empower Jace 2`, `[+1]: Add {R}`, `51.70%`) and `Way of the Necromancer` (`{1}{B}`, `Empower Jace 2`, `52.53%`) only grant 2 loyalty counters and no board presence.
- **Planeswalker Tutor Traps (`Loyal Tutor`, `Entrust the Spark`)**: Because Jace tokens are **tokens** and `FRA` contains zero Common or Uncommon Planeswalker cards (only 8 Rare/Mythic Planeswalkers in the entire set), cards that search your library for a *"planeswalker card"* (`Loyal Tutor` `46.25%`, `Entrust the Spark` `47.21%`) fail to find a card in >90% of Limited decks.

### 2.3 `Prepare` Mechanics (`layout: "prepare"`, Rule 702.193a) in `SOS` vs. `FRA`
While `SOS` featured 5 enemy colleges, `FRA` features the 5 **allied** fractured colleges (`WU` Fatehold, `UB` Theorix, `BR` Stingerquill, `RG` Konstrari, `GW` Vigorbloom), each with a signature hybrid `Prepare` spell:
- **High-Impact Board-Affecting `Prepare` Spells (`WU` & `GW`)**:
  - `Peer Review` (`{2}{W/U}` Sorcery — *Create a 2/2 Cadet token, Surveil 1*) and `Seed Suture` (`{G/W}` Sorcery — *Put a +1/+1 counter on target creature, gain 1 life*, identical to `SOS`'s `Adventurous Eater` spell) directly add stats to the board. Creatures that enter prepared with these spells (`Fatehold Chronologist // Peer Review` `57.09%`, `Prudent Fateseer // Peer Review` `57.79%`, `Vigorbloom Vanguard // Seed Suture` `57.28%`, `Blossom-Blessed Angel // Seed Suture` `56.16%`, `Semester Foreseer // Peer Review` `56.03%`, `Emergency Phytomedic // Seed Suture` `55.90%`) are premier 2-for-1 curve plays.
- **Low-Impact / Setup `Prepare` Spells (`UB`, `BR`, `RG`)**:
  - `Omit Variables` (`{U/B}` — *Mill 3 cards*), `Vicious Verse` (`{B/R}` — *1 damage to target opponent*), and `Soul Tether` (`{2}{R/G}` — *3 mana to create a Heartwood mana rock*) do not affect creatures on the board by themselves, resulting in significantly lower win rates for their Common carriers (`53.1%–54.3%`).

---

## 3. Intermediate Artifacts & Verification Figures

All intermediate scripts, datasets, and figures are saved both in the repository (`src/`, `analysis/`) and in `/opt/cursor/artifacts/`:

- `src/extract_historical.py` & `analysis/historical_game_stats.json`: Vectorized 17Lands game statistics across all 10 historical sets (`2,935` cards).
- `src/build_card_catalog.py` & `analysis/card_catalog.json`: Canonical Scryfall + 17Lands catalog (`100%` historical match + `285` `FRA` main-set + `10` `SPG` cards).
- `src/env_simulation.py`: Pairwise combat & removal survival simulation engine.
- `src/featurize_and_model.py` & `analysis/historical_oof_predictions.csv` & `analysis/fra_ml_predictions.csv`: 141-feature multi-task stacked LOSO-CV ensemble.
- `src/generate_final_predictions.py` & `predictions.csv` & `predictions.json` & `analysis/fra_final_predictions.csv`: Final calibrated GiH win rate predictions.

<img src="/opt/cursor/artifacts/loso_cv_evaluation.png" alt="Leave-One-Set-Out Cross-Validation Evaluation across 10 Historical Sets" />

<img src="/opt/cursor/artifacts/fra_environment_and_mechanics.png" alt="Reality Fracture Environment, Rarity, Color, and Mechanics Breakdown" />

<img src="/opt/cursor/artifacts/fra_top_bottom_cards.png" alt="Reality Fracture Top 15 Bombs and Top 10 Commons/Uncommons vs Traps" />

---

## 4. Complete Reality Fracture (`FRA` + `SPG`) Predicted 17Lands GiH Win Rate Table

| Card | Predicted GiH Win Rate | Predicted GiH (%) | Set | CN | Rarity | Colors | Mana Cost | Type Line |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Emrakul, the Exigent Doom | 0.5944 | 59.44% | FRA | 1 | Mythic | C | {10} | Legendary Creature — Eldrazi |
| Academic Ascent | 0.5508 | 55.08% | FRA | 2 | Common | W | {1}{W} | Instant |
| Blossom-Blessed Angel // Seed Suture | 0.5599 | 55.99% | FRA | 3 | Common | W | {3}{W} // {G/W} | Creature — Angel Cleric // Sorcery |
| Campus Crier | 0.5589 | 55.89% | FRA | 4 | Common | W | {1}{W} | Creature — Human Advisor |
| Enlightened Confidant | 0.5970 | 59.70% | FRA | 5 | Mythic | W | {1}{W} | Creature — Kor Cleric |
| Fateshaper Aspirant | 0.5526 | 55.26% | FRA | 6 | Common | W | {4}{W} | Creature — Rhino Cleric |
| Flickering Hound | 0.5842 | 58.42% | FRA | 7 | Rare | W | {3}{W} | Creature — Dog |
| Generous Revival | 0.5655 | 56.55% | FRA | 8 | Uncommon | W | {2}{W} | Sorcery |
| Germinate Recruits | 0.5924 | 59.24% | FRA | 9 | Rare | W | {2}{W} | Instant |
| Graft Surgeon | 0.5636 | 56.36% | FRA | 10 | Common | W | {2}{W} | Creature — Human Cleric |
| Guiding Hydra | 0.5944 | 59.44% | FRA | 11 | Rare | W | {X}{W} | Creature — Hydra Horror |
| Hexhaven Battalion | 0.5666 | 56.66% | FRA | 12 | Common | W | {4}{W}{W} | Sorcery |
| Kindred Judgment | 0.5924 | 59.24% | FRA | 13 | Rare | W | {5}{W}{W} | Sorcery |
| Loyal Tutor | 0.4625 | 46.25% | FRA | 14 | Rare | W | {W} | Instant |
| Memory Trap | 0.5697 | 56.97% | FRA | 15 | Common | W | {2}{W} | Enchantment |
| Predictive Preparations | 0.5565 | 55.65% | FRA | 16 | Common | W | {1}{W} | Sorcery |
| Prophesied End | 0.5655 | 56.55% | FRA | 17 | Uncommon | W | {1}{W} | Instant |
| Refute Destiny | 0.5257 | 52.57% | FRA | 18 | Uncommon | W | {1}{W} | Sorcery |
| Repurposed Enforcer | 0.5936 | 59.36% | FRA | 19 | Rare | W | {1}{W} | Creature — Human Soldier |
| Return to the Light Realms | 0.5148 | 51.48% | FRA | 20 | Mythic | W | {7}{W}{W} | Sorcery |
| Shatterwing Pegasus | 0.5446 | 54.46% | FRA | 21 | Common | W | {2}{W} | Creature — Pegasus |
| Surgical Precision | 0.5685 | 56.85% | FRA | 22 | Common | W | {1}{W} | Sorcery |
| Unflinching Hortimancer | 0.5591 | 55.91% | FRA | 23 | Common | W | {1}{W} | Creature — Human Cleric |
| Your Fate Ends Here | 0.5629 | 56.29% | FRA | 24 | Uncommon | W | {2}{W} | Instant |
| Countersculpt | 0.5603 | 56.03% | FRA | 25 | Uncommon | U | {U}{U} | Instant |
| Cruel Calculations | 0.5275 | 52.75% | FRA | 26 | Rare | U | {2}{U} | Sorcery |
| Cryotheory Adept | 0.5492 | 54.92% | FRA | 27 | Common | U | {1}{U} | Creature — Human Wizard |
| Diviner of Victory // Unwind History | 0.5770 | 57.70% | FRA | 28 | Rare | U | {U} // {1}{U} | Creature — Dwarf Wizard // Sorcery |
| Divining Duelist | 0.5472 | 54.72% | FRA | 29 | Common | U | {2}{U} | Creature — Merfolk Wizard |
| Icy Reception | 0.5601 | 56.01% | FRA | 30 | Common | U | {1}{U} | Instant |
| Infinite Coursework | 0.5525 | 55.25% | FRA | 31 | Common | U | {2}{U} | Enchantment — Aura |
| Jace's Machinations | 0.5919 | 59.19% | FRA | 32 | Rare | U | {2}{U} | Instant |
| Mindseeker Oculus | 0.5706 | 57.06% | FRA | 33 | Common | U | {2}{U} | Creature — Homunculus |
| Perfected Theory | 0.5510 | 55.10% | FRA | 34 | Uncommon | U | {U} | Instant |
| Plan for All Outcomes | 0.5706 | 57.06% | FRA | 35 | Uncommon | U | {3}{U} | Enchantment |
| Precise Redaction | 0.5129 | 51.29% | FRA | 36 | Uncommon | U | {1}{U} | Instant |
| Protege's Awakening | 0.5686 | 56.86% | FRA | 37 | Common | U | {3}{U} | Sorcery |
| Seasoned Cryomancer | 0.5969 | 59.69% | FRA | 38 | Mythic | U | {1}{U}{U} | Creature — Human Wizard |
| Semester Foreseer // Peer Review | 0.5593 | 55.93% | FRA | 39 | Common | U | {3}{U} // {2}{W/U} | Creature — Human Wizard // Sorcery |
| Sphinx of False Conclusions | 0.6108 | 61.08% | FRA | 40 | Rare | U | {2}{U}{U} | Creature — Sphinx Illusion |
| Sphinx's Approach | 0.5577 | 55.77% | FRA | 41 | Common | U | {1}{U}{U} | Instant |
| Surveillance Phantasm | 0.5553 | 55.53% | FRA | 42 | Common | U | {1}{U} | Creature — Bird Illusion |
| The Theorist, Jace Beleren | 0.6491 | 64.91% | FRA | 43 | Mythic | U | {2}{U}{U} | Legendary Planeswalker — Jace |
| Theorist's Proxy | 0.5823 | 58.23% | FRA | 44 | Rare | U | {1}{U} | Creature — Illusion |
| Undulating Witness | 0.5372 | 53.72% | FRA | 45 | Common | U | {4}{U} | Creature — Serpent |
| Unsummon | 0.5598 | 55.98% | FRA | 46 | Common | U | {U} | Instant |
| Variable Chaser // Arc of Fortune | 0.5894 | 58.94% | FRA | 47 | Rare | U | {2}{U} // {2}{U} | Creature — Human Wizard // Sorcery |
| Apex Witchstalker | 0.5332 | 53.32% | FRA | 48 | Common | B | {4}{B}{B} | Creature — Wolf |
| Bloodline Recollector // Ancestral Craving | 0.5576 | 55.76% | FRA | 49 | Mythic | B | {1}{B} // {B} | Creature — Vampire Warlock // Instant |
| Break Under Pressure | 0.5674 | 56.74% | FRA | 50 | Uncommon | B | {2}{B} | Instant |
| Cast Away Doubt | 0.5454 | 54.54% | FRA | 51 | Common | B | {2}{B} | Sorcery |
| Dark Matter Manipulator | 0.5704 | 57.04% | FRA | 52 | Rare | B | {B} | Creature — Human Warlock |
| Darklight Phoenix | 0.5853 | 58.53% | FRA | 53 | Mythic | B | {3}{B} | Creature — Phoenix |
| Extended Absence | 0.5701 | 57.01% | FRA | 54 | Common | B | {3}{B} | Instant |
| Extrapolate the Impossible | 0.5050 | 50.50% | FRA | 55 | Rare | B | {1}{B} | Sorcery |
| Last Gasp | 0.5743 | 57.43% | FRA | 56 | Common | B | {1}{B} | Instant |
| Lich's Relic | 0.5963 | 59.63% | FRA | 57 | Rare | B | {B} | Artifact — Equipment |
| Multiply by Zero | 0.5649 | 56.49% | FRA | 58 | Uncommon | B | {1}{B} | Instant |
| Overwrite the Multiverse | 0.6106 | 61.06% | FRA | 59 | Mythic | B | {4}{B}{B} | Sorcery |
| Rampart Hunter | 0.5294 | 52.94% | FRA | 60 | Common | B | {3}{B} | Creature — Horror |
| Rank Rat | 0.5421 | 54.21% | FRA | 61 | Common | B | {1}{B} | Creature — Zombie Rat |
| Rewrite Regrets | 0.5428 | 54.28% | FRA | 62 | Uncommon | B | {3}{B} | Sorcery |
| Rise of the Deathbringer | 0.6057 | 60.57% | FRA | 63 | Rare | B | {4}{B} | Instant |
| Sanctum Lurker | 0.6038 | 60.38% | FRA | 64 | Rare | B | {2}{B} | Creature — Horror |
| Screeching Soulbreaker | 0.5285 | 52.85% | FRA | 65 | Common | B | {2}{B} | Creature — Siren Bard |
| Silence the Echo | 0.5551 | 55.51% | FRA | 66 | Common | B | {1}{B} | Sorcery |
| Solve for Disappointment | 0.5357 | 53.57% | FRA | 67 | Common | B | {1}{B} | Sorcery |
| Terminal Criticism | 0.5294 | 52.94% | FRA | 68 | Uncommon | B | {1}{B} | Instant |
| Theoretical Necromancer | 0.5346 | 53.46% | FRA | 69 | Common | B | {2}{B} | Creature — Vampire Warlock |
| Void Extrapolator // Omit Variables | 0.5332 | 53.32% | FRA | 70 | Common | B | {1}{B} // {U/B} | Creature — Aetherborn Warlock // Sorcery |
| Vraska's Final Mercy | 0.5874 | 58.74% | FRA | 71 | Rare | B | {B}{B} | Sorcery |
| Ajani's Anguish | 0.5735 | 57.35% | FRA | 72 | Rare | R | {X}{R} | Enchantment |
| Artifist Acumen | 0.5240 | 52.40% | FRA | 73 | Common | R | {R} | Sorcery |
| Awaken the Inferno | 0.5564 | 55.64% | FRA | 74 | Common | R | {4}{R} | Sorcery |
| Blazing Crescendo | 0.5297 | 52.97% | FRA | 75 | Common | R | {1}{R} | Instant |
| Chandra's Emberling | 0.5201 | 52.01% | FRA | 76 | Common | R | {2}{R} | Creature — Gremlin Elemental |
| Command the Stage | 0.5475 | 54.75% | FRA | 77 | Uncommon | R | {2}{R} | Sorcery |
| Craterclaw Colossus | 0.5724 | 57.24% | FRA | 78 | Mythic | R | {4}{R}{R}{R} | Artifact Creature — Beast Construct |
| Curse-Marred Demon | 0.5756 | 57.56% | FRA | 79 | Rare | R | {2}{R}{R} | Creature — Demon |
| Draconic Visitor | 0.6013 | 60.13% | FRA | 80 | Rare | R | {3}{R}{R} | Creature — Dragon |
| Eardrum Rattler | 0.5334 | 53.34% | FRA | 81 | Common | R | {1}{R} | Creature — Human Bard |
| Essence Burn | 0.5248 | 52.48% | FRA | 82 | Uncommon | R | {1}{R} | Instant |
| Face Yourself | 0.5258 | 52.58% | FRA | 83 | Rare | R | {5}{R}{R} | Sorcery |
| Fulminous Forte | 0.5702 | 57.02% | FRA | 84 | Uncommon | R | {2}{R} | Instant |
| Hallway Heckler // Vicious Verse | 0.5208 | 52.08% | FRA | 85 | Common | R | {2}{R} // {B/R} | Creature — Elemental Sorcerer // Sorcery |
| Heartstring Puller | 0.5505 | 55.05% | FRA | 86 | Common | R | {3}{R} | Creature — Elf Sorcerer |
| Identity Echo | 0.5274 | 52.74% | FRA | 87 | Rare | R | {2}{R} | Enchantment |
| Master of Barbs | 0.5598 | 55.98% | FRA | 88 | Rare | R | {1}{R} | Creature — Lizard Bard |
| No Admittance | 0.5698 | 56.98% | FRA | 89 | Common | R | {1}{R} | Sorcery |
| Pompous Battlemage // Improvised Act | 0.5528 | 55.28% | FRA | 90 | Rare | R | {R} // {R} | Creature — Goblin Sorcerer // Sorcery |
| Pyre Rhymer // Molten Tide | 0.5489 | 54.89% | FRA | 91 | Rare | R | {1}{R}{R} // {R} | Creature — Elemental Sorcerer // Instant |
| Skilled Battlecarver | 0.5297 | 52.97% | FRA | 92 | Common | R | {1}{R} | Creature — Human Warrior |
| Stingcaster Mage | 0.5860 | 58.60% | FRA | 93 | Mythic | R | {1}{R} | Creature — Human Wizard |
| Tether Technician | 0.5220 | 52.20% | FRA | 94 | Common | R | {4}{R} | Creature — Minotaur Artificer |
| Violent Echoes | 0.5696 | 56.96% | FRA | 95 | Uncommon | R | {2}{R}{R} | Instant |
| Wrath of the Bloodmane | 0.5640 | 56.40% | FRA | 96 | Common | R | {2}{R} | Instant |
| Arcane Amphisbaena | 0.5596 | 55.96% | FRA | 97 | Common | G | {1}{G} | Creature — Snake |
| Bestial Incursion | 0.5628 | 56.28% | FRA | 98 | Common | G | {3}{G} | Sorcery |
| Budding Insurgent | 0.5484 | 54.84% | FRA | 99 | Common | G | {2}{G} | Creature — Dryad Scout |
| Carnivorous Cultivator // Enroot | 0.5690 | 56.90% | FRA | 100 | Rare | G | {1}{G} // {G} | Creature — Elf Warlock // Sorcery |
| Compel Brutality | 0.5636 | 56.36% | FRA | 101 | Common | G | {1}{G} | Instant |
| Flourishing Grapple | 0.5213 | 52.13% | FRA | 102 | Uncommon | G | {G} | Instant |
| Gardenize | 0.4847 | 48.47% | FRA | 103 | Rare | G | {1}{G}{G} | Enchantment |
| Greenhouse Propagator | 0.5492 | 54.92% | FRA | 104 | Common | G | {2}{G} | Creature — Cat Druid |
| Heartwood Crafter // Soul Tether | 0.5602 | 56.02% | FRA | 105 | Uncommon | G | {G} // {2}{R/G} | Creature — Elf Artificer // Sorcery |
| Hexhaven Invigorator | 0.5891 | 58.91% | FRA | 106 | Mythic | G | {G}{G}{G}{G} | Creature — Chimera Horror |
| Hungering Puppetbeast | 0.6044 | 60.44% | FRA | 107 | Rare | G | {3}{G}{G} | Artifact Creature — Beast Construct |
| Hunter's Axe | 0.5381 | 53.81% | FRA | 108 | Uncommon | G | {G} | Artifact — Equipment |
| Inspired Tethermage | 0.5381 | 53.81% | FRA | 109 | Common | G | {2}{G} | Creature — Elf Warrior |
| Omnipresence | 0.5093 | 50.93% | FRA | 110 | Mythic | G | {5}{G}{G}{G} | Enchantment |
| Puppet Crafting | 0.5112 | 51.12% | FRA | 111 | Rare | G | {1}{G} | Enchantment — Aura |
| Restore with Empathy | 0.5577 | 55.77% | FRA | 112 | Uncommon | G | {2}{G} | Instant |
| Simulacrum Shaper | 0.5804 | 58.04% | FRA | 113 | Rare | G | {1}{G}{G} | Creature — Elf Druid |
| Something Worth Saving | 0.5497 | 54.97% | FRA | 114 | Common | G | {1}{G} | Instant |
| Sureshot Sower | 0.5614 | 56.14% | FRA | 115 | Common | G | {1}{G} | Creature — Human Archer |
| Tarmogoyf | 0.5881 | 58.81% | FRA | 116 | Mythic | G | {1}{G} | Creature — Lhurgoyf |
| Tethermage's Advantage | 0.5385 | 53.85% | FRA | 117 | Common | G | {G} | Instant |
| Verdant Kraken | 0.6134 | 61.34% | FRA | 118 | Rare | G | {4}{G}{G}{G} | Creature — Plant Kraken |
| Vinelasher Adept | 0.5565 | 55.65% | FRA | 119 | Common | G | {4}{G}{G} | Creature — Rhino Soldier |
| Wrecking Gecko | 0.5490 | 54.90% | FRA | 120 | Common | G | {4}{G} | Artifact Creature — Lizard Construct |
| Aerid Konstrari | 0.6088 | 60.88% | FRA | 121 | Mythic | RG | {1}{R}{G}{G} | Legendary Creature — Elder Sphinx |
| Avatar of Burgeoning Echoes | 0.5872 | 58.72% | FRA | 122 | Mythic | GU | {G}{U} | Creature — Avatar |
| Blessed Ghoul | 0.5669 | 56.69% | FRA | 123 | Common | WB | {W/B} | Creature — Zombie Cleric |
| Bloombrute | 0.5742 | 57.42% | FRA | 124 | Uncommon | GW | {2}{G}{W} | Creature — Plant Elemental |
| Charge the Sanctum | 0.5411 | 54.11% | FRA | 125 | Common | RW | {2}{R/W} | Instant |
| Clash of Elements | 0.5537 | 55.37% | FRA | 126 | Uncommon | UR | {1}{U}{R} | Instant |
| Craftwork Crusher | 0.5820 | 58.20% | FRA | 127 | Uncommon | RG | {3}{R}{R}{G}{G} | Artifact Creature — Boar Construct |
| Denzilore Fatehold | 0.6229 | 62.29% | FRA | 128 | Mythic | WU | {1}{W}{U}{U} | Legendary Creature — Elder Sphinx |
| Desperate Futurescribe | 0.5717 | 57.17% | FRA | 129 | Uncommon | WU | {2}{W}{U} | Creature — Kor Scout |
| Emergency Phytomedic // Seed Suture | 0.5593 | 55.93% | FRA | 130 | Common | GW | {G/W} // {G/W} | Creature — Dryad Cleric // Sorcery |
| Entrust the Spark | 0.4721 | 47.21% | FRA | 131 | Rare | GU | {3}{G}{U} | Sorcery |
| Fatehold Charm | 0.5825 | 58.25% | FRA | 132 | Uncommon | WU | {W}{U} | Instant |
| Fatehold Chronologist // Peer Review | 0.5696 | 56.96% | FRA | 133 | Common | WU | {1}{W/U} // {2}{W/U} | Creature — Bird Wizard // Sorcery |
| Ferocity of the Hunt | 0.5402 | 54.02% | FRA | 134 | Common | BG | {1}{B/G} | Enchantment — Aura |
| Frostbite Pyromental | 0.5805 | 58.05% | FRA | 135 | Rare | UR | {U}{R}{R} | Creature — Elemental |
| Grim Repriser | 0.5555 | 55.55% | FRA | 136 | Uncommon | BR | {B}{R} | Creature — Zombie Bard |
| Ingris Stingerquill | 0.6037 | 60.37% | FRA | 137 | Mythic | BR | {B}{R}{R} | Legendary Creature — Elder Sphinx |
| Konstrari Charm | 0.5651 | 56.51% | FRA | 138 | Uncommon | RG | {R}{G} | Instant |
| Konstrari Improviser // Soul Tether | 0.5409 | 54.09% | FRA | 139 | Common | RG | {1}{R/G} // {2}{R/G} | Creature — Human Artificer // Sorcery |
| Kwia Vigorbloom | 0.6339 | 63.39% | FRA | 140 | Mythic | GW | {3}{G}{W}{W} | Legendary Creature — Elder Sphinx |
| Mind Meanderer | 0.5764 | 57.64% | FRA | 141 | Uncommon | GU | {3}{G}{U}{U} | Creature — Bird Fish Illusion |
| Null Summoner | 0.5861 | 58.61% | FRA | 142 | Rare | UB | {2}{U}{B} | Creature — Human Warlock |
| Paradox Shaper // Omit Variables | 0.5527 | 55.27% | FRA | 143 | Uncommon | UB | {1}{U/B} // {U/B} | Creature — Octopus Wizard // Sorcery |
| Primal Witchstalker | 0.5556 | 55.56% | FRA | 144 | Uncommon | BG | {1}{B}{G} | Creature — Wolf |
| Proctor of Potential | 0.5891 | 58.91% | FRA | 145 | Rare | WU | {W}{U} | Creature — Human Cleric |
| Prudent Fateseer // Peer Review | 0.5774 | 57.74% | FRA | 146 | Uncommon | WU | {1}{W/U}{W/U} // {2}{W/U} | Creature — Dwarf Wizard // Sorcery |
| Recursive Recruitment | 0.5812 | 58.12% | FRA | 147 | Uncommon | UB | {2}{U}{B} | Sorcery |
| Solarium Sentry | 0.5811 | 58.11% | FRA | 148 | Rare | GW | {G}{W} | Creature — Cat Soldier |
| Solitary Cell | 0.5920 | 59.20% | FRA | 149 | Rare | RW | {R}{W} | Artifact |
| Stingerquill Charm | 0.5737 | 57.37% | FRA | 150 | Uncommon | BR | {B}{R} | Instant |
| Stingerquill Voxmancer // Vicious Verse | 0.5396 | 53.96% | FRA | 151 | Uncommon | BR | {B/R} // {B/R} | Creature — Goblin Sorcerer // Sorcery |
| Stinging Vitriol | 0.5391 | 53.91% | FRA | 152 | Rare | BR | {B}{R} | Sorcery |
| Tam's Resistance | 0.5649 | 56.49% | FRA | 153 | Common | GU | {1}{G/U} | Sorcery |
| Tenured Tethermage | 0.5725 | 57.25% | FRA | 154 | Rare | RG | {1}{R}{G} | Creature — Human Artificer |
| Theorix Charm | 0.5809 | 58.09% | FRA | 155 | Uncommon | UB | {U}{B} | Instant |
| Theorix Metamage // Omit Variables | 0.5354 | 53.54% | FRA | 156 | Common | UB | {2}{U/B} // {U/B} | Creature — Shade Wizard // Sorcery |
| Twinned Vision | 0.5599 | 55.99% | FRA | 157 | Common | UR | {1}{U/R} | Instant |
| Twisted Fates | 0.5788 | 57.88% | FRA | 158 | Uncommon | WB | {2}{W}{W}{B} | Sorcery |
| Uldaros Theorix | 0.6012 | 60.12% | FRA | 159 | Mythic | UB | {3}{U}{B}{B} | Legendary Creature — Elder Sphinx |
| Vigorbloom Charm | 0.5755 | 57.55% | FRA | 160 | Uncommon | GW | {G}{W} | Instant |
| Vigorbloom Vanguard // Seed Suture | 0.5728 | 57.28% | FRA | 161 | Uncommon | GW | {1}{G/W} // {G/W} | Creature — Troll Druid // Sorcery |
| Vindictive Triumph | 0.5890 | 58.90% | FRA | 162 | Rare | WB | {W}{B}{B} | Instant |
| Warrior's Blades | 0.5685 | 56.85% | FRA | 163 | Uncommon | RW | {2}{R}{W} | Artifact — Equipment |
| Whiplash Wordsmith // Vicious Verse | 0.5234 | 52.34% | FRA | 164 | Common | BR | {3}{B/R} // {B/R} | Creature — Vampire Sorcerer // Sorcery |
| Woodwork Prodigy // Soul Tether | 0.5385 | 53.85% | FRA | 165 | Uncommon | RG | {2}{R/G} // {2}{R/G} | Creature — Cat Druid // Sorcery |
| Afterthought Sentry | 0.5353 | 53.53% | FRA | 166 | Common | C | {2} | Artifact Creature — Gargoyle |
| Archive Arbiter | 0.5400 | 54.00% | FRA | 167 | Uncommon | C | {6} | Artifact Creature — Sphinx |
| Codie, Ravenous Codex | 0.5398 | 53.98% | FRA | 168 | Rare | WUBRG | {3} | Legendary Artifact Creature — Book Construct |
| The Echoverse Fulcrum | 0.5844 | 58.44% | FRA | 169 | Mythic | C | {2} | Legendary Artifact |
| Eye of Jace | 0.5498 | 54.98% | FRA | 170 | Uncommon | C | {1} | Artifact |
| Keeper of the Quiet Hour | 0.5266 | 52.66% | FRA | 171 | Common | C | {3} | Artifact Creature — Chimera |
| Living Library | 0.5413 | 54.13% | FRA | 172 | Common | C | {2} | Artifact Creature — Book Illusion |
| Medic's Kitesail | 0.5334 | 53.34% | FRA | 173 | Common | C | {2} | Artifact — Equipment |
| Murmuring Volume | 0.5269 | 52.69% | FRA | 174 | Common | C | {3} | Artifact — Book |
| Dedicated Commons | 0.5465 | 54.65% | FRA | 175 | Common | RW | nan | Land |
| Deserted Beach | 0.5711 | 57.11% | FRA | 176 | Rare | WU | nan | Land |
| Fatehold Annex | 0.5563 | 55.63% | FRA | 177 | Common | WU | nan | Land |
| Formidable Commons | 0.5450 | 54.50% | FRA | 178 | Common | BG | nan | Land |
| Hall of Echoes | 0.5540 | 55.40% | FRA | 179 | Rare | Land | nan | Land |
| Haunted Ridge | 0.5604 | 56.04% | FRA | 180 | Rare | BR | nan | Land |
| Hexhaven Dueling Arena | 0.5253 | 52.53% | FRA | 181 | Uncommon | Land | nan | Land |
| Innovative Commons | 0.5485 | 54.85% | FRA | 182 | Common | UR | nan | Land |
| Konstrari Annex | 0.5423 | 54.23% | FRA | 183 | Common | RG | nan | Land |
| Meticulous Commons | 0.5498 | 54.98% | FRA | 184 | Common | WB | nan | Land |
| Overgrown Farmland | 0.5690 | 56.90% | FRA | 185 | Rare | GW | nan | Land |
| Rockfall Vale | 0.5626 | 56.26% | FRA | 186 | Rare | RG | nan | Land |
| Roiling Canopy | 0.5455 | 54.55% | FRA | 187 | Rare | G | nan | Land |
| Room of Refuge | 0.5433 | 54.33% | FRA | 188 | Common | Land | nan | Land |
| Shipwreck Marsh | 0.5659 | 56.59% | FRA | 189 | Rare | UB | nan | Land |
| Stingerquill Annex | 0.5403 | 54.03% | FRA | 190 | Common | BR | nan | Land |
| Theorist's Sanctum | 0.5592 | 55.92% | FRA | 191 | Rare | U | nan | Land — Island |
| Theorix Annex | 0.5503 | 55.03% | FRA | 192 | Common | UB | nan | Land |
| Transformative Commons | 0.5537 | 55.37% | FRA | 193 | Common | GU | nan | Land |
| Vigorbloom Annex | 0.5521 | 55.21% | FRA | 194 | Common | GW | nan | Land |
| Ajani Resolute | 0.6120 | 61.20% | FRA | 195 | Mythic | W | {1}{W} | Legendary Planeswalker — Ajani |
| Danitha, Sword of Hope | 0.5494 | 54.94% | FRA | 196 | Uncommon | W | {2}{W} | Legendary Creature — Human Knight |
| Ghalta the Immovable | 0.5595 | 55.95% | FRA | 197 | Uncommon | W | {8}{W} | Legendary Creature — Elder Dinosaur |
| Gideon's Memorial | 0.5767 | 57.67% | FRA | 198 | Rare | W | {1}{W} | Legendary Artifact |
| Koth of the Homestead | 0.5553 | 55.53% | FRA | 199 | Uncommon | W | {2}{W} | Legendary Creature — Human Citizen |
| Liliana the Faultless | 0.5779 | 57.79% | FRA | 200 | Rare | W | {W} | Legendary Creature — Human Cleric |
| Lyra, Archangel of Dawn | 0.5859 | 58.59% | FRA | 201 | Rare | W | {2}{W} | Legendary Creature — Angel Knight |
| Rescue Girl, First Responder | 0.5440 | 54.40% | FRA | 202 | Uncommon | W | {2}{W} | Legendary Creature — Human Cleric |
| Saheeli, Consul of Oversight | 0.5696 | 56.96% | FRA | 203 | Uncommon | W | {3}{W}{W} | Legendary Creature — Human Advisor |
| Teyo, Lightshield Expert | 0.5714 | 57.14% | FRA | 204 | Uncommon | W | {1}{W} | Legendary Creature — Human Cleric |
| Thalia, the Survivor | 0.5635 | 56.35% | FRA | 205 | Uncommon | W | {3}{W} | Legendary Creature — Human Soldier |
| Tomik, Orzhov Lawmage | 0.5633 | 56.33% | FRA | 206 | Uncommon | W | {1}{W} | Legendary Creature — Human Advisor |
| Way of the Healer | 0.5770 | 57.70% | FRA | 207 | Uncommon | W | {3}{W} | Legendary Enchantment |
| Way of the Mentor | 0.5572 | 55.72% | FRA | 208 | Uncommon | W | {2}{W} | Legendary Enchantment |
| Yoshimaru, Beloved Companion | 0.5624 | 56.24% | FRA | 209 | Uncommon | W | {2}{W} | Legendary Creature — Dog |
| Yuriko, Blade of the Mighty | 0.5503 | 55.03% | FRA | 210 | Uncommon | W | {3}{W} | Legendary Creature — Human Samurai |
| Arni, Humble Scribe | 0.5512 | 55.12% | FRA | 211 | Uncommon | U | {2}{U} | Legendary Creature — Human Wizard |
| Chandra, Chill of Compliance | 0.5976 | 59.76% | FRA | 212 | Mythic | U | {1}{U}{U} | Legendary Planeswalker — Chandra |
| Fblthp, Impossibly Lost | 0.5597 | 55.97% | FRA | 213 | Uncommon | U | {1}{U} | Legendary Creature — Homunculus |
| Geist of Saint Thalia | 0.5526 | 55.26% | FRA | 214 | Uncommon | U | {1}{U} | Legendary Creature — Spirit Cleric |
| Hapatra, the Desert Frost | 0.5657 | 56.57% | FRA | 215 | Uncommon | U | {3}{U} | Legendary Creature — Human Wizard |
| Jace, Reality Sculptor | 0.6141 | 61.41% | FRA | 216 | Rare | U | {3}{U}{U} | Legendary Planeswalker — Jace |
| Lyra, Tolarian Archangel | 0.5980 | 59.80% | FRA | 217 | Rare | U | {1}{U}{U} | Legendary Creature — Angel Wizard |
| Proft, Consulting Detective | 0.5591 | 55.91% | FRA | 218 | Uncommon | U | {1}{U} | Legendary Creature — Human Detective |
| Ruric Thar, Biomagus | 0.5638 | 56.38% | FRA | 219 | Uncommon | U | {4}{U}{U} | Legendary Creature — Ogre Crab Wizard |
| Samut, Tyrant of Naktamun | 0.5687 | 56.87% | FRA | 220 | Rare | U | {1}{U} | Legendary Creature — Human Wizard |
| Tetsuko Umezawa, Fugitive | 0.5532 | 55.32% | FRA | 221 | Uncommon | U | {1}{U} | Legendary Creature — Human Rogue |
| Traxos, Academy Guardian | 0.5422 | 54.22% | FRA | 222 | Uncommon | U | {3}{U} | Legendary Artifact Creature — Dragon Construct |
| Way of the Cryomancer | 0.5471 | 54.71% | FRA | 223 | Uncommon | U | {2}{U} | Legendary Enchantment |
| Way of the Mind Sculptor | 0.5635 | 56.35% | FRA | 224 | Uncommon | U | {4}{U} | Legendary Enchantment |
| Yargle, Goliath of Otaria | 0.5271 | 52.71% | FRA | 225 | Uncommon | U | {4}{U} | Legendary Creature — Frog Spirit |
| Yuriko, Hope from the Shadows | 0.5740 | 57.40% | FRA | 226 | Uncommon | U | {U} | Legendary Creature — Human Ninja |
| Danitha, Spear of Agony | 0.5268 | 52.68% | FRA | 227 | Uncommon | B | {2}{B} | Legendary Creature — Human Knight |
| Gallia, Tragic Host | 0.5454 | 54.54% | FRA | 228 | Uncommon | B | {1}{B} | Legendary Creature — Zombie Satyr |
| Garruk, Veiled Butcher | 0.6267 | 62.67% | FRA | 229 | Mythic | B | {3}{B}{B} | Legendary Planeswalker — Garruk |
| Gideon the Oathless | 0.5633 | 56.33% | FRA | 230 | Rare | B | {2}{B} | Legendary Creature — Human Mercenary |
| Liliana the Repentant | 0.5635 | 56.35% | FRA | 231 | Rare | B | {1}{B} | Legendary Creature — Human Warlock |
| Loot, the Anomaly | 0.5478 | 54.78% | FRA | 232 | Uncommon | B | {2}{B} | Legendary Creature — Beast Horror |
| Mabel, Bitter Recluse | 0.5603 | 56.03% | FRA | 233 | Uncommon | B | {B} | Legendary Creature — Mouse Warlock |
| Massacre Girl, Most Wanted | 0.5384 | 53.84% | FRA | 234 | Uncommon | B | {4}{B} | Legendary Creature — Human Assassin |
| Proft, Sinister Mastermind | 0.5569 | 55.69% | FRA | 235 | Uncommon | B | {2}{B} | Legendary Creature — Human Rogue |
| Teyo, Diamondblade Mage | 0.5590 | 55.90% | FRA | 236 | Uncommon | B | {3}{B} | Legendary Creature — Human Warlock |
| Tinybones, Pocket Nuisance | 0.5513 | 55.13% | FRA | 237 | Uncommon | B | {2}{B} | Legendary Creature — Skeleton Rogue |
| Way of the Deathbringer | 0.5523 | 55.23% | FRA | 238 | Uncommon | B | {2}{B} | Legendary Enchantment |
| Way of the Necromancer | 0.5228 | 52.28% | FRA | 239 | Uncommon | B | {1}{B} | Legendary Enchantment |
| Winter, Tormented Loner | 0.5454 | 54.54% | FRA | 240 | Uncommon | B | {2}{B} | Legendary Creature — Human Warlock |
| Yargle, Glutton of Urborg | 0.5333 | 53.33% | FRA | 241 | Uncommon | B | {4}{B} | Legendary Creature — Frog Spirit |
| Ajani Unrelenting | 0.6403 | 64.03% | FRA | 242 | Mythic | R | {4}{R}{R} | Legendary Planeswalker — Ajani |
| Arni, Renowned Champion | 0.5194 | 51.94% | FRA | 243 | Uncommon | R | {3}{R} | Legendary Creature — Human Berserker |
| Chandra, Torch of Defiance | 0.6185 | 61.85% | FRA | 244 | Mythic | R | {2}{R}{R} | Legendary Planeswalker — Chandra |
| Gallia, the Merrymaker | 0.5399 | 53.99% | FRA | 245 | Uncommon | R | {1}{R} | Legendary Creature — Satyr |
| Jiang Yanggu, Alone | 0.5268 | 52.68% | FRA | 246 | Uncommon | R | {4}{R} | Legendary Creature — Human Berserker |
| Kiora of Fire and Ashes | 0.5742 | 57.42% | FRA | 247 | Uncommon | R | {4}{R}{R} | Legendary Creature — Merfolk Noble |
| Koth, the Geomancer | 0.5297 | 52.97% | FRA | 248 | Uncommon | R | {2}{R} | Legendary Creature — Human Warrior |
| Marwyn, the Clearcutter | 0.5403 | 54.03% | FRA | 249 | Uncommon | R | {R} | Legendary Creature — Elf Warrior |
| Pia, Determined Rebuilder | 0.5572 | 55.72% | FRA | 250 | Uncommon | R | {2}{R} | Legendary Creature — Human Artificer |
| Samut, Hazoret's Champion | 0.5462 | 54.62% | FRA | 251 | Rare | R | {1}{R} | Legendary Creature — Human Warrior Cleric |
| Tetsuko Umezawa, Pursuer | 0.5214 | 52.14% | FRA | 252 | Uncommon | R | {3}{R} | Legendary Creature — Human Mercenary |
| Tomik, Izzet Sparkmage | 0.5370 | 53.70% | FRA | 253 | Uncommon | R | {1}{R} | Legendary Creature — Human Wizard |
| Way of the Pyromancer | 0.5170 | 51.70% | FRA | 254 | Uncommon | R | {1}{R} | Legendary Enchantment |
| Way of the Warlord | 0.5567 | 55.67% | FRA | 255 | Uncommon | R | {2}{R} | Legendary Enchantment |
| Winter, Team Player | 0.5311 | 53.11% | FRA | 256 | Uncommon | R | {4}{R} | Legendary Creature — Human Warrior |
| Edgar, Moonlit Sovereign | 0.5614 | 56.14% | FRA | 257 | Uncommon | G | {3}{G}{G} | Legendary Creature — Werewolf Noble |
| Fblthp, Knows the Way | 0.5600 | 56.00% | FRA | 258 | Uncommon | G | {X}{G}{G} | Legendary Creature — Homunculus Scout |
| Garruk, Curse Breaker | 0.6368 | 63.68% | FRA | 259 | Mythic | G | {3}{G}{G} | Legendary Planeswalker — Garruk |
| Ghalta the Unstoppable | 0.5627 | 56.27% | FRA | 260 | Uncommon | G | {8}{G} | Legendary Creature — Elder Dinosaur |
| Jiang Yanggu, Never Alone | 0.5743 | 57.43% | FRA | 261 | Uncommon | G | {3}{G} | Legendary Creature — Human Druid |
| Loot, the Nexus | 0.5459 | 54.59% | FRA | 262 | Uncommon | G | {2}{G} | Legendary Creature — Beast Noble |
| Marwyn, the Preserver | 0.5564 | 55.64% | FRA | 263 | Uncommon | G | {1}{G} | Legendary Creature — Elf Druid |
| Pia, Aether Ascetic | 0.5466 | 54.66% | FRA | 264 | Uncommon | G | {2}{G} | Legendary Creature — Human Druid |
| Ruric Thar, Magecrusher | 0.5654 | 56.54% | FRA | 265 | Uncommon | G | {5}{G}{G} | Legendary Creature — Ogre Warrior |
| Titanbones, Towering Heart | 0.5424 | 54.24% | FRA | 266 | Uncommon | G | {3}{G} | Legendary Creature — Skeleton Druid |
| Way of the Paradox | 0.5429 | 54.29% | FRA | 267 | Uncommon | G | {2}{G} | Legendary Enchantment |
| Way of the Wildspeaker | 0.5745 | 57.45% | FRA | 268 | Uncommon | G | {4}{G} | Legendary Enchantment |
| Yoshimaru, Scrappy Stray | 0.5737 | 57.37% | FRA | 269 | Uncommon | G | {1}{G} | Legendary Creature — Dog |
| Edgar, Ancient Bloodlord | 0.5655 | 56.55% | FRA | 270 | Uncommon | WB | {W}{B} | Legendary Creature — Vampire Noble |
| Hapatra, the Desert Fang | 0.5595 | 55.95% | FRA | 271 | Uncommon | BG | {2}{B}{B}{G} | Legendary Creature — Human Cleric |
| Karn, Gilded Guardian | 0.5776 | 57.76% | FRA | 272 | Rare | WUBRG | {2/W}{2/U}{2/B}{2/R}{2/G} | Legendary Artifact Creature — Golem |
| Kiora of Salt and Sand | 0.5763 | 57.63% | FRA | 273 | Uncommon | GU | {1}{G}{U} | Legendary Creature — Merfolk Noble |
| Mabel, Valley Hero | 0.5745 | 57.45% | FRA | 274 | Uncommon | RW | {1}{R}{W} | Legendary Creature — Mouse Soldier |
| Saheeli, Jewel of Avishkar | 0.5680 | 56.80% | FRA | 275 | Uncommon | UR | {2}{U}{R} | Legendary Creature — Human Artificer |
| Tam, the Possibility | 0.5737 | 57.37% | FRA | 276 | Rare | GU | {1}{G}{U} | Legendary Creature — Gorgon Wizard |
| Vraska, Soul of Stone | 0.5662 | 56.62% | FRA | 277 | Rare | WUR | {U}{R}{W} | Legendary Creature — Gorgon Wizard |
| Vraska, the Cutting Glare | 0.5767 | 57.67% | FRA | 278 | Rare | BG | {B}{B}{G} | Legendary Creature — Gorgon Assassin |
| Karn, Argent Defender | 0.5141 | 51.41% | FRA | 279 | Rare | C | {2} | Legendary Artifact Creature — Golem |
| Traxos, Scourge Eternal | 0.5300 | 53.00% | FRA | 280 | Uncommon | C | {4} | Legendary Artifact Creature — Dragon Construct |
| Plains | 0.5633 | 56.33% | FRA | 281 | Common | W | nan | Basic Land — Plains |
| Island | 0.5574 | 55.74% | FRA | 283 | Common | U | nan | Basic Land — Island |
| Swamp | 0.5519 | 55.19% | FRA | 285 | Common | B | nan | Basic Land — Swamp |
| Mountain | 0.5491 | 54.91% | FRA | 287 | Common | R | nan | Basic Land — Mountain |
| Forest | 0.5547 | 55.47% | FRA | 289 | Common | G | nan | Basic Land — Forest |
| Eye of Ugin | 0.4760 | 47.60% | SPG | 159 | Mythic | Land | nan | Legendary Land |
| Austere Command | 0.6089 | 60.89% | SPG | 160 | Mythic | W | {4}{W}{W} | Sorcery |
| Flesh Duplicate | 0.5762 | 57.62% | SPG | 161 | Mythic | U | {U}{U} | Creature — Shapeshifter Rebel |
| Consign to Memory | 0.5178 | 51.78% | SPG | 162 | Mythic | U | {U} | Instant |
| Sublime Epiphany | 0.6321 | 63.21% | SPG | 163 | Mythic | U | {4}{U}{U} | Instant |
| Consider | 0.5683 | 56.83% | SPG | 164 | Mythic | U | {U} | Instant |
| Necrodominance | 0.4837 | 48.37% | SPG | 165 | Mythic | B | {B}{B}{B} | Legendary Enchantment |
| Mind Twist | 0.5614 | 56.14% | SPG | 166 | Mythic | B | {X}{B} | Sorcery |
| Splinter Twin | 0.5378 | 53.78% | SPG | 167 | Mythic | R | {2}{R}{R} | Enchantment — Aura |
| Root Maze | 0.4682 | 46.82% | SPG | 168 | Mythic | G | {G} | Enchantment |
