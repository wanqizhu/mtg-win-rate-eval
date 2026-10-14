#!/usr/bin/env python3
"""Generate the comprehensive methodology report and single unified prediction table in PREDICTIONS.md."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

OUT_DIR = Path("/workspace/analysis")


def main() -> None:
    df = pd.read_csv(OUT_DIR / "fra_final_predictions.csv")
    # Sort FRA main set by collector number (1..290), then Special Guests (1..10)
    df["set_order"] = df["set_code"].map({"FRA": 0, "SPG": 1})
    df.sort_values(by=["set_order", "cn_num", "card_name"], inplace=True)

    table_lines = [
        "| Card | Predicted GiH Win Rate | Predicted GiH (%) | Set | CN | Rarity | Colors | Mana Cost | Type Line |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for _, r in df.iterrows():
        table_lines.append(
            f"| {r['card_name']} | {r['predicted_gih_wr']:.4f} | {r['predicted_gih_pct']} | {r['set_code']} | {r['collector_number']} | {r['rarity']} | {r['colors']} | {r['mana_cost']} | {r['type_line']} |"
        )
    md_table = "\n".join(table_lines)

    report = """# Reality Fracture (`FRA`) 17Lands Games-in-Hand (GiH) Win Rate Predictions & Methodology Report

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

<<MD_TABLE>>
""".replace("<<MD_TABLE>>", md_table)

    Path("/workspace/PREDICTIONS.md").write_text(report)
    print(f"Wrote /workspace/PREDICTIONS.md ({len(report):,} bytes, {len(df)} card rows in single table).")


if __name__ == "__main__":
    main()
