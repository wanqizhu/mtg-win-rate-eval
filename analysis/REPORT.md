# Reality Fracture (FRA): GIH WR prediction report

Every Reality Fracture card gets a predicted 17lands Premier Draft GIH WR (games-in-hand win rate). That covers all 280 main-set cards plus the 10 special guests. The predictions are in `predictions.csv`, along with each card's expert rating, ML deviation and a one-line rationale, and in `PREDICTIONS.md`.

Each prediction blends two signals:
- a card-by-card expert evaluation of FRA in the context of its draft environment;
- a feature-based machine-learning model trained on the 10 historical sets.

The blend weights are calibrated on two **blind** tests (SOS and ECL). For each, every card was rated before any card-level result from that set was viewed. Both the ratings and the leave-one-set-out ML predictions were then compared with what actually happened.

Reproduce the final step with `python final_blend.py`.

## 1. Realized win rates for the 10 reference sets (`compute_hist_stats.py`)
- Per-card 17lands statistics (GP, OH, GD, GIH, GNS, IWD) were recomputed from the raw public game data in `data/17lands`.
- Premier Draft was used for every set except OM1, which only has Pick-Two Draft.
- Outputs are in `out/stats_<SET>.csv` and `out/colors_<SET>.csv`.
- Sets used: HOB, MSH, SOS, TMT, ECL, TLA, OM1, EOE, FIN, TDM.
- **Card-mean GIH by set:** 55.4–56.2 for all sets except TMT (57.3).
- **Rarity priors (deviation from set mean):**

| Rarity | Mean deviation | Spread (SD) |
|---|---|---|
| Common | −1.0 | 2.5 |
| Uncommon | −0.4 | 3.3 |
| Rare | +1.0 | 4.7 |
| Mythic | +3.0 | 4.9 |

## 2. Historical category study (`category_analysis.py`, `print_set_results.py`, `out/set_results/*.txt`)
The per-set result dumps were used to build analog tables for each card archetype.

| Archetype | Example | GIH WR / note |
|---|---|---|
| Common O-ring | Banishing Light | 58.2 |
| Common O-ring | Mothwing Shroud | 59.7 |
| Cheap common cantrip creature | Sibsig Appraiser | 58.9 |
| Cheap common cantrip creature | Codecracker Hound | 60.2 |
| Common burn | Pinecone Strike | 59.4 |
| Common burn | Molten Exhale | 57.9 |
| Common burn | Bombard | 54.7 |
| Taplands / gain lands | — | 54.3–57.6 |
| Bomb planeswalkers and mythics | — | 63–68 |
| Equipment, auras, narrow build-arounds | — | well below the set mean (narrow build-arounds 46–50) |

- **SOS's "prepare" mechanic returns in FRA.** Mid-cost prepare value creatures underperformed in SOS: Campus Composer 54.2, Landscape Painter 53.8, several around 51.5. Cheap, efficient ones did well: Elite Interceptor 59.6.
- ECL was kept out of this study so it could serve as a second blind test.

## 3. ML model (`features.py`, `ml_model.py`)
- **Features (about 100 per card) from Scryfall data:**
  - rarity, mana value, colors and pips, hybrid costs, card types;
  - power and toughness relative to cost;
  - about 15 keywords;
  - regex text flags: removal type (destroy, exile, damage, −X/−X, fight, bounce, tap-down), card draw, loot, tokens, ETB/death/attack triggers, pumps, anthems, counters, lifegain, landcycling, ramp, tutor, discard, mill, surveil, recursion, sacrifice costs, enters tapped, dual land;
  - FRA-relevant flags: Empower, prepare, planeswalker synergy, loyalty, legend synergy, noncreature and self-mill synergy.
- **Target:** GIH WR minus the set's card mean, with sample weights of min(GIH games, 5000)/5000.
- **Models:** LightGBM and Ridge, both validated leave-one-set-out (LOSO).

| Held-out set | ECL | EOE | FIN | HOB | MSH | OM1 | SOS | TDM | TLA | TMT | All |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM r | .58 | .41 | .41 | .31 | .57 | .52 | .47 | .55 | .55 | .52 | **.487** |
| Ridge r | .48 | .22 | .37 | .22 | .52 | .47 | .44 | .57 | .52 | .49 | .426 |

LOSO predictions are in `out/ml_loso.csv`. FRA predictions, from a fit on all 10 sets, are in `out/fra_ml.csv`.

## 4. Expert evaluation of FRA (`blind/fra_expert_ratings.csv`, with a rationale per card)
The full set (`sets/fra.txt`, `sets/fra_guests.txt`) and the comprehensive rules were read, and every card was rated against the environment. The ratings were written before the per-card ML output for FRA was viewed. The main judgments:

- **Empower Jace N is the central mechanic.** It creates or grows a blue Jace planeswalker token with "−1: Surveil 1" and "−3: Draw a card".
  - Empower 3 or more in the main phase is roughly a cantrip.
  - The token is a blue permanent, so it can be hit by Refute Destiny and Terminal Criticism.
  - It can be sacrificed for value, for example by Silence the Echo and Winter.
  - It makes the 10 common dual lands ("Commons" and "Annex") enter untapped. These were rated above typical taplands.
- **Factions:**
  - Fatehold (WU): surveil.
  - Theorix (UB): threshold and mill.
  - Stingerquill (BR): noncombat damage.
  - Konstrari (RG): Heartwood artifacts.
  - Vigorbloom (GW): lifegain and counters.
  - Enemy pairs get hybrid commons.
  - With many duals, Jace tokens, Murmuring Volume, Room of Refuge and Heartwood tokens, fixing is abundant. Gold and three-color cards are therefore penalized less than usual.
- **Color depth:**
  - White and black are deep in removal: Memory Trap, Surgical Precision, Last Gasp, Extended Absence, No Admittance, Prophesied End.
  - Blue has card advantage and tempo.
  - Red has three common removal spells but a historically negative color prior.
  - Green is creature-heavy with weak tricks.
- **Color hosers** (cards that punish an opposing color) each hit about 70% of opponents, so they were rated as somewhat-conditional removal.
- **Planeswalkers:** the bombs (Ajani Unrelenting, both Garruks, The Theorist, Chandra, Torch of Defiance) were rated 63–64. The "Way of the X" enchantments were rated by how their Empower loyalty turns into cards or tokens.
- **Narrow cards:** cards that need planeswalker cards to work, such as Loyal Tutor and Entrust the Spark, were rated very low.

## 5. Blind calibration tests (`blind/sos_blind_ratings.csv`, `blind/ecl_blind_ratings.csv`, `eval_blind.py`)
For SOS and then ECL, every card was rated **before any card-level result from that set was viewed**. The git history shows each ratings file committed before its evaluation. Both the ratings and the LOSO ML model were then compared with actual GIH WR:

| Blind set | Expert r | ML (LOSO LightGBM) r | Expert + ML r (in-sample) |
|---|---|---|---|
| SOS (264 cards) | 0.42 | 0.47 | 0.51 |
| ECL (265 cards) | 0.57 | 0.58 | 0.66 |

Lessons from the blind tests:
- The two signals are only partly correlated, so blending them clearly beats either alone.
- Raw expert ratings are compressed: their spread is smaller than the actual spread.
- The expert ratings systematically underrate rares and mythics relative to commons. ECL had a common mean error of −1.1 and a rare mean error of +0.8. The ML model's rarity prior corrects this.
- In ECL, the ratings were too harsh on tribal synergy payoffs (Gathering Stone, Flamebraider, Chronicle of Victory, the lords) and on odd rares and mythics (Mirrorform, Glen Elendra's Answer).

## 6. Blend (`final_blend.py`, `out/blend_cv.csv`, `out/final_blend.log`)
Several linear blends were compared. Each was fitted on one blind set and scored on the other (cross-set validation):

| Model | Mean out-of-set r | Weighted RMSE |
|---|---|---|
| **expert + LightGBM (chosen)** | **0.576** | **2.87** |
| expert + LightGBM + rarity dummies | 0.554 | 2.88 |
| expert + LightGBM + Ridge + rarity | 0.539 | 2.93 |
| LightGBM alone | 0.522 | 2.94 |
| expert + rarity | 0.510 | 2.95 |
| expert alone | 0.498 | 3.09 |

- z-scored expert ratings and separate weights for rare/mythic versus common/uncommon were also tested. Both were slightly worse, so the simplest model was kept.
- **Final fit (pooled across both sets, weighted by games):**

  dev = −0.23 + 0.332 × (expert − mean expert) + 0.755 × LightGBM deviation

- **Final prediction** = 56.02 + dev, where 56.02 is the average card-mean GIH of the 10 reference sets.
- The predictions are deliberately shrunk, as the blind tests show is optimal for squared error. Their spread is 2.3 points, against about 3.5–4 for realized values.

| Rarity | Predicted mean | Historical mean |
|---|---|---|
| Common | 55.1 | about 55.0 |
| Uncommon | 55.5 | about 55.5 |
| Rare | 56.8 | about 57.3 |
| Mythic | 59.0 | about 59–61 |

## 7. Special guests (`guest_calibration.py`, `out/guest_calibration.*`)
- All 10 FRA guests are printed at mythic, but the ML model only saw main-set rarity.
- To check whether that inflates them, all 147 historical bonus-sheet and guest cards with at least 500 games in hand were predicted with LOSO models and compared with their realized GIH.
- Weighted by games, the residual for bonus-sheet mythics is +0.1, so the model is essentially unbiased for them. Bonus-sheet cards overall run about 1.0 above the ML prediction. The measured shift for each printed rarity is applied to the guests.

## 8. Outcome overview

| Group | Cards | Pred GIH WR |
|---|---|---|
| Top | Ajani Unrelenting / Kwia Vigorbloom / Garruk, Veiled Butcher / Uldaros Theorix / The Theorist, Jace Beleren | 64.3 / 63.9 / 63.4 / 63.1 / 63.0 |
| Best commons | Memory Trap / Surgical Precision / No Admittance / Extended Absence / Awaken the Inferno | 57.3 / 57.2 / 57.1 / 56.9 / 56.9 |
| Bottom | Gardenize / Identity Echo / Root Maze / Arni, Renowned Champion / Medic's Kitesail | 50.8 / 52.0 / 52.3 / 52.4 / 52.5 |

**Predicted means by color:**

| Color | Mean |
|---|---|
| Blue | 56.5 |
| Multicolor | 56.5 |
| White | 56.4 |
| Black | 56.3 |
| Red | 55.7 |
| Green | 55.7 |
| Colorless | 54.2 |

## 9. Artifacts

| File | Contents |
|---|---|
| `out/stats_*.csv`, `out/colors_*.csv` | Realized card and color stats for the reference sets |
| `out/historical_features.csv`, `out/fra_features.csv` | Feature matrices |
| `out/ml_loso.csv`, `out/fra_ml.csv` | LOSO and FRA ML predictions |
| `blind/*_blind_ratings.csv`, `out/blind_eval_*.csv` | Blind calibration ratings and their evaluations |
| `blind/fra_expert_ratings.csv` | FRA expert ratings with per-card notes |
| `out/blend_cv.csv`, `out/final_blend.log` | Blend selection |
| `out/guest_calibration.*` | Guest-sheet bias check |
| `out/fra_blend_full.csv` | All per-card components |
| `sets/*.txt` | Readable card-text dumps |

## 10. Caveats
- The FRA set mean is assumed to equal the historical average. A set-wide shift would move every card equally.
- Color-level strength in a new format is the largest source of shared error.
- Based on the blind tests, expect a per-card mean absolute error of about 2.3 points.
