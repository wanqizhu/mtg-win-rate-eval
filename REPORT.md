# Reality Fracture (FRA) — predicted 17lands GIH win rates

Model: Claude Fable 5.1 (branch `cursor/claude-fable-5-1-reality-fracture-gih-e8bf`).

Deliverable: `predictions/reality_fracture_gih_predictions.csv` — one row per card
(280 main-set cards, 5 basic lands, 10 Special Guests that appear in FRA boosters),
column `predicted_gih_wr` is the predicted Premier Draft "game in hand" win rate as a
fraction (e.g. `0.581`). `predictions/reality_fracture_gih_predictions.md` is the same
table in Markdown. `work/predictions_fra_detailed.csv` adds the raw expert number, the
statistical-model baseline and a one-line rationale per card.

No information about FRA other than the Scryfall dump, the comprehensive rules and the
prior-set 17lands data on disk was used.

## 1. Data preparation

| Step | Script | Output |
|---|---|---|
| Per-card stats (GIH/OH/GD/GP/GNS WR, IWD, ALSA, ATA) for all 10 prior sets and every format in `17lands/` (Premier/Trad/PickTwo/Sealed), computed from `game_data_public.*.csv.gz` (`opening_hand_X + drawn_X + tutored_X > 0` → GIH) and `draft_data_public.*` | `work/scripts/compute_card_stats.py` | `work/stats/<SET>_<FORMAT>_card_stats.csv`, `work/stats/set_summary.csv` |
| Oracle-text feature extraction (rarity, cmc, colors, types, P/T, keywords, regex features for removal/draw/ETB/tokens/lifegain/etc.) | `work/scripts/features.py` | — |
| Join stats with features for 2 730 historical cards (Premier Draft; PickTwo for OM1) | `work/scripts/build_training.py` | `work/training_cards.csv` |
| Full FRA card dump (all 280 main-set cards with oracle text) and SOS dump with realised stats | `work/scripts/dump_set_with_stats.py` | `work/fra_cards_full.txt`, `work/fra_main_cards.json`, `work/sos_cards_with_stats.txt` |
| Analog lookup tool (regex over historical oracle text → GIH WR, n, ALSA) | `work/scripts/query_analogs.py` | used interactively |

Leakage check: `manifests/leakage_report.json` confirms zero FRA rows in the 17lands data.
FRA has no bonus sheet; the 10 Special Guests (SPG) are included because SPG cards show up in
17lands card data for prior sets (e.g. ECL Bitterblossom), flagged `section = special_guest`.

## 2. Statistical baseline (weak prior)

`work/scripts/train_model.py` trains, on set-centred GIH WR (card WR minus set mean):

* LightGBM on ~150 hand-crafted features (lr 0.03, 15 leaves, 600 rounds, weight √n_gih),
* TF-IDF (1-2 grams of normalised oracle text) + Ridge,
* blended 0.6 / 0.4.

Leave-one-set-out CV (`work/model_cv_report.csv`): blend MAE 0.0256 vs 0.0287 for the
"predict the set mean" null, mean correlation ≈ 0.46 (range 0.26 HOB – 0.55 TLA).
Top features: rarity, text length, power/toughness, cmc, ETB, tokens, flying, legendary.

The baseline is useful for rarity/cmc/stat-line priors but is blind to everything that
matters in FRA (empower Jace, prepare, planeswalker payoffs, color hosers, do-nothing
constructed rares). Its FRA predictions are in `work/model_predictions_fra.csv` and are
kept only as a sanity check — the final numbers are the expert review below
(correlation between the two: 0.59, mean |diff| 1.7 points).

## 3. Format analysis (what drives the expert numbers)

* **Structure.** 194 "main" cards + 86 legendary alternate-reality characters (#195–280:
  66 U / 14 R / 6 M) occupying a dedicated slot. Five allied-colour Elder Sphinx factions:
  Fatehold WU (scry/surveil + counters), Theorix UB (self-mill/threshold), Stingerquill BR
  (noncombat damage to opponents, 2/2 Cadet tokens), Konstrari RG (Heartwood mana
  artifacts), Vigorbloom GW (+1/+1 counters, lifegain). Enemy pairs are supported by gold
  uncommon legends (WB sacrifice, UR spells/Thopters, BG graveyard, RW go-wide, GU
  planeswalkers).
* **Empower Jace N** (CR 701.71): creates a 0-loyalty blue Jace token with `−1: surveil 1`
  and `−3: draw a card`, then adds N counters, usable the same turn at sorcery speed.
  Valuation used: N ≥ 3 = an immediate card, N ≥ 6 = two cards over two turns, with a
  discount for delay/vulnerability; plus a large "you control a planeswalker" bonus because
  it untaps the 10 common dual lands, enables Sanctum Lurker, Mind Meanderer, Compel
  Brutality mode 2, Kiora, Avatar, the *Way of …* enchantments, Face Yourself, Tomik.
* **Prepare** (CR 722): a creature that enters prepared is a body plus a castable spell
  copy; SOS showed prepared commons over-perform their raw stats. Repreparing (upkeep)
  creatures (Paradox Shaper, Voxmancer, Woodwork Prodigy) are repeatable engines.
* **Removal density.** Common removal is deep in W/B/R (Memory Trap, Surgical Precision,
  Last Gasp, Extended Absence, Silence the Echo, No Admittance, Wrath of the Bloodmane,
  Awaken the Inferno, Infinite Coursework) — so x/1 bodies are penalised and cheap
  card-advantage/2-for-1s are rewarded. Green has the least (Compel Brutality).
* **Colour hosers.** Cycle of five "destroy/exile a <colour or colour> permanent"
  uncommons: ~70 % of opposing 2-colour decks contain one of the two colours but only
  ~40 % of their creatures are hit → priced 52–54.
* **Historical anchors** (from `query_analogs.py`): 4-mana exile instants 56.3–57.0;
  O-ring commons ≈ 58.5; 2-mana 3-damage 55.5–57.5; −3/−3 for 2 ≈ 57.7 (Last Gasp 57.8 in
  SOS); 2-mana destroy with a drawback for the opponent 47–51 (Harsh Annotation, Pongify);
  counterspells ≈ 56.4; 3-mana draw-two 56.4–57.1 (Quick Study, Waterbending Lesson);
  landcycling commons ≈ 55.7; 5-cmc 5/5 fliers ≈ 61; direct reprints Last Gasp 57.8 (SOS)
  and Blazing Crescendo 52.9 (MSH); planeswalkers 53–69; charms 54–60.

## 4. Card-by-card expert review

`work/scripts/expert_predictions.py` holds a hand-written number and rationale for each
of the 295 cards. Every card was evaluated in the context above (which archetype plays it,
what it does against the common removal suite, how often its conditions are met), not as a
vacuum heuristic. Typical considerations recorded in the rationales: stat line vs. curve
slot, evasion, 2-for-1 potential, dependence on a shell (e.g. Cruel Calculations needs
self-mill, Germinate Recruits needs lifegain), castability (Hexhaven Invigorator GGGG,
Karn's hybrid cost), and constructed-only designs that are blanks in limited (Loyal Tutor,
Extrapolate the Impossible, Entrust the Spark, Gardenize, Omnipresence, Root Maze).

## 5. Calibration (`work/scripts/finalize_predictions.py`)

Realised per-set means of card GIH WR (cards with ≥ 500 games) are 55.6–56.2 %; realised
spreads (≥ 100 games) are sd ≈ 2.6 (common), 3.4 (uncommon), 5.6 (rare), 5.6 (mythic).
Raw expert numbers were, as expected, shrunk (sd 1.3 / 1.7 / 3.3 / 3.9), so the final step
rescales deviations from the rarity mean by 1.35 / 1.20 / 1.05 / 1.00 and sets rarity means
to 55.15 / 55.6 / 56.2 / 58.2 (MSE-optimal spread ≈ expected correlation × realised sd).
Lands and basics are not rescaled (their GIH WR tracks the set mean). Result
(`work/calibration_report.txt`): overall mean 55.8 %, common 55.2 ± 1.6, uncommon
55.6 ± 2.0, rare 56.1 ± 3.4, mythic 58.2 ± 3.8.

Top predictions: Kwia Vigorbloom 64.0, Aerid Konstrari 63.0, Chandra Torch of Defiance /
Denzilore Fatehold / The Theorist Jace Beleren 62.0, Vraska the Cutting Glare /
Curse-Marred Demon / Draconic Visitor 61.5. Best commons: Memory Trap 59.4, Last Gasp 58.7,
then the 4-damage/exile/6-damage removal suite at 58.1. Bottom: Entrust the Spark, Gardenize,
Loyal Tutor 46.8, Omnipresence 48.0.

## 6. Reproduce

```bash
PY=/opt/mtg-data/venv/bin/python
ln -s /opt/mtg-data data
printf '%s\n' TMT OM1 HOB MSH ECL EOE TDM TLA SOS FIN | xargs -P 4 -I{} sh -c "$PY work/scripts/compute_card_stats.py {} > work/stats_{}.log 2>&1"
$PY work/scripts/build_training.py
$PY work/scripts/train_model.py          # baseline + LOSO CV
$PY work/scripts/finalize_predictions.py # expert table -> predictions/
```

## 7. Full prediction table

See `predictions/reality_fracture_gih_predictions.md` (also reproduced in the PR description).
