# Reality Fracture (FRA) — 17lands GiH win-rate predictions

Model/branch: `muse-spark-1.3` (`cursor/muse-spark-1-3-c70e`)

Primary deliverable: `predictions/fra_gih_predictions.csv` — one row per card,
295 rows = 285 unique Reality Fracture cards (by `oracle_id`, basics included)
plus the 10 Special Guest (`spg`) cards that appear in the bonus slot.
`predicted_gih_wr` is the predicted 17lands PremierDraft Game-In-Hand win rate.

Because 17lands names modal double-faced / preparation cards by their front face
(e.g. game data uses `Diviner of Victory`, Scryfall uses
`Diviner of Victory // Unwind History` — verified on TLA/FIN historical data),
the table carries both `name_full` (Scryfall) and `name_17lands` (front-face
short name) plus `oracle_id`, `collector_number`, `type_line`, `mana_cost` so the
predictions join correctly however the evaluator keys cards.

## 1. Data used (all local, no set lookup)

- 17lands PremierDraft `game_data_public.*.PremierDraft.csv.gz` for 9 sets
  (HOB, MSH, SOS, TMT, ECL, TLA, EOE, FIN, TDM) + OM1 `PickTwoDraft` game data
  (excluded from training — see below). GiH computed as: game counted when the
  card's `opening_hand_*` or `drawn_*` column is numerically > 0
  (the demo script's `not in {"", "0"}` check misfires on float-formatted zeros
  such as EOE's `Burgeoning`/`Magus of the Moon` columns, which briefly made
  those look like every-game staples; fixed with float parsing).
- 17lands `cards.csv` / `abilities.csv` for set-membership reference.
- Scryfall per-set extracts for all 10 historical sets + `fra.jsonl.gz` (462
  printings = 285 unique `oracle_id`s, 5 of which are basics) +
  `fra-special-guests.jsonl` (10 cards) + `set_objects.json`.
- Comprehensive Rules text (used only to understand the new mechanics:
  Preparation Cards §722, Empower Jace §701.71, Behold §701.4, Heartwood §111.10x).

## 2. Target construction and name mapping

Per set, GiH WR = wins / games-in-hand over the chosen game file
(HOB 323k games @ 55.9–56.2% base rates across sets; full per-set tables in
`analysis/gih2_*.json`). Game-data card columns were intersected with the set's
Scryfall names, mapping front-face names to full modal names
(verified: FIN/TLA game data contains zero `//` names; every modal card appears
under its front face). Non-set bonus-sheet columns (e.g. TLA's Noxious Gearhulk,
SOS's Burst Lightning, MSH's Sword of Fire and Ice) were excluded from training
but confirm bonus-slot cards do get GiH tracking — hence SPG predictions are
included. OM1 matched only 35/232 game columns to its Scryfall set (PickTwo
mixed-pool format, no `cards.csv` entry) and was excluded from training.

Training set: 2,338 rows (HOB 193 … FIN 298 matched cards/set).
Card-average GiH means per set: 0.551–0.569, grand mean 0.5573.

## 3. Predictive model (prior)

Features per card from its canonical Scryfall object: CMC/log-CMC, colors,
multicolor, rarity one-hots, card-type one-hots (creature/land/instant/sorcery/
enchantment/artifact/planeswalker/legendary/basic), power/toughness (+sum,
stat-efficiency vs CMC), keyword counts, 30+ text-signal flags (flying,
removal words, draw, counter, token, surveil, sacrifice, search, reanimate,
copy, cycling, flashback, landfall, prowess, …), fine extras (vanilla-like,
ETB, unconditional vs conditional removal, sweeper, burn, aura/equipment/saga,
mythic/rare-bomb, cantrip), plus TF-IDF (1–2 grams, 2000 feats, min_df=4,
reminder text stripped) over oracle text + type line.

Target: `gih_wr − set_mean`, sample-weighted by √games-in-hand.
Final prior = 0.6·Ridge(α=30, numeric+TF-IDF) + 0.4·HistGradientBoosting
(numeric), re-offset by the set mean. Leave-one-set-out RMSE = **0.0369**
vs 0.0413 for the always-predict-set-mean baseline (per-set: ECL .0374, EOE
.0347, FIN .0355, HOB .0494, MSH .0349, SOS .0325, TDM .0335, TLA .0324,
TMT .0418; correlations ≈ 0.37–0.56). Scripts: `analysis/train_*.py`,
`analysis/build_features.py`; training rows: `analysis/train_rows.json`.

Known blind spots of the prior (all addressed in §5): `Empower Jace`,
`prepared`, `Heartwood`, `behold a Jace` are out-of-vocabulary for the TF-IDF
(min_df filtering, no historical occurrences), so the prior prices them at ~0;
narrow color-hate removal/counters get generic removal credit; preparation-card
flexibility (modal 2-for-1) has no historical analogue in the text model.

## 4. Reality Fracture set context (from spoiler + rules only)

- **Empower Jace (35 cards):** creates (first time) a blue Jace planeswalker
  token (`−1`: Surveil 1, `−3`: Draw a card) with N loyalty. Repeatable
  surveil/draw engine available at common (Academic Ascent, Mindseeker Oculus,
  Hexhaven Battalion, …). Implications: card-advantage games, Jace-conditional
  cards (`behold`, loyalty-matter) play at near face value; answers hitting
  planeswalkers (Vindictive Triumph, Solitary Cell, Lich's Relic, Extended
  Absence, …) carry a premium; Mabel, Bitter Recluse answers loyalty directly.
- **Prepared / preparation cards (24+):** creature enters prepared → a castable
  copy of the inset spell sits in exile (rules §722.3c) — modal 2-for-1s in the
  Adventure/Foretell family, which historically overperform. Premium scaled to
  spell quality; upkeep/attack/threshold re-prepare engines (Paradox Shaper,
  Stingerquill Voxmancer, Bloodline Recollector, Hexhaven Dueling Arena) rated
  as repeatable value.
- **Heartwood tokens (RG artifact mana):** fixing/ramp glue; Draconic Visitor
  (artifact tokens → 5/5 Dragons) and artifact-count payoffs (Craterclaw
  Colossus, Pia, Karn) graded with Heartwood online; Tenured Tethermage as
  ramp; Crafter/Improviser/Prodigy lines as fixing dorks.
- **Surveil (47 cards) + threshold/flashback/mill payoffs:** Dark Matter
  Manipulator, Tarmogoyf, Null Summoner, Loot/Proft lines, Something Worth
  Saving,Recursive Recruitment all graded with a full graveyard in mind.
- **Structure:** main set #1–194 (81 commons) + 91-card legendary bonus sheet
  (#195–280: 66 uncommon legends) + 5 basics + 20 lands. Legendary density (95)
  is high but precedented (MSH 132, FIN 115); legend-rule clashes treated as a
  small drag on filler legends, offset by legend synergies (Fateshaper
  Aspirant, Wrath of the Bloodmane discount, Legendary Sorceries matter).
  Curve: 2-drops dominate (61 @ CMC 2); removal is medium-depth with heavy
  conditional/color-hate share (Refute Destiny, Terminal Criticism, Essence
  Burn, Flourishing Grapple, Precise Redaction discounted accordingly).
  Lands: slow-dual rare cycle (near-untapped duals, +0.012) and
  planeswalker-untapped common Annex cycle (+0.014 given Jace ubiquity).

## 5. Card-by-card review and final adjustments

Every one of the 295 cards was read (full texts in `analysis/fra_review.txt`)
and scored against the prior. `analysis/apply_adjustments.py` holds the complete
per-card adjustment ledger (`MANUAL` dict, also persisted per-row as
`adjustment` in `analysis/fra_final.json`): mechanic premiums (Empower scaled
by N/cost, prepared +0.008–0.016 by spell quality, Heartwood +0.006/token,
landcycling/flashback smoothing), narrow-hate discounts (−0.006 to −0.014),
aura/build-around haircuts (Gardenize, Omnipresence, Puppet Crafting,
Entrust the Spark, Extrapolate), equipment-filler repair (Hunter's Axe,
Medic's Kitesail, Artifist Acumen cantrip), bomb confirmation (Garruks,
Jace Beleren, Uldaros/Aerid, Kindred Judgment, Overwrite, Austere Command,
Sublime Epiphany, Consider in a surveil set), and SPG-specific calls
(Eye of Ugin with ~no Eldrazi support −0.030; Root Maze symmetric −0.008).
Final level recentered by −0.0083 to the historical grand mean (0.5575) so
mechanic optimism cannot inflate the set level — only relative ordering changes.

Final spread: mean 0.5575, sd 0.023, range 0.504–0.637; rarity medians —
common 0.550, uncommon 0.546 (dragged by the 66-card filler-legend bonus sheet),
rare 0.570, mythic 0.585. Top: Garruk, Curse Breaker 0.637, Avatar of
Burgeoning Echoes 0.626, Jace Beleren 0.616, Garruk Veiled Butcher 0.615,
Austere Command 0.610, Diviner of Victory 0.610. Bottom: Gardenize 0.504,
Root Maze 0.505, Karn Argent Defender 0.512, Medic's Kitesail 0.515.

## 6. Diagnostics and risks

- `analysis/fra_base.json` (pure model prior) vs `analysis/fra_final.json`
  (prior + adjustments) are both committed for audit.
- Biggest risks: (a) Jace-token equilibrium — if Jace is answered cleanly and
  often, Empower premiums (+0.004–0.022) are 0.5–1pt rich; (b) prepared-spell
  pacing — if exiled copies expire under pressure more than Adventure analogues
  suggest, prepared premiums are rich; (c) bonus-sheet draft frequency — if the
  91 legends flow at a different rate than main-set uncommons, their
  low-sample realized rates will be noisier than predicted; (d) SPG pull rate —
  10 guests predicted at full confidence but realized samples will be thin.
- Deliberately shrunk toward the mean (sd 0.023 vs historical realized 0.039):
  optimal under squared error given per-card sampling noise in the truth.
