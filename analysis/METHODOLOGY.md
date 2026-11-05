# Star Trek Premier Draft GIH predictions — GPT-6

The deliverable is [`predictions.csv`](../predictions.csv): one row per card, with `predicted_gih_win_rate` expressed as a fraction. For example, `0.580` means **58.0%**. The table has 311 unique names and is ordered by the base printing's collector number. Predictions are rounded to 0.1 percentage point; that precision is an output convention, not a claim of that degree of accuracy.

## Scope and evidence boundary

All card information and outcomes came from the frozen local snapshot. No Star Trek results, reviews, pick orders, articles, images, or other information were fetched. Network use was limited to installing CatBoost from PyPI and publishing this work to the requested GitHub repository. Card prices, EDHREC rankings, purchase links, and other market/popularity fields were not model inputs.

The supplied TRK extract has 576 English printings and 311 unique names. Each name has an Arena printing in the supplied data. Alternate art and other cosmetic printings are collapsed. All base cards 1–311 are covered, including the five basic lands. The snapshot contains no Star Trek Special Guests extract. The 30 SDS cards are excluded: their supplied `games` fields are paper-only, and the SDS set object lacks an Arena code. This is a scope decision from the supplied snapshot, not a separately verified Arena collation announcement. Uniformly false `booster` flags in the TRK extract are not used to exclude the main set.

The September Comprehensive Rules were read together with the November 4 Star Trek release notes. In particular, rules 601.2b, 702.184 and 721 establish modal choices and station operation; the release notes supply the new beam, dilemma, assimilation and federation rules. Focused card-specific rulings resolved the important unusual cases below. [`input_manifest.json`](input_manifest.json) records source hashes and software versions. The existing frozen leakage report was checked for zero target-set 17lands rows/files.

## Historical labels

I aggregated **4,392,417 games** from FIN, EOE, OM1, TLA, ECL, TMT, SOS, MSH, HOB and FRA. Nine use Premier Draft; OM1 uses its available Pick Two Draft format. The event and expansion columns are asserted for every input chunk. No sealed, traditional draft, replay, or later online data is mixed into the labels.

For card c in game g, exposure is:

```
seen(g,c) = opening_hand(g,c) + drawn(g,c) > 0
GIH(c)   = sum(won(g) * seen(g,c)) / sum(seen(g,c))
```

This counts a game once even if several copies were seen. It follows the supplied `demo_winrate.py`, including exclusion of tutored-only exposure. Copy-weighted rates, opening-hand rates, drawn rates, deck rates, sample sizes and improvement-when-drawn diagnostics are also retained. The copy-weighted alternative is a diagnostic, not the target definition.

The aggregation also records rank distributions, on-play outcomes, turn distributions and main-color archetype outcomes. GIH is observational: it includes draft selection, deck construction, color strength, player selection and game-duration effects. It is not a causal estimate of replacing a card in an otherwise identical deck. I therefore model the observed win rate directly rather than treating a generic card-power score as the answer.

A consequential data repair was matching OM1's `printed_name` aliases to its original card names. Without it only 40 historical OM1 examples joined; with it 184 do. Missing bonus-sheet card text is not invented. The supervised pool contains **2,369 set/card observations with at least 400 GIH games and supplied matching rules text**. Another 289 observations passing the sample threshold have no matching supplied text and are listed in [`unmatched_historical.csv`](unmatched_historical.csv).

## Learned models and validation

The regression response is in percentage points, normalized to remove each historical set's overall population baseline:

```
y = 100 * historical_GIH - 100 * historical_set_game_WR + 55.5
```

The target baseline is then restored using the equal-set mean of the nine historical Premier Draft population rates: **55.7663%**. This avoids claiming that the unknown target population's baseline can be measured before release. OM1 informs card evaluation but does not determine this Premier Draft population prior.

Input features include mana value, pips, color requirements, hybrid flexibility, printed stats, card types, rarity, creature versus dormant Spacecraft status, token material, evasion, protection, entry/death/attack triggers, removal, draw, filtering, recursion, mana production, modal abilities and related rules patterns. Rules text is also represented with word unigrams/bigrams and character n-grams. Exact full-name self references and reminder text are normalized. Cosmetic and commercial metadata are omitted.

I compared:

* word/character TF-IDF plus numerical features with Ridge regularization 1, 4, 16 and 64;
* numerical-feature Ridge models;
* CatBoost depths 4 and 6, using numerical features plus 100 text-SVD components;
* similarity-weighted nearest-text neighbors, which performed worse and were not selected.

Validation holds out an **entire set**, refitting vocabulary, scaling, SVD and regressors on the other sets. It is not a random split of cards within the same environment. Historical reprints in other sets remain legitimate training examples. This is leave-one-set-out validation, not a chronological prospective backtest.

Training weights are `GIH_n / (GIH_n + 1800)`, reducing noise from tiny samples without letting the most-played common dominate all rare cards. The selected machine ensemble is:

```
M = 0.40 * Ridge(alpha=1)
  + 0.25 * Ridge(alpha=4)
  + 0.175 * CatBoost(depth=4)
  + 0.175 * CatBoost(depth=6)
```

Across 2,369 held-out observations, its RMSE is **3.33 percentage points**, versus **3.83** for a constant normalized baseline. Its reliability-weighted RMSE is **3.03**. Full fold results and individual model predictions are retained.

I also reran all nine Premier Draft validation folds without OM1. On exactly the same 2,185 test observations, inclusion gives RMSE **3.320**, versus **3.343** without it; weighted RMSE is **3.019** versus **3.042**. The difference is small but favors retaining the repaired historical data. The final machine model includes OM1.

## Human evaluation and empirical calibration

Every target card has an individual assessment in [`target_expert.tsv`](target_expert.tsv). Grades distinguish entry value, repeated advantage, defensive floor, mana demands, support requirements, likely home, and specific failure modes. This is where a new mechanic's meaning and actual supporting cards enter the forecast, rather than expecting text similarity alone to infer them.

To put these judgments onto a realized GIH scale, I also graded 308 historical cards in two deterministic rarity-stratified samples. Of these, 302 have eligible matched outcomes. Grades were assigned before querying the samples' calibration residuals. Some historical examples were previously visible or recognizable, and the second sample was assessed after learning from the first sample's errors. These judgments are consequently **not a fully blinded independent experiment**. The historical calibration scores are frozen in the artifacts rather than silently rewritten to match outcomes.

The selected calibration is a small Ridge model of grade, lower/upper grade hinges, mana value, land status, gold requirements, expensive creatures without an immediate use, and setup enchantments. Immediate uses recognize entry material, removal, counters, exploration, haste, beginning-of-combat effects, copying and cycling. A disabling Aura is not classified as an engine that waits for future triggers. The small feature set is intended to prevent fitting a separate arbitrary rule for every historical example.

Using set-held-out calibration on these 302 examples:

| Predictor | RMSE, percentage points |
| --- | ---: |
| Machine ensemble on this same sample | 3.63 |
| Calibrated human model alone | 3.02 |
| 80% calibrated human + 20% machine | **2.94** |

The selected blend's MAE is **2.32 points**, reliability-weighted RMSE **2.78**, and mean error **+0.07**. A 65% human blend marginally improves the weighted metric but worsens unweighted RMSE. Since the requested table covers every card, the final ordinary weight is 80% human. Critical unusual-wording cases use 90% human. That latter weighting and the final target-only adjustments are judgment calls, not separately validated performance gains.

Several calibration specifications and weights were compared using these folds, so these numbers contain model-selection optimism. They describe the historical sample, not guaranteed accuracy on TRK, and do not evaluate the final manual target corrections.

## Environment and interactions

The ten color pairs were evaluated as overlapping packages, not ten sealed silos. Hybrid cards often fit multiple pairs; true gold cards impose stronger mana demands. Colorless artifacts connect several strategies.

| Pair | Principal plan and constraints considered |
| --- | --- |
| WU | Spacecraft with entry value, cheap artifact bodies, station/untap tools, tempo and ETB replay. Multiple dormant ships still need a creature board. |
| WB | Borg and artifact-creature material, token bodies, lords, affinity, sacrifice and artifact recursion. Borg do not contribute federation types. |
| UB | Modal spells and ETB creatures plus repeatable dilemmas; Seven of Nine and Garak reward actual mode choices rather than any spell. |
| UR | Noncreature casts, Officers, cheap filtering and beam spells; station can tap creatures without requiring them to survive combat. |
| BR | Klingon bodies and temporary hasty tokens, sacrifice outlets and growth; theft is much stronger when the stolen creature can be sacrificed. |
| BG | Self-mill, recursion and cards leaving the graveyard; beam exile and graveyard activations feed Vash, Sela and Stalker. |
| RG | Ramp, temporary or repeated land entry, landfall creatures and expensive finishers; repeated New Planets add landfall but only one new name. |
| GW | Permanent counters, counter protection, lifegain and token boards; some engines need a creature already present to matter. |
| GU | Fixing, ramp, distinct land names and value creatures; five or seven names cannot be assumed from ordinary two-color basics. |
| RW | Eight-total-power attacks, tokens, power boosts and tempo; charging ships competes with attacking even when their future power helps the threshold. |

The supplied pool has considerable modal support: the compatible UB pool contains 21 modal nonlands across all rarities, plus modal lands, and repeatable modal attack triggers. A deck with six modal cards, conditional on seeing a payoff, has about an **88%** chance to see at least one modal card and **55%** to see at least two among its other cards by turn six on the play. This is access, not a promise that both can be cast after the payoff. Tal Shiar Operative provides repeated triggers without requiring another card for every dilemma. Modal lands matter too; choosing multiple modes would still be only one dilemma.

Beam has three distinct costs: mana for the graveyard spell, temporarily losing a creature's board presence, and paying to replay it. Returning an ETB creature can be profitable, especially with cheap bodies and spare mana, but is not free flashback. Tokens can pay the return cost but disappear. Scotty grants beam only to castable instant/sorcery spells, only during your main phase, and cannot return himself for his granted beam ability. Marooned prevents payment of a beam return cost with the enchanted permanent.

For federation, all types on a non-Borg creature count, including both species and job; all Borg creatures are ignored. Controlled board samples show a two-type payoff plus two other creatures commonly reaching roughly **4.4–5.3 types**, depending on colors. This supports Saavik and Hoshi while giving little basis for treating Sarek's fifteen-type payoff as routine. These are conditional board samples, not predicted probabilities of maintaining that board through interaction.

The unique-name simulations use 40-card, 17-land decks and sequence new names first. With two basic names and three distinct nonbasic names, turn-seven mean names are **2.88**, and five names occur only **2.5%** of the time. Adding a guaranteed New Planet on turn four raises the mean to **3.88** and the five-name probability to **21.6%**. Additional basics, more nonbasics and ramp are tested separately. Thus the first New Planet is valuable, but making three copies is not three different names. These assumptions are deliberately explicit and should not be mistaken for exact draft-deck frequencies.

Station scenarios condition on a ship in the opening hand, then model land drops, a rarity-weighted creature curve, entry tokens and either no attrition or 80% creature survival per turn. All available bodies may be used to charge, including summoning-sick creatures. Under the attrition scenario, D-7's three-counter engine is usually ready by turn three, while the larger ships often need one or two additional turns after casting. Charging consumes real power that could have attacked or blocked. The simulator omits colored-mana constraints, most other abilities, ramp, discover hits and strategic combat decisions. It measures consistency and opportunity cost, **not game win rates**, and is not used as a substitute for card judgment.

## Unusual cards and final corrections

The final settings retain explicit, small point adjustments for cases where learned text and calibrated grade still miss the actual interaction. The exact values and reasons are in [`final_settings.json`](final_settings.json); every term is decomposed in [`prediction_components.csv`](prediction_components.csv).

Examples of rule-sensitive decisions:

* Picard's anthem and token creation from both creature and Spacecraft casts can snowball with ordinary cards; the seven-Officer tutor is upside rather than a prerequisite. Historical Agent Bishop and Mighty Mutanimals help anchor the upper tail.
* Spock doubles every card draw, including routine draw steps and draw-two spells. He is stronger than a conditional once-per-turn draw engine.
* Holo-Engineer replaces token characteristics completely: Food, Clues, Treasures, Officers and even land tokens become 3/3 Illusion artifacts. It does not retain their original abilities. This supplies enormous combat material but can remove fixing or other token utility.
* Beverly Crusher doubles her own lifegain trigger, giving two counters per event; she still needs an actual source of lifegain.
* Kobayashi Maru's fourth upkeep ordinarily forces the loss mode. Its useful homes require a fast finish, blink, sacrifice, or a way to counter that trigger. Selection into decks with such outs is why its forecast is above treating the loss as completely unavoidable.
* Quark prohibits sacrificing Treasure tokens for **any** reason, including other sacrifice costs. Beam can remove Quark and unlock stored mana, but Treasures are not immediately functioning ramp while he remains.
* The Measure of a Man is temporary removal. The City on the Edge of Forever phases out both boards, and ending its lock can return opposing permanents before yours.
* Mirror Universe Intrusion's explicit kicker is `{4}{U}{B}`. The release notes resolve the inconsistent shorter reminder in the JSON text; the kicked spell costs nine total mana.
* The Holodeck still requires paying the replayed permanent's cost. Infinite Diversity requires nine mana, a suitable graveyard spell and a surviving diverse board. These are not treated as unconditional bombs because their payoff text is impressive.

Color-pair strength was separately tested from predicted common/uncommon quality and structural pool features. Its out-of-set predictive power is weak: the best strongly regularized structural model has archetype RMSE about **1.75 points**, compared with **1.80** for zero adjustment. Therefore only half of its centered forecast is applied, producing small context shifts rather than confident declarations that an entire color will dominate. Basic lands receive the population baseline plus this context shift, rather than a spell-like power grade. The color experiment reuses precomputed card-level out-of-fold features rather than refitting that entire feature pipeline inside every outer color-validation fold, so its small gain is exploratory rather than a fully nested validation result.

## Verification and practical limits

[`verification.json`](verification.json) records passing coverage, uniqueness, units, rounding and source-scope checks. An independent Python CSV-reader implementation recounts all **297,964 FRA Premier Draft games** for a mythic, a common and a basic land and exactly reproduces their GIH counts and wins. It encounters 110,776 multiple-copy exposures among those checks, verifying that the aggregation does not accidentally count each copy as another game. The output SHA-256 is recorded in both verification and prediction summary artifacts.

The final forecasts range from **44.1% to 66.5%**. Their narrower spread than raw historical rates is intentional regression toward conditional expectations, not an assertion that all realized cards will fall inside this range. Remaining uncertainty includes the future player population, actual draft/deck frequencies, color balance, nonlinear combinations, metagame adaptation, and rare-card sampling noise. No full rules-complete draft/combat simulator or prospective TRK results were used.

## Reproduction

Python 3.12 was used. Pinned package versions are in [`scripts/prediction/requirements.txt`](../scripts/prediction/requirements.txt). Set `MTG_DATA_ROOT` if the frozen snapshot is stored elsewhere.

```bash
python -m pip install -r scripts/prediction/requirements.txt
python scripts/prediction/prepare.py
OPENBLAS_NUM_THREADS=1 python scripts/prediction/model.py --full
OPENBLAS_NUM_THREADS=1 python scripts/prediction/calibrate.py
OPENBLAS_NUM_THREADS=1 python scripts/prediction/refine_calibration.py
OPENBLAS_NUM_THREADS=1 python scripts/prediction/color_context.py
OPENBLAS_NUM_THREADS=1 python scripts/prediction/environment.py
OPENBLAS_NUM_THREADS=1 python scripts/prediction/ablate_om1.py
python scripts/prediction/finalize.py
python scripts/prediction/verify_predictions.py
```

Human assessments and final adjustments are fixed inputs, not regenerated by training. Random seeds are fixed in the model and simulations. The compressed normalized card text is included for inspection; the preparation script regenerates the uncompressed working file from the supplied snapshot.

## Publication provenance

The complete table and its SHA-256 were finalized and verified before discovering that the requested `gpt-6` remote branch already existed. Publishing inspected only Git refs, tree paths, object hashes and PR identity metadata. No existing prediction or analysis contents were read or used. The branch history and older analysis directory are preserved; the current submission replaces `predictions.csv`, and the report and artifacts directly under `analysis/` describe this independently generated submission.
