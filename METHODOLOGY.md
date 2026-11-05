# Star Trek Premier Draft GIH predictions

The table is `predictions/trk_premier_draft_gih.csv` (311 rows, one per unique card). `gih_win_rate` is a probability, four decimals, matching `scripts/demo_winrate.py`. The same rows are in `predictions/trk_premier_draft_gih.md`. A one-line reason for every card is in `analysis/prediction_notes.csv`.

Predicted mean across the 311 cards is 0.5585. That sits on the nine-set Premier Draft player win rate of 0.5577. By rarity the means are common 0.551, uncommon 0.553, rare 0.566, mythic 0.588. Spreads are in `analysis/final_distribution.json`.

## What is being predicted

Games in hand, Premier Draft, all ranks, no rank filter. A card is in hand when `opening_hand_<card>` or `drawn_<card>` is present and not `"0"`. The rate is the fraction of those games the recorded player won. Opening-hand and drawn games are pooled. This is raw GIH, not improvement-when-drawn and not deck win rate.

The pool is the English premiere printing of collector numbers 1–311. That block is contiguous: 15 commons in each color, then gold, artifacts, planets, shocks, and basics. Alternate arts start at 312 (surgefoil, showcase, borderless, headliners, Klingon printings). Stardates and the commander decks are not in the draft. `booster` is false on every TRK row because the set is unreleased, so it is not a pool filter. Standard legality is not either: only basics and shocks are legal, and both are in the draft. One row per card name.

Through the Omenpaths has no Premier Draft file (Pick-Two only) and is not in the training labels. The other nine sets are.

## What the last nine drafts actually look like

`src/compute_historical_gih.py` streams each Premier Draft game file and writes card-level GIH to `/tmp/mtg_analysis/historical_gih/`. Set win rates:

| Set | Games | Player WR |
| --- | ---: | ---: |
| FRA | 297964 | 0.5542 |
| HOB | 323114 | 0.5616 |
| MSH | 377514 | 0.5608 |
| SOS | 582914 | 0.5538 |
| TMT | 180204 | 0.5680 |
| ECL | 536566 | 0.5499 |
| TLA | 562924 | 0.5601 |
| EOE | 563418 | 0.5532 |
| FIN | 769659 | 0.5573 |

On-play share is about half. Mean game length is about 9.2 turns, except HOB at 8.85. TRK is not built like HOB. Removal exists in every color, so the planning length is about 9.5 turns: a six-drop has to win the game or replace itself, and a two-drop with no text is a 0.53 card, not a 0.50 card.

Cards with at least 500 GIH, unweighted:

| Rarity | Mean GIH | Std | p10 | p90 |
| --- | ---: | ---: | ---: | ---: |
| Common | 0.551 | 0.025 | 0.517 | 0.580 |
| Uncommon | 0.556 | 0.033 | 0.515 | 0.596 |
| Rare | 0.571 | 0.049 | 0.508 | 0.628 |
| Mythic | 0.595 | 0.046 | 0.534 | 0.652 |

The practical ceiling for a format bomb is the mid-60s. A few outliers reach 0.70–0.74 (FRA Ajani, MSH Civil War, TMT Jitte). The floor for a card that is merely bad, rather than a 7-mana french vanilla, is about 0.48–0.52. Basics track color strength inside a set and usually sit within a point or two of the set win rate (mean 0.560, std 0.009 across 45 basics). Shocks do not default to 0.60. The ten shocks in EOE and ECL average 0.543, from Breeding Pool at 0.568 down to Sacred Foundry at 0.523. Tapped gain-lands in MSH sit around 0.55–0.57.

## Why a text model is not the submission

`src/predict_gih.py` joins those labels to Scryfall, builds structural features plus a text field, and leave-one-set-out stacks LightGBM, histogram gradient boosting, ridge, and a 25-neighbor model. Training rows require at least 200 GIH and are weighted by `sqrt(n)` clipped to 8000, so commons dominate the loss. Bonus-sheet reprints are included and labeled rarity `bonus`, because the printed rarity of a random reprint is the wrong availability signal. Front-face names are used, which is what 17lands records for double-faced cards.

Leave-one-set-out, 2556 cards: MAE 0.0235, RMSE 0.0315, Spearman 0.463, against a null MAE of 0.0275 from predicting each card's set mean. Full numbers are in `analysis/cv_metrics.json`. The stack beats the mean. It does not separate bombs from traps. Predicted standard deviation is about half the actual standard deviation at every rarity (commons 0.011 vs 0.023, mythics 0.020 vs 0.048). Stretching the tails by a constant did not reduce MAE, so the compression is not a simple scale factor. `sqrt(n)` weighting and noisy effect regexes make shrinkage the MAE-optimal fit.

The misses are the ones that matter for a new set (`analysis/loo_worst_errors.csv`):

- EOE Dawnsire, a 5-mana 20/20 with station 20 and no enters ability, was predicted near 0.68 and realized 0.489. Printed power on a spacecraft is the stationed body. The model treated it as a creature.
- MSH Civil War (0.703) and TMT Umezawa's Jitte (0.717) were predicted in the mid-0.50s. Narrow build-arounds and true bombs are under-called.
- HOB Gleaming Splendor (0.390) was predicted at 0.598. Do-nothing enchantments with long text look like value.

Feature correlations on cards with at least 400 GIH tell the same story. Rarity and oracle length beat hand-built effect scores. `removal_eff` correlates at 0.04 and `ca_raw` at 0.07, because the regexes fire on the wrong cards. Color-environment features rank high in gain importance and still leave large leave-one-set-out color bias (`analysis/loo_color_bias.csv`: HOB white commons were predicted 3.4 points too high). Whole colors are not shifted by more than about a point.

`src/expert_rate.py` was a second attempt: score each card in points, then fit a ridge on the points. The fit learned a negative quadratic. High scores were punished. Mr. Spock, the double-draw mythic, came out as one of the worst cards in the set at 0.485. That file is kept so the failure is reproducible. It is not used.

The submitted rates are card-by-card judgments on a fixed scale, one point of card quality to one point of GIH, anchored to the measured analogues above. The gradient-boosted scores were a prior and a disagreement check, not an average. Blending them back in would pull Discovery and Spock toward 0.58, which is the error the model makes on every bomb.

## How this format changes the usual rates

Several TRK words are not keywords in the 25 September 2026 Comprehensive Rules. The reminder text is the definition.

- Beam me up: cast the instant or sorcery from the graveyard for the printed beam cost, and return a creature you control to hand, then exile the spell. A beam cost of `{1}{B}` or `{1}{G}` is a real second copy when the format is full of 1/1s. A beam cost of `{3}{R}{R}{R}` is reminder text. Scotty gives every instant and sorcery in the graveyard beam-me-up equal to its mana cost during your main phase, and he cannot be the creature you bounce.
- Face a dilemma: you face one dilemma when you choose one or more modes, including modal enters abilities. The payoffs (Sisko, Garak, Seven of Nine, Jett Reno, Caretaker, Lorca, the Operative) get credit for how often a normal deck of this set actually chooses a mode. Modal commons are dense, so the trigger is a real once-a-turn effect in the deck that wants it, and a blank in a deck of vanilla creatures.
- Strange new worlds counts differently named lands, not basic land types. A normal two-color deck presents about four names by the midgame. A deck that tries (planets, shocks, New Planet, the check lands) presents six or seven. Every New Planet token has the same name, so the token adds at most one.
- Assimilate puts a creature card from a graveyard onto the battlefield with a +1/+1 counter. It is a Borg artifact creature and loses its other types.
- Federation counts creature types among non-Borg creatures. A normal board is two to four types. Payoffs that need five types (Sarek, the draw half of Saavik's damage) are narrow.
- Rule 614.5: Spock's "if you would draw a card, draw two cards instead" doubles each draw once. It does not loop.
- Discover 5, printed on U.S.S. Discovery, exiles until a nonland of mana value 5 or less and casts it for free or puts it in hand. That is a spell, not a rummage.
- "Outside the game" on Arena limited is the sideboard. The Holodeck plays a sideboard or graveyard permanent, and you still pay for it.
- Token creatures have mana value 0. Engineered Explosives cast for X=0 wipes Officer, Klingon, Borg, and Illusion tokens and leaves lands alone.

Fixing is the other environmental fact. Ten planets (tapped gain-lands, one per pair), ten shocks, five check lands, five anomaly sac-lands, and Unexplored Territory. Gold cards take about half a point of penalty, not two or three. Splashing a removal spell is normal. That lifts Signpost uncommons and three-color cards relative to a fix-light format, and it lifts "strange new worlds" cards whose X is 5 rather than 3.

Spacecraft are scored the way EOE spacecraft behaved, not the way their printed power reads. Warmaker Gunship (enters and deals damage) was 0.620. Uthros Scanship (draw two, discard one) was 0.571. Dawnsire (no enters ability, station 20) was 0.489. Galvanizing Sawship (no enters ability) was 0.499. The stationed body is ignored unless the station number is small enough that a single creature turns it on. D-7 makes a hasty 2/1 at three charge counters. Discovery's discover is an enters ability and does not need station at all.

Color lean is small, on purpose. Leave-one-set-out color residuals of three points are common and are not predictable from the card text. White, green, and blue look slightly deeper (removal, explore, card flow). Red's burn is good and its french-vanilla creatures are not. Basics: Plains 0.566, Forest 0.564, Island 0.562, Swamp 0.558, Mountain 0.554.

## Anchors the rates are tied to

Clean O-rings cluster at 0.582–0.598 (EOE Banishing Light 0.582, n=57627; FIN White Auracite 0.596; MSH Web Up 0.598). Malfunctioning Holodeck exiles any nonland and has ward 2, so it is the top of that cluster at 0.600, which is also the common ceiling. 70,000 Light-Years from Home gives them a land, so it is 0.570 rather than 0.590.

Hard removal: one-mana bounce FRA Unsummon 0.580, so the two-mana Energize is 0.570. Last Gasp-style shrink spells sit near 0.576; Eject the Warp Core is −2/−2, or −6/−6 if you sacrifice a body, at 0.576. Four-mana unconditional destroy (He's Dead, Jim) is 0.578. Lightning Strike was 0.575 in TLA and 0.555 in MSH. Transporter Malfunction is that spell; the six-mana beam cost is ignored. 0.580.

Cantrips on permanents: FIN Combat Tutorial ("target player draws two") 0.591, EOE Cryogen Relic 0.592 because it draws on the way out as well. Tea, Earl Grey, Hot draws once. 0.576. Suurok-Class draws once and is a ship afterward. 0.580. Open Communications is a one-mana sorcery cantrip with a usable beam cost. 0.574.

Cheap equipment without an enters ability averages 0.542, and a lot of it is under 0.50 (Meltstrider's Gear 0.502). Tricorder is +1/+1, equip 1, surveil on attack, and makes no body. 0.556. Type-2 Phaser and Bat'leth get to keep their enters interaction (a stun, a −2/−2), so they are 0.562 and 0.564.

True anthem lords, not the regex that matches every +1/+1, are 0.64–0.66 (HOB Fíli 0.663, Bard's Company 0.641). Weak lords are 0.55. Picard is an anthem and a token on every creature or ship, on a 2/3 for three. 0.648. Deanna Troi puts an extra +1/+1 counter on every creature that enters. 0.634.

No card in the nine sets says "if you would draw a card, draw two cards instead," and none explores. Those two mythics are not copied from a neighbor. Spock is a 2/4 for four with no enters ability, so a removal spell blanks him, and the replacement is the most broken ongoing effect in the file. 0.650. Janeway explores as she enters, then gives an extra land and explores every creature. 0.644. Seven of Nine loots every turn on a 0/4 for two, and the loot becomes a draw at five differently named lands, which this format can present. 0.638. Discovery discovers 5 on enter, before anyone can remove it, and again on attack once stationed to 6. 0.658.

Ramp: EOE Larval Scoutlander 0.597. Voyager puts two lands onto the battlefield for four mana. 0.604. Final Frontier is a Rampant Growth for a basic or a Planet, and beam `{1}{G}` recasts it. 0.582. Warp Speed makes lands enter untapped in a format of ten tapped planets, and finds a land. 0.596.

Shocks follow pair quality inside the historical 0.523–0.568 band: Breeding Pool 0.560, Hallowed Fountain 0.558, Temple Garden 0.554, Watery Grave 0.554, Overgrown Tomb 0.552, Stomping Ground 0.550, Godless Shrine 0.548, Steam Vents 0.540, Sacred Foundry 0.536, Blood Crypt 0.534. Planets are a little under the average of their colors because they enter tapped, and a little over a generic tap-land because gain 1 and a unique name both matter here. Vulcan 0.564, Risa 0.562, Earth 0.560, down to Qo'noS 0.548. Unexplored Territory taps for colorless and costs one to find a tapped basic. Evolving Wilds was 0.550 and Terramorphic Expanse 0.566. 0.554.

## Cards the text model gets wrong, and why the rate moved

The gradient-boosted prior is `analysis/trk_raw_predictions.csv`. Mean absolute gap versus the submitted rate is about 1.7 points. The large gaps are the point of doing this by card.

- U.S.S. Discovery, prior 0.585. `ca_raw` saw the discover and the trees still shrunk it. Submitted 0.658.
- Mr. Spock, prior 0.587. No training analogue, and the ridge expert had him at 0.485. Submitted 0.650.
- Captain Kathryn Janeway, prior 0.582. "Explore" and "additional land" were not in the card-advantage features. Submitted 0.644.
- Captain Jean-Luc Picard, prior 0.607. Anthem regexes average 0.587 because they match bad lords. Submitted 0.648.
- Support Mission, prior 0.508. Every creature enters with a +1/+1 counter. Submitted 0.594.
- Vengeance Mission, prior 0.494. They lose 1 life on every death. Submitted 0.548.
- Seven of Nine (uncommon), prior 0.534. She draws when you choose a mode, and this set is full of modes. Submitted 0.596.
- Wesley Crusher, prior 0.634. A 2/2 for four with two small modes. Submitted 0.560.
- The Destruction of Praxis, prior 0.616. −X/−X hits your creatures too. Submitted 0.556.
- H.M.S. Bounty, prior 0.594. The enters ability searches for a Whale. There is one Whale in the set, and you may not be playing it. This is the Dawnsire failure mode. Submitted 0.496.
- Quark, prior 0.567. You cannot sacrifice Treasures, so the tokens he makes do not make mana, and he turns off every other Treasure. Submitted 0.508.
- Georgiou, prior 0.638. She is a real mythic: you don't lose for having 0 or less life, she draws and loses 3 each end step, and at 0 life she makes ten Officers. She is also a 4/6 for six that spends your life total to get there. Submitted 0.612, not 0.638.
- Peace in Our Time, prior 0.539. The opponent chooses whether to draw, and you need twelve counters. Submitted 0.476.
- Infinite Diversity, prior 0.565. Nine mana, no cost reduction, and it needs a spell already in the grave. Submitted 0.498.
- Deep Space 9, prior 0.563. Nine mana. If it resolves, permanents come off the top every upkeep. Most games it is a card in hand. Submitted 0.518.
- The Holodeck, prior 0.511. Seven mana to play one sideboard or graveyard permanent a turn, and you pay its cost. Submitted 0.546.
- Engineered Explosives, prior around the mid-0.55s. X=0 kills every token creature in a token format. Submitted 0.606.
- Tricorder, prior 0.531. Cheap equipment is not Meltstrider's Gear, and it is not White Mage's Staff either. Submitted 0.556.

Other judgments that are easy to misread from the type line:

- Kobayashi Maru gives you a bounce, a draw-two, and two tokens across three upkeeps, and then you lose the game. The value is real and the lose clause shows up in long games. 0.578.
- Q exiles a random nonland and lets you cast copies of it by discarding. That is not a free spell. The 4/4 is the floor. 0.574.
- Dr. Beverly Crusher doubles lifegain triggers. The lifegain in the set is mostly 1 life from a planet. 0.552.
- Death Is Irrelevant is eight mana with no reduction. 0.524. Crystalline Entity is an eight-mana wrath of nonartifact creatures on a 7/7 flyer. Extinguisher Battleship, a cleaner eight-mana wipe, was 0.583. This one hits your own nonartifacts. 0.558.
- U.S.S. Reliant reanimates a creature of mana value 4 or less on enter. Fell Gravship was 0.600 for a worse enters ability. 0.624.
- D-7 Battle Cruiser makes a hasty 2/1 every combat at three charge counters, which is one creature tapped. 0.616.
- Romulan Bird-of-Prey is hexproof while it is not a creature and puts two charge counters on itself whenever you cast a noncreature spell. 0.618.
- U.S.S. Enterprise-D impulses the first time you put charge counters on it each turn. Stationing it is the cost of the card. 0.610.
- Mirror Universe Intrusion is a permanent copy of their creature for three mana, not a threaten. 0.620.
- Holo-Engineer is two 3/3s for four, and later tokens become 3/3 Illusions. That replacement also eats Clues and New Planets, which is priced in. 0.618.
- Captain Pike's power and toughness are your land count, and you play lands off the top. 0.622.
- Deanna is listed above. Dr. Joseph M'Benga doubles counters and gives those creatures hexproof, and he does not make counters himself. 0.612.
- Jadzia Dax is an 0/1 the turn you cast her, then adds green equal to her counters every turn. 0.602.
- Elim Garak is a 3/3 with menace that drains 2 whenever you face a dilemma. 0.612.
- Salt Vampire exiles a card from their hand on a 2/2 with lifelink. 0.612.
- The Wrath of Khan deals 4, steals that creature for a combat, loots, then they sacrifice. 0.616.
- Starfleet Spanner Jockey is a 3/3 for two that attacks once you have cast a noncreature spell. 0.596, near the common ceiling with the Holodeck.
- Reckless Impulse is the usual exile-two spell, at common. 0.594.
- Highly Illogical counters mana value 2 or less for two mana and kicks into a hard counter. 0.598.
- Test of Conviction is a team counter or "they draw one, you draw two." 0.594.
- Test of Priorities is surveil 2 and draw, or a blink. 0.586.

Signpost uncommons are scored as the card you actually cast, with fixing good enough that the gold cost is cheap. The Duras Sisters (0.600), Spock, Always Your Friend (0.602), Captain Michael Burnham (0.602), Dal R'El (0.600), and Hemmer (0.598) are the top of that group. Worf, Honor-Bound Warrior (0.510), Saavik (0.496), and Hoshi Sato (0.506) are bodies that do not do enough for their cost. Sarek needs five creature types before he draws, in three colors, on an 0/5. 0.502.

## What was deliberately not used

No prices, no EDHREC, no draft pick order, and no public win rate, spoiler, or set review for Star Trek. Internet access was used only to install pandas, numpy, scikit-learn, lightgbm, and scipy into `/opt/mtg-data/venv`. The 17lands files were not refreshed. Features and judgments are from the card text, the Comprehensive Rules, and the nine historical Premier Draft files.

## Files

| Path | What it is |
| --- | --- |
| `predictions/trk_premier_draft_gih.csv` | The submitted table |
| `predictions/trk_premier_draft_gih.md` | The same table |
| `src/hand_rates.py` | The 311 rates and the one-line reason for each |
| `src/emit_predictions.py` | Writes the table from those rates and the Scryfall file |
| `src/compute_historical_gih.py` | GIH aggregation for the nine Premier Draft files |
| `src/predict_gih.py` | Leave-one-set-out stack. Prior, not the submission |
| `src/expert_rate.py` | Rejected point-score regression (negative quadratic) |
| `analysis/cv_metrics.json` | Stack MAE, Spearman, per-rarity and per-set bias |
| `analysis/run_meta.json` | Set win rates and the training setup |
| `analysis/trk_raw_predictions.csv` | Stack scores for all 311 cards |
| `analysis/loo_worst_errors.csv` | Largest historical misses |
| `analysis/loo_color_bias.csv` | Common-card residual by set and color |
| `analysis/feature_importance.csv` | LightGBM gain |
| `analysis/trk_review.txt` | Every TRK card with stack score, features, and neighbors |
| `analysis/trk_expert_raw.csv` | The rejected expert-regression scores |
| `analysis/expert_cv.json` | That fit's coefficients and MAE |
| `analysis/prediction_notes.csv` | Submitted rate plus the reason string |
| `analysis/final_distribution.json` | Mean and spread of the submitted table |
