# Star Trek (TRK) Premier Draft: predicted 17lands GIH win rates

Model: Claude Fable 5.1 (branch `cursor/claude-fable-5-1-trk-gih-predictions-800e`).

Deliverable: [`trk_gih_predictions.csv`](trk_gih_predictions.csv) / [`trk_gih_predictions.md`](trk_gih_predictions.md),
one row per card available in TRK Premier Draft (main set #1-306 plus the five basic lands #307-311; 311 rows)
with a predicted GIH (games-in-hand) win rate.

Everything below was produced from the data on disk only: `/opt/mtg-data/scryfall/sets/trk.jsonl.gz`,
`/opt/mtg-data/rules/MagicCompRules-20260925.txt`, the ten historical 17lands dumps, and the Scryfall bulk file for
matching historical card attributes. No external information about Star Trek was consulted.

## 1. Which cards are in the draft pool

`trk.jsonl.gz` contains 586 printings. Collector numbers 1-306 are the main set (the normal draft booster: 96 commons
incl. 10 Planet duals, 110 uncommons, 80 rares, 25 mythics, counting the 20 rare/mythic lands) and 307-311 are the
basics. Everything above 311 is a showcase/borderless/extended/promo reprint of a 1-311 card. Stardates (`SDS`) is a
paper-only masterpiece sheet and is not part of Arena Premier Draft, so it is excluded. There are no Special Guests.
Full oracle text of every card as used in the evaluation: [`artifacts/trk_cards.txt`](artifacts/trk_cards.txt).

## 2. Historical ground truth (10 sets)

[`scripts/compute_card_stats.py`](scripts/compute_card_stats.py) streams each `game_data_public.<SET>.PremierDraft.csv.gz`
(PickTwoDraft for OM1, which has no Premier Draft file) and reproduces the 17lands card metrics from the raw game rows:
GIH WR (opening hand + drawn + tutored), OH WR, GD WR, GP WR, GNS WR, IWD, play rate and average copies per deck. Output:
[`artifacts/historical/<SET>_<EVENT>_cards.csv`](artifacts/historical/) and `_summary.json` (games, overall win rate,
average game length, colour-pair game counts). Sets: FRA, HOB, MSH, SOS, TMT, ECL, TLA, OM1, EOE, FIN; 4.4M games total.

[`scripts/build_dataset.py`](scripts/build_dataset.py) joins those stats with Scryfall attributes (rarity, cost, colours,
type line, P/T, keywords, oracle text), matching by name against the set file first and the bulk `default-cards` file as a
fallback for bonus-sheet cards. Result: [`artifacts/historical/all_cards.csv`](artifacts/historical/all_cards.csv)
(2,100 in-set cards with >= 500 GIH games).

Key facts extracted from the history and used as anchors (all numbers normalised to a format whose GIH-weighted mean is
0.5665, the median of the ten sets; raw set means range 0.555-0.581):

| Quantity | Value |
| --- | --- |
| Median GIH WR by rarity (non-land) | common 0.556, uncommon 0.561, rare 0.572, mythic 0.598 |
| 10th / 90th percentile by rarity | C 0.520/0.579, U 0.516/0.599, R 0.499/0.627, M 0.544/0.651 |
| Colour effect (mono-coloured cards) | red ~ -0.02 vs the other colours; white/green/black ~ +0.00-0.01 |
| Lands | basics 0.555-0.560; gain-1-life taplands ~ mean -0.002; shocklands ~ mean -0.01; colourless utility lands below mean |
| Average game length | 8.9-10.1 turns (median 9.2) |
| Card classes | O-ring commons +0.02; 4-mana instant kill + surveil +0.01-0.02; tuck/bounce-to-library removal +0.01; "Reckless Impulse" impulse draw -0.03; equipment and +X/+0 auras -0.03 to -0.06; counterspells ~0; vanilla common creatures -0.01 to -0.025; 3-mana common/uncommon spacecraft in EOE +0.01 to +0.03; -2/-2 ETB equipment (Crude Bent Blade) +0.037; kill-but-give-tokens (Harsh Annotation) -0.05 |

## 3. Machine-learned baseline

[`scripts/features.py`](scripts/features.py) turns a card into ~150 features: cost/pips/colours, P/T and P/T-per-mana,
types, 25 keyword flags, ~90 regex oracle-text flags (removal classes, card draw, tokens, counters, ETB, evasion, ramp,
recursion, sacrifice, "can't block", etc.) and derived indicators (removal_any, card_adv, evasive_creature, ...).

[`scripts/train_model.py`](scripts/train_model.py) trains three models on the target *GIH WR minus set weighted mean* with
sample weight sqrt(GIH games) capped at 200, evaluated with leave-one-set-out cross-validation:

| Model | wMAE | wRMSE | corr |
| --- | --- | --- | --- |
| LightGBM (600 trees, 15 leaves) | 0.0206 | 0.0272 | 0.516 |
| Ridge on dense features | 0.0215 | 0.0284 | 0.455 |
| TF-IDF (1-2 grams) + rarity/cmc ridge | 0.0294 | 0.0388 | 0.331 |
| Blend 0.45/0.25/0.30 | 0.0208 | 0.0276 | 0.506 |

Baseline (predict the mean) wMAE is 0.0318, so the model explains roughly half of the variance between cards without
seeing any TRK data. Per-set correlation of the blend ranges 0.44-0.60 (OM1 0.09 on only 35 matched cards). Full report:
[`artifacts/model_cv_report.txt`](artifacts/model_cv_report.txt); out-of-fold predictions:
[`artifacts/historical/oof_predictions.csv`](artifacts/historical/oof_predictions.csv); TRK scores:
[`artifacts/trk_model_baseline.csv`](artifacts/trk_model_baseline.csv).

The model is deliberately not the final answer: it cannot see that a 9-mana copy spell is unplayable, that a build-around
needs a critical mass of enablers, or how the TRK environment (tokens, artifacts, dilemmas) changes a card's value. Its
strongest TRK disagreements with the expert pass are almost all narrow rares/mythics that it rates near the rare median.

[`scripts/analogs.py`](scripts/analogs.py) also produced TF-IDF nearest historical neighbours for every TRK card
([`artifacts/trk_analogs_compact.txt`](artifacts/trk_analogs_compact.txt)); these were used as a lookup aid, together with
[`scripts/lookup.py`](scripts/lookup.py) for ad-hoc "how did cards like X do" queries on the historical table.

## 4. Format analysis (what TRK Premier Draft looks like)

Derived from reading all 306 cards and the rules (the rules file predates the set, so TRK keywords were read from reminder
text; Station/Spacecraft follow the EOE implementation):

* **Mechanics.** *Strange new worlds* (differently named lands you control; the 10 common Planet duals, Unexplored
  Territory, basics and rare lands feed it - a normal two-colour deck has 3 names early and 4-5 late). *Face a dilemma*
  (any modal spell or ability, including many "when this enters, choose one" commons, so dilemma payoffs trigger often).
  *Beam me up* (rebuy an ETB by bouncing). *Station* spacecraft (tap a creature to charge; become creatures at a
  threshold). *Landfall*, *Federation* (count creature types), *Assimilate*/Borg artifact creatures, +1/+1 counters,
  "total power 8 or greater", and "card leaves your graveyard" triggers.
* **Archetypes.** WB Borg artifact creatures, RW go-wide / power 8, UR noncreature spells (Officer tokens), BG graveyard,
  GW +1/+1 counters, RG landfall, GU strange new worlds / ramp, WU spacecraft/station, UB dilemmas, BR Klingon
  sacrifice/aggro.
* **Board texture.** Unusually many 1/1 Officer tokens and 2/1 Klingon tokens are produced (Picard, Enterprise-A,
  Gamemaster, Troubling Tribbles, D-7, L'Rell, Duras Sisters, Cha'DIch, Georgiou). That raises the value of small
  sweep effects (D'Ghor Bloodletter, Khan Despot, Encounter the Unknown), anthems (Picard, Sarcophagus, Troi) and
  token converters (Holo-Engineer, Khan Vengeful Heart, Automated Warfare System), and lowers the value of x/1
  attackers.
* **Removal.** White: 70,000 Light-Years (2-mana exile), Malfunctioning Holodeck (O-ring with ward), Command Decision,
  Set Phasers, Excelsior. Black: Resistance Is Futile (2-mana kill), He's Dead Jim, Eject the Warp Core (Vayne's
  Treachery reprint), Bat'leth, Impossible Choice, De-evolve. Red: Plasma Cascade (Tweeze), Transporter Malfunction,
  Khaaan, Planetary Bombardment, Genesis Effect. Green: Common Goal (double bite), Amok Time. Blue: Guidance Failure
  (tuck), Marooned, Enterprise NX-01. Removal is plentiful and cheap, which keeps raw creature stats from carrying
  games and favours value engines and ETB bodies.
* **Mana.** Ten Planet taplands at common plus Unexplored Territory and shocklands: splashing is easier than average and
  the Planets will be played ~1.5-2 per deck (TMT/MSH tapland precedent).
* **Speed.** Expected ~9 turns (midrange), many 4-5 mana value creatures; the 6-7 mana top end (Horta, Humpback Whales,
  Cold-Blooded Crew, Rogue AI) has basic landcycling to hedge.
* **Colour read.** Red's creature quality is the weakest and several red rares are narrow; white and green have the
  deepest commons; blue has the best card-advantage engines; black has the best cheap removal.

## 5. Card-by-card expert evaluation

[`scripts/expert_eval.py`](scripts/expert_eval.py) (exported as [`artifacts/trk_expert_eval.csv`](artifacts/trk_expert_eval.csv))
assigns every one of the 311 cards an absolute GIH WR estimate with a one-line rationale. Each estimate considered: rate
(stats per mana vs. the format's bodies), the quality and frequency of its effect, which archetype(s) want it and how
often a drafter ends up there, how the token-heavy, artifact-heavy, dilemma-rich environment interacts with it, how many
enablers a build-around has, and the closest historical analogues and their realised win rates (Section 2 anchors).

Examples of format-driven calls:

* `D'Ghor Bloodletter` (2/2 for 3 with a -1/-1 sweep or edict) is rated 0.575 rather than the model's 0.547 because the
  format is full of 1/1 Officers.
* `Holo-Engineer` (0.636 final) turns every Officer, Clue, Treasure and Food into a 3/3.
* `Impossible Choice` is held near 0.55 despite being an instant kill: the two 1/1s it hands over mirror Harsh
  Annotation (0.509) and matter more in a token/anthem format.
* `Kobayashi Maru Scenario`, `Infinite Diversity`, `Peace in Our Time`, `Brazen Bluff`, `Combadge`, `The Holodeck`,
  `Deep Space 9` are placed in the 0.46-0.50 band that unplayable rares occupy historically (2-5th percentile).
* Planets at 0.565 (gain-life taplands historically sit at the format mean), shocklands and basics ~0.556.

## 6. Calibration and blending

[`scripts/finalize.py`](scripts/finalize.py), detail table [`artifacts/trk_final_detail.csv`](artifacts/trk_final_detail.csv):

1. **Within-rarity quantile mapping.** The raw expert scores were compressed relative to history (rare median 0.557 vs
   historical 0.572; mythic 0.583 vs 0.598; too few uncommons above 0.60). Non-land cards are ranked within rarity and
   mapped onto the pooled historical distribution for that rarity (Hazen plotting positions). The calibrated expert score
   is 0.5 x raw + 0.5 x mapped, which keeps the expert ordering while moving the rarity levels and tails toward their
   historical shape without fully trusting the extremes of my own ranking. Lands are exempt (their win rates cluster
   near the mean regardless of rarity).
2. **Model blend.** final = calibrated expert + 0.2 x clip(model - calibrated expert, +-0.05), i.e. the ML baseline can
   move a card by at most +-0.01.
3. **Red adjustment.** -0.003 for mono-red cards (the historical red penalty is larger, but the expert scores already
   carry most of it).

Resulting distribution (311 cards): mean 0.558, std 0.028, common 0.499-0.591 (median 0.556), uncommon 0.481-0.603
(median 0.556), rare 0.457-0.636 (median 0.561), mythic 0.476-0.655 (median 0.588). This matches the historical shape
in Section 2.

## 7. Top of the table

| Card | Pred. |
| --- | --- |
| U.S.S. Enterprise-A, Free Spirit | 0.655 |
| Excalbian Gamemaster | 0.642 |
| Holo-Engineer | 0.636 |
| Captain Jean-Luc Picard | 0.636 |
| Mr. Spock, Science Officer | 0.629 |
| Khan, Vengeful Heart | 0.618 |
| Kira Nerys, Firebrand | 0.617 |
| U.S.S. Discovery, at Black Alert | 0.617 |
| The Wrath of Khan | 0.616 |
| Miles O'Brien, Transporter Ops | 0.610 |

Best commons: Malfunctioning Holodeck 0.591, Suurok-Class Science Vessel 0.582, He's Dead Jim 0.580, Command Decision
0.579, Eject the Warp Core 0.577, Set Phasers to... 0.574, Guidance Failure 0.574, Marooned 0.573, DOT-7 Repair Squad
0.573, Cha'DIch Investigator 0.573, Common Goal 0.573.

## 8. Reproducing

The scripts were run from a scratch directory `/tmp/trk` with a venv containing numpy, pandas, scikit-learn, lightgbm and
scipy; they reference that path. Order: `compute_card_stats.py <SET> <EVENT>` for each set, `build_dataset.py`,
`train_model.py`, `analogs.py`, `expert_eval.py`, `finalize.py`, `make_outputs.py`.

## Known limitations

* No TRK play data exists, so archetype popularity and format speed are inferred from card text; if the format turns
  out much faster or slower than ~9 turns, expensive value cards and cheap aggressive cards move in opposite directions.
* The 17lands sample at release is dominated by the first two weeks; cards that need the format to be "solved"
  (build-arounds like Troubling Tribbles, Sarek, Jadzia Dax) usually underperform their eventual ceiling.
* Low-play narrow rares have noisy realised GIH WR; their predictions are pinned to the historical band for such cards
  rather than to a specific number.
