# Star Trek (TRK) Premier Draft — Predicted 17lands GIH win rates

**Branch:** `cursor/muse-spark-807c` · **Model:** Muse Spark
**Output:** `trk_gih_predictions.csv` — 311 rows (every distinct TRK card available in Premier Draft), `card,predicted_gih_wr` sorted descending.
**Headline stats:** mean 0.5685, sd 0.0174, min 0.5150 (`Peace in Our Time`), max 0.6270 (`U.S.S. Enterprise-A, Free Spirit`).

## 1. What was predicted

- Target: 17lands **game-in-hand (GIH) win rate** for TRK Premier Draft: for each card, P(win | card in opening hand or drawn), computed exactly as 17lands does (any `opening_hand_<card>`/`drawn_<card>` column nonzero).
- Card universe: all **311 distinct front-face names** in the TRK Scryfall extract (`scryfall/sets/trk.jsonl.gz`, deduped by lowest collector number; split-card fronts joined to game names). Includes 5 basics, 10 Planet gainlands, 10 shocks, 5 anomaly lands + Unexplored Territory, and all commons/uncommons/rares/mythics. Excludes Stardates (SDS masterpiece sheet, not in draft) and showcase-duplicate printings (same name, one row).
- No 17lands TRK rows were used (verified zero by `leakage_report.json`); no external lookup of the new set was performed — only the on-disk Scryfall text, rules, and the ten historical sets.

## 2. Data & intermediate artifacts (`analysis/`)

| Artifact | How built | What it is |
|---|---|---|
| `01_compute_gih.py` → `gih_by_set.json` | Row-wise scan of each set's Premier Draft game CSV (OM1: PickTwoDraft, its only draft queue). Pairs `opening_hand_*`/`drawn_*` columns per card. | Per-card GIH wins/total for 9 PremierDraft sets + OM1; per-file game counts and win rates (FRA 297,964 games @ 55.42% … FIN 769,659 @ 55.73%). 9 PremierDraft pools: 193–363 cards each. |
| `02_featurize.py` → `features.pkl` | Parse Scryfall JSON for all 11 sets (dedup by front-face name), ~106 structured features + raw text. Reminder text `(…)` stripped for TF-IDF. | Structured: cmc, pips, colors, rarity, types (creature/artifact/spacecraft/planet/saga/aura/equipment…), P/T + stat efficiency, 20+ keywords (flying…ward-N, station threshold), text stats, flags (choose-one/dilemma/beam/landcycling/strange/kicker/quest/counters), value signals (draw, tokens, lifegain, destroy/exile/bounce/damage/tap/stun/counterspell, discard, ramp, treasure/clue, mill, fight, extra turn, alt-win, ETB, attack triggers, sac outlets). |
| `03_model.py` / `04_model_v2.py` → `train*.pkl`, `model*.pkl` | Ridge (TF-IDF 1–2grams, 1200 terms + standardized structured) + HistGradientBoosting (structured only), 50/50 ensemble; leave-one-set-out CV. OM1 dropped after proving its Scryfall extract is Spider-Man cards (only 35/232 names overlap) while all 9 other sets join ~100%. | LOSO CV (9 sets, gih≥100, n=2,338): Ridge(α=100) RMSE 0.0382 corr 0.417; GBM RMSE 0.0377 corr 0.430; **ensemble RMSE 0.0369 corr 0.462**. Per-rarity residual: commons 2.45%, uncommons 3.4%, rares 5.5%, myths 5.1% — rares/myths need judgment on top of the model. |
| `05_final_predict.py` → `trk_pred_v2.pkl` | Fit ensemble on all 9 sets; per-rarity linear calibration (true ~ pred on CV preds; slopes 0.66–0.80) to de-shrink sensibly. | Calibrated raw predictions: commons mean 56.05%, uncommons 56.46%, rares 58.22%, myths 59.69%. |
| `06_adjust.py` → `trk_pred_final.pkl`, `trk_gih_predictions.csv` / `analysis/trk_gih_predictions.csv` | 50 evidence-backed manual overrides (see §4), magnitudes ±0.5–5%. | Final predictions (see §5). |

Historical context used for calibration: basics mean 56.02% (range 54.2–58.1%); gainland analogues (MSH/TMT tapped duals + life) mean **57.30%**; shock analogues mean **54.2%** (52.3–56.8%); 3-mana exile-until-leaves mean **58.06%** (Banishing Light 58.24%, Web Up 59.85%); EOE Spacecraft n=21 range 46.9–62.0% (low station + strong ETB ≈ 60%, station 8–20 + weak ETB ≈ 47–53%); 9-mana non-cheating cards mean 53.9%; wish/outside-game no-body cards ≈ 51.5%; 0-power utility creatures ≈ 52–55%.

## 3. Format read (from TRK card text only)

- **Removal is thin/conditional at common:** W kills power-4+ or cantrips; B has 2-mana −2/−2 (kicked −6/−6) and 4-mana kill; R has 1-mana 2 (4 if a creature died) and 3-mana 3 + loot; G fights; U bounces/freezes; the premium common answer is the colorless Banishing Light with Ward 2 (`Malfunctioning Holodeck`). Expect creature-centric, bomb-relevant games.
- **Spacecraft/station (28 cards):** thresholds 5–12. Cheap cantrip/tutor ships (Suurok station-5 draws; Oberth station-6 Plains-tutors; Class-F station-6 ramps) play as value artifacts first, creatures later. Station 8+ without immediate ETB is the EOE trap (47–53%).
- **Key mechanics:** beam-me-up (11, graveyard value + bounce synergy), dilemma/modal density (~15+ choose-one enablers powering Seven of Nine / Obsidian Order / Garak payoffs), strange-new-worlds (discount scales with distinct land names; 10 Planets + anomalies + New Planet tokens make 3–5 routine), landcycling-{2} (5 cards: premium floor), Planet gainlands (10, common fixing + life), shocks (10, rare), Federation/creature-type, quest counters, Officer/Klingon tokens, Investigate/Clue + Surveil smoothing.
- **Color depth (model commons means):** W 56.33 > U 56.12 > G 55.82 ≈ B 55.79 > R 55.32. No explicit color correction applied — predictions carry color only through card quality.

## 4. Manual adjustments (50 cards, all with historical analogues)

- **Jank/uncastable down (−2.5 to −5.5pp):** `Peace in Our Time` 56.28→51.50 (group-hug alt-win, no board); `The Holodeck` 55.86→52.50 (7-mana wish, no body; wish analogues 51.5); `Deep Space 9` 57.42→53.50 (9 mana + station 9; EOE station-8–20 analogues 47–53); `Infinite Diversity` 58.97→53.50 (9-mana win-more); `Live Long and Prosper` 58.17→54.50, `Branching Timeline` 58.06→54.50 (7–8 mana, no board); `Crystalline Entity` 58.80→56.50 (8-mana symmetrical wrath); `Starfleet Academy` 58.89→55.50 (often just {C}); `Brazen Bluff` 55.93→53.50 (opponent's-choice, RR); `Stable Wormhole`/`Sickbay` −1.4/−1.5 (slow no-body engines); `Cloak and Phaser` 57.48→56.00; `Collation Trimatrix` 58.33→56.00; `Gwyndala` 59.39→57.50 (weak equipment theme).
- **Shocks to historical band (−2.7 to −3.5pp → 53.8–54.5):** all ten (historical shocks 52.3–56.8, mean 54.2; model had 57.1–57.3).
- **0/1 bodies down:** `Emergency Medical Hologram` 57.34→54.50; `Data, New Life` 56.49→54.00.
- **Removal/bodies up (+0.5 to +1.6pp):** `Malfunctioning Holodeck` 57.31→58.50 (Banishing-Light analogues 58.06 + Ward 2); `Dominion Assault Force` 55.42→57.00 (4/4 vigilance + modal); `Eject the Warp Core` →57.20; `He's Dead, Jim` →57.50; `Planetary Bombardment` →57.30; `Command Decision` →57.10; `Cha'DIch` →57.30; `Horta` 55.32→56.80; `Cold-Blooded Crew` →57.00; `Common Goal`/`Plasma Cascade`/`Dahar`/`Photosynthetic` +0.6–0.7.
- **Spacecraft to EOE analogues:** `Class-F Shuttlecraft` 54.63→55.80; `Voyager` 55.32→56.50; `Suurok` →57.80; `NX-01` →56.70; `Cerritos` →56.50; `Warp Core` →56.50; `Ravenous Cosmozoan` →56.60.
- **Basics equalized to 56.20** (historical basics mean 56.02).
- Deliberately **not** adjusted: quest-mission enchantments (only 4 TLA analogues, bimodal 45–63 — kept model 53.4–54.5), gainlands (kept 57.61; MSH/TMT analogues mean 57.30 support it), top bombs (Enterprise-A 62.70, Inner Light 62.15, Spock 61.81 — within historical mythic/rare top range 64–66).

## 5. Final predictions

`trk_gih_predictions.csv`: 311 rows, mean 0.5685, sd 0.0174, median 0.5671. Top: Enterprise-A 0.6270, Inner Light 0.6215, Spock Science Officer 0.6181, City on the Edge 0.6174, Khan Vengeful 0.6120. Bottom: Peace 0.5150, Holodeck 0.5250, Support Mission 0.5340, Deep Space 9 / Infinite Diversity / Brazen Bluff 0.5350. Full table in the CSV; per-card lineage (model → calibrated → adjusted) is reproducible via `analysis/01–06` scripts.

## 6. Limitations & risks

- Text-only model LOSO corr 0.46: ~55% of GIH variance is context (archetype depth, color balance, play patterns) the model cannot see. Rare/mythic RMSE ~5% — individual bombs can still miss by 3–5pp.
- Mean-level risk: final mean 56.85% sits at the top of the historical card-average band (55.2–57.0, TMT 56.99). Kept deliberately (TRK versatility/cycling resembles high-power TMT); if TRK plays average, predictions run ~0.5–1.0pp hot.
- Color order (W>U>G≈B>R) comes from card quality only; a wrong color read would shift same-color cards ±1pp together.
- Repro: `/opt/mtg-data/venv/bin/python analysis/01_compute_gih.py` (≈10 min, all sets) then `02→06` in order; outputs land in `analysis/` with `trk_gih_predictions.csv` copied to repo root for the deliverable.
