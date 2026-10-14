# Reality Fracture Premier Draft GIH predictions

Predicted statistic: **game-in-hand win rate** from 17lands Premier Draft logs. A game counts for a card when that card was in the opening hand or drawn later. The rate is wins divided by those games. It is a proportion in `[0, 1]`, not a percent. One row per unique card name (285 cards). Art variants share a rate. Basics are included. The separate special-guest sheet (`spg`) is not part of the FRA card list and is not in the table.

The table is `predictions/reality_fracture_gih.csv`. `analysis/card_scores.csv` keeps the gradient-boosted prior next to the final number.

## How historical GIH was measured

Premier Draft game logs for HOB, MSH, SOS, TMT, ECL, TLA, EOE, FIN, and TDM, plus OM1 Pick-Two Draft (that set has no Premier file). For each card column, a game counts when `opening_hand_*` or `drawn_*` is non-zero. That is the same definition as `scripts/demo_winrate.py`.

Games-weighted Premier Draft win rate, OM1 excluded: **0.5559**. 17lands users win more than half their games, so an average card sits near 0.56, not 0.50. Cards with at least 200 games in hand (2,604 of them) have mean GIH 0.559 and standard deviation 0.040. A lot of that spread is sampling noise. Among cards with at least 8,000 games the standard deviation falls to **0.029**, the 5th percentile is **0.523**, and the 95th is **0.615**. Mythics with at least 1,500 games have median GIH **0.595** and mean **0.596**. The final FRA distribution was lined up to those stable numbers: mean 0.563, std 0.027, p05 0.518, p95 0.612, mythic median 0.601.

| Rarity | Historical mean excess over the set win rate | FRA prediction mean |
| --- | --- | --- |
| Common | −0.0055 | 0.555 |
| Uncommon | −0.0011 | 0.559 |
| Rare | +0.0080 | 0.569 |
| Mythic | +0.024 (median GIH 0.595 on large samples) | 0.590 (median 0.601) |

## Model, and why it is only a prior

Card text from Scryfall was joined to those rates (front-face names for double-faced cards, bonus-sheet reprints from the bulk file). About 100 features: mana value, stats versus curve, rarity, types, keywords, and text patterns for draw, removal, tokens, evasion, enters-the-battlefield, and so on. Empower was translated into a fractional “virtual draw” from the loyalty rules, because a fresh Jace token can spend 3 loyalty to draw.

A gradient-boosted tree, trained leave-one-set-out, had weighted MAE **0.0257** and RMSE **0.0347** on excess win rate. Blending it with a 12-neighbor average did not meaningfully beat the tree. The prior is useful and systematically too tight for this set:

- Feature importance collapsed onto text length, mana value, and rarity. “Draw a card” barely correlated with excess win rate (+0.03), so the tree does not price card advantage cleanly.
- Empower Jace does not exist in the training sets, so the tree cannot learn it except through the virtual-draw hack, which it mostly ignores.
- Neighbors for unique cards were often unrelated bodies. Emrakul was treated as a generic large creature. Yargle was treated as a random five-drop. The Theorist was averaged with failed planeswalkers (Ral 0.541, Tezzeret 0.532) and bombs (Ugin 0.685, Elspeth 0.668).

The published number is therefore a card-by-card judgment, started from that prior (mean absolute move 0.018) and then set to match the anchors below. `analysis/predict_gih.py` is the record of every final rate.

## The format the rates assume

Reality Fracture drafts as five allied colleges, each with a shared prepare spell:

- **Fatehold (WU)** — surveil and scry, flying, Peer Review (a 2/2 Cadet plus surveil).
- **Theorix (UB)** — mill, threshold, graveyard, Omit Variables (mill three).
- **Stingerquill (BR)** — noncombat damage, Vicious Verse (1 damage).
- **Konstrari (RG)** — Heartwood mana rocks, Soul Tether.
- **Vigorbloom (GW)** — lifegain and +1/+1 counters, Seed Suture.

Enemy gold cards exist but are not the supported decks, so they are a little harder to cast and a little less synergistic.

**Empower Jace** (rule 701.71) creates or grows one blue Jace token whose abilities are `[-1]: Surveil 1` and `[-3]: Draw a card`. Loyalty can be used the turn the token is made, and only once that turn. Empower 1–2 is selection. Empower 3 is a cantrip if you cash the token in. Empower 5–8 draws and leaves a token that draws again later. Because surveil, mill, and the token are everywhere, the format is grindy: removal and repeatable card advantage outperform one-shot aggro, and a planeswalker-matters line is turned on by a common token rather than by a rare walker.

**Prepare** (rule 722) is not Adventure. The creature has to be on the battlefield to cast a copy of the spell, and it unprepares when you do. Removal on the creature blanks the spell. Secrets of Strixhaven already has this mechanic, so those realized rates are the template, not a guess from Adventures or Omens. Re-preparing every upkeep is only as good as the spell. SOS cards that re-prepare on a weak spell landed around 0.52–0.55.

Blue is the deepest color because both of its colleges use the Jace/surveil engine and its commons include real card flow. Red’s college is the most all-in on a weak enabler (1 damage), so red filler is a notch worse. That shows up as about half a point on basics (Island 0.562, Mountain 0.553) and on the allied versus enemy tap-lands.

## Anchors taken from the ten sets

These are realized Premier Draft GIH rates, not impressions.

| Card or class | Set | GIH | How it was used |
| --- | --- | --- | --- |
| Last Gasp | SOS | 0.577 | FRA Last Gasp, same card |
| Lightning Strike | TLA / MSH | 0.575 / 0.555 | No Admittance is this plus Empower 1, so 0.574 |
| Banishing Light | EOE | 0.582 | Memory Trap |
| Quick Study | SOS | 0.562 | Sphinx’s Approach is the same draw-two at a slightly harder cost, 0.568 |
| Counterspell | FIN | 0.591 | Countersculpt is a hard counter plus a loyalty counter, 0.594 |
| Blazing Crescendo | MSH | 0.529 | Same card. Tricks that do not draw are punished. FRA copy 0.538 |
| Adventurous Eater | SOS | 0.544 | Same prepare spell as Seed Suture. Emergency Phytomedic 0.546 |
| Elite Interceptor | SOS | 0.596 | Ceiling for a prepare common whose spell draws a card. Nothing at common in FRA is that spell |
| Studious First-Year | SOS | 0.569 | Prepare common with a ramp spell |
| Slow lands (Shattered Sanctum and the rest) | SOS | 0.564–0.577 | The five allied slow lands, 0.574 each |
| Gain-lands and tapped duals | TMT and others | about 0.56–0.58 | Planeswalker-conditional duals: 0.568 allied, 0.554 enemy. They enter tapped until a Jace token exists, so they are not untapped duals |
| Planeswalkers on large samples | several | 0.53–0.69 | Good four- and five-mana walkers 0.61–0.66. Failed walkers sit under 0.55. Uncastable eight-mana spells and 10-drops sit near 0.48–0.50 even when the text is absurd |

Board wipes on large samples cluster around 0.59–0.67 when they are actually castable. Overwrite the Multiverse is a six-mana exile-all that also dumps that many loyalty counters onto Jace, so it is with the good wipes (0.634), not with Return to the Light Realms (nine mana, 0.480) or Omnipresence (eight mana, 0.478).

## How individual effects were priced

**Empower.** Added on top of the spell’s board effect, using the draw anchors rather than a flat “card advantage is premium” rule. One-shot Divination-rate cards in this data are only about 0.55–0.57. Repeatable engines are what clear 0.60. Protege’s Awakening (Empower 6 and draw a card) is 0.580: two cards now, token left at 3. Jace’s Machinations (Empower 8, and loyalty at instant speed) is 0.614. A combat trick with Empower 2, such as Academic Ascent, stays a trick (0.556).

**Jace token payoffs.** Sanctum Lurker stops the token dying at 0 loyalty and gives it a ping. That is a real engine and a fragile 3/2, rated 0.608 rather than as a mythic. The ten Way enchantments all Empower on entry and staple an activated ability onto every walker, including the token. The cheap ones with a live ability (Cryomancer’s spell copy, Pyromancer’s red mana, Mentor’s lifegain loyalty) are 0.576–0.590. The five-mana ones are lower.

**Removal.** Efficient removal is the most reliable high-GIH effect in the training slices. Last Gasp 0.577, Multiply by Zero (base power and toughness 0/0) 0.586, Fulminous Forte 0.594, the five charms 0.572–0.596. Color-hate removal and counters are maindeck-dead in two or three of the five colleges: Precise Redaction 0.516, Terminal Criticism 0.544, Essence Burn 0.548. Refute Destiny hits green or blue, which is most of the format, so it is not sideboard-only (0.554).

**Prepare.** Rated from the SOS spell, not from the creature’s rarity. Seed Suture (+1/+1 and gain 1) stays near Adventurous Eater. Peer Review and Soul Tether are fine and not exciting. Omit Variables is good only because Theorix wants the mill. Paradox Shaper re-prepares every upkeep and mills three for one mana on a 1/3, which is the best of that cycle (0.570), and it is still not a bomb: the creature has to live. Diviner of Victory is the rare where both halves are good in this format (a one-drop that grows whenever you surveil, and a bounce), 0.606.

**Creatures.** A common two-drop with no text sits near 0.54–0.55. A 2/3 flyer for three (Shatterwing Pegasus) is 0.562. A 3/3 for two mana (Solarium Sentry) is 0.600 because that rate is genuinely ahead of the curve, not because it is rare. Yargle, Glutton of Urborg is a 9/3 for five that dies to the format’s common 3-damage and −3/−3 removal, so 0.570 rather than a 0.62 bomb. The blue 3/9 vanilla is a five-mana wall (0.522). Samut, Hazoret’s Champion gives every creature haste on a two-mana 2/2 (0.616).

**Lands.** Deserted Beach, Haunted Ridge, Overgrown Farmland, Rockfall Vale, and Shipwreck Marsh are exactly the five allied pairs and are the SOS slow lands, so they get that set’s rate (0.574). Theorist’s Sanctum is an Island that enters untapped once you have a Jace and spends mana to Empower (0.582). Roiling Canopy is a tapped Forest with a six-Forest pump (0.505). Hall of Echoes taps for colorless (0.540). Karn, Gilded Guardian costs ten and five colors (0.462). Emrakul’s channel does not reduce its cost; it is a ten-drop people will keep in hand (0.498).

**The top of the set.** The Theorist, Jace Beleren draws an extra card on every opponent’s draw step in a duel, on a four-mana walker that also makes a body and bounces. That is the Ugin/Elspeth tier (0.656), not the median mythic. Kwia Vigorbloom is a 6/6 flying lifelink that makes a Lotus whenever you gain life (0.642). Chandra, Torch of Defiance is the known four-mana walker with mana, impulse, and removal (0.632). Avatar of Burgeoning Echoes is a two-drop whose landfall Empower 2 turns land drops into a draw engine (0.630).

## What this will get wrong

GIH is not card quality in a vacuum. It is the win rate of the decks that draw the card. If a college is much stronger or weaker than the allied-pair read above, every card in that color moves together and these rates will miss by about a point. If prepare plays more like Adventure than it did in Secrets of Strixhaven, the prepare commons are low. If players refuse to play the ten-drops, those GIH samples will be tiny and noisier than a point estimate can capture. The bonus-sheet legends (collector numbers 195–280) are rated as draft cards; if they are opened much less often, their GIH still means the same thing but the sample will be smaller.
