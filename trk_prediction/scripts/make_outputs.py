"""Write the deliverable prediction table (CSV + Markdown) from trk_final_detail.csv."""
import pandas as pd

SRC = '/tmp/trk/trk_final_detail.csv'
OUT_DIR = '/agent/repos/mtg-win-rate-eval/trk_prediction'

d = pd.read_csv(SRC)
d['cn_num'] = pd.to_numeric(d.collector_number, errors='coerce')
d = d.sort_values('cn_num')
out = d[['collector_number', 'name', 'rarity', 'colors', 'mana_cost', 'type_line', 'final']].rename(
    columns={'final': 'predicted_gih_wr'})
out['colors'] = out.colors.fillna('')
out['mana_cost'] = out.mana_cost.fillna('')
out['predicted_gih_wr'] = out.predicted_gih_wr.round(4)
out.to_csv(f'{OUT_DIR}/trk_gih_predictions.csv', index=False)

lines = ['# Star Trek (TRK) Premier Draft - predicted 17lands GIH win rates', '',
         f'{len(out)} cards: main set #1-306 plus the five basic lands (#307-311). Sorted by collector number.',
         'Prediction = probability of winning a game in which the card was in the opening hand or drawn.', '',
         '| # | Card | Rarity | Cost | Type | Predicted GIH WR |', '|---|---|---|---|---|---|']
for r in out.itertuples():
    lines.append(f'| {r.collector_number} | {r.name} | {r.rarity} | {r.mana_cost} | {r.type_line} | {r.predicted_gih_wr:.3f} |')
lines += ['', '## Top 40 by predicted GIH WR', '', '| Rank | Card | Rarity | Predicted GIH WR |', '|---|---|---|---|']
for i, r in enumerate(out.sort_values('predicted_gih_wr', ascending=False).head(40).itertuples(), 1):
    lines.append(f'| {i} | {r.name} | {r.rarity} | {r.predicted_gih_wr:.3f} |')
open(f'{OUT_DIR}/trk_gih_predictions.md', 'w').write('\n'.join(lines) + '\n')
print(out.predicted_gih_wr.describe())
