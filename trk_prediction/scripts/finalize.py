"""Combine expert evaluation, within-rarity quantile calibration and the ML baseline into final predictions."""
import numpy as np, pandas as pd

FORMAT_MEAN = 0.5665          # median GIH-weighted set mean over the 10 historical sets
W_QMAP = 0.5                  # weight on quantile-mapped expert score (vs raw expert score)
W_MODEL = 0.2                 # weight on ML baseline in the final blend
RED_ADJ = -0.003              # mono-red historically underperforms by ~0.02; expert scores already carry most of it

attrs = pd.read_csv('/tmp/trk/trk_attrs.csv')
expert = pd.read_csv('/tmp/trk/trk_expert_eval.csv')
model = pd.read_csv('/tmp/trk/trk_model_baseline.csv')[['name', 'm_lgb', 'm_ridge', 'm_text', 'm_blend']]
hist = pd.read_csv('/tmp/trk/hist/all_cards.csv')

hist = hist[(hist.gih >= 500) & (hist.in_set_file == True)].copy()
set_mean = hist.groupby('set').apply(lambda g: np.average(g.gih_wr, weights=g.gih))
hist['abs_norm'] = FORMAT_MEAN + hist.gih_wr - hist.set.map(set_mean)

df = attrs.merge(expert, on='name', how='left').merge(model, on='name', how='left')
df['is_basic'] = df.type_line.str.startswith('Basic Land')
df['is_land'] = df.type_line.str.contains('Land')
df['model_abs'] = FORMAT_MEAN + df.m_blend

# Within-rarity rank -> historical quantile mapping. Lands are excluded: their win rates cluster near the
# format mean regardless of rarity, so rarity-based mapping would be inappropriate for them.
df['qmapped'] = np.nan
for rar, g in df[~df.is_land].groupby('rarity'):
    ref = np.sort(hist.loc[(hist.rarity == rar) & ~hist.type_line.str.contains('Land', na=False), 'abs_norm'].values)
    ranks = g.expert_pred.rank(method='average')
    p = (ranks - 0.5) / len(g)                       # Hazen plotting positions
    df.loc[g.index, 'qmapped'] = np.quantile(ref, p)

df['cal_expert'] = np.where(df.is_land, df.expert_pred,
                            (1 - W_QMAP) * df.expert_pred + W_QMAP * df.qmapped)
# The ML baseline knows nothing about narrowness / build-around requirements, so its pull is capped at +-0.01.
model_delta = (df.model_abs - df.cal_expert).clip(-0.05, 0.05).fillna(0.0)
df['final'] = df.cal_expert + W_MODEL * model_delta
mono_red = df.colors.fillna('') == 'R'
df.loc[mono_red, 'final'] += RED_ADJ
df['final'] = df['final'].round(4)

df = df.sort_values('final', ascending=False)
cols = ['name', 'collector_number', 'rarity', 'colors', 'mana_cost', 'type_line', 'expert_pred', 'qmapped',
        'cal_expert', 'model_abs', 'm_lgb', 'm_ridge', 'm_text', 'final', 'rationale']
df[cols].to_csv('/tmp/trk/trk_final_detail.csv', index=False)

if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
    print(df.groupby('rarity')['final'].describe().round(3))
    print('overall mean', df.final.mean().round(4), 'std', df.final.std().round(4))
    print(df[['name', 'rarity', 'colors', 'expert_pred', 'qmapped', 'model_abs', 'final']].to_string())
