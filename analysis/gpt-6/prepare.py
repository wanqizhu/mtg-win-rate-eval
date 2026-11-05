"""Derive offline, game-presence GIH targets; no network or target outcomes."""
from pathlib import Path
import concurrent.futures as cf
import gzip, json, hashlib
import numpy as np
import pandas as pd

ROOT = Path('/workspace/mtg-data')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts'
SETS = ['FIN', 'EOE', 'OM1', 'TLA', 'ECL', 'TMT', 'SOS', 'MSH', 'HOB', 'FRA']

def cards(code):
    path = ROOT / 'scryfall/sets' / f'{code.lower()}.jsonl.gz'
    rows = [json.loads(s) for s in gzip.open(path, 'rt')]
    rows.sort(key=lambda c: (not c.get('collector_number','').isdigit(), int(c['collector_number']) if c.get('collector_number','').isdigit() else 99999))
    seen, result = set(), []
    keys = ['name','oracle_name','printed_name','oracle_id','collector_number','mana_cost','cmc','type_line','oracle_text','colors','color_identity','keywords','power','toughness','loyalty','defense','layout','card_faces','rarity','games','released_at']
    for c in rows:
        if c.get('layout') == 'reversible_card':
            c={**c,**c['card_faces'][0]}
            c.pop('card_faces',None)
        if code == 'OM1' and c.get('printed_name'):
            c['oracle_name']=c['name'];c['name']=c['printed_name']
        if c['name'] in seen: continue
        seen.add(c['name'])
        result.append({'set':code, **{k:c[k] for k in keys if k in c}})
    return result

def aggregate(code):
    fmt = 'PickTwoDraft' if code == 'OM1' else 'PremierDraft'
    path = ROOT / '17lands/sets' / code / f'game_data_public.{code}.{fmt}.csv.gz'
    header = pd.read_csv(path, nrows=0).columns.tolist()
    names = [k[len('opening_hand_'):] for k in header if k.startswith('opening_hand_')]
    prefixes = ['opening_hand_', 'drawn_', 'tutored_', 'deck_']
    cols = [p+n for p in prefixes for n in names if p+n in header]
    meta = ['won','expansion','event_type','main_colors','splash_colors','num_turns','on_play','num_mulligans']
    dtype = {c:'int16' for c in cols}
    count = np.zeros((len(names),7),dtype=np.int64)
    total = wins = turns = 0
    pairs = {}
    for frame in pd.read_csv(path, usecols=cols+meta, dtype=dtype, chunksize=12000):
        assert set(frame.expansion) == {code}
        assert set(frame.event_type) == {fmt}
        w = frame.won.astype(str).str.lower().isin(['true','1']).to_numpy().astype(np.int64)
        op = frame[['opening_hand_'+n for n in names]].to_numpy()>0
        dr = frame[['drawn_'+n for n in names]].to_numpy()>0
        tu = frame.reindex(columns=['tutored_'+n for n in names],fill_value=0).to_numpy()>0
        deck = frame[['deck_'+n for n in names]].to_numpy()>0
        hand = op | dr
        count[:,0] += hand.sum(axis=0)
        count[:,1] += hand.T @ w
        count[:,2] += deck.sum(axis=0)
        count[:,3] += deck.T @ w
        count[:,4] += op.sum(axis=0)
        count[:,5] += op.T @ w
        count[:,6] += (tu & ~hand).sum(axis=0)
        total += len(w); wins += w.sum(); turns += frame.num_turns.sum()
        for color, ix in frame.groupby('main_colors',dropna=False).indices.items():
            key=str(color)
            val=pairs.setdefault(key, [0,0,0])
            val[0]+=len(ix);val[1]+=int(w[ix].sum());val[2]+=int(frame.iloc[ix].num_turns.sum())
    df=pd.DataFrame(count,columns=['gih_games','gih_wins','deck_games','deck_wins','oh_games','oh_wins','tutored_only_games'])
    df.insert(0,'name',names); df.insert(0,'set',code);df['format']=fmt
    df['gih_wr']=df.gih_wins/df.gih_games
    df['deck_wr']=df.deck_wins/df.deck_games
    df['oh_wr']=df.oh_wins/df.oh_games
    df['set_wr']=wins/total
    df.to_csv(OUT/f'labels_{code}.csv',index=False)
    summary={'set':code,'format':fmt,'games':total,'wins':int(wins),'win_rate':wins/total,'mean_turns':float(turns/total),'cards_observed':int((df.gih_games>0).sum()),'color_pairs':pairs}
    (OUT/f'summary_{code}.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='color_pairs'}),flush=True)
    return summary

def main():
    OUT.mkdir(exist_ok=True)
    allcards=[c for s in SETS+['TRK','SDS'] for c in cards(s)]
    (OUT/'cards.json').write_text(json.dumps(allcards,ensure_ascii=False,indent=2)+'\n')
    with cf.ProcessPoolExecutor(max_workers=3) as pool:
        summaries=list(pool.map(aggregate,SETS))
    pd.concat([pd.read_csv(OUT/f'labels_{s}.csv') for s in SETS]).to_csv(OUT/'historical_labels.csv',index=False)
    (OUT/'historical_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    sources=[]
    for s in SETS+['TRK','SDS']:
        p=ROOT/'scryfall/sets'/f'{s.lower()}.jsonl.gz'
        sources.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
    for p in (ROOT/'rules').glob('*.txt'):
        sources.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
    for s in SETS:
        fmt='PickTwoDraft' if s=='OM1' else 'PremierDraft'
        p=ROOT/'17lands/sets'/s/f'game_data_public.{s}.{fmt}.csv.gz'
        sources.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
    (OUT/'source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')

if __name__=='__main__':main()
