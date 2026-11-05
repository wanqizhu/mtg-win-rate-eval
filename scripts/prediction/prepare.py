"""Build auditable, offline card and historical outcome tables."""
from pathlib import Path
import argparse, collections, gzip, hashlib, json, os
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('MTG_DATA_ROOT','/workspace/mtg-data'))
OUT = REPO / 'analysis'
SETS = ['FIN','EOE','OM1','TLA','ECL','TMT','SOS','MSH','HOB','FRA']

def canonical(code):
    cards = [json.loads(x) for x in gzip.open(DATA/'scryfall'/'sets'/f'{code.lower()}.jsonl.gz','rt')]
    cards.sort(key=lambda c: (int(c['collector_number']) if c['collector_number'].isdigit() else 99999))
    by_name = {}
    for c in cards:
        if c['name'] in by_name: continue
        fields = ['name','oracle_id','collector_number','rarity','mana_cost','cmc','type_line',
                  'oracle_text','colors','color_identity','power','toughness','loyalty','keywords',
                  'layout','card_faces','games','printed_name','printed_text','printed_type_line']
        by_name[c['name']] = {k:c[k] for k in fields if k in c}
        by_name[c['name']]['set'] = code
    return list(by_name.values())

def aggregate(code):
    path = DATA/'17lands'/'sets'/code/f'game_data_public.{code}.PremierDraft.csv.gz'
    event = 'PremierDraft'
    if code == 'OM1':
        path = path.with_name(f'game_data_public.{code}.PickTwoDraft.csv.gz')
        event = 'PickTwoDraft'
    head = pd.read_csv(path, nrows=0).columns
    names = [s.removeprefix('opening_hand_') for s in head if s.startswith('opening_hand_')]
    prefixes = ['opening_hand_', 'drawn_', 'deck_']
    cols = ['expansion','event_type','won','num_turns','on_play','main_colors','splash_colors','rank','draft_id']
    cols += [p+n for p in prefixes for n in names]
    counts = np.zeros((len(names),10),dtype=np.int64)
    # columns: gih, gih wins, deck games/wins, OH games/wins, drawn games/wins, copy exposure/wins
    totals=collections.Counter(); colors=collections.defaultdict(lambda:np.zeros(3,dtype=np.int64))
    color_card=collections.defaultdict(lambda:np.zeros((len(names),4),dtype=np.int64))
    drafts=set(); turns=collections.Counter(); ranks=collections.Counter()
    for chunk in pd.read_csv(path,usecols=cols,chunksize=10000):
        assert chunk.expansion.eq(code).all() and chunk.event_type.eq(event).all()
        won=chunk.won.to_numpy().astype(bool); w=won.astype(np.int64)
        oh=chunk[['opening_hand_'+n for n in names]].to_numpy(dtype=np.int16)
        dr=chunk[['drawn_'+n for n in names]].to_numpy(dtype=np.int16)
        de=chunk[['deck_'+n for n in names]].to_numpy(dtype=np.int16)
        seen=(oh+dr)>0; deck=de>0; opening=oh>0; drawn=dr>0; copies=oh+dr
        for k,a in enumerate([seen,deck,opening,drawn,copies]):
            counts[:,2*k] += a.sum(axis=0)
            counts[:,2*k+1] += a.T @ w
        totals['games']+=len(chunk); totals['wins']+=int(w.sum())
        totals['play_games']+=int(chunk.on_play.sum());totals['play_wins']+=int(w[chunk.on_play].sum())
        totals['turn_sum']+=int(chunk.num_turns.sum())
        drafts.update(chunk.draft_id)
        turns.update(chunk.num_turns.tolist());ranks.update(chunk['rank'].fillna('unknown').tolist())
        for color,inds in chunk.groupby('main_colors').indices.items():
            colors[str(color)] += [len(inds),int(w[inds].sum()),int(chunk.num_turns.iloc[inds].sum())]
            color_card[str(color)][:,0] += seen[inds].sum(axis=0)
            color_card[str(color)][:,1] += seen[inds].T @ w[inds]
            color_card[str(color)][:,2] += deck[inds].sum(axis=0)
            color_card[str(color)][:,3] += deck[inds].T @ w[inds]
    cols=['gih_n','gih_wins','deck_n','deck_wins','oh_n','oh_wins','drawn_n','drawn_wins','copy_n','copy_wins']
    frame=pd.DataFrame(counts,columns=cols);frame.insert(0,'name',names);frame.insert(0,'set',code)
    for label in ['gih','deck','oh','drawn','copy']:
        frame[label+'_wr']=frame[label+'_wins']/frame[label+'_n'].replace(0,np.nan)
    frame['iwd']=frame.gih_wr-(frame.deck_wins-frame.gih_wins)/(frame.deck_n-frame.gih_n).replace(0,np.nan)
    frame.to_csv(OUT/f'historical_{code}.csv',index=False)
    cc=[]
    for color,c in color_card.items():
        for i,name in enumerate(names):
            if c[i,0]: cc.append([code,color,name,*map(int,c[i])])
    pd.DataFrame(cc,columns=['set','main_colors','name','gih_n','gih_wins','deck_n','deck_wins']).to_csv(OUT/f'colors_{code}.csv',index=False)
    summary={'set':code,'event_type':event,**dict(totals),'drafts':len(drafts),
             'win_rate':totals['wins']/totals['games'], 'mean_turns':totals['turn_sum']/totals['games'],
             'colors':{k:{'games':int(v[0]),'wins':int(v[1]),'wr':v[1]/v[0],'mean_turns':v[2]/v[0]} for k,v in colors.items()},
             'turn_histogram':dict(turns),'rank_counts':dict(ranks), 'source_file':str(path.relative_to(DATA)),
             'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
    (OUT/f'summary_{code}.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(code,totals['games'],round(summary['win_rate'],5),round(summary['mean_turns'],2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cards-only',action='store_true');parser.add_argument('--sets',nargs='*',default=SETS)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    cards=sum([canonical(s) for s in SETS+['TRK','SDS']],[])
    (OUT/'cards.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cards))
    trk=[c for c in cards if c['set']=='TRK']
    (OUT/'target_cards.txt').write_text('\n\n'.join(f"{c['collector_number']}. {c['name']} [{c['rarity']}] {c.get('mana_cost','')} {c.get('power','')}/{c.get('toughness','')}\n{c['type_line']}\n{c.get('oracle_text','')}" for c in trk)+'\n')
    if args.cards_only:return
    for s in args.sets:
        aggregate(s)
    frames=[pd.read_csv(OUT/f'historical_{s}.csv') for s in SETS if (OUT/f'historical_{s}.csv').exists()]
    pd.concat(frames,ignore_index=True).to_csv(OUT/'historical_all.csv',index=False)

if __name__=='__main__':main()
