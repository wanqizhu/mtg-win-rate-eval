#!/usr/bin/env python3
"""Build training table: Scryfall features + GiH targets for historical sets."""
import gzip, json, re
from pathlib import Path
import numpy as np

SETS = ["HOB","MSH","SOS","TMT","ECL","TLA","EOE","FIN","TDM"]  # exclude OM1 (odd pool)
# load gih2
import json as j

def full_text(card):
    # combine oracle_text + faces
    parts=[]
    if card.get('oracle_text'): parts.append(card['oracle_text'])
    for f in (card.get('card_faces') or []):
        if f.get('oracle_text'): parts.append(f['oracle_text'])
    return "\n".join(parts)

def canonical_map(code):
    with gzip.open(f"/opt/mtg-data/scryfall/sets/{code.lower()}.jsonl.gz",'rt') as f:
        cards=[json.loads(l) for l in f if l.strip()]
    by_full={c['name']:c for c in cards}
    by_first={}
    for c in cards:
        ff=c['name'].split('//')[0].strip() if '//' in c['name'] else c['name']
        # keep lowest cmc? just first encountered; prefer non-promo? all non-promo here except few
        if ff not in by_first:
            by_first[ff]=c
    return by_full, by_first

def parse_pt(v):
    if v is None: return None
    s=str(v).strip()
    if s in ('*','?',''): return None
    # handle like "1+*" or "*/2"? take leading number
    m=re.match(r'^-?\d+', s)
    if m: return float(m.group(0))
    return None

def features_for(card):
    t=(card.get('type_line') or '')
    txt=full_text(card)
    low=txt.lower()
    cmc=float(card.get('cmc') or 0)
    mc=card.get('mana_cost') or ''
    # pips
    pips=re.findall(r'\{([^}]+)\}', mc)
    # for DFC prepare, mana_cost like "{U} // {1}{U}" - count all
    npips=len(pips)
    # generic
    gen=0
    for p in pips:
        if p.isdigit(): gen+=int(p)
        elif p=='X': pass
    colors=card.get('colors') or []
    ci=card.get('color_identity') or []
    rar=card.get('rarity')
    pow_=parse_pt(card.get('power')); tou=parse_pt(card.get('toughness'))
    is_creature='Creature' in t
    is_land='Land' in t
    is_instant='Instant' in t; is_sorcery='Sorcery' in t
    is_ench='Enchantment' in t; is_art='Artifact' in t; is_pw='Planeswalker' in t
    is_legend='Legendary' in t
    is_basic='Basic Land' in t
    kw=card.get('keywords') or []
    # text signals
    def has(*terms):
        return int(any(term in low for term in terms))
    feats={
        'cmc':cmc,
        'log_cmc':float(np.log1p(cmc)),
        'n_colors':len(colors),
        'is_multicolor':int(len(colors)>=2),
        'is_colorless_nonland':int(len(colors)==0 and not is_land),
        'W':int('W' in colors),'U':int('U' in colors),'B':int('B' in colors),'R':int('R' in colors),'G':int('G' in colors),
        'rarity':rar,
        'is_creature':int(is_creature),'is_land':int(is_land),
        'is_instant':int(is_instant),'is_sorcery':int(is_sorcery),
        'is_enchantment':int(is_ench),'is_artifact':int(is_art),'is_planeswalker':int(is_pw),
        'is_legendary':int(is_legend),'is_basic':int(is_basic),
        'power':pow_ if pow_ is not None else -1,
        'toughness':tou if tou is not None else -1,
        'pt_sum':(pow_+tou) if (pow_ is not None and tou is not None) else -1,
        'has_pt':int(pow_ is not None),
        'n_keywords':len(kw),
        'text_len':len(txt),'text_words':len(txt.split()),
        'has_flying':int('flying' in low),
        'has_trample':int('trample' in low),
        'has_haste':int('haste' in low),
        'has_vigilance':int('vigilance' in low),
        'has_deathtouch':int('deathtouch' in low),
        'has_menace':int('menace' in low),
        'has_reach':int('reach' in low),
        'has_lifelink':int('lifelink' in low),
        'has_ward':int('ward' in low),
        'has_flash':int('flash' in low),
        'has_draw':int(re.search(r'\bdraw\b', low) is not None),
        'has_destroy':int('destroy' in low),
        'has_exile':int('exile' in low),
        'has_counter':int('counter target' in low or 'counter all' in low),
        'has_token':int('token' in low),
        'has_treasure':int('treasure' in low),
        'has_surveil':int('surveil' in low),
        'has_scry':int('scry' in low),
        'has_mill':int(' mill' in low or low.startswith('mill')),
        'has_sacrifice':int('sacrifice' in low),
        'has_discard':int('discard' in low),
        'has_lifegain':int('you gain' in low and 'life' in low),
        'has_damage':int('damage' in low),
        'has_fight':int(' fight' in low),
        'has_search':int('search your library' in low),
        'has_return_gy':int('return' in low and 'graveyard' in low),
        'has_copy':int(' copy' in low),
        'has_proliferate':int('proliferate' in low),
        'has_cycling':int('cycling' in low),
        'has_flashback':int('flashback' in low),
        'has_landfall':int('landfall' in low),
        'has_prowess':int('prowess' in low),
        'has_flying_hate':int('with flying' in low),
        'n_removal_words':low.count('destroy')+low.count('exile'),
    }
    return feats, txt

rows=[]
for s in SETS:
    gih=j.loads(Path(f"/workspace/analysis/gih2_{s}.json").read_text())
    by_full, by_first=canonical_map(s)
    for e in gih['cards']:
        if e['match'].startswith('non-set'): continue
        key=e['card_17lands']
        if key in by_full: card=by_full[key]
        elif key in by_first: card=by_first[key]
        else: continue
        feats,txt=features_for(card)
        rows.append({"set":s,"name_17lands":key,"scryfall_name":card['name'],
                      "gih":e['gih'],"gih_wr":e['gih_wr'], **feats, "oracle_combined":txt,
                      "cmc_raw":card.get('cmc'),"mana_cost":card.get('mana_cost'),
                      "type_line":card.get('type_line')})
print(f"rows={len(rows)}")
# per-set stats
import collections
for s in SETS:
    rr=[r for r in rows if r['set']==s]
    import statistics
    wrs=[r['gih_wr'] for r in rr]
    print(s, len(rr), f"mean={statistics.mean(wrs):.4f}", f"sd={statistics.pstdev(wrs):.4f}",
          f"min={min(wrs):.4f}", f"max={max(wrs):.4f}")
import pickle
with open("/workspace/analysis/train_rows.json","w") as f:
    json.dump(rows,f)
print("saved")
