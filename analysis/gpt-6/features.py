"""Features use only rules text and printed characteristics, never market metadata."""
import re
import numpy as np
from collections import Counter

COLORS='WUBRG'
PATTERNS={
 'etb':r'when [^.\n]* enters', 'attack_trigger':r'whenever [^.\n]* attacks',
 'dies_trigger':r'when(?:ever)? [^.\n]* dies', 'repeat_end':r'at the beginning of (?:your|each) end step',
 'upkeep':r'at the beginning of (?:your|each) upkeep', 'combat_trigger':r'at the beginning of combat',
 'cast_trigger':r'whenever [^.\n]* casts? ', 'draw_trigger':r'whenever you draw',
 'lifegain_trigger':r'whenever you gain life', 'counter_trigger':r'whenever [^.\n]*counter',
 'draw_one':r'draw a card','draw_two':r'draw two cards','draw_many':r'draw (?:three|four|five|x|that many|cards)',
 'draw_repeated':r'(?:whenever|at the beginning)[^.\n]*draw',
 'loot':r'draw[^.\n]*then discard', 'discard_cost':r'(?:discard a card:|as an additional cost[^.\n]*discard)',
 'opp_discard':r'(?:opponent|player) discards? ', 'self_discard':r'discard (?:a|two|your) ',
 'destroy_creature':r'destroy (?:target|all|each|up to)[^.\n]*creature',
 'exile_creature':r'exile (?:target|all|each|up to)[^.\n]*(?:creature|nonland permanent)',
 'negative_stats':r'gets? -\d+/-\d+', 'damage':r'deals? [^.\n]*damage',
 'damage_creature':r'damage to (?:target creature|any target)', 'damage_face':r'damage to (?:each opponent|target opponent|each player)',
 'destroy_artifact':r'destroy[^.\n]*artifact', 'destroy_enchant':r'destroy[^.\n]*enchantment',
 'fight':r'fights? ', 'bite':r'deals? damage equal to [^.\n]*power',
 'removal_restriction':r'(?:power|toughness|mana value) (?:\d+|x) or (?:less|greater)',
 'counterspell':r'counter target (?:spell|noncreature|creature)', 'soft_counter':r'unless [^.\n]*pays',
 'bounce':r'return target [^.\n]* to its owner.s hand',
 'tap_enemy':r'tap (?:target|up to)[^.\n]*creature', 'stun':r'stun counter',
 'pacify':r'(?:can.t attack or block|can.t attack|doesn.t untap|loses all abilities)',
 'recursion_hand':r'return[^.\n]*from your graveyard to your hand',
 'reanimate':r'(?:return|put)[^.\n]*from (?:your|a) graveyard[^.\n]*battlefield',
 'grave_cast':r'(?:cast|play)[^.\n]*from your graveyard',
 'flicker':r'exile[^.\n]*you control',
 'sac_outlet':r'sacrifice (?:another|a|an)[^.:\n]*:',
 'sac_cost':r'as an additional cost[^.\n]*sacrifice',
 'sac_effect':r'(?:opponent|player) sacrifices',
 'tutor':r'search your library', 'land_to_hand':r'(?:land|plains|island|swamp|mountain|forest) card[^.\n]*put it into your hand',
 'ramp_land':r'(?:land|plains|island|swamp|mountain|forest) card[^.\n]*(?:battlefield|onto)',
 'mana_dork':r'\{t\}: add', 'mana_any':r'mana of any color',
 'cost_reduction':r'costs?[^.\n]*less', 'cheat':r'without paying',
 'impulse':r'(?:may|can) play (?:that|those|the exiled)',
 'token':r'create', 'creature_token':r'creature tokens?',
 'two_tokens':r'create two', 'three_tokens':r'create three',
 'repeat_token':r'(?:whenever|at the beginning)[^.\n]*create',
 'treasure':r'(?:create|sacrifice)[^.\n]*treasure', 'food':r'(?:create|sacrifice)[^.\n]*food',
 'clue':r'investigate|clue token',
 'pump_counter':r'\+1/\+1 counter', 'pump_team':r'creatures you control get',
 'counter_team':r'counter on each', 'buff_aura':r'enchanted creature gets \+',
 'equipment':r'equipped creature', 'equipped_draw':r'equipped creature[^.\n]*draw',
 'selfmill':r'mill (?:two|three|four|five|six|x)',
 'surveil':r'surveil', 'scry':r'scry', 'lifegain':r'gain (?:\d+|x|that much) life',
 'life_loss':r'you lose [0-9]', 'drain':r'opponent loses',
 'cant_block':r'(?:this creature|cardname) can.t block', 'cant_attack':r'can.t attack (?:alone|unless)',
 'must_sac':r'(?:end step|upkeep)[^.\n]*sacrifice', 'lose_game':r'you lose the game',
 'extra_turn':r'extra turn', 'control_steal':r'gain control of', 'boardwipe':r'(?:destroy|exile) all|all creatures get -',
 'legendary_payoff':r'(?:legendary [^.\n]*you control|legendary spell|legendary card)',
 'artifact_payoff':r'(?:artifact[^.\n]*you control|artifact spell|artifact card)',
 'noncreature_payoff':r'whenever you cast a noncreature',
 'counters_payoff':r'creatures?[^.\n]*with (?:a |\+1/\+1 )?counters?',
 'grave_leave':r'(?:card|cards) leaves? your graveyard',
 'modal':r'choose (?:one|two|three|one or more)',
 'repeat_modal':r'(?:whenever|at the beginning)[^.\n]*choose (?:one|two)',
 'once_turn':r'only once each turn', 'sorcery_only':r'only as a sorcery',
 'threshold':r'threshold', 'federation':r'federation', 'lands_named':r'differently named lands',
 'dilemma':r'whenever you face a dilemma', 'power_eight':r'total power 8',
 'station':r'\bstation\b', 'landfall':r'landfall', 'seek_out':r'seek out',
 'flashback':r'flashback', 'beam':r'beam me up', 'cycling':r'cycling',
 'affinity':r'affinity', 'adventure':r'adventure', 'kicker':r'kicker',
}

def text(c):
    t=c.get('oracle_text','')
    if not t: t='\n'.join(f.get('oracle_text','') for f in c.get('card_faces',[]))
    names=[c['name'],c.get('oracle_name','')]+[f.get('name','') for f in c.get('card_faces',[])]
    for name in names:
        if name:t=t.replace(name,'CARDNAME')
        short=name.split(',')[0]
        if len(short)>3:t=t.replace(short,'CARDNAME')
    t=re.sub(r'\([^)]*\)','',t)
    return t.lower()

def number(s, default=0):
    try:return float(s)
    except (ValueError,TypeError):return default

def features(c, extended=True):
    t=text(c); typ=c['type_line'].lower()
    front=c.get('card_faces',[c])[0]
    cost=front.get('mana_cost',c.get('mana_cost',''))
    colors=front.get('colors',c.get('colors',[]))
    p=number(front.get('power',c.get('power')),0);h=number(front.get('toughness',c.get('toughness')),0)
    mv=number(c.get('cmc')); syms=re.findall(r'\{([^}]+)\}',cost)
    d={'mv':mv,'power':p,'toughness':h,'pt':p+h,'p_minus_t':p-h,
       'pt_per_mana':(p+h)/max(mv,1),'p_per_mana':p/max(mv,1),'t_per_mana':h/max(mv,1),
       'mana_x':int('X' in cost),'n_colors':len(colors),'n_identity':len(c.get('color_identity',[])),
       'hybrid':int('/' in cost),'colored_pips':sum(any(x in s for x in COLORS) for s in syms),
       'max_pips':max([sum(x in s for s in syms) for x in COLORS]),
       'variable_pt':int('*' in str(front.get('power',''))+str(front.get('toughness',''))),
       'text_length':len(t),'lines':len(t.split('\n')),'modes':t.count('•'),
       'loyalty':number(c.get('loyalty')),'faces':len(c.get('card_faces',[])),
       'creature_types':len(typ.split(' — ')[-1].split()) if 'creature' in typ else 0}
    for color in COLORS:d['color_'+color]=int(color in colors)
    for rarity in ['common','uncommon','rare','mythic']:d['rarity_'+rarity]=int(c['rarity']==rarity)
    for ct in ['creature','artifact','enchantment','instant','sorcery','land','planeswalker','battle','legendary','basic','saga','aura','equipment','vehicle','spacecraft']:
        d['type_'+ct]=int(ct in typ)
    for kw in ['flying','vigilance','lifelink','deathtouch','menace','trample','haste','flash','reach','defender','ward','hexproof','indestructible','first strike','double strike','unblockable']:
        d['kw_'+kw]=int(kw in t)
    for k,pat in PATTERNS.items():d[k]=len(re.findall(pat,t))
    station_levels=[int(x) for x in re.findall(r'(\d+)\+',t)] if 'station' in t else []
    d['station_min']=min(station_levels,default=0);d['station_max']=max(station_levels,default=0)
    d['station_pt_efficiency']=(p+h)/(mv+(d['station_max']/3)) if mv+(d['station_max']/3)>0 else 0
    for n in range(1,8):
        d[f'mv_{n}']=int(mv==n)
    for k in ['draw_one','draw_two','token','recursion_hand','destroy_creature','exile_creature','damage_creature','surveil','lifegain','ramp_land']:
        d['etb_'+k]=int(bool(d['etb'])*bool(d[k]))
    for k in ['kw_flying','kw_lifelink','kw_deathtouch','kw_trample','kw_double strike']:
        d[k+'_power']=d[k]*p
    if not extended:return d
    # Parse immediate body/value separately from conditional future triggers.
    nums={'a':1,'an':1,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
    def numword(x):return nums.get(x,number(x,0))
    initial_p,initial_t=p,h
    for m in re.finditer(r'enters (?:the battlefield )?with (a|one|two|three|four|five|six|\d+) ([+-]\d+)/([+-]\d+) counters?',t):
        initial_p+=numword(m[1])*float(m[2]);initial_t+=numword(m[1])*float(m[3])
    d['initial_power']=initial_p;d['initial_toughness']=initial_t
    d['initial_pt_per_mana']=(initial_p+initial_t)/max(mv,1)
    lines=t.split('\n')
    immediate=' '.join(l for l in lines if re.search(r'when .* enters',l))
    if ('instant' in typ or 'sorcery' in typ) and 'creature' not in typ:
        immediate=t
    d['immediate_length']=len(immediate)
    for k,pat in PATTERNS.items():
        if k in ['draw_one','draw_two','draw_many','token','clue','recursion_hand','reanimate','opp_discard','destroy_creature','exile_creature','damage_creature','negative_stats','surveil','scry','ramp_land','land_to_hand','pump_counter','lifegain']:
            d['immediate_'+k]=len(re.findall(pat,immediate))
    d['modal_body']=d['modal']*d['type_creature']
    d['job_select']=int('job select' in t)
    d['initial_tokens']=0;d['initial_token_power']=0;d['initial_token_toughness']=0
    for m in re.finditer(r'create (a|one|two|three|four|five|six|\d+) (\d+)/(\d+)',immediate):
        d['initial_tokens']+=numword(m[1]);d['initial_token_power']+=numword(m[1])*int(m[2]);d['initial_token_toughness']+=numword(m[1])*int(m[3])
    d['initial_board_power']=initial_p+d['initial_token_power']
    d['initial_board_toughness']=initial_t+d['initial_token_toughness']
    d['initial_board_efficiency']=(d['initial_board_power']+d['initial_board_toughness'])/max(mv,1)
    d['conditional_if']=t.count(' if ');d['conditional_unless']=t.count('unless')
    d['other_creature_required']=int('another target creature you control' in t)
    d['combat_connect_required']=int('deals combat damage to a player' in t)
    d['attack_required']=int('whenever' in t and 'attacks' in t)
    d['each_player_draw']=int(bool(re.search(r'each player draws|opponent draws',t)))
    d['expensive_engine']=mv*int(bool(d['upkeep'] or d['repeat_end'] or d['cast_trigger'] or d['repeat_token']))
    d['activation_count']=t.count(':')
    d['tap_ability']=int('{t}' in t and ':' in t)
    d['symmetric']=int('each player' in t or 'all creatures' in t)
    d['recur_without_cost']=int('at the beginning of' in t and not re.search(r'you may pay|if you',t))
    d['card_select']=int(bool(re.search(r'look at the top .*put .*into your hand',t)))
    for kw in ['flying','trample','lifelink','vigilance','menace','reach','haste','deathtouch','flash','first strike','double strike','ward']:
        d['intrinsic_'+kw]=int(any(l.startswith(kw) or (',' in l and len(l)<70 and kw in l) for l in lines))
    return d

def feature_matrix(cards, extended=True):
    import pandas as pd
    f=pd.DataFrame([features(c,extended=extended) for c in cards]).fillna(0)
    # Environment measurements depend only on each set's common/uncommon cards.
    for s in sorted(set(c['set'] for c in cards)):
        ix=[i for i,c in enumerate(cards) if c['set']==s]
        ci=[i for i in ix if cards[i]['rarity'] in ['common','uncommon'] and not f.iloc[i].type_basic]
        if not ci:ci=ix
        env=f.iloc[ci]
        for key in ['type_artifact','type_creature','type_spacecraft','modal','pump_counter','lifegain','creature_token','noncreature_payoff','grave_leave','federation','lands_named','dilemma','station','landfall','beam','type_land']:
            val=env[key].clip(0,1).mean()
            f.loc[ix,'env_'+key]=val
        for key,support in [('artifact_payoff','type_artifact'),('dilemma','modal'),('station','type_creature'),('lifegain_trigger','lifegain'),('counters_payoff','pump_counter'),('grave_leave','beam')]:
            f.loc[ix,'support_'+key]=f.loc[ix,key]*f.loc[ix,'env_'+support]
    return f
