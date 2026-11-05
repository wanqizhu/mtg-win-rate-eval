"""Card rules and mana features. No prices, ranks, external ratings or outcomes."""
import re
import numpy as np

NUMBERS={'a':1,'an':1,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'x':3}

def number(s):
    try:return float(s)
    except (ValueError,TypeError):return NUMBERS.get(str(s).lower(),0)

def get_text(c):
    faces=c.get('card_faces') or [c]
    out=[]
    for f in faces:
        t=f.get('oracle_text','').replace(f.get('name','__missing__'),'CARDNAME')
        # Reminder text is redundant, and sometimes much longer than the actual effect.
        t=re.sub(r'\([^()]*\)','',t)
        out.append(t)
    return '\n'.join(out)

def features(c):
    faces=c.get('card_faces') or [c];f=faces[0]
    typ=f.get('type_line',c.get('type_line',''));text=get_text(c).lower()
    cost=f.get('mana_cost',c.get('mana_cost',''))
    cmc=float(c.get('cmc',0));power=number(f.get('power'));toughness=number(f.get('toughness'))
    symbols=re.findall(r'\{([^}]+)\}',cost)
    colors=c.get('colors') or f.get('colors',[])
    colored=[s for s in symbols if re.search('[WUBRG]',s)]
    # Hybrid is castable using either of its colors; it is not a gold restriction.
    required=set(s for s in colored if len(s)==1)
    d={'mv':cmc,'power':power,'toughness':toughness,'power_unknown':f.get('power') is not None and '*' in f['power'],
       'toughness_unknown':f.get('toughness') is not None and '*' in f['toughness'],
       'power_eff':power/max(cmc,1),'toughness_eff':toughness/max(cmc,1),
       'body_eff':(power+toughness)/max(cmc,1),'n_colors':len(colors),'n_required_colors':len(required),
       'n_pips':len(colored),'pip_density':len(colored)/max(cmc,1),'max_pips':max([colored.count(s) for s in set(colored)]+[0]),
       'hybrid':any('/' in s for s in symbols),'x_cost':'X' in cost,'legendary':'Legendary' in typ,
       'rarity':c['rarity'],'color_combo':''.join(colors) or 'C', 'layout':c.get('layout','normal'),
       'length':len(text),'lines':len(text.splitlines()),'modes':text.count('•'),
       'faces':len(faces),'loyalty':number(f.get('loyalty')),'subtypes':len(typ.split('—')[-1].split()) if '—' in typ else 0}
    for color in 'WUBRG':d['color_'+color]=color in colors;d['identity_'+color]=color in c.get('color_identity',[])
    for ty in ['Creature','Artifact','Enchantment','Instant','Sorcery','Land','Planeswalker','Battle','Equipment','Aura','Spacecraft','Vehicle','Saga','Basic','Planet']:
        d['type_'+ty]=ty in typ
    for word in ['flying','reach','lifelink','deathtouch','menace','trample','haste','first strike','double strike',
                 'vigilance','defender','flash','ward','hexproof','indestructible','surveil','scry','investigate','mill','explores',
                 'landfall','affinity','convoke','kicker','flashback','cycling','landcycling','beam me up','station','warp',
                 'teamwork','sneak','airbending','earthbending','firebending','waterbending','blight','empower','converge',
                 'federation','strange new worlds','dilemma','seek out','finality','changeling','discover']:
        d['kw_'+word]=word in text
    patterns={
      'etb':r'when[^.\n]* enters',
      'dies':r'when[^.\n]* dies',
      'attack_trigger':r'whenever[^.\n]* attacks?',
      'upkeep':r'at the beginning of[^.\n]*upkeep',
      'endstep':r'at the beginning of[^.\n]*end step',
      'draw':r'draw(?:s)? (?:a|one|two|three|four|five|x|that many) cards?',
      'discard_self':r'(?:you may |then )?discard (?:a|one|two|three|your hand)',
      'discard_opp':r'(?:opponent|player)[^.\n]* discards?',
      'token':r'create(?:s)?[^.\n]*token',
      'clue':r'investigate|clue token',
      'treasure':r'treasure token',
      'food':r'food token',
      'recursion':r'(?:return|put)[^.\n]*(?:graveyard)[^.\n]*(?:hand|battlefield)',
      'reanimate':r'(?:return|put)[^.\n]*(?:graveyard)[^.\n]*battlefield',
      'tutor':r'search your library',
      'ramp':r'(?:land cards?|planet card)[^.\n]*onto the battlefield',
      'mana_dork':r'\{t\}: add',
      'removal':r'(?:destroy|exile) (?:up to one |another )?target (?:nonland permanent|creature|artifact|nonartifact creature)',
      'destroy':r'destroy target',
      'exile_opp':r'exile[^.\n]*(?:opponent controls|target creature)',
      'bounce':r'return target[^.\n]*owner.s hand',
      'counterspell':r'counter target[^.\n]*spell',
      'fight':r'fights? (?:up to one )?target|damage equal to[^.\n]*power',
      'burn':r'deals? (?:[0-9]+|x|twice x) damage',
      'shrink':r'gets? -[0-9x]+/-[1-9x]',
      'tap_down':r'tap (?:up to [a-z]+ )?target[^.\n]*|stun counter',
      'negative_aura':r'(?:doesn.t untap|can.t (?:attack|block))',
      'protect':r'gains?[^.\n]*(?:hexproof|indestructible)',
      'evasion':r'can.t be blocked',
      'counters':r'\+1/\+1 counter',
      'anthem':r'(?:other )?creatures you control get \+[0-9]/\+[0-9]',
      'steal':r'gain control|assimilate',
      'sac_outlet':r'sacrifice (?:another|a) (?:artifact|creature|nonland permanent)',
      'grave_exile':r'exile (?:this|target|a)[^.\n]*graveyard',
      'cost_reduce':r'costs?[^.\n]*less to',
      'draw_engine':r'(?:whenever|at the beginning)[^.\n]*draw',
      'global_removal':r'destroy all|all creatures get -|damage to each creature',
      'modal_repeat':r'whenever[^.\n]*choose one',
      'tapped_trigger':r'whenever[^.\n]*becomes tapped',
      'nontoken_restrict':r'nontoken',
      'basic_search':r'basic land|basic plains|basic island|basic swamp|basic mountain|basic forest',
      'enters_tapped':r'(?:this land|CARDNAME|cardname) enters tapped',
      'sac_land':r'sacrifice (?:this land|a land)',
      'life_gain':r'(?:you )?gain (?:[0-9]+|x|that much) life',
      'lose_game':r'you lose the game',
      'cant_block':r'this creature can.t block',
      'only_sorcery':r'only as a sorcery',
      'noncreature_payoff':r'whenever[^.\n]*cast[^.\n]*noncreature',
      'artifact_payoff':r'artifact creatures you control|whenever[^.\n]*artifact[^.\n]*enters',
    }
    for k,p in patterns.items():d[k]=len(re.findall(p,text))
    for action in ['draw','damage','gain','surveil','scry','mill']:
        pat=(rf'{action} (\d+|a|an|one|two|three|four|five|six|seven|x)'+(' cards?' if action=='draw' else ''))
        d[action+'_magnitude']=sum(number(n) for n in re.findall(pat,text))
    d['token_power']=0
    d['token_toughness']=0
    for count,p,t in re.findall(r'create (a|an|one|two|three|four|five|six|ten|x|\d+) (?:tapped )?(\d+)/(\d+)',text):
        d['token_power']+=number(count)*int(p);d['token_toughness']+=number(count)*int(t)
    d['etb_card']=bool(d['etb'] and (d['draw'] or d['token'] or d['recursion'] or d['removal'] or d['tutor']))
    d['station_min']=min([int(x) for x in re.findall(r'(\d+)\+',text)] or [0])
    d['station_max']=max([int(x) for x in re.findall(r'(\d+)\+',text)]+[0])
    d['true_creature_power']=power if d['type_Creature'] else 0
    d['true_creature_toughness']=toughness if d['type_Creature'] else 0
    d['flying_body']=power*bool(d['kw_flying'])
    d['life_link_body']=power*bool(d['kw_lifelink'])
    d['splashable']=len(required)<=1 and len(colored)<=1
    return d

def text_input(c):
    faces=c.get('card_faces') or [c]
    # Omit proper names, illustration metadata, collector numbers and all market data.
    return '\n'.join(f.get('type_line','').split('—')[0]+' '+get_text({**c,'card_faces':[f]}) for f in faces).lower()
