"""Card-by-card Premier Draft GIH rates for Star Trek (TRK).

Rates are probabilities. One point of judgment is one point of GIH.
They are anchored to measured analogues from the nine Premier Draft
sets, then adjusted for this format: abundant fixing, token fuel for
beam-me-up, modal spells as dilemma triggers, unique land names, and
spacecraft value living in the enters ability rather than the stationed
body. See METHODOLOGY.md.
"""

from __future__ import annotations

# collector number -> predicted GIH win rate
RATES: dict[int, float] = {
    # White
    1: 0.570,   # exile + they get a land. Premium, but not a clean O-ring.
    2: 0.498,   # 3/1 that tutors extra copies you often don't have
    3: 0.558,   # 2/3 for 2, artifacts make Borg on damage. Build-around.
    4: 0.584,   # two bodies for 1W
    5: 0.554,   # lifelink bear, extra counters only if you tap him in a counter deck
    6: 0.564,   # token per dilemma. Bear plus a real payoff in the modal deck.
    7: 0.648,   # anthem and an Officer on every creature or ship. Format lord.
    8: 0.552,   # kill a big creature or draw. Never dead, removal is conditional.
    9: 0.582,   # 2/1 vigilance for W, and artifact creatures share vigilance
    10: 0.530,  # tokens when nontoken artifacts enter. Slow enchantment.
    11: 0.634,  # every creature enters with a +1/+1 counter
    12: 0.536,  # 3/3 flyer for 4 with a once-a-turn untap
    13: 0.552,  # lifegain payoff. Planets gain 1, but the theme is thin.
    14: 0.536,  # 4 mana 2/2. The counter-shield is real only in a counter deck.
    15: 0.540,  # 0/1 that puts a counter and gains 1
    16: 0.628,  # 6/6 plus a token per unique land
    17: 0.534,  # 3/2 vigilance for 4, counter or a conditional draw
    18: 0.528,  # tutor an aura, equipment, or ship. Narrow, beam cost is high.
    19: 0.536,  # 4/5 for 5, temporary anthem, landcycling
    20: 0.554,  # 3/3 flyer that blinks one of your permanents
    21: 0.556,  # 3/3 for 3. The power-8 counter is a bonus.
    22: 0.600,  # O-ring any nonland, with ward 2. Best common removal.
    23: 0.554,  # temporary exile, a Clue, then they get it back
    24: 0.610,  # bounce your own permanent and draw if you did
    25: 0.508,  # one-shot fog that costs a creature
    26: 0.576,  # enters, find a Plains. Station 6 is the extra.
    27: 0.476,  # they choose to draw or attack. Twelve counters is not a plan.
    28: 0.546,  # 4/3 for 3 that needs a friend to attack
    29: 0.540,  # 2/2 that taps to pump by his own power
    30: 0.522,  # double strike for artifacts on a 4-mana 2/3
    31: 0.548,  # two counters. Beam cost asks for another creature.
    32: 0.546,  # look at six for an artifact. Whiffs a lot.
    33: 0.496,  # 5 mana 2/2 flash. X damage is usually 2 or 3.
    34: 0.550,  # stun, or 3 to an attacker. Flexible trick, not a removal spell.
    35: 0.552,  # blank a removal spell and leave a counter
    36: 0.532,  # 4/4 for 5, flying only on your turn
    37: 0.548,  # first strike bear. Draws only if a ship or equipment entered.
    38: 0.532,  # 2/1. The counter needs a power-8 attack.
    39: 0.594,  # team counter, or they draw 1 and you draw 2
    40: 0.572,  # look at five for a Plains or a cheap permanent
    41: 0.492,  # 0/1s that can't block and can't attack for value without an anthem
    42: 0.534,  # 4/5 for 5 with a conditional first-strike pump
    43: 0.626,  # three Officers now, ward 2, 5/5 flying lifelink at station 6
    44: 0.574,  # destroy a tapped creature on enter
    45: 0.574,  # 2/2 for 2 that taps something
    # Blue
    46: 0.550,  # loot on spells, then a free draw once it has three counters
    47: 0.558,  # 5/6 ward for 6. The steal needs a dilemma that turn.
    48: 0.536,  # 3/3 ward 1 for 4
    49: 0.530,  # two 2/2s and a loot, discounted by unique lands. Often stuck.
    50: 0.570,  # flash 2/1 that loots
    51: 0.580,  # soft counter, or tap and draw. Never dead.
    52: 0.532,  # 0/1 that grows when you draw. Dies before it matters.
    53: 0.610,  # draw X at six mana. X is 4 or 5 in the deck that plays it.
    54: 0.522,  # five-mana clone
    55: 0.548,  # 3/4 for 4, surveil 2 or get a ship back
    56: 0.570,  # two-mana bounce. The beam cost is too high to plan around.
    57: 0.538,  # 1/5 that sometimes steals a land from the top
    58: 0.570,  # stun on enter, flyer at station 6
    59: 0.576,  # flying 1/1 that pumps or surveils
    60: 0.508,  # four-mana tuck, one mana off if it is attacking
    61: 0.524,  # makes one equipment token. The copy ability is narrow.
    62: 0.598,  # counter mana value 2 or less, or kick into a hard counter
    63: 0.496,  # the Whale tutor misses. Station 8 and a conditional extra turn.
    64: 0.506,  # 5 mana 3/3 that looks at a few cards
    65: 0.540,  # 5/6 ward with landcycling. The body is the last resort.
    66: 0.536,  # 4/3 for 5 that bounces
    67: 0.574,  # 1/2 that surveils when tapped and grows on dilemmas
    68: 0.578,  # three turns of bounce, draw 2, and tokens, then you lose
    69: 0.632,  # grows on every noncreature spell and spends counters to draw
    70: 0.558,  # doubles charge counters. A 3/2 outside the ship deck.
    71: 0.560,  # draw two and a token, or draw two and untap. Four mana.
    72: 0.486,  # lock aura that does not even tap unless you pay 3
    73: 0.512,  # 2/1 flyer for 3. The dilemma counter is small.
    74: 0.650,  # every draw is two draws, on a 2/4. No enters ability, so removal blanks it.
    75: 0.552,  # 2/4 for 3. Guessing a mana value rarely draws more than once.
    76: 0.574,  # one-mana cantrip. Beam {2}{U} is a real second copy with a token.
    77: 0.618,  # hexproof while parked, charges itself on noncreature spells
    78: 0.502,  # both modes hand the other player a large profit
    79: 0.624,  # every instant and sorcery in the graveyard is flashback for a bounce
    80: 0.638,  # 0/4 for 2 that loots every turn, and draws clean once you have five lands
    81: 0.596,  # 3/3 for 2 that attacks once you have cast a spell
    82: 0.562,  # flash 3/1. Stifle is narrow; the stun mode is the card.
    83: 0.580,  # draw a card on enter, fly at station 5
    84: 0.514,  # 3/2 for 3 with an expensive tap ability
    85: 0.586,  # surveil 2 and draw, or blink your own creature
    86: 0.550,  # loot two, get a spell back, shuffle graves
    87: 0.536,  # 5/7 flyer for 7
    88: 0.506,  # they play one of yours, you play two of theirs. Often their card is better.
    89: 0.560,  # flash 2/2 for 4, choose two small modes. Flexible, not a bomb.
    90: 0.576,  # cast artifacts off the top, ships enter with two counters
    # Black
    91: 0.572,  # 6/6 that makes them discard, and cycles for -4/-4
    92: 0.494,  # +1/+0 anthem on a 4-mana 2/4
    93: 0.514,  # 2/2 flyer that draws if you sacrifice something
    94: 0.564,  # -2/-2 on enter, then a +2/+0 sword
    95: 0.518,  # 1/1 menace that grows once a turn when something dies
    96: 0.534,  # temporary reanimation of a small creature
    97: 0.538,  # 4/4 and a hasty 2/1 for 5
    98: 0.538,  # draw if you can tap three artifacts
    99: 0.530,  # power equals artifact creatures. Often a 1/2.
    100: 0.550,  # exile a grave card or Investigate
    101: 0.552,  # -1/-1 to everyone else, or each player sacrifices
    102: 0.524,  # eight mana, no reduction. Indestructible artifacts if you cast it.
    103: 0.556,  # -X/-X to every creature, including yours
    104: 0.492,  # -4/-0 aura. The threshold exile is slow.
    105: 0.576,  # -2/-2, or -6/-6 if you sacrifice a body or an artifact
    106: 0.612,  # 3/3 menace that drains 2 whenever you choose a mode
    107: 0.548,  # 3/1 that makes them discard
    108: 0.540,  # 5/4 for 5 that buys itself back from the grave on a dilemma
    109: 0.536,  # sacrifice outlet that only pumps Klingons
    110: 0.612,  # you don't lose for life. She draws, bleeds, then makes ten tokens.
    111: 0.562,  # +2/+0 that draws if the creature dies
    112: 0.562,  # a Treasure whenever a creature dies, and he grows off them
    113: 0.544,  # 7 mana 6/6. Doubles life they already lost.
    114: 0.578,  # destroy and surveil 1. Four mana, unconditional.
    115: 0.536,  # artifacts draw on damage. A 4/4 for 5 while you wait.
    116: 0.552,  # they get two tokens, or you draw two and everyone loses 2
    117: 0.574,  # they sacrifice, you Investigate, maybe two 2/2s
    118: 0.562,  # 3/1 for 2. Naming a card almost never matters.
    119: 0.592,  # 5/3 that draws two if you sacrifice a token or artifact
    120: 0.530,  # 2/5 that drains 1 on a dilemma
    121: 0.508,  # 2/3 with a menace activation
    122: 0.604,  # cast an artifact creature from the grave once a turn
    123: 0.580,  # look at two, keep one. Beam {1}{B} is cheap enough to use.
    124: 0.558,  # 1/1 lifelink that draws from the grave
    125: 0.572,  # destroy a nonartifact, or return a small artifact creature
    126: 0.534,  # 6/5 with landcycling
    127: 0.612,  # 2/2 lifelink that exiles a card from their hand
    128: 0.562,  # attacks exile a grave card, and leaving the grave makes a 2/2
    129: 0.554,  # 2/1 you can cast from the grave once
    130: 0.562,  # two creatures back, or exile a card from their hand
    131: 0.548,  # 2/2 that grants deathtouch or surveils on attack
    132: 0.552,  # they give you a card or lose 4. Comes back when you face a dilemma.
    133: 0.566,  # 2/1 that leaves a Borg
    134: 0.584,  # exile a grave card and draw, or return a creature
    135: 0.624,  # reanimate mana value 4 or less on enter. Station 10 is a bonus.
    136: 0.548,  # they lose 1 whenever a creature dies, 2 after seven deaths
    # Red
    137: 0.538,  # makes an Officer whenever you cast a spell. Starts as an 0/3.
    138: 0.550,  # 2/1 first strike that eventually deals 2
    139: 0.572,  # hasty 2/1. Beam {2}{R} is usable with a token.
    140: 0.516,  # they choose whether to counter their spell or give you a free one
    141: 0.532,  # 2/3 that pumps one creature
    142: 0.610,  # loot, a token, or a pump, on enter and on every attack
    143: 0.602,  # 3/3 trample that pings on landfall and grows on the second land
    144: 0.498,  # +2/+0 aura
    145: 0.616,  # station 3, then a hasty 2/1 every combat
    146: 0.534,  # 4/3 that deals 2 if you sacrifice something
    147: 0.560,  # temporary copies of their creatures. Good when X is 2 or more.
    148: 0.572,  # impulse two, or 2 damage to every creature and player
    149: 0.540,  # 3/3 haste that bounces a land to replay one
    150: 0.538,  # landfall impulse. Four mana before it does anything.
    151: 0.542,  # 6 damage that costs you a land
    152: 0.532,  # 7/6 trample with landcycling
    153: 0.498,  # nine mana, no discount, needs a spell already in the grave
    154: 0.582,  # {X}{R} deals 2X to a creature and exiles it
    155: 0.560,  # 5/4 trample and a shock to their board
    156: 0.534,  # 4/3 that makes one creature unable to block
    157: 0.566,  # 2/2 that copies one creature's triggers while she stays tapped
    158: 0.584,  # 3/2 trample for 2 that deals 1 when it dies
    159: 0.544,  # a Mountain, a temporary anthem, then you sacrifice a land
    160: 0.538,  # 7 mana 4/4 that pumps the team by your land count
    161: 0.542,  # +3/+2 or destroy an artifact
    162: 0.576,  # 2 damage for one mana, 4 if something already died
    163: 0.532,  # 2/2 that gets +1/+1 until end of turn on landfall
    164: 0.566,  # 3 damage and a rummage
    165: 0.574,  # 4/4. The exiled card is cast by discarding, not for free.
    166: 0.508,  # Treasures can't be sacrificed, so the tokens he makes don't make mana
    167: 0.538,  # flying haste 3/3 that gets cheaper with spells in the grave
    168: 0.594,  # exile two, play them until next turn
    169: 0.504,  # +2/+0, cheaper only while you are being attacked
    170: 0.524,  # +1/+0 anthem. Station 10 is not a body you get for free.
    171: 0.554,  # taps for red, and can make a Treasure
    172: 0.536,  # 4/4 reach that impulses one card
    173: 0.568,  # hasty 1/1 that Investigates if it dies on your turn
    174: 0.556,  # 3/2 pinger that untaps when you cast a spell
    175: 0.508,  # 5/3 that sacrifices a permanent after it attacks
    176: 0.570,  # hasty 2/1, impulses on death, and comes back from the grave once
    177: 0.564,  # double strike on a Klingon, or discard and draw two
    178: 0.580,  # 3 damage to any target. The beam cost is six mana.
    179: 0.554,  # two lands for one combat, then they leave
    180: 0.558,  # threaten each opponent's small creature
    181: 0.616,  # 4 damage and a one-turn steal, a loot, then they sacrifice
    # Green
    182: 0.558,  # a 3/3 artifact. Beam {3}{G} is expensive.
    183: 0.550,  # fight, gain 3, return a creature
    184: 0.530,  # 2/2 with a pump that gets cheaper
    185: 0.556,  # put the top X permanents onto the battlefield. Eight mana.
    186: 0.612,  # 3/3 and a 1/1, plus a pump that scales with creature types
    187: 0.644,  # extra land, and every creature explores, including her
    188: 0.622,  # power and toughness equal your lands, and you play lands off the top
    189: 0.534,  # 4/4 for 6 with landcycling and two counters for someone else
    190: 0.576,  # one or two creatures bite
    191: 0.546,  # 3/4 that adds two mana or puts a counter
    192: 0.548,  # +3/+3 trample, or they sacrifice an artifact or enchantment
    193: 0.564,  # 1/3 that makes a New Planet
    194: 0.552,  # 4/4 vigilance, exile a grave card or surveil
    195: 0.546,  # 3/1 and a counter. Trample for creatures that already have counters.
    196: 0.612,  # counters are doubled, and those creatures have hexproof
    197: 0.614,  # flash 3/2. Return a card, or persist another creature.
    198: 0.582,  # land or Planet onto the battlefield. Beam {1}{G} recasts it.
    199: 0.576,  # mill four, keep a permanent
    200: 0.566,  # ramp one, then turn lands into more lands
    201: 0.618,  # two 3/3s, and later tokens become 3/3 Illusions
    202: 0.528,  # 6 mana 2/2 that returns a permanent
    203: 0.546,  # a counter, untap, and +1/+1
    204: 0.602,  # grows and adds that much green every turn. Dies as an 0/1.
    205: 0.572,  # landfall gains 1, and the second land Investigates
    206: 0.540,  # an Officer when they target your creature
    207: 0.572,  # 1/1 deathtouch
    208: 0.568,  # 6/6 reach, vigilance, deathtouch for 5
    209: 0.546,  # a Food and a counter whenever you gain life
    210: 0.536,  # 5/5 reach that makes an Officer from the grave
    211: 0.584,  # your creatures tap for two green. Six mana, plus an 0/1.
    212: 0.568,  # flash 3/2 that gives hexproof
    213: 0.536,  # 4/3 that gets +2/+2 until end of turn on landfall
    214: 0.544,  # 7/5 that gains some life
    215: 0.558,  # a tapped any-color land whenever you attack
    216: 0.594,  # every creature enters with a +1/+1 counter, later two
    217: 0.570,  # surveil 2, and a counter if you exile a grave card
    218: 0.572,  # 4/4 trample that puts two counters on landfall
    219: 0.556,  # mill three, get a land back or a Treasure
    220: 0.542,  # exile a grave card, or destroy a small artifact or enchantment
    221: 0.658,  # discover 5 on enter, before anyone can remove it, and again on attack
    222: 0.604,  # two lands onto the battlefield
    223: 0.596,  # your lands enter untapped, and you find a land or Planet
    224: 0.576,  # 1/3 that taps for any color
    225: 0.542,  # 3/2 that grows on landfall
    # Gold
    226: 0.554,  # draw two, exile one, three times
    227: 0.608,  # affinity, and it exiles a creature on enter. Assimilate is the dream.
    228: 0.564,  # phases the board out for a few turns, including your own
    229: 0.534,  # 2/2 for 4. Worms that can't block.
    230: 0.538,  # 4/3 for 5, stun or two Clues
    231: 0.600,  # 3/3. Landfall adds mana, and a land draws a card.
    232: 0.538,  # 3/1 for 4 that returns a creature
    233: 0.550,  # draws when you cast an enchantment. A 3/3 the rest of the time.
    234: 0.536,  # 5 mana 3/3 that spreads counters
    235: 0.600,  # 3/2 for 2 that leaves a hasty 2/1
    236: 0.570,  # 2/4 that untaps something
    237: 0.566,  # grows when creatures die, and sacrificing a Klingon draws
    238: 0.550,  # temporary copy if you cast a noncreature spell
    239: 0.598,  # 2/3 that loots when it taps. Ships have hexproof on your turn.
    240: 0.568,  # haste, and mill five to find a land
    241: 0.608,  # reanimate a green or white creature, then draw off combat damage
    242: 0.614,  # 4/3 trample. Your creatures enter attacking.
    243: 0.560,  # 3/3 that impulses at power 8 and sacrifices to blow up a permanent
    244: 0.542,  # double your life and draw your hand. Seven mana.
    245: 0.548,  # +1/+1 for artifact creatures on a 1/1, plus a shock from hand
    246: 0.556,  # unblockable or surveil 2
    247: 0.546,  # a temporary 2/1 every combat
    248: 0.620,  # a permanent copy of their creature for three mana
    249: 0.612,  # levels from a 2/2 into a 3/4 looter into extra land drops
    250: 0.574,  # 3/4 trample that draws when a countered creature hits
    251: 0.502,  # 0/5 for three colors. Draws only at five creature types.
    252: 0.596,  # draws a card whenever you choose a mode
    253: 0.602,  # look at four, take a noncreature spell
    254: 0.580,  # */3 reach vigilance for two mana. Power is your land count.
    255: 0.596,  # surveil every upkeep, draw when the graveyard moves
    256: 0.510,  # 1/1 that doubles only after you have paid to grow him
    257: 0.538,  # each player draws and takes 2, three times
    258: 0.546,  # exile a card they can still cast for two more
    # Artifacts and ships
    259: 0.562,  # any-color rock that can become a 2/4 flyer
    260: 0.534,  # two equipment, evasion only if both attack. Equip 2 each.
    261: 0.558,  # 8 mana 7/7 flyer. Wrath of nonartifact creatures if you cast it.
    262: 0.550,  # 2/1 that surveils
    263: 0.518,  # nine mana. Permanents off the top every upkeep if you ever cast it.
    264: 0.606,  # X=0 destroys token creatures. Scales up to small permanents.
    265: 0.546,  # seven mana. Play one sideboard or grave permanent a turn, for its cost.
    266: 0.526,  # colorless mana for artifacts only, plus a loot
    267: 0.562,  # a Food, and a copy of any token
    268: 0.506,  # five mana. Counters equal to life gained this turn.
    269: 0.576,  # draw a card. Later, 2 damage or 3 life.
    270: 0.556,  # +1/+1 and surveil for one mana, equip 1. No enters ability.
    271: 0.562,  # stun a creature on enter. Equip 3 is the backup.
    272: 0.534,  # two 1/1s for 5. A 4/4 flyer at station 6.
    273: 0.610,  # impulse the first time you put counters on it each turn
    274: 0.558,  # find a basic, and put two charge counters on a ship
    # Lands. Planets are tapped gain-lands. Shocks track pair quality, not 60%.
    275: 0.554,  # Andoria, UR
    276: 0.538,  # colorless, drain 2 and a New Planet for 3B
    277: 0.552,  # Bajor, RW
    278: 0.534,  # Blood Crypt. Weaker pair, two life.
    279: 0.560,  # Breeding Pool
    280: 0.558,  # Cardassia Prime, WB
    281: 0.560,  # Earth, GU
    282: 0.564,  # white check land, gain 2 or surveil
    283: 0.556,  # Ferenginar, BG
    284: 0.534,  # draw two for 5U
    285: 0.548,  # Godless Shrine
    286: 0.558,  # Hallowed Fountain
    287: 0.562,  # blue check land
    288: 0.554,  # Orion, RG
    289: 0.552,  # Overgrown Tomb
    290: 0.564,  # green check land
    291: 0.548,  # Qo'noS, BR
    292: 0.554,  # red check land
    293: 0.562,  # Risa, GW
    294: 0.556,  # Romulus, UB
    295: 0.536,  # Sacred Foundry
    296: 0.530,  # 2 damage for 4R
    297: 0.540,  # colorless land. Five mana for a Federation ritual.
    298: 0.540,  # Steam Vents
    299: 0.550,  # Stomping Ground
    300: 0.536,  # two Officers for 4W
    301: 0.554,  # Temple Garden
    302: 0.548,  # mill four, keep a permanent, for 3G
    303: 0.554,  # taps for colorless. One mana and a tap finds a basic, tapped.
    304: 0.558,  # black check land. Can put a creature on top.
    305: 0.564,  # Vulcan, WU
    306: 0.554,  # Watery Grave
    307: 0.566,  # Plains
    308: 0.562,  # Island
    309: 0.558,  # Swamp
    310: 0.554,  # Mountain
    311: 0.564,  # Forest
}
