#!/usr/bin/env python3
"""Print a set's cards sorted by GIH WR with compact text, for qualitative study."""
import sys

import pandas as pd

OUT = "/workspace/analysis/out"
code = sys.argv[1]
h = pd.read_csv(f"{OUT}/historical_features.csv")
h = h[(h["set"] == code) & (h.t_basic == 0) & (h.gih >= 300)].copy()
h["gih_wr"] *= 100
h["gd"] = h["gd_wr"] * 100
h["oh"] = h["oh_wr"] * 100
h["gns"] = h["gns_wr"] * 100
h["share"] = h["gih"] / h["set_games"] * 100
h = h.sort_values("gih_wr", ascending=False)
for _, r in h.iterrows():
    txt = str(r.oracle).replace("\n", " / ")
    txt = txt if len(txt) < 230 else txt[:230] + "…"
    pt = f" {int(r.power) if r.power == r.power else '*'}/{int(r.toughness) if r.toughness == r.toughness else '*'}" if r.t_creature else ""
    main = "" if r.in_main_set else " [BONUS]"
    print(f"{r.gih_wr:5.1f} gih{int(r.gih):6d} iwd{(r.gih_wr - r.gns):5.1f} {r.rarity_str[0].upper()} {r['name']}{main} {r.mana_cost} | {r.type_line}{pt} :: {txt}")
