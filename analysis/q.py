#!/usr/bin/env python3
"""Query historical cards by regex over oracle text / type line / name and show win rates.

Usage: q.py [-t TYPE_REGEX] [-r RARITY] [-n NAME_REGEX] [-c CMC] [-s SET] [ORACLE_REGEX]
Shows rel = GIH WR minus the set's mean GIH WR (cards with >= 500 GIH games).
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("text", nargs="?", default=None)
ap.add_argument("-t", "--type")
ap.add_argument("-r", "--rarity")
ap.add_argument("-n", "--name")
ap.add_argument("-c", "--cmc", type=float)
ap.add_argument("-s", "--set")
ap.add_argument("-k", "--limit", type=int, default=60)
ap.add_argument("-v", "--verbose", action="store_true")
ap.add_argument("--sort", default="rel")
a = ap.parse_args()

d = pd.read_csv("out/hist_joined.csv")
d = d[(d.gih >= 500) & ~d.type_line.fillna("").str.contains("Basic Land")]
d["rel"] = d.gih_wr - d.groupby("set").gih_wr.transform("mean")
m = pd.Series(True, index=d.index)
if a.text:
    m &= d.oracle.fillna("").str.contains(a.text, flags=re.I, regex=True)
if a.type:
    m &= d.type_line.fillna("").str.contains(a.type, flags=re.I, regex=True)
if a.rarity:
    m &= d.rarity.str[0] == a.rarity[0]
if a.name:
    m &= d.name.str.contains(a.name, flags=re.I, regex=True)
if a.cmc is not None:
    m &= d.cmc == a.cmc
if a.set:
    m &= d.set == a.set.upper()
r = d[m].sort_values(a.sort, ascending=False)
print(f"n={len(r)} mean_rel={r.rel.mean():.4f} median_rel={r.rel.median():.4f}")
for _, x in r.head(a.limit).iterrows():
    print(f"{x.rel:+.3f} {x.gih_wr:.3f} {x.set} {x.rarity[0].upper()} {x.colors if isinstance(x.colors, str) else '-':5} {x.mana_cost if isinstance(x.mana_cost, str) else '':12} {x['name']} | {x.type_line} {'' if pd.isna(x.power) else str(x.power)+'/'+str(x.toughness)}")
    if a.verbose:
        print("      " + str(x.oracle).replace("\n", " / ")[:400])
