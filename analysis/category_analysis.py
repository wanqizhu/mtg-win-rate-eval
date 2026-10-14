#!/usr/bin/env python3
"""Pooled category effects on GIH WR deviation (excluding held-out sets)."""
import sys

import numpy as np
import pandas as pd

OUT = "/workspace/analysis/out"
exclude = set(sys.argv[1:])
h = pd.read_csv(f"{OUT}/historical_features.csv")
h = h[(h.in_main_set == 1) & (h.t_basic == 0) & (h.gih >= 500) & ~h["set"].isin(exclude)].copy()
h["set_mean"] = h["set"].map(h.groupby("set").gih_wr.mean())
h["y"] = (h.gih_wr - h.set_mean) * 100


def cat(r):
    if r.t_land:
        return "land_dual" if r.dual_land else "land_other"
    if r.t_creature:
        return f"creature_cmc{int(min(r.cmc, 6))}"
    if r.t_planeswalker:
        return "planeswalker"
    if r.removal_any or r.wrath or r.bounce or r.counterspell:
        if r.counterspell:
            return "counterspell"
        if r.bounce and not r.removal_any:
            return "bounce"
        return "removal_spell"
    if r.draw_n >= 2:
        return "draw2plus_spell"
    if r.pump_eot:
        return "pump_trick"
    if r.t_equipment:
        return "equipment"
    if r.t_aura:
        return "aura"
    if r.t_artifact:
        return "artifact_other"
    if r.t_enchantment:
        return "enchantment_other"
    return "spell_other"


h["cat"] = h.apply(cat, axis=1)
pd.set_option("display.width", 200)
g = h.groupby(["cat", "rarity_str"]).y.agg(["mean", "std", "count"]).round(2)
print(g.unstack("rarity_str").to_string())
print(h.groupby("cat").y.agg(["mean", "std", "count"]).round(2).sort_values("mean").to_string())
print(h.groupby("n_colors").y.agg(["mean", "std", "count"]).round(2))
print(h[h.t_creature == 1].groupby(pd.cut(h.pt_minus_cmc, [-20, -3, -1, 0, 1, 2, 20])).y.agg(["mean", "count"]).round(2))
for colname in ["c_W", "c_U", "c_B", "c_R", "c_G"]:
    print(colname, h[(h[colname] == 1) & (h.n_colors == 1)].groupby("set").y.mean().round(2).to_dict())
