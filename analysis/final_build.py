#!/usr/bin/env python3
"""Apply adjustments to model predictions and write the final submission table."""
import json, csv
import numpy as np
from pathlib import Path

preds=json.loads(Path("analysis/trk_model_predictions.json").read_text())
adj=json.loads(Path("analysis/adjustments.json").read_text())
by={p["name"]:p for p in preds}
missing=[k for k in adj if k not in by]
assert not missing, missing

PLANETS=["Andoria","Bajor","Cardassia Prime","Earth, Sector 001","Ferenginar","Orion","Qo'noS","Risa","Romulus","Vulcan"]
OUTPOSTS=["Federation Outpost","Ni'Var Science Academy","Q Continuum","Quark's Bar","The Unicomplex"]
SHOCKS=["Blood Crypt","Breeding Pool","Godless Shrine","Hallowed Fountain","Overgrown Tomb","Sacred Foundry","Steam Vents","Stomping Ground","Temple Garden","Watery Grave"]

rows=[]
for p in preds:
    v=p["model_gih_wr"]; notes=[]
    if p["name"] in adj:
        v+=adj[p["name"]]["delta"]; notes.append(adj[p["name"]]["reason"])
    if p["name"] in PLANETS:
        v+=0.008; notes.append("Group: Planet tapped duals — historical gain-1 tapped duals average ~+0.01 centred, and each distinct Planet name feeds Strange-new-worlds/landfall in TRK")
    if p["name"] in OUTPOSTS:
        v+=0.004; notes.append("Group: monocolour 'enters untapped with 2+ other lands' lands with an ETB effect — efficient fixing plus value")
    if p["name"] in SHOCKS:
        # smooth noisy per-colour-pair historical anchors toward the pooled shockland evidence (~0.550)
        v = 0.65*v + 0.35*0.550; notes.append("Group: shockland historical anchors pooled/smoothed (per-pair samples are colour-confounded)")
    v=float(np.clip(v, 0.44, 0.68))
    rows.append({"card_name":p["name"],"set":p["pool"],"rarity":p["rarity"],"mana_cost":p["mana_cost"],
                 "type_line":p["type_line"],"predicted_gih_win_rate":round(v,4),
                 "model_raw":p["model_gih_wr"],"adjustment_note":"; ".join(notes)})
rows.sort(key=lambda r:-r["predicted_gih_win_rate"])
with open("output/predictions.csv","w",newline="") as f:
    wr=csv.DictWriter(f, fieldnames=["card_name","set","rarity","mana_cost","type_line","predicted_gih_win_rate"])
    wr.writeheader()
    for r in rows: wr.writerow({k:r[k] for k in wr.fieldnames})
Path("output/predictions_full.json").write_text(json.dumps(rows,indent=1))
vals=np.array([r["predicted_gih_win_rate"] for r in rows])
print(f"n={len(rows)} mean={vals.mean():.4f} median={np.median(vals):.4f} min={vals.min():.3f} max={vals.max():.3f} sd={vals.std():.4f}")
from collections import Counter
print(Counter(r["set"] for r in rows))
for r in rows[:25]: print(f"{r['predicted_gih_win_rate']:.3f} {r['card_name']}")
print("...")
for r in rows[-15:]: print(f"{r['predicted_gih_win_rate']:.3f} {r['card_name']}")
