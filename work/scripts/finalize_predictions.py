#!/usr/bin/env python3
"""
Merge expert predictions with card metadata and the statistical baseline,
report calibration diagnostics, and write the final prediction table.

Outputs
  work/predictions_fra_detailed.csv   (all columns incl. model baseline + rationale)
  predictions/reality_fracture_gih_predictions.csv  (final deliverable table)
  work/calibration_report.txt
"""
import json, gzip, sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from expert_predictions import PRED, SPECIAL_GUESTS, BASICS

ROOT = "/workspace"
cards = json.load(open(f"{ROOT}/work/fra_main_cards.json"))
model = pd.read_csv(f"{ROOT}/work/model_predictions_fra.csv").set_index("name")

rows = []
for c in cards:
    name = c["name"]
    if name not in PRED:
        raise SystemExit(f"missing prediction for {name}")
    p, why = PRED[name]
    m = model.loc[name] if name in model.index else None
    rows.append(dict(
        collector_number=int(c["collector_number"]), name=name, rarity=c["rarity"],
        colors="".join(c["colors"]) or "C", mana_cost=c["mana_cost"], cmc=c["cmc"],
        type_line=c["type_line"], section="main",
        model_blend=round(100 * m["model_blend"], 2) if m is not None else np.nan,
        model_lgb=round(100 * m["model_lgb"], 2) if m is not None else np.nan,
        model_txt=round(100 * m["model_txt"], 2) if m is not None else np.nan,
        predicted_gih_wr=p, rationale=why,
    ))

scry = [json.loads(l) for l in gzip.open(f"{ROOT}/data/scryfall/sets/fra.jsonl.gz", "rt")]
seen = set()
for r in sorted(scry, key=lambda r: int(r["collector_number"]) if r["collector_number"].isdigit() else 9999):
    if "Basic Land" in r.get("type_line", "") and r["name"] not in seen and 281 <= int(r["collector_number"]) <= 290:
        seen.add(r["name"])
        p, why = BASICS[r["name"]]
        rows.append(dict(collector_number=int(r["collector_number"]), name=r["name"], rarity="common",
                         colors="C", mana_cost="", cmc=0.0, type_line=r["type_line"], section="basic",
                         model_blend=np.nan, model_lgb=np.nan, model_txt=np.nan,
                         predicted_gih_wr=p, rationale=why))

for l in open(f"{ROOT}/data/scryfall/sets/fra-special-guests.jsonl"):
    r = json.loads(l)
    p, why = SPECIAL_GUESTS[r["name"]]
    rows.append(dict(collector_number=1000 + int(r["collector_number"]), name=r["name"], rarity=r["rarity"],
                     colors="".join(r.get("colors", [])) or "C", mana_cost=r.get("mana_cost", ""), cmc=r["cmc"],
                     type_line=r["type_line"], section="special_guest",
                     model_blend=np.nan, model_lgb=np.nan, model_txt=np.nan,
                     predicted_gih_wr=p, rationale=why))

df = pd.DataFrame(rows).sort_values("collector_number").reset_index(drop=True)
df["expert_raw"] = df.predicted_gih_wr

# ---------------- calibration: rarity-level mean/spread adjustment ----------------
# The raw expert numbers are deliberately conservative (shrunk toward the mean).
# Historically (cards with >=100 GIH games) the realised spread is
#   common sd ~2.6, uncommon ~3.4, rare ~5.6, mythic ~5.6
# while the raw expert sd is 1.3 / 1.7 / 3.3 / 3.9.  With an expected
# expert-vs-truth correlation of roughly 0.6-0.7 the MSE-optimal prediction
# spread is ~0.65 x realised, so we stretch commons and uncommons (which are
# the most compressed) and nudge rarity means to the historical levels.
# Lands and basics are excluded: their GIH WR is essentially "set mean".
STRETCH = {"common": (55.15, 1.35), "uncommon": (55.60, 1.20), "rare": (56.20, 1.05), "mythic": (58.20, 1.00)}
is_land = df.type_line.str.contains("Land")
adj = (df.section == "main") & ~is_land
for rar, (target_mean, k) in STRETCH.items():
    sel = adj & (df.rarity == rar)
    raw_mean = df.loc[sel, "expert_raw"].mean()
    df.loc[sel, "predicted_gih_wr"] = target_mean + k * (df.loc[sel, "expert_raw"] - raw_mean)
df["predicted_gih_wr"] = df.predicted_gih_wr.round(1)

# ---------------- calibration diagnostics ----------------
main = df[df.section == "main"]
hist = pd.read_csv(f"{ROOT}/work/training_cards.csv")
hist = hist[hist.n_gih >= 500]
hist["gih"] = 100 * hist.gih_wr
hist_set_mean = hist.groupby("set").gih.mean()

lines = []
lines.append("FRA expert predictions: calibration report")
lines.append("=" * 60)
lines.append(f"main-set cards: {len(main)}   overall mean = {main.predicted_gih_wr.mean():.2f}   sd = {main.predicted_gih_wr.std():.2f}")
lines.append(f"historical per-set mean of card GIH WR (n>=500): {hist_set_mean.round(2).to_dict()}")
lines.append("")
lines.append("By rarity (prediction vs historical pooled, n>=500):")
for r in ["common", "uncommon", "rare", "mythic"]:
    a = main[main.rarity == r].predicted_gih_wr
    h = hist[hist.rarity == r].gih
    lines.append(f"  {r:9s} pred mean {a.mean():.2f} sd {a.std():.2f} (n={len(a)})   hist mean {h.mean():.2f} sd {h.std():.2f} (n={len(h)})")
lines.append("")
lines.append("By color (main set):")
for col, g in main.groupby("colors"):
    if len(g) >= 5:
        lines.append(f"  {col:4s} n={len(g):3d} mean {g.predicted_gih_wr.mean():.2f}")
lines.append("")
m2 = main.dropna(subset=["model_blend"])
corr = np.corrcoef(m2.predicted_gih_wr, m2.model_blend)[0, 1]
lines.append(f"Correlation expert vs statistical baseline (blend): {corr:.3f}  (n={len(m2)})")
lines.append(f"Mean abs difference: {np.abs(m2.predicted_gih_wr - m2.model_blend).mean():.2f}")
d = (m2.predicted_gih_wr - m2.model_blend)
m2 = m2.assign(diff=d.round(2))
lines.append("")
lines.append("Largest expert > model disagreements:")
for _, r in m2.sort_values("diff", ascending=False).head(15).iterrows():
    lines.append(f"  {r['name'][:38]:38s} {r.rarity[:1]} expert {r.predicted_gih_wr:5.1f} model {r.model_blend:5.1f}  (+{r['diff']:.1f})")
lines.append("Largest expert < model disagreements:")
for _, r in m2.sort_values("diff").head(15).iterrows():
    lines.append(f"  {r['name'][:38]:38s} {r.rarity[:1]} expert {r.predicted_gih_wr:5.1f} model {r.model_blend:5.1f}  ({r['diff']:.1f})")
lines.append("")
lines.append("Top 20 predicted cards:")
for _, r in main.sort_values("predicted_gih_wr", ascending=False).head(20).iterrows():
    lines.append(f"  {r.predicted_gih_wr:5.1f}  {r['name']} ({r.rarity})")
lines.append("Bottom 15 predicted cards:")
for _, r in main.sort_values("predicted_gih_wr").head(15).iterrows():
    lines.append(f"  {r.predicted_gih_wr:5.1f}  {r['name']} ({r.rarity})")

report = "\n".join(lines)
print(report)
open(f"{ROOT}/work/calibration_report.txt", "w").write(report + "\n")

df.to_csv(f"{ROOT}/work/predictions_fra_detailed.csv", index=False)
os.makedirs(f"{ROOT}/predictions", exist_ok=True)
final = df[["collector_number", "name", "rarity", "colors", "mana_cost", "type_line", "section", "predicted_gih_wr"]].copy()
final["collector_number"] = final.collector_number.where(final.section != "special_guest", "SPG-" + (final.collector_number - 1000).astype(str))
final["predicted_gih_wr"] = (final.predicted_gih_wr / 100).round(4)
final.to_csv(f"{ROOT}/predictions/reality_fracture_gih_predictions.csv", index=False)
print(f"\nwrote {len(final)} rows")
