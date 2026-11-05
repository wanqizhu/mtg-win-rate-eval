#!/usr/bin/env python3
"""Combine manual card ratings with the baseline model into final TRK GIH WR predictions.

rel_pred = c + B_MANUAL * manual_adj + B_MODEL * model
  * B_MANUAL / B_MODEL come from OLS on the pooled FRA + HOB blind tests (score_blind.py).
  * manual_adj moves the TRK manual ratings halfway toward the per-rarity means I used in the
    blind tests, so the calibrated coefficients see ratings on the same scale they were fit on.
  * c is chosen so the main-set mean relative win rate is zero (rel is defined vs the card mean).
gih_pred = SET_MEAN + rel_pred
Outputs ../predictions/trk_gih_predictions.csv and ../predictions/trk_gih_predictions.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SET_MEAN = 0.560
B_MANUAL = 0.73
B_MODEL = 0.72
RARITY_SHIFT_WEIGHT = 0.5
BLIND_MANUAL_RARITY_MEAN = {"common": -0.0061, "uncommon": -0.0003, "rare": 0.0046, "mythic": 0.0218}
BASICS = {"Plains": 0.006, "Island": 0.000, "Swamp": 0.002, "Mountain": -0.008, "Forest": -0.003}
OUT = Path(__file__).resolve().parent.parent / "predictions"


def load_manual(path: str = "trk_manual_ratings.txt") -> pd.DataFrame:
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        name, val, note = line.rstrip("\n").split("|", 2)
        rows.append((name, float(val), note))
    return pd.DataFrame(rows, columns=["name", "manual", "note"])


def main() -> None:
    t = pd.read_csv("out/baseline_trk.csv")
    t = t.merge(load_manual(), on="name", how="left", validate="one_to_one")
    assert t.manual.notna().all(), t[t.manual.isna()].name.tolist()

    main_set = t.sheet == "TRK"
    trk_rar_mean = t[main_set].groupby("rarity").manual.mean()
    shift = t.rarity.map(lambda r: BLIND_MANUAL_RARITY_MEAN[r] - trk_rar_mean[r])
    t["manual_adj"] = t.manual + RARITY_SHIFT_WEIGHT * shift

    raw = B_MANUAL * t.manual_adj + B_MODEL * t.pred_ens
    c = -raw[main_set].mean()
    t["rel_pred"] = c + raw
    t["gih_wr_pred"] = SET_MEAN + t.rel_pred

    basics = pd.DataFrame({"name": list(BASICS), "rel_pred": list(BASICS.values())})
    basics["gih_wr_pred"] = SET_MEAN + basics.rel_pred
    basics["rarity"] = "basic"
    basics["sheet"] = "TRK"
    basics["collector_number"] = ["307", "308", "309", "310", "311"]
    basics["note"] = "basic land: historical basic-land offset by color"

    cols = ["collector_number", "name", "sheet", "rarity", "colors", "mana_cost", "type_line",
            "manual", "manual_adj", "pred_ens", "rel_pred", "gih_wr_pred", "note"]
    final = pd.concat([t, basics], ignore_index=True)[cols]
    final = final.sort_values("gih_wr_pred", ascending=False).reset_index(drop=True)
    final.insert(0, "rank", np.arange(1, len(final) + 1))

    OUT.mkdir(exist_ok=True)
    final.round(4).to_csv(OUT / "trk_gih_predictions.csv", index=False)

    lines = ["| Rank | # | Card | Sheet | Rarity | Color | Predicted GIH WR |",
             "|---:|---|---|---|---|---|---:|"]
    for _, r in final.iterrows():
        col = r.colors if isinstance(r.colors, str) else ("—" if r.rarity != "basic" else "—")
        nm = str(r["name"]).replace("|", "\\|")
        lines.append(f"| {r['rank']} | {r.collector_number} | {nm} | {r.sheet} | {r.rarity} | {col} | {r.gih_wr_pred * 100:.1f}% |")
    (OUT / "trk_gih_predictions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    m = final[final.sheet == "TRK"]
    print(f"intercept c={c:+.4f}  n={len(final)} (main {len(m)}, SDS {len(final) - len(m)})")
    print("mean pred main", m.gih_wr_pred.mean().round(4), "sd", m.gih_wr_pred.std().round(4))
    print(final.groupby("rarity").gih_wr_pred.agg(["size", "mean"]).round(4))
    print(final.head(25)[["name", "rarity", "manual", "pred_ens", "gih_wr_pred"]].round(3).to_string())
    print(final.tail(15)[["name", "rarity", "manual", "pred_ens", "gih_wr_pred"]].round(3).to_string())


if __name__ == "__main__":
    main()
