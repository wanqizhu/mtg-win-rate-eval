#!/usr/bin/env python3
"""Join per-card 17lands stats (Premier Draft, or PickTwo for OM1) with Scryfall features
for the 10 historical sets. Output work/training_cards.csv"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import extract, set_features  # noqa: E402

STATS = Path("/workspace/work/stats")
SETS = {"TDM": "PremierDraft", "FIN": "PremierDraft", "EOE": "PremierDraft", "OM1": "PickTwoDraft",
        "TLA": "PremierDraft", "ECL": "PremierDraft", "TMT": "PremierDraft", "SOS": "PremierDraft",
        "MSH": "PremierDraft", "HOB": "PremierDraft"}


def load_default_index(names: set[str]) -> dict:
    """Look up cards missing from the set file (bonus sheets etc.) in default-cards."""
    found = {}
    with gzip.open("/opt/mtg-data/scryfall/default-cards.jsonl.gz", "rt") as f:
        for line in f:
            c = json.loads(line)
            if c.get("lang") != "en":
                continue
            n = c.get("name")
            fn = n.split(" // ")[0] if n else n
            for key in (n, fn):
                if key in names and key not in found:
                    found[key] = c
    return found


def main() -> None:
    frames = []
    missing_all = set()
    for s, fmt in SETS.items():
        st = pd.read_csv(STATS / f"{s}_{fmt}_card_stats.csv")
        feats = set_features(s)
        feats["front_name"] = feats["name"].str.split(" // ").str[0]
        recs = feats.to_dict("records")
        by_name = {r["name"]: r for r in recs}
        by_front = {r["front_name"]: r for r in recs}
        rows = []
        missing = set()
        for r in st.to_dict("records"):
            fr = by_name.get(r["name"]) or by_front.get(r["name"])
            if fr is None:
                missing.add(r["name"])
                continue
            d = dict(fr)
            d.update({k: r[k] for k in st.columns if k != "name"})
            d["card_name_17l"] = r["name"]
            rows.append(d)
        if missing:
            idx = load_default_index(missing)
            for r in st.to_dict("records"):
                if r["name"] in missing and r["name"] in idx:
                    d = extract(idx[r["name"]])
                    d["set"] = s
                    d["front_name"] = d["name"].split(" // ")[0]
                    d["bonus_sheet"] = 1
                    d.update({k: r[k] for k in st.columns if k != "name"})
                    d["card_name_17l"] = r["name"]
                    rows.append(d)
                    missing.discard(r["name"])
        print(s, "matched", len(rows), "missing", len(missing), sorted(missing)[:10])
        missing_all |= missing
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    df["bonus_sheet"] = df.get("bonus_sheet", 0).fillna(0).astype(int)
    df = df[~df["type_line"].str.contains("Basic Land", na=False)]
    df.to_csv("/workspace/work/training_cards.csv", index=False)
    print(df.shape)
    print(df.groupby("set").agg(n=("name", "size"), gih=("gih_wr", "mean"), games=("n_gih", "sum")))


if __name__ == "__main__":
    main()
