#!/usr/bin/env python3
"""Text/feature baseline: predict a card's GIH WR relative to its set mean.

Leave-one-set-out CV over the 10 historical sets, then fit on all and predict Star Trek.
Outputs out/baseline_cv.csv, out/baseline_trk.csv.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cards_lib import cn_int, full_text, load_set, oracle, faces
from features import featurize

warnings.filterwarnings("ignore")


def load_hist() -> pd.DataFrame:
    d = pd.read_csv("out/hist_joined.csv")
    d = d[(d.gih >= 500) & ~d.type_line.fillna("").str.contains("Basic Land")].copy()
    d["rel"] = d.gih_wr - d.groupby("set").gih_wr.transform("mean")
    d["bonus"] = (d.source == "bonus").astype(float)
    return d.reset_index(drop=True)


def load_trk() -> pd.DataFrame:
    rows = []
    seen = set()
    for c in sorted(load_set("trk"), key=cn_int):
        if c["name"] in seen or "Basic Land" in c["type_line"]:
            continue
        seen.add(c["name"])
        f0 = faces(c)[0]
        rows.append({
            "name": c["name"], "collector_number": c["collector_number"], "rarity": c["rarity"],
            "mana_cost": f0.get("mana_cost", c.get("mana_cost", "")), "cmc": c["cmc"],
            "colors": "".join(c.get("colors", [])), "color_identity": "".join(c["color_identity"]),
            "type_line": c["type_line"], "power": f0.get("power"), "toughness": f0.get("toughness"),
            "layout": c["layout"], "keywords": ";".join(c.get("keywords", [])), "oracle": oracle(c),
            "full_text": full_text(c), "bonus": 0.0, "sheet": "TRK",
        })
    for c in sorted(load_set("sds"), key=cn_int):
        f0 = faces(c)[0]
        rows.append({
            "name": c["name"], "collector_number": "SDS-" + c["collector_number"], "rarity": c["rarity"],
            "mana_cost": f0.get("mana_cost", c.get("mana_cost", "")), "cmc": c["cmc"],
            "colors": "".join(c.get("colors", [])), "color_identity": "".join(c["color_identity"]),
            "type_line": c["type_line"], "power": f0.get("power"), "toughness": f0.get("toughness"),
            "layout": c["layout"], "keywords": ";".join(c.get("keywords", [])), "oracle": oracle(c),
            "full_text": full_text(c), "bonus": 1.0, "sheet": "SDS",
        })
    return pd.DataFrame(rows)


def norm_text(s: pd.Series, names: pd.Series) -> pd.Series:
    out = []
    for t, n in zip(s.fillna(""), names):
        t = t.replace(n, "CARDNAME")
        for part in n.split(","):
            if part.strip():
                t = t.replace(part.strip(), "CARDNAME")
        out.append(t.lower())
    return pd.Series(out, index=s.index)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    ftr = featurize(train)
    fte = featurize(test)
    ftr["bonus"] = train.bonus.values
    fte["bonus"] = test.bonus.values
    w = np.sqrt(np.minimum(train.gih.values, 20000))
    y = train.rel.values
    out = {}
    gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
    gbm.fit(ftr, y, sample_weight=w)
    out["gbm"] = gbm.predict(fte)
    sc = StandardScaler().fit(ftr)
    tf = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=20000, sublinear_tf=True)
    ttr = tf.fit_transform(norm_text(train.oracle, train.name) + " " + train.type_line.fillna("").str.lower())
    tte = tf.transform(norm_text(test.oracle, test.name) + " " + test.type_line.fillna("").str.lower())
    Xtr = hstack([csr_matrix(sc.transform(ftr)), ttr]).tocsr()
    Xte = hstack([csr_matrix(sc.transform(fte)), tte]).tocsr()
    rid = Ridge(alpha=30.0).fit(Xtr, y, sample_weight=w)
    out["ridge"] = rid.predict(Xte)
    out["ens"] = 0.5 * out["gbm"] + 0.5 * out["ridge"]
    return out


def main() -> None:
    d = load_hist()
    preds = {k: np.zeros(len(d)) for k in ["gbm", "ridge", "ens"]}
    for s in sorted(d.set.unique()):
        te = d.set == s
        p = fit_predict(d[~te], d[te])
        for k in preds:
            preds[k][te.values] = p[k]
    for k in preds:
        d[f"pred_{k}"] = preds[k]
    rows = []
    for s in sorted(d.set.unique()) + ["ALL"]:
        sub = d if s == "ALL" else d[d.set == s]
        r = {"set": s, "n": len(sub)}
        for k in preds:
            r[f"r_{k}"] = np.corrcoef(sub.rel, sub[f"pred_{k}"])[0, 1]
            r[f"mae_{k}"] = (sub.rel - sub[f"pred_{k}"]).abs().mean()
        r["mae_zero"] = sub.rel.abs().mean()
        rows.append(r)
    res = pd.DataFrame(rows)
    print(res.round(4).to_string())
    d.to_csv("out/baseline_cv.csv", index=False)
    trk = load_trk()
    p = fit_predict(d, trk)
    for k in p:
        trk[f"pred_{k}"] = p[k]
    trk.drop(columns=["full_text"]).to_csv("out/baseline_trk.csv", index=False)
    print(trk.sort_values("pred_ens", ascending=False)[["name", "rarity", "pred_gbm", "pred_ridge", "pred_ens"]].head(30).to_string())
    print(trk.sort_values("pred_ens")[["name", "rarity", "pred_gbm", "pred_ridge", "pred_ens"]].head(20).to_string())


if __name__ == "__main__":
    main()
