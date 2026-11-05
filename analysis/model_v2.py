#!/usr/bin/env python3
"""Improved baseline: hand features + set-context synergy features + TF-IDF ridge + LightGBM.

Set-context features are computed within each set (historical or Star Trek), so they transfer:
  * tribe_support: how many other cards in the set mention one of this card's creature subtypes
  * mech_share: how many other cards in the set share this card's rarest mechanic words
  * color_depth: mean hand-feature quality score of the card's color at common/uncommon (proxy)
Outputs out/v2_cv.csv and out/v2_trk.csv.
"""

from __future__ import annotations

import re
import sys
import warnings
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baseline_model import load_hist, load_trk, norm_text
from features import featurize

warnings.filterwarnings("ignore")

STOP = set("""a an the of to and or target creature creatures you your that this it its is are be with for on in
each player opponent opponents control controls card cards until end turn may if from into as at beginning when whenever
put gets get then can't can't up one two three four x other another their they them have has gain gains+ battlefield
hand library graveyard mana cost spell spells counter counters +1/+1 damage deals deal life draw any instead
less more than only those all number equal among total under owner's owner don't do does not non""".split())

EXTRA = {
    "x_wrath": r"destroy all|exile all creatures|all creatures get -|deals? \d+ damage to each (?:other )?creature|each player sacrifices",
    "x_pw": r"planeswalker",
    "x_scaling": r"for each|equal to the number",
    "x_lose_game": r"you lose the game",
    "x_win_game": r"you win the game",
    "x_cast_free": r"without paying",
    "x_extra_turn": r"extra turn",
    "x_steal": r"gain control of",
    "x_tutor_any": r"search your library for a card",
    "x_drawback_sac": r"sacrifice (?:a|another) (?:nonland permanent|land|creature)",
    "x_symmetric": r"each player",
    "x_opp_draws": r"(?:target opponent|that player|its controller) draws",
}


def mech_words(text: str, name: str) -> set[str]:
    t = text.lower().replace(name.lower(), " ")
    t = re.sub(r"\(.*?\)", " ", t)
    words = set(re.findall(r"[a-z][a-z'\-]{3,}", t))
    return {w for w in words if w not in STOP}


def subtypes(type_line: str) -> list[str]:
    if "—" not in type_line:
        return []
    out = []
    for face in type_line.split("//"):
        if "—" in face and "Creature" in face.split("—")[0]:
            out += [w for w in face.split("—")[1].split() if w]
    return out


def set_context(df: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    res = pd.DataFrame(index=df.index, columns=["tribe_support", "mech_share", "mech_rare_share"], dtype=float)
    for g in group.unique():
        idx = df.index[group == g]
        texts = {i: str(df.at[i, "oracle"]) for i in idx}
        words = {i: mech_words(texts[i], str(df.at[i, "name"])) for i in idx}
        wc = Counter(w for i in idx for w in words[i])
        n = len(idx)
        for i in idx:
            subs = subtypes(str(df.at[i, "type_line"]))
            sup = 0
            for s in subs:
                pat = re.compile(rf"\b{re.escape(s.lower())}s?\b")
                sup += sum(1 for j in idx if j != i and pat.search(texts[j].lower()))
            res.at[i, "tribe_support"] = np.log1p(sup)
            ws = [wc[w] - 1 for w in words[i] if wc[w] >= 2]
            res.at[i, "mech_share"] = np.log1p(np.mean(ws)) if ws else 0.0
            rare_ws = sorted(ws)[:3]
            res.at[i, "mech_rare_share"] = np.log1p(np.mean(rare_ws)) if rare_ws else 0.0
        res.loc[idx, "mech_share"] -= res.loc[idx, "mech_share"].mean()
        res.loc[idx, "tribe_support"] = res.loc[idx, "tribe_support"]
    return res.astype(float)


def build_features(df: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    f = featurize(df)
    text = df.oracle.fillna("").str.lower()
    for k, pat in EXTRA.items():
        f[k] = text.str.contains(pat, regex=True).astype(float)
    f["bonus"] = df.bonus.values
    ctx = set_context(df, group)
    f = pd.concat([f, ctx], axis=1)
    return f


def fit_predict(Ftr, ytr, wtr, Ttr, Fte, Tte, seed=0):
    gbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.025, num_leaves=15, min_child_samples=15,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                            random_state=seed, verbose=-1)
    gbm.fit(Ftr, ytr, sample_weight=wtr)
    p_gbm = gbm.predict(Fte)
    sc = StandardScaler().fit(Ftr)
    tf = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=25000, sublinear_tf=True)
    Xtr = hstack([csr_matrix(sc.transform(Ftr)), tf.fit_transform(Ttr)]).tocsr()
    Xte = hstack([csr_matrix(sc.transform(Fte)), tf.transform(Tte)]).tocsr()
    rid = Ridge(alpha=30.0).fit(Xtr, ytr, sample_weight=wtr)
    p_rid = rid.predict(Xte)
    return p_gbm, p_rid


def main() -> None:
    d = load_hist()
    trk = load_trk()
    Fh = build_features(d, d.set)
    Ft = build_features(trk, trk.sheet.where(trk.sheet == "TRK", "TRK"))
    Th = norm_text(d.oracle, d.name) + " " + d.type_line.fillna("").str.lower()
    Tt = norm_text(trk.oracle, trk.name) + " " + trk.type_line.fillna("").str.lower()
    w = np.sqrt(np.minimum(d.gih.values, 20000))
    pg = np.zeros(len(d))
    pr = np.zeros(len(d))
    for s in sorted(d.set.unique()):
        te = (d.set == s).values
        a, b = fit_predict(Fh[~te], d.rel.values[~te], w[~te], Th[~te], Fh[te], Th[te])
        pg[te], pr[te] = a, b
    d["v2_gbm"], d["v2_ridge"] = pg, pr
    d["v2_ens"] = 0.5 * pg + 0.5 * pr
    old = pd.read_csv("out/baseline_cv.csv")[["set", "name", "pred_ens"]]
    d = d.merge(old, on=["set", "name"], how="left")
    rows = []
    for s in sorted(d.set.unique()) + ["ALL"]:
        sub = d if s == "ALL" else d[d.set == s]
        r = {"set": s}
        for k in ["v2_gbm", "v2_ridge", "v2_ens", "pred_ens"]:
            r[f"r_{k}"] = np.corrcoef(sub.rel, sub[k])[0, 1]
        r["mae_v2"] = (sub.rel - sub.v2_ens).abs().mean()
        r["mae_v1"] = (sub.rel - sub.pred_ens).abs().mean()
        rows.append(r)
    print(pd.DataFrame(rows).round(4).to_string())
    d.to_csv("out/v2_cv.csv", index=False)
    gbm_seeds = []
    rid = None
    for seed in range(5):
        a, b = fit_predict(Fh, d.rel.values, w, Th, Ft, Tt, seed=seed)
        gbm_seeds.append(a)
        rid = b
    trk["v2_gbm"] = np.mean(gbm_seeds, axis=0)
    trk["v2_ridge"] = rid
    trk["v2_ens"] = 0.5 * trk.v2_gbm + 0.5 * trk.v2_ridge
    trk = pd.concat([trk.reset_index(drop=True), Ft[["tribe_support", "mech_share", "mech_rare_share"]].reset_index(drop=True)], axis=1)
    trk.drop(columns=["full_text"]).to_csv("out/v2_trk.csv", index=False)
    imp = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.025, num_leaves=15, min_child_samples=15,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, verbose=-1,
                            importance_type="gain").fit(Fh, d.rel.values, sample_weight=w)
    fi = pd.Series(imp.feature_importances_, index=Fh.columns).sort_values(ascending=False)
    print((fi / fi.sum()).head(25).round(3).to_string())


if __name__ == "__main__":
    main()
