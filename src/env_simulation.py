#!/usr/bin/env python3
"""Environment-level combat simulation, removal survival analysis, and color-pair synergy scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/workspace/analysis")
RARITY_WEIGHTS = {"common": 6.0, "uncommon": 2.2, "rare": 0.6, "mythic": 0.2}


def simulate_combat_outcome(f1: dict, f2: dict) -> float:
    """Simulate 1v1 combat between creature 1 and creature 2 using their extracted feature dicts."""
    p1, t1 = f1["power"], f1["toughness"]
    p2, t2 = f2["power"], f2["toughness"]
    if t1 <= 0 or t2 <= 0:
        return 0.0

    # Defenders cannot attack unless conditional
    can_atk1 = f1["kw_defender"] == 0
    can_atk2 = f2["kw_defender"] == 0

    c1_unblockable = can_atk1 and f1["kw_flying"] >= 0.8 and f2["kw_flying"] == 0 and f2["kw_reach"] == 0
    c2_unblockable = can_atk2 and f2["kw_flying"] >= 0.8 and f1["kw_flying"] == 0 and f1["kw_reach"] == 0

    if c1_unblockable and not c2_unblockable:
        return 0.85 if p1 >= p2 else 0.35
    if c2_unblockable and not c1_unblockable:
        return -0.85 if p2 >= p1 else -0.35

    fs1 = f1["kw_first_strike"] >= 0.8 or f1["kw_double_strike"] >= 0.8
    fs2 = f2["kw_first_strike"] >= 0.8 or f2["kw_double_strike"] >= 0.8
    dt1 = f1["kw_deathtouch"] >= 0.8
    dt2 = f2["kw_deathtouch"] >= 0.8

    dmg1_first = p1 if fs1 else 0.0
    dmg2_first = p2 if fs2 else 0.0

    c2_dies_first = (dmg1_first >= t2) or (dmg1_first > 0 and dt1)
    c1_dies_first = (dmg2_first >= t1) or (dmg2_first > 0 and dt2)

    if c2_dies_first and not c1_dies_first:
        c1_dies, c2_dies = False, True
    elif c1_dies_first and not c2_dies_first:
        c1_dies, c2_dies = True, False
    else:
        tot1 = p1 * (2.0 if f1["kw_double_strike"] >= 0.8 else 1.0)
        tot2 = p2 * (2.0 if f2["kw_double_strike"] >= 0.8 else 1.0)
        c2_dies = (tot1 >= t2) or (tot1 > 0 and dt1)
        c1_dies = (tot2 >= t1) or (tot2 > 0 and dt2)

    if c2_dies and not c1_dies:
        return (0.90 if can_atk1 else 0.45) + 0.05 * f1["kw_lifelink"]
    if c1_dies and not c2_dies:
        return (-0.90 if can_atk2 else -0.45) - 0.05 * f2["kw_lifelink"]
    if c1_dies and c2_dies:
        cmc_diff = np.clip((f2["eff_cmc"] - f1["eff_cmc"]) * 0.25, -0.6, 0.6)
        return float(cmc_diff)
    return float(np.clip((p1 - p2) * 0.1 + (f1["kw_lifelink"] - f2["kw_lifelink"]) * 0.25, -0.5, 0.5))


def compute_set_environment_features(
    cards_dict: dict[str, dict], base_feats_by_name: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Compute environment-contextual combat, removal survival, and color pool features for a set."""
    creatures = []
    removals = []
    for cname, c in cards_dict.items():
        if c.get("is_bonus_sheet") or "Basic Land" in c.get("type_line", ""):
            continue
        f = base_feats_by_name[cname]
        rw = RARITY_WEIGHTS.get((c.get("rarity") or "common").lower(), 1.0)
        if f["has_body"] > 0 and f["eff_cmc"] > 0:
            creatures.append((cname, c, f, rw))
        if f["is_any_removal"] > 0:
            removals.append((cname, c, f, rw))

    avg_creat_tou = (
        np.average([f["toughness"] for _, _, f, _ in creatures], weights=[rw for _, _, _, rw in creatures])
        if creatures
        else 3.0
    )
    avg_creat_pow = (
        np.average([f["power"] for _, _, f, _ in creatures], weights=[rw for _, _, _, rw in creatures])
        if creatures
        else 2.6
    )
    flyer_share = (
        np.average([f["kw_flying"] for _, _, f, _ in creatures], weights=[rw for _, _, _, rw in creatures])
        if creatures
        else 0.15
    )

    env_feats: dict[str, dict[str, float]] = {}

    for cname, c in cards_dict.items():
        f = base_feats_by_name[cname]
        combat_score = 0.0
        same_cmc_combat = 0.0
        removal_survival = 0.0
        removal_kill_share = 0.0

        if f["has_body"] > 0 and creatures:
            near_scores, near_wts = [], []
            all_scores, all_wts = [], []
            for oname, oc, of, orw in creatures:
                if oname == cname:
                    continue
                out = simulate_combat_outcome(f, of)
                all_scores.append(out)
                all_wts.append(orw)
                if abs(of["eff_cmc"] - f["eff_cmc"]) <= 1.0:
                    near_scores.append(out)
                    near_wts.append(orw)
            combat_score = float(np.average(all_scores, weights=all_wts)) if all_wts else 0.0
            same_cmc_combat = float(np.average(near_scores, weights=near_wts)) if near_wts else combat_score

            surv_scores, surv_wts = [], []
            for rname, rc, rf, rrw in removals:
                killed = False
                if rf["uncond_destroy_exile"] > 0 or rf["is_sweeper"] >= 0.8 or rf["is_pacifism"] > 0:
                    killed = True
                elif rf["burn_dmg"] > 0 and rf["burn_dmg"] >= f["toughness"]:
                    killed = True
                elif rf["minus_tou"] > 0 and rf["minus_tou"] >= f["toughness"]:
                    killed = True
                elif rf["cond_destroy_exile"] > 0:
                    killed = True
                elif rf["is_bite"] > 0 and avg_creat_pow >= f["toughness"]:
                    killed = True

                if f["kw_indestructible"] > 0 and rf["minus_tou"] < f["toughness"] and "exile" not in (rc.get("oracle_text") or "").lower():
                    killed = False
                if f["kw_hexproof"] > 0 and rf["is_sweeper"] < 0.8:
                    killed = False

                if not killed:
                    surv_scores.append(1.0)
                else:
                    etb_val = 0.5 if (f["has_etb_trigger"] > 0 or f["enters_prepared"] > 0 or f["empower_jace_n"] >= 3) else 0.0
                    mana_adv = np.clip((rf["eff_cmc"] + f["kw_ward"] - f["eff_cmc"]) * 0.25, -0.5, 0.5)
                    surv_scores.append(float(etb_val + mana_adv))
                surv_wts.append(rrw)
            removal_survival = float(np.average(surv_scores, weights=surv_wts)) if surv_wts else 0.0

        if f["is_any_removal"] > 0 and creatures:
            kills, kwts = [], []
            for oname, oc, of, orw in creatures:
                if f["uncond_destroy_exile"] > 0 or f["is_sweeper"] >= 0.8 or f["is_pacifism"] > 0:
                    k = 1.0
                elif f["burn_dmg"] > 0:
                    k = 1.0 if f["burn_dmg"] >= of["toughness"] else 0.0
                elif f["minus_tou"] > 0:
                    k = 1.0 if f["minus_tou"] >= of["toughness"] else 0.0
                elif f["is_bite"] > 0:
                    k = 1.0 if avg_creat_pow >= of["toughness"] else 0.3
                else:
                    k = 0.65
                if f["is_color_hose"] > 0:
                    k *= 0.40
                kills.append(k)
                kwts.append(orw)
            removal_kill_share = float(np.average(kills, weights=kwts))

        env_feats[cname] = {
            "env_combat_score": combat_score,
            "env_same_cmc_combat": same_cmc_combat,
            "env_removal_survival": removal_survival,
            "env_removal_kill_share": removal_kill_share,
            "env_tou_vs_set_avg": (f["toughness"] - avg_creat_tou) if f["has_body"] > 0 else 0.0,
            "env_pow_vs_set_avg": (f["power"] - avg_creat_pow) if f["has_body"] > 0 else 0.0,
            "env_reach_vs_flyer_density": f["kw_reach"] * flyer_share,
        }

    return env_feats
