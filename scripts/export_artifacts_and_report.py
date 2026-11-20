#!/usr/bin/env python3
"""Export intermediate artifacts, final CSVs, and PREDICTIONS.md markdown report for Star Trek (TRK + SDS) Premier Draft GIH win-rate evaluation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

REPO_DIR = Path("/agent/repos/mtg-win-rate-eval")
ART_DIR = REPO_DIR / "artifacts"
CURSOR_ART_DIR = Path("/opt/cursor/artifacts")


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    CURSOR_ART_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("/tmp/trk_final_predictions.csv")

    # Save clean CSV of final predictions with all intermediate sub-model columns
    export_cols = [
        "set",
        "collector_number",
        "card",
        "rarity",
        "mana_cost",
        "cmc",
        "type_line",
        "colors",
        "predicted_gih_wr",
        "predicted_gih_pct",
        "pred_ml_stacked",
        "pred_lgb",
        "pred_hgb",
        "pred_et",
        "pred_iwd",
        "pred_oh_dr",
        "pred_ridge",
        "pred_tfidf",
        "env_adjustment",
        "context_adjustment",
        "nearest_historical_analogs",
        "evaluation_rationale",
    ]
    df_export = df[export_cols].copy()
    df_export.to_csv(ART_DIR / "trk_gih_predictions.csv", index=False)
    df_export.to_csv(CURSOR_ART_DIR / "trk_gih_predictions.csv", index=False)

    # Copy intermediate JSON artifacts
    for src_name, dst_name in [
        ("/tmp/trk_loso_summary.json", "loso_cv_diagnostics.json"),
        ("/tmp/trk_env_simulation.json", "trk_environment_simulation.json"),
        ("/tmp/historical_17l_summary.json", "historical_10sets_summary.json"),
    ]:
        if Path(src_name).exists():
            shutil.copyfile(src_name, ART_DIR / dst_name)
            shutil.copyfile(src_name, CURSOR_ART_DIR / dst_name)

    # Build the comprehensive PREDICTIONS.md file containing Methodology, Intermediate Artifacts, and the Single Unified Table
    loso = json.loads((ART_DIR / "loso_cv_diagnostics.json").read_text())
    sim = json.loads((ART_DIR / "trk_environment_simulation.json").read_text())

    md_lines = [
        "# Star Trek (`TRK` & `SDS`) Premier Draft 17Lands GIH Win-Rate Predictions",
        "",
        "## 1. Executive Summary & Methodology",
        "",
        "To predict realized **17Lands Premier Draft Games-in-Hand (GIH) win rates** for every card available in **Star Trek (`TRK`)** Premier Draft (all **311 `TRK` main-set cards** plus all **30 `SDS` (`Stardates`) bonus-sheet cards**, **341 cards total**), we built a three-stage quantitative pipeline trained on **4,394,160 games** (`2,944` unique set-card records) across the 10 preceding 17Lands Premier Draft sets (`FRA`, `HOB`, `MSH`, `SOS`, `TMT`, `ECL`, `TLA`, `OM1`, `EOE`, and `FIN`):",
        "",
        "1. **Stage 1 — Multi-Task & Multi-Model Machine Learning Ensemble (`scripts/trk_model_pipeline.py`)**:",
        "   - Matched 100% of the 2,944 historical 17Lands card rows to their exact Scryfall metadata (including resolving `OM1` Arena `printed_name` vs paper `SPM` `name` and bonus-sheet reprints).",
        "   - Engineered **95+ Limited-specific features**, including:",
        "     - **Effective ETB Board Stats**: Parses token creation (`1/1 Officer`, `1/1 Borg`, `2/1 Klingon`, `2/2 Romulan`, `3/3 Illusion`) and automatic ETB `+1/+1` counters so multi-body cards (`Assembly Specialist`, `Holo-Engineer`, `U.S.S. Enterprise-A`, `Captain Jonathan Archer`, `Cha'DIch Investigator`) are evaluated on total immediate board power/toughness.",
        "     - **Set-Contextual Combat & Removal Dominance**: Evaluates every creature's effective P/T and every removal spell's damage/toughness reach against the exact Common (`3x`) + Uncommon (`1x`) creature toughness distribution of its set.",
        "     - **Spacecraft & `Station` Mechanics (Rules `702.184` & `721`)**: Explicitly models `Station` charge-counter thresholds (`2+` through `12+`), immediate ETB/static abilities prior to stationing, and cross-set transfer from `Edge of Eternities` (`EOE`), which also featured `Spacecraft` and `Station`.",
        "     - **Decomposed 17Lands Sub-Targets**: Trains both direct GIH win-rate regressors (`LightGBM`, `HistGradientBoosting`, `ExtraTrees`, `Ridge` tabular, and `TF-IDF` 1–2 gram oracle text `Ridge`) AND two structural 17Lands decompositions:",
        "       1. $\\widehat{\\text{GIH\\_WR}}_{\\text{IWD}} = \\widehat{\\text{GNS\\_WR}} + \\widehat{\\text{IWD}}$ (separating deck-color/archetype base win rate when not seen from true in-hand Improvement When Drawn).",
        "       2. $\\widehat{\\text{GIH\\_WR}}_{\\text{OH+DR}} = 0.37 \\cdot \\widehat{\\text{OH\\_WR}} + 0.63 \\cdot \\widehat{\\text{DR\\_WR}}$ (separating Opening Hand curve/mulligan impact from Drawn topdeck impact).",
        "   - Combined all 7 sub-models via a non-negative **Rarity-Stratified Ridge Meta-Stacker** plus an empirical **Reprint Prior** for cards with historical 17Lands data (`10` Shock Lands, `5` Basic Lands, and `3` `SDS` reprints).",
        "",
        "2. **Stage 2 — `TRK` Limited Environment & Monte Carlo Gameplay Simulator (`scripts/trk_environment_sim.py`)**:",
        "   - **Toughness & Removal Reach Math**: In `TRK` (weighted `3x` Common, `1x` Uncommon), **26.3% of creatures have Toughness 1** (plus `1/1 Officer`, `1/1 Borg`, and `2/1 Klingon` tokens), **48.8% have Toughness $\\le 2$**, **73.7% have Toughness $\\le 3$**, and **88.8% have Toughness $\\le 4$**. This makes 1-damage sweepers (`Khan, Despot by Design`, `D'Ghor Bloodletter`) and 2-to-4 damage removal (`Eject the Warp Core`, `Set Phasers to . . .`, `Planetary Bombardment`, `Plasma Cascade`, `Bat'leth`, `The Wrath of Khan`, `Encounter the Unknown`) premium.",
        "   - **Artifact Creature Density**: **26.8% of all C/U creatures in `TRK` are Artifact Creatures** (and 100% of mono-colored C/U artifact creatures are concentrated in `W` [8], `B` [8], and `U` [5], with `0` in `R` and `0` in `G`), making `WB` Borg/Artifact Creatures (`Locutus, the One Who Speaks`, `Assembly Specialist`, `Primary Unimatrix Adjunct`, `Overflow Submatrix`, `Borg Cube`) the #1 most cohesive synergy deck in the format, while turning modal artifact destruction (`Perils of the Past`, `Hull Breach`, `70,000 Light-Years from Home`) into maindeck removal.",
        "   - **`Strange new worlds` Monte Carlo Simulation ($N=20,000$ 40-card decks)**: A standard 2-color deck with 2 Planet duals averages only `2.32` differently named lands on Turn 4 and `2.50` on Turn 6 ($P(\\ge 5) = 0.0\\%$), whereas a dedicated `GU`/`RG` Planet/ramp deck averages `3.47` distinct lands on Turn 4 and `4.34` on Turn 6 ($P(\\ge 5) = 44.2\\%$). Linearly scaling payoffs (`The Traveler`, `Cloak and Phaser`, `Captain Pike`, `The Destruction of Praxis`, `Excalbian Gamemaster`) and the **10 Common `Land — Planet` gain-1-life duals** (analogous to `MSH`'s common gain-1-life duals, which averaged `56.5%` GIH WR) perform strongly.",
        "   - **`Station` Threshold Simulation ($N=10,000$ combat curves)**: Low intermediate thresholds (`Station 2+`, `3+`, `5+`, `6+`) come online in `1.4–2.5` turns, whereas high thresholds (`Station 8+`, `10+`, `12+`) require `3.2–4.6` turns of tapping creatures at sorcery speed. Spacecraft with strong ETB/passive effects (`U.S.S. Discovery`, `Borg Cube`, `U.S.S. Enterprise-A`, `U.S.S. Reliant`, `U.S.S. Enterprise-D`, `U.S.S. Voyager`, `U.S.S. Excelsior`, `D-7 Battle Cruiser`, `Oberth-Class Starship`, `Suurok-Class Science Vessel`) excel, whereas 9-mana `Deep Space 9` (`48.20%`) is a trap.",
        "",
        "3. **Stage 3 — Card-by-Card Contextual Evaluation & Empirical Rarity Calibration (`scripts/generate_trk_predictions.py`)**:",
        "   - Audited all 341 cards against the 2026-09-25 Comprehensive Rules and historical 17Lands twins, capturing card-specific mechanics such as `*/*` dynamic stats (`Captain Pike`, `The Traveler`, `Collective Drone`), `Kobayashi Maru Scenario`'s 4th-upkeep `'You lose the game'` trigger (`48.64%`), `Quark, Humble Proprietor` locking Treasures (`49.43%`), `Infinite Diversity`'s uncastable `{7}{R}{R}` cost (`45.35%`), and `U.S.S. Enterprise-D`'s repeatable turn-3 Station impulse draw (`60.42%`).",
        "   - Calibrated each rarity stratum's mean and spread to the 10-set 17Lands empirical distribution (`Common: mean 55.36%, std 1.91%`; `Uncommon: mean 55.55%, std 3.03%`; `Rare: mean 56.38%, std 4.42%`; `Mythic: mean 58.35%, std 5.79%`; `SDS Bonus Sheet: mean 56.62%, std 3.08%`).",
        "",
        "---",
        "",
        "## 2. Intermediate Artifacts & Cross-Validation Diagnostics",
        "",
        "- `artifacts/trk_gih_predictions.csv`: Full 341-row dataset with every sub-model prediction (`pred_lgb`, `pred_hgb`, `pred_et`, `pred_iwd`, `pred_oh_dr`, `pred_ridge`, `pred_tfidf`, `pred_ml_stacked`), environment adjustment, contextual adjustment, top 3 nearest historical analog cards, and evaluation rationale.",
        "- `artifacts/loso_cv_diagnostics.json`: Leave-One-Set-Out cross-validation metrics across the 10 historical sets (`2,516` main-set cards with $\\text{GIH} \\ge 200$):",
        f"  - **Overall LOSO-CV**: $\\text{{RMSE}} = {loso['overall']['rmse']:.4f}$, $\\text{{MAE}} = {loso['overall']['mae']:.4f}$, $\\text{{Pearson }} r = {loso['overall']['pearson_r']:.4f}$, $\\text{{Spearman }} \\rho = {loso['overall']['spearman_rho']:.4f}$",
        f"  - **Common ($n={loso['by_rarity']['common']['n_cards']}$)**: $\\text{{RMSE}} = {loso['by_rarity']['common']['rmse']:.4f}$, $\\text{{MAE}} = {loso['by_rarity']['common']['mae']:.4f}$, $r = {loso['by_rarity']['common']['pearson_r']:.4f}$",
        f"  - **Uncommon ($n={loso['by_rarity']['uncommon']['n_cards']}$)**: $\\text{{RMSE}} = {loso['by_rarity']['uncommon']['rmse']:.4f}$, $\\text{{MAE}} = {loso['by_rarity']['uncommon']['mae']:.4f}$, $r = {loso['by_rarity']['uncommon']['pearson_r']:.4f}$",
        f"  - **Rare ($n={loso['by_rarity']['rare']['n_cards']}$)**: $\\text{{RMSE}} = {loso['by_rarity']['rare']['rmse']:.4f}$, $\\text{{MAE}} = {loso['by_rarity']['rare']['mae']:.4f}$, $r = {loso['by_rarity']['rare']['pearson_r']:.4f}$",
        f"  - **Mythic ($n={loso['by_rarity']['mythic']['n_cards']}$)**: $\\text{{RMSE}} = {loso['by_rarity']['mythic']['rmse']:.4f}$, $\\text{{MAE}} = {loso['by_rarity']['mythic']['mae']:.4f}$, $r = {loso['by_rarity']['mythic']['pearson_r']:.4f}$",
        "- `artifacts/trk_environment_simulation.json`: Monte Carlo simulation outputs for `Strange new worlds`, `Station` thresholds, and color/archetype metrics.",
        "- `artifacts/historical_10sets_summary.json`: Historical 10-set 17Lands dataset summary.",
        "",
        "---",
        "",
        "## 3. Complete Predicted 17Lands GIH Win-Rate Table (`TRK` & `SDS` — All 341 Cards)",
        "",
        "| Set | # | Card | Rarity | Mana Cost | Type Line | Predicted GIH WR | ML Base | Nearest Historical 17Lands Analogs | Contextual Evaluation Notes |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]

    for _, r in df.iterrows():
        scode = str(r["set"])
        cn = int(r["cn_int"])
        card = str(r["card"])
        rar = str(r["rarity"]).capitalize()
        mc = str(r["mana_cost"]) if pd.notna(r["mana_cost"]) and str(r["mana_cost"]) != "" else "—"
        tl = str(r["type_line"])
        pred_pct = str(r["predicted_gih_pct"])
        ml_pct = f"{float(r['pred_ml_stacked'])*100:.2f}%"
        analogs = str(r["nearest_historical_analogs"]).replace("|", ";")
        notes = str(r["evaluation_rationale"]).replace("|", "/")
        md_lines.append(
            f"| `{scode}` | {cn} | **{card}** | {rar} | `{mc}` | {tl} | **{pred_pct}** | {ml_pct} | {analogs} | {notes} |"
        )

    report_text = "\n".join(md_lines) + "\n"
    (REPO_DIR / "PREDICTIONS.md").write_text(report_text)
    (CURSOR_ART_DIR / "PREDICTIONS.md").write_text(report_text)
    print(f"Wrote PREDICTIONS.md ({len(md_lines)} lines) and exported artifacts to {ART_DIR} and {CURSOR_ART_DIR}")


if __name__ == "__main__":
    main()
