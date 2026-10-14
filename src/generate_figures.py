#!/usr/bin/env python3
"""Generate diagnostic figures for LOSO-CV evaluation and Reality Fracture (FRA) predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

OUT_DIR = Path("/workspace/analysis")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path("/opt/cursor/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8fafc",
    "axes.grid": True,
    "grid.alpha": 0.35,
})


def make_fig1_loso_cv() -> None:
    df = pd.read_csv(OUT_DIR / "historical_oof_predictions.csv")
    stats = json.loads((OUT_DIR / "historical_game_stats.json").read_text())
    main_df = df[df["is_bonus_sheet"] == 0.0].copy()

    # Calibrate sample-weight intercept per set
    in_univ = {"TDM", "EOE", "OM1", "ECL", "SOS"}
    main_df["oof_cal"] = main_df["oof_pred"]
    for s in main_df["set"].unique():
        m = main_df["set"] == s
        target = 0.5570 if s in in_univ else (0.5590 if m.sum() >= 240 else 0.5630)
        main_df.loc[m, "oof_cal"] -= (main_df.loc[m, "oof_pred"].mean() - target)

    yt = main_df["gih_wr"].to_numpy()
    yp = main_df["oof_cal"].to_numpy()
    yh = main_df["oof_heuristic"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=160)

    # Panel A: Out-of-fold scatter
    ax = axes[0]
    rar_colors = {0.0: "#64748b", 1.0: "#0284c7", 2.0: "#d97706", 3.0: "#dc2626"}
    rar_labels = {0.0: "Common", 1.0: "Uncommon", 2.0: "Rare", 3.0: "Mythic"}
    for r_num in [0.0, 1.0, 2.0, 3.0]:
        sub = main_df[main_df["rarity_num"] == r_num]
        ax.scatter(
            sub["oof_cal"], sub["gih_wr"],
            c=rar_colors[r_num], label=rar_labels[r_num],
            alpha=0.45, s=16, edgecolors="none"
        )
    ax.plot([0.44, 0.68], [0.44, 0.68], "k--", lw=1.2, label="Ideal y = x")
    r_val = pearsonr(yt, yp)[0]
    rho_val = spearmanr(yt, yp)[0]
    mae_val = mean_absolute_error(yt, yp)
    rmse_val = root_mean_squared_error(yt, yp)
    ax.set_xlim(0.47, 0.65)
    ax.set_ylim(0.43, 0.69)
    ax.set_xlabel("LOSO-CV Predicted GiH Win Rate")
    ax.set_ylabel("Realized 17lands GiH Win Rate")
    ax.set_title(f"A. Out-of-Fold Predictions (n=2,512 Main-Set Cards)\nr={r_val:.3f}, rho={rho_val:.3f}, MAE={mae_val:.4f}, RMSE={rmse_val:.4f}")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # Panel B: Per-Set Pearson r Comparison (Ensemble vs Heuristic)
    ax = axes[1]
    sets_order = ["TDM", "ECL", "TLA", "MSH", "TMT", "OM1", "FIN", "SOS", "EOE", "HOB"]
    r_ens, r_heu = [], []
    for s in sets_order:
        sub = main_df[main_df["set"] == s]
        r_ens.append(pearsonr(sub["gih_wr"], sub["oof_cal"])[0])
        r_heu.append(pearsonr(sub["gih_wr"], sub["oof_heuristic"])[0])
    x = np.arange(len(sets_order))
    w = 0.38
    ax.bar(x - w / 2, r_heu, width=w, color="#94a3b8", label="Rarity + CMC Heuristic")
    ax.bar(x + w / 2, r_ens, width=w, color="#2563eb", label="Stacked Multi-Task Ensemble")
    ax.set_xticks(x)
    ax.set_xticklabels(sets_order)
    ax.set_ylim(0, 0.70)
    ax.set_ylabel("Within-Set Pearson Correlation (r)")
    ax.set_title(f"B. Within-Set Correlation Across All 10 Sets\nMean r: {np.mean(r_ens):.3f} (Ensemble) vs. {np.mean(r_heu):.3f} (Heuristic)")
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel C: In-Universe vs Universes Beyond 17lands Baseline Win Rate
    ax = axes[2]
    all_sets = ["ECL", "TDM", "EOE", "SOS", "OM1", "FIN", "TLA", "MSH", "HOB", "TMT"]
    g_wrs = []
    bar_cols = []
    for s in all_sets:
        k = f"{s}_PremierDraft" if f"{s}_PremierDraft" in stats else f"{s}_PickTwoDraft"
        g_wrs.append(stats[k]["game_win_rate"] * 100.0)
        bar_cols.append("#0f766e" if s in in_univ else "#7c3aed")
    bars = ax.bar(all_sets, g_wrs, color=bar_cols, alpha=0.88)
    ax.axhline(55.24, color="#0f766e", ls="--", lw=1.4, label="In-Universe Mean (55.24%)")
    ax.axhline(56.16, color="#7c3aed", ls=":", lw=1.4, label="Universes Beyond Mean (56.16%)")
    ax.set_ylim(54.0, 57.3)
    ax.set_ylabel("17lands Overall Game Win Rate (%)")
    ax.set_title("C. Set Baseline Calibration: In-Universe vs. UB\nFRA is an In-Universe Sequel to SOS (55.38%)")
    ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    for p in (FIG_DIR / "loso_cv_evaluation.png", ARTIFACTS_DIR / "loso_cv_evaluation.png"):
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def make_fig2_fra_environment() -> None:
    fra_df = pd.read_csv(OUT_DIR / "fra_final_predictions.csv")
    hist_df = pd.read_csv(OUT_DIR / "historical_oof_predictions.csv")
    sos_df = hist_df[(hist_df["set"] == "SOS") & (hist_df["is_bonus_sheet"] == 0.0)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=160)

    # Panel A: Rarity boxplots (FRA Predicted vs SOS Realized)
    ax = axes[0]
    rarities = ["Common", "Uncommon", "Rare", "Mythic"]
    fra_main = fra_df[fra_df["set_code"] == "FRA"]
    fra_data = [fra_main[fra_main["rarity"] == r]["predicted_gih_wr"].to_numpy() * 100 for r in rarities]
    sos_data = [sos_df[sos_df["rarity_num"] == idx]["gih_wr"].to_numpy() * 100 for idx in range(4)]

    pos_fra = np.arange(4) * 2.2 - 0.35
    pos_sos = np.arange(4) * 2.2 + 0.35
    bp1 = ax.boxplot(fra_data, positions=pos_fra, widths=0.55, patch_artist=True, showfliers=True)
    bp2 = ax.boxplot(sos_data, positions=pos_sos, widths=0.55, patch_artist=True, showfliers=True)
    for box in bp1["boxes"]:
        box.set(facecolor="#2563eb", alpha=0.75)
    for box in bp2["boxes"]:
        box.set(facecolor="#94a3b8", alpha=0.65)
    ax.set_xticks(np.arange(4) * 2.2)
    ax.set_xticklabels(rarities)
    ax.set_ylabel("GiH Win Rate (%)")
    ax.set_title("A. GiH Win Rate by Rarity: FRA (Predicted) vs. SOS (Realized)")
    ax.plot([], [], color="#2563eb", lw=6, label="FRA Predicted (n=285)")
    ax.plot([], [], color="#94a3b8", lw=6, label="SOS Realized (n=269)")
    ax.legend(loc="lower right", fontsize=8.5)

    # Panel B: Mono-color & Allied College Common/Uncommon Strength in FRA
    ax = axes[1]
    cu = fra_main[fra_main["rarity"].isin(["Common", "Uncommon"]) & (~fra_main["type_line"].str.contains("Basic Land"))]
    groups = ["W", "U", "G", "B", "R", "WU", "GW", "UB", "BR", "RG"]
    means = []
    cols_bar = []
    palette = {
        "W": "#eab308", "U": "#0284c7", "G": "#16a34a", "B": "#475569", "R": "#dc2626",
        "WU": "#38bdf8", "GW": "#4ade80", "UB": "#6366f1", "BR": "#f43f5e", "RG": "#f97316"
    }
    for g in groups:
        sub = cu[cu["colors"] == g]
        means.append(sub["predicted_gih_wr"].mean() * 100.0)
        cols_bar.append(palette[g])
    ax.bar(groups, means, color=cols_bar, alpha=0.88)
    ax.axhline(cu["predicted_gih_wr"].mean() * 100.0, color="k", ls="--", lw=1.2, label="FRA C/U Mean (55.20%)")
    ax.set_ylim(53.0, 57.8)
    ax.set_ylabel("Mean C/U Predicted GiH Win Rate (%)")
    ax.set_title("B. FRA Color & Allied College C/U Depth\nWU Fatehold & GW Vigorbloom Lead Common/Uncommon Pool")
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel C:FRA Key Mechanics Comparison
    ax = axes[2]
    cat = json.loads((OUT_DIR / "card_catalog.json").read_text())
    mech_groups = {
        "Empower Jace\n(N >= 4)": [],
        "Empower Jace\n(N = 1..3)": [],
        "Prepare:\nPeer Review / Seed": [],
        "Prepare:\nOmit / Verse / Tether": [],
        "Removal\n(C/U Top Tier)": [],
        "Color Hosers &\nPW Tutors": [],
    }
    pred_map = fra_main.set_index("card_name")["predicted_gih_wr"].to_dict()
    for cname, c in cat["fra_main"].items():
        wr = pred_map[cname] * 100.0
        txt = (c.get("oracle_text") or "").lower()
        if "empower jace" in txt:
            if any(f"empower jace {k}" in txt for k in ["4", "5", "6", "7", "8", "x"]):
                mech_groups["Empower Jace\n(N >= 4)"].append(wr)
            else:
                mech_groups["Empower Jace\n(N = 1..3)"].append(wr)
        if c.get("layout") == "prepare":
            if "peer review" in cname.lower() or "seed suture" in cname.lower():
                mech_groups["Prepare:\nPeer Review / Seed"].append(wr)
            elif any(x in cname.lower() for x in ["omit variables", "vicious verse", "soul tether"]):
                mech_groups["Prepare:\nOmit / Verse / Tether"].append(wr)
        if cname in {"Last Gasp", "Memory Trap", "No Admittance", "Extended Absence", "Fulminous Forte", "Violent Echoes", "Multiply by Zero", "Wrath of the Bloodmane"}:
            mech_groups["Removal\n(C/U Top Tier)"].append(wr)
        if cname in {"Loyal Tutor", "Entrust the Spark", "Refute Destiny", "Precise Redaction", "Terminal Criticism", "Essence Burn", "Flourishing Grapple"}:
            mech_groups["Color Hosers &\nPW Tutors"].append(wr)

    labels_m = list(mech_groups.keys())
    vals_m = [np.mean(mech_groups[k]) for k in labels_m]
    c_m = ["#0284c7", "#38bdf8", "#16a34a", "#94a3b8", "#d97706", "#dc2626"]
    y_pos = np.arange(len(labels_m))
    ax.barh(y_pos, vals_m, color=c_m, alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_m, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(55.97, color="k", ls="--", lw=1.2, label="FRA Mean (55.97%)")
    ax.set_xlim(48.0, 60.0)
    ax.set_xlabel("Mean Predicted GiH Win Rate (%)")
    ax.set_title("C. FRA Mechanics & Functional Group Breakdown")
    ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    for p in (FIG_DIR / "fra_environment_and_mechanics.png", ARTIFACTS_DIR / "fra_environment_and_mechanics.png"):
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def make_fig3_top_bottom() -> None:
    fra_df = pd.read_csv(OUT_DIR / "fra_final_predictions.csv")
    fra_main = fra_df[fra_df["set_code"] == "FRA"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), dpi=160)

    # Panel A: Top 15 Main-Set Cards + Top 5 Commons/Uncommons
    top15 = fra_main.sort_values("predicted_gih_wr", ascending=False).head(15)
    ax = axes[0]
    y = np.arange(len(top15))
    ax.barh(y, top15["predicted_gih_wr"] * 100.0, color="#1d4ed8", alpha=0.85, label="Final Calibrated GiH WR")
    ax.scatter(top15["ml_ensemble_pred"] * 100.0, y, color="#f59e0b", zorder=5, s=42, label="Raw Tabular ML Pred")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.card_name} [{r.rarity[0]}]" for _, r in top15.iterrows()], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(54.0, 66.5)
    ax.set_xlabel("Predicted GiH Win Rate (%)")
    ax.set_title("A. Top 15 Predicted Reality Fracture (FRA) Bombs")
    ax.legend(loc="lower right", fontsize=8.5)

    # Panel B: Top 10 Commons/Uncommons vs Bottom 8 Traps
    top_cu = fra_main[fra_main["rarity"].isin(["Common", "Uncommon"])].sort_values("predicted_gih_wr", ascending=False).head(10)
    bot_traps = fra_main.sort_values("predicted_gih_wr", ascending=True).head(8)
    comb = pd.concat([top_cu, bot_traps], ignore_index=True)

    ax = axes[1]
    y2 = np.arange(len(comb))
    cols2 = ["#059669" if idx < 10 else "#dc2626" for idx in range(len(comb))]
    ax.barh(y2, comb["predicted_gih_wr"] * 100.0, color=cols2, alpha=0.85)
    ax.scatter(comb["ml_ensemble_pred"] * 100.0, y2, color="#f59e0b", zorder=5, s=42, label="Raw Tabular ML Pred")
    ax.axvline(55.97, color="k", ls="--", lw=1.1, label="FRA Set Mean (55.97%)")
    ax.set_yticks(y2)
    ax.set_yticklabels([f"{r.card_name.split(' // ')[0]} [{r.rarity[0]}]" for _, r in comb.iterrows()], fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlim(44.5, 60.5)
    ax.set_xlabel("Predicted GiH Win Rate (%)")
    ax.set_title("B. Top 10 FRA Commons/Uncommons (Green) vs. Top 8 Traps (Red)")
    ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    for p in (FIG_DIR / "fra_top_bottom_cards.png", ARTIFACTS_DIR / "fra_top_bottom_cards.png"):
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_fig1_loso_cv()
    make_fig2_fra_environment()
    make_fig3_top_bottom()
    print("Generated all 3 diagnostic figures in /opt/cursor/artifacts/ and /workspace/analysis/figures/")
