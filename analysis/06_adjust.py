"""Apply evidence-backed manual adjustments to calibrated model predictions.
Each adjustment has historical analogue justification. Magnitudes conservative."""
import pandas as pd
from pathlib import Path

ANALYSIS=Path("/agent/repos/mtg-win-rate-eval/analysis")
trk=pd.read_pickle(ANALYSIS/"trk_pred_v2.pkl")
trk=trk.set_index("name")

# name -> (new_value, reason)
ADJ={
 # --- clear jank / uncastable (wish/alt-win/9-mana analogues ~51-53%) ---
 "Peace in Our Time": (0.5150, "3W group-hug alt-win, no board; wish/no-body analogues 51-52%"),
 "The Holodeck": (0.5250, "7-mana wish/graveyard engine, no body; wish analogues 51.5%"),
 "Deep Space 9": (0.5350, "9-mana station-9 spacecraft; EOE high-station analogues 47-49%, 9-mana mean 53.9%"),
 "Infinite Diversity": (0.5350, "9-mana win-more sorcery; 9-mana non-cheating sorceries ~50-53%"),
 "Live Long and Prosper": (0.5450, "7-mana no-board lifegain/draw"),
 "Branching Timeline": (0.5450, "8-mana slow permanent-cheat"),
 "Crystalline Entity": (0.5650, "8-mana symmetrical wrath; keeps flyer-body premium over 9-mana mean"),
 "Cloak and Phaser": (0.5600, "9-mana discounted to ~5; 2 bodies+loot fair not premium"),
 "Starfleet Academy": (0.5550, "mythic land often just colorless; utility-land low end"),
 "Brazen Bluff": (0.5350, "opponent-choice counterspell, RR narrow"),
 "Stable Wormhole": (0.5450, "2G attack-trigger ramp, slow"),
 "Sickbay": (0.5400, "5-mana no-body lifegain/counters"),
 # --- 0-power bodies overvalued ---
 "Emergency Medical Hologram": (0.5450, "W 0/1; 0-power utility creatures 52-55%, 1-mana fragility"),
 "Data, New Life": (0.5400, "1U 0/1 needs draw triggers; fragile grower"),
 # --- shocks overvalued (historical shocks mean 54.2%, max 56.8%) ---
 "Blood Crypt": (0.5400, "shockland calibration to historical 52-55% band"),
 "Breeding Pool": (0.5450, "shockland calibration; Breeding historically top shock"),
 "Godless Shrine": (0.5400, "shockland calibration"),
 "Hallowed Fountain": (0.5380, "shockland calibration"),
 "Overgrown Tomb": (0.5420, "shockland calibration"),
 "Sacred Foundry": (0.5380, "shockland calibration; historically lowest shock"),
 "Steam Vents": (0.5410, "shockland calibration"),
 "Stomping Ground": (0.5440, "shockland calibration"),
 "Temple Garden": (0.5380, "shockland calibration"),
 "Watery Grave": (0.5430, "shockland calibration"),
 # --- narrow engines slightly overvalued ---
 "Collation Trimatrix": (0.5600, "B 1/1 tap-3-artifacts draw engine; slow"),
 "Gwyndala, the Progeny": (0.5750, "1U equipment theme, weak support"),
 # --- undervalued premium removal (Banishing Light analogues 58.1%) ---
 "Malfunctioning Holodeck": (0.5850, "3-mana Banishing Light + Ward 2, colorless; analogues 58.1%"),
 "Eject the Warp Core": (0.5720, "1B premium flexible removal (-2/-2 to -6/-6)"),
 "He's Dead, Jim": (0.5750, "unconditional kill + surveil at 4B fair premium"),
 "Planetary Bombardment": (0.5730, "R 1-mana 2-4 damage efficient"),
 "Command Decision": (0.5710, "1W modal kill-big-or-cycle versatility"),
 "Common Goal": (0.5650, "2G 1-2 target fight; fight mean 56.1%"),
 # --- undervalued bodies/value ---
 "Dominion Assault Force": (0.5700, "4G 4/4 vigilance + modal; 4-mana mean 56.2% with vigilance premium"),
 "Cha'DIch Investigator": (0.5730, "5B 4/4 + 2/1 haste = 6 power across bodies"),
 "Horta": (0.5680, "6RR 7/6 trample + landcycling floor"),
 "Cold-Blooded Crew": (0.5700, "6GG immediate 6/6 spread + evasion + cycling floor"),
 "Photosynthetic Physician": (0.5650, "5G 5/5 reach + recursion"),
 "Dahar Master": (0.5650, "4R 4/3 + shock outlet"),
 "Plasma Cascade": (0.5650, "3R 3-damage + loot flexibility"),
 # --- spacecraft recalibration to EOE analogues ---
 "Class-F Shuttlecraft": (0.5580, "3-mana rock + 2/4 flyer; EOE 3-mana spacecraft 56.7%"),
 "U.S.S. Voyager, Lone Wanderer": (0.5650, "4G double-land ramp + body; strong in landfall"),
 "Suurok-Class Science Vessel": (0.5780, "3U cantrip spacecraft station-5; EOE cantrip ships 57-60%"),
 "Enterprise NX-01": (0.5670, "4U removal ETB + evasive body"),
 "U.S.S. Cerritos, Following Up": (0.5650, "5-mana 3 bodies + flyer"),
 "Warp Core": (0.5650, "2-mana fixing + station accelerator"),
 "Ravenous Cosmozoan": (0.5660, "cost-reduced 3/3 flying haste often 3-4 mana"),
 # --- basics equalized to historical basics mean 56.0% ---
 "Plains": (0.5620, "basics equalized; historical basics mean 56.02%"),
 "Island": (0.5620, "basics equalized"),
 "Swamp": (0.5620, "basics equalized"),
 "Mountain": (0.5620, "basics equalized"),
 "Forest": (0.5620, "basics equalized"),
}
applied=[]
for name,(val,reason) in ADJ.items():
    if name not in trk.index:
        print("MISSING:",name); continue
    old=float(trk.loc[name,"pred_calibrated"])
    trk.loc[name,"pred_final"]=val
    applied.append((name,old,val,reason))
# default: pred_final = pred_calibrated
if "pred_final" not in trk.columns:
    trk["pred_final"]=trk["pred_calibrated"]
else:
    trk["pred_final"]=trk["pred_final"].fillna(trk["pred_calibrated"])
# report
print(f"adjusted {len(applied)} cards")
import pandas as pd
for name,old,val,reason in sorted(applied,key=lambda x: x[2]-x[1]):
    print(f"{val-old:+.4f} {name}: {old:.4f} -> {val:.4f} | {reason}")
trk["adjustment"]=trk["pred_final"]-trk["pred_calibrated"]
print(trk["pred_final"].describe())
trk.to_pickle(ANALYSIS/"trk_pred_final.pkl")
# final table
final=trk.reset_index()[["name","rarity","mana_cost","type_line","pred_final"]].rename(columns={"name":"card","pred_final":"predicted_gih_wr"})
final=final.sort_values("predicted_gih_wr",ascending=False)
final.to_csv(ANALYSIS/"trk_gih_predictions.csv",index=False)
print("wrote", ANALYSIS/"trk_pred_final.pkl", "and trk_gih_predictions.csv", len(final))
