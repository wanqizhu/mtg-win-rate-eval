# MTG win-rate eval

Offline dataset for predicting limited win rates of **Star Trek** (`TRK`) without any 17lands performance data for that set.

The card and rules context for the new set is included. Historical 17lands draft, game, and replay files are included for the ten previous limited sets. Star Trek rows that ship inside 17lands' global card list are removed before the file is stored.

## Where the data lives

Canonical path: `/opt/mtg-data` (also linked from `/workspace/data` on boot).

| Path | Contents |
| --- | --- |
| `17lands/sets/<CODE>/` | Public 17lands draft, game, and replay gzip CSVs |
| `17lands/cards/cards.csv` | Arena card id map with Star Trek rows removed |
| `17lands/cards/abilities.csv` | Ability id map |
| `scryfall/sets/<code>.jsonl.gz` | English Scryfall card objects for that set, including `trk` and historical `fra` |
| `scryfall/sets/sds.jsonl.gz` | Stardates, the Star Trek masterpiece sheet (Scryfall, not 17lands) |
| `scryfall/sets/trk-special-guests.jsonl` | Special Guests released with Star Trek, when Scryfall has any |
| `scryfall/default-cards.jsonl.gz` | Full Scryfall default-cards bulk used to build the extracts |
| `rules/MagicCompRules-20260925.txt` | Comprehensive Rules, 25 September 2026 |
| `rules/MagicCompRules-20260925.pdf` | Same document, PDF |
| `manifests/leakage_report.json` | Full scan showing zero Star Trek 17lands rows |

17lands public datasets are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Scryfall data and the Comprehensive Rules are used under the Wizards of the Coast Fan Content Policy. This project is not affiliated with Wizards of the Coast or 17lands.

## Historical sets

The ten limited sets immediately before Star Trek. Marvel's Spider-Man (`SPM`) has no 17lands public dump; Arena limited for that window is Through the Omenpaths (`OM1`). Tarkir: Dragonstorm (`TDM`) aged out of the ten-set window when Reality Fracture became historical.

| Code | Set |
| --- | --- |
| FRA | Reality Fracture |
| HOB | The Hobbit |
| MSH | Marvel Super Heroes |
| SOS | Secrets of Strixhaven |
| TMT | Teenage Mutant Ninja Turtles |
| ECL | Lorwyn Eclipsed |
| TLA | Avatar: The Last Airbender |
| OM1 | Through the Omenpaths |
| EOE | Edge of Eternities |
| FIN | Final Fantasy |

Reality Fracture (`FRA`, released 2026-10-02) now has public 17lands draft and game files, dated 2026-10-19. That dump has no replay files. Lorwyn Eclipsed replay files are included; the previous allowlist had stopped before those objects.

## Star Trek

Star Trek (`TRK`, released 2026-11-13) has Scryfall cards and the rules, and no 17lands files. A HEAD of the Premier Draft game file returns a non-200 status, and the body is not stored. Scryfall's set object lists 586 printings. Stardates (`SDS`, 30 cards) is the masterpiece sheet released with the set and is stored as card text only.

Related Scryfall products, also excluded from 17lands storage: Star Trek Commander (`TRC`), Star Trek Tokens (`TTRK`), and Star Trek Commander Tokens (`TTRC`). `YTRK` and `PTRK` are reserved even though Scryfall has no such sets yet.

Forbidden 17lands codes: `TRK`, `TRC`, `YTRK`, `PTRK`, `TTRK`, `TTRC`, `SDS`.

Scryfall has no Special Guests (`SPG`) with release date 2026-11-13. `scripts/download_datasets.py` searches `set:spg date=2026-11-13` and writes `scryfall/sets/trk-special-guests.jsonl` only when that search returns cards.

## Commands

```bash
/opt/mtg-data/venv/bin/python /opt/mtg-data/scripts/verify_dataset.py
/opt/mtg-data/venv/bin/python /opt/mtg-data/scripts/demo_winrate.py
```

`verify_dataset.py` scans every 17lands export's `expansion` column and exits non-zero if any Star Trek row is present. `demo_winrate.py` computes games-in-hand win rates for Reality Fracture Premier Draft and prints sample Star Trek cards.

Do not refresh 17lands from the network in this environment. A later public dump may include Star Trek win rates. `scripts/download_datasets.py` only knows the frozen allowlist in `manifests/17lands_allowlist.json` and still strips Star Trek card-list rows, but the snapshot is the source of truth. The dataset is not in git; agents are expected to boot from the snapshot that already contains `/opt/mtg-data`.
