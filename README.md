# MTG win-rate eval

Offline dataset for predicting limited win rates of **Reality Fracture** (`FRA`) without any 17lands performance data for that set.

The card and rules context for the new set is included. Historical 17lands draft, game, and replay files are included for the ten previous limited sets. Reality Fracture rows that shipped inside 17lands' global card list are removed before the file is stored.

## Where the data lives

Canonical path: `/opt/mtg-data` (also linked from `/workspace/data` on boot).

| Path | Contents |
| --- | --- |
| `17lands/sets/<CODE>/` | Public 17lands draft, game, and replay gzip CSVs |
| `17lands/cards/cards.csv` | Arena card id map with Reality Fracture rows removed |
| `17lands/cards/abilities.csv` | Ability id map |
| `scryfall/sets/<code>.jsonl.gz` | English Scryfall card objects for that set, including `fra` |
| `scryfall/sets/fra-special-guests.jsonl` | Special Guests released with Reality Fracture (Scryfall, not 17lands) |
| `scryfall/default-cards.jsonl.gz` | Full Scryfall default-cards bulk used to build the extracts |
| `rules/MagicCompRules-20260925.txt` | Comprehensive Rules, 25 September 2026 |
| `rules/MagicCompRules-20260925.pdf` | Same document, PDF |
| `manifests/leakage_report.json` | Full scan showing zero Reality Fracture 17lands rows |

17lands public datasets are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Scryfall data and the Comprehensive Rules are used under the Wizards of the Coast Fan Content Policy. This project is not affiliated with Wizards of the Coast or 17lands.

## Historical sets

The ten limited sets immediately before Reality Fracture, skipping the Powered Cube (a cube, not a set). Marvel's Spider-Man (`SPM`) has no 17lands public dump; Arena limited for that window is Through the Omenpaths (`OM1`).

| Code | Set |
| --- | --- |
| HOB | The Hobbit |
| MSH | Marvel Super Heroes |
| SOS | Secrets of Strixhaven |
| TMT | Teenage Mutant Ninja Turtles |
| ECL | Lorwyn Eclipsed |
| TLA | Avatar: The Last Airbender |
| OM1 | Through the Omenpaths |
| EOE | Edge of Eternities |
| FIN | Final Fantasy |
| TDM | Tarkir: Dragonstorm |

Reality Fracture (`FRA`, released 2026-10-02) has Scryfall cards and the rules, and no 17lands files. Forbidden 17lands codes: `FRA`, `FRC`, `YFRA`, `PFRA`, `TFRA`, `TFRC`.

## Commands

```bash
/opt/mtg-data/venv/bin/python /opt/mtg-data/scripts/verify_dataset.py
/opt/mtg-data/venv/bin/python /opt/mtg-data/scripts/demo_winrate.py
```

`verify_dataset.py` scans every 17lands export's `expansion` column and exits non-zero if any Reality Fracture row is present. `demo_winrate.py` computes games-in-hand win rates for The Hobbit Premier Draft and prints sample Reality Fracture cards.

The 17lands global card list originally contained Reality Fracture rows (432 `FRA`, 16 `FRC`, 15 `TFRA`). Those rows are deleted before `cards.csv` is stored. A full scan of every kept 17lands file's `expansion` column is recorded in `manifests/leakage_report.json`.

Do not refresh 17lands from the network in this environment. A later public dump may include Reality Fracture win rates. `scripts/download_datasets.py` only knows the frozen allowlist in `manifests/17lands_allowlist.json` and still strips Reality Fracture card-list rows, but the snapshot is the source of truth. The dataset is not in git; agents are expected to boot from the snapshot that already contains `/opt/mtg-data`.
