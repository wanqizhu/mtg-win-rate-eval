#!/usr/bin/env python3
"""Prove the offline dataset has historical 17lands data and zero Star Trek 17lands data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/opt/mtg-data")
FORBIDDEN = frozenset({"TRK", "TRC", "YTRK", "PTRK", "TTRK", "TTRC", "SDS"})
HISTORICAL = ["FRA", "HOB", "MSH", "SOS", "TMT", "ECL", "TLA", "OM1", "EOE", "FIN"]
TARGET_SET = "trk"


def die(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def expansion_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    with open_text(path) as handle:
        header_line = handle.readline()
        if not header_line:
            die(f"empty file {path}")
        header = next(csv.reader([header_line]))
        if "expansion" not in header:
            return counts
        index = header.index("expansion")
        if index != 0:
            pass
        else:
            for line in handle:
                if not line.strip():
                    continue
                if line[0] == '"':
                    end = line.find('"', 1)
                    value = line[1:end]
                else:
                    comma = line.find(",")
                    value = line[:comma] if comma >= 0 else line.strip()
                counts[value] += 1
            return counts
    with open_text(path) as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if len(row) <= index:
                continue
            counts[row[index]] += 1
    return counts


def path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in path.relative_to(ROOT / "17lands").parts:
        for token in part.replace(".", " ").replace("_", " ").replace("-", " ").split():
            tokens.add(token.upper())
    return tokens


def iter_17lands_files() -> list[Path]:
    base = ROOT / "17lands"
    if not base.is_dir():
        die("missing /opt/mtg-data/17lands")
    return [path for path in base.rglob("*") if path.is_file()]


def check_filenames(files: list[Path]) -> None:
    for path in files:
        leaked = path_tokens(path) & FORBIDDEN
        if leaked:
            die(f"17lands path contains forbidden set code {sorted(leaked)}: {path}")


def check_cards_csv() -> dict:
    path = ROOT / "17lands" / "cards" / "cards.csv"
    if not path.exists():
        die("missing filtered 17lands cards.csv")
    counts = expansion_counts(path)
    leaked = {code: counts[code] for code in FORBIDDEN if counts[code]}
    if leaked:
        die(f"17lands card list still contains Star Trek rows: {leaked}")
    return {"rows": sum(counts.values()), "expansions": len(counts), "forbidden_rows": 0}


def check_historical_files(files: list[Path], full: bool) -> dict:
    by_set: dict[str, int] = {code: 0 for code in HISTORICAL}
    scanned = []
    forbidden_rows = {code: 0 for code in sorted(FORBIDDEN)}
    for path in files:
        name = path.name
        for code in HISTORICAL:
            if f".{code}." in name or f"/{code}/" in str(path):
                by_set[code] += 1
        if not full:
            continue
        if path.suffix == ".py":
            continue
        if path.suffix not in {".gz", ".csv"}:
            continue
        if path.name in {"abilities.csv", "dungeon.csv"}:
            continue
        counts = expansion_counts(path)
        if not counts and path.name != "cards.csv":
            if path.parent.name == "sets" or "/sets/" in str(path):
                die(f"no expansion column in {path}")
        for code in FORBIDDEN:
            forbidden_rows[code] += counts[code]
        scanned.append(
            {
                "path": str(path.relative_to(ROOT)),
                "rows": sum(counts.values()),
                "expansions": sorted(counts),
            }
        )
    missing = [code for code, count in by_set.items() if count == 0]
    if missing:
        die(f"missing 17lands files for {missing}")
    leaked = {code: count for code, count in forbidden_rows.items() if count}
    if leaked:
        die(f"17lands rows with Star Trek expansions: {leaked}")
    return {
        "files_per_set": by_set,
        "scanned_files": len(scanned),
        "forbidden_rows": forbidden_rows,
    }


def count_jsonl_gz(path: Path, expected_set: str) -> tuple[int, dict | None]:
    count = 0
    sample = None
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            card = json.loads(line)
            if card.get("set") != expected_set:
                die(f"non-{expected_set} card in {path.name}: {card.get('set')} {card.get('name')}")
            count += 1
            if sample is None:
                sample = {
                    "name": card.get("name"),
                    "mana_cost": card.get("mana_cost"),
                    "type_line": card.get("type_line"),
                }
    return count, sample


def check_scryfall() -> dict:
    trk = ROOT / "scryfall" / "sets" / "trk.jsonl.gz"
    if not trk.exists():
        die("missing Scryfall Star Trek extract")
    count, sample = count_jsonl_gz(trk, TARGET_SET)
    if count < 250:
        die(f"Scryfall TRK extract too small: {count}")
    sds_path = ROOT / "scryfall" / "sets" / "sds.jsonl.gz"
    if not sds_path.exists():
        die("missing Scryfall Stardates extract")
    sds_count, _sds_sample = count_jsonl_gz(sds_path, "sds")
    if sds_count < 20:
        die(f"Scryfall Stardates extract too small: {sds_count}")
    historical = {}
    for code in HISTORICAL:
        path = ROOT / "scryfall" / "sets" / f"{code.lower()}.jsonl.gz"
        if not path.exists():
            die(f"missing Scryfall extract for {code}")
        with gzip.open(path, "rt") as handle:
            historical[code] = sum(1 for line in handle if line.strip())
        if historical[code] < 50:
            die(f"Scryfall extract for {code} too small")
    guests = ROOT / "scryfall" / "sets" / "trk-special-guests.jsonl"
    guest_count = 0
    if guests.exists():
        guest_count = sum(1 for line in guests.read_text().splitlines() if line.strip())
    return {
        "trk_cards": count,
        "trk_sample": sample,
        "stardates": sds_count,
        "historical_counts": historical,
        "special_guests": guest_count,
    }


def check_rules() -> dict:
    text_path = ROOT / "rules" / "MagicCompRules-20260925.txt"
    pdf_path = ROOT / "rules" / "MagicCompRules-20260925.pdf"
    if not text_path.exists() or not pdf_path.exists():
        die("missing comprehensive rules")
    head = text_path.read_text(errors="replace")[:2500]
    if "Magic" not in head:
        die("comprehensive rules text is unreadable")
    return {
        "txt_bytes": text_path.stat().st_size,
        "pdf_bytes": pdf_path.stat().st_size,
        "head": head.splitlines()[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Skip the full gzip expansion scan")
    args = parser.parse_args()
    files = iter_17lands_files()
    check_filenames(files)
    cards = check_cards_csv()
    historical = check_historical_files(files, full=not args.fast)
    scryfall = check_scryfall()
    rules = check_rules()
    report = {
        "ok": True,
        "mode": "fast" if args.fast else "full",
        "target_set": "TRK",
        "seventeenlands_file_count": len(files),
        "star_trek_seventeenlands_files": 0,
        "star_trek_seventeenlands_rows": 0,
        "cards": cards,
        "historical": {key: value for key, value in historical.items() if key != "scanned_files"}
        | {"scanned_files": historical["scanned_files"]},
        "scryfall": scryfall,
        "rules": {key: value for key, value in rules.items() if key != "head"},
        "rules_head": rules["head"],
    }
    out = ROOT / "manifests" / "leakage_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.fast:
        if not out.exists():
            die("full leakage report is missing; run verify_dataset.py without --fast")
        previous = json.loads(out.read_text())
        if previous.get("mode") != "full" or not previous.get("ok"):
            die("stored leakage report is not a successful full scan")
        if previous.get("target_set") != "TRK":
            die("stored leakage report is not for Star Trek")
        if previous.get("star_trek_seventeenlands_rows") != 0:
            die("stored leakage report shows Star Trek 17lands rows")
        if previous.get("star_trek_seventeenlands_files") != 0:
            die("stored leakage report shows Star Trek 17lands files")
    else:
        out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "rules_head"}, indent=2))
    print("verify ok")


if __name__ == "__main__":
    main()
