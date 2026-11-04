#!/usr/bin/env python3
"""Download the frozen offline dataset for win-rate prediction.

This script only fetches URLs in manifests/17lands_allowlist.json plus Scryfall
and the comprehensive rules. It refuses Star Trek 17lands files, and it strips
Star Trek rows out of the global 17lands card list. Reality Fracture is historical.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/opt/mtg-data")
REPO = Path("/workspace")
ALLOWLIST = REPO / "manifests" / "17lands_allowlist.json"
UA = "mtg-win-rate-eval/1.0 (offline limited win-rate dataset)"

# Star Trek and its related 17lands set codes. Never store these game files.
# YTRK and PTRK are not Scryfall sets yet; they are reserved so a later card
# list cannot keep Alchemy or promo rows.
FORBIDDEN_17LANDS = frozenset({"TRK", "TRC", "YTRK", "PTRK", "TTRK", "TTRC", "SDS"})

HISTORICAL_SETS = [
    "FRA",
    "HOB",
    "MSH",
    "SOS",
    "TMT",
    "ECL",
    "TLA",
    "OM1",
    "EOE",
    "FIN",
]
TARGET_SET = "TRK"
TARGET_NAME = "Star Trek"
# Stardates is the Star Trek masterpiece sheet. Card text only; no 17lands files.
SIDE_SETS = ["SDS"]
SET_OBJECT_CODES = [code.lower() for code in HISTORICAL_SETS] + [
    "trk",
    "trc",
    "sds",
    "ttrk",
    "ttrc",
]
SPECIAL_GUEST_QUERY = "set:spg date=2026-11-13"
SPECIAL_GUEST_RELEASED = "2026-11-13"
OBSOLETE = [
    ROOT / "17lands" / "sets" / "TDM",
    ROOT / "scryfall" / "sets" / "tdm.jsonl.gz",
    ROOT / "scryfall" / "sets" / "fra-special-guests.jsonl",
]


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def assert_url_allowed(url: str, expansion: str | None = None) -> None:
    url = url.strip()
    if expansion and expansion.upper() in FORBIDDEN_17LANDS:
        die(f"refusing forbidden expansion {expansion}: {url}")
    upper = url.upper()
    for code in FORBIDDEN_17LANDS:
        if f".{code}." in upper or f"/{code}." in upper or f"_{code}." in upper:
            die(f"refusing forbidden 17lands URL ({code}): {url}")
    if "STAR" in upper and "TREK" in upper and "17LANDS" in upper:
        die(f"refusing star trek 17lands URL: {url}")


def curl(url: str, dest: Path) -> None:
    url = url.strip()
    assert_url_allowed(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    cmd = [
        "curl",
        "-fL",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "-A",
        UA,
        "-o",
        str(partial),
        url,
    ]
    subprocess.run(cmd, check=True)
    partial.replace(dest)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_head(url: str) -> tuple[int, int | None]:
    cmd = [
        "curl",
        "-sI",
        "-A",
        UA,
        "--max-time",
        "60",
        "-w",
        "\nHTTPCODE:%{http_code}",
        url,
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    text = completed.stdout or ""
    status = 0
    if "HTTPCODE:" in text:
        try:
            status = int(text.strip().split("HTTPCODE:")[-1].strip() or "0")
        except ValueError:
            status = 0
    length = None
    for line in text.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                length = int(line.split(":", 1)[1].strip())
            except ValueError:
                length = None
    return status, length


def download_one(item: dict) -> dict:
    expansion = item["expansion"]
    url = item["url"].strip()
    assert_url_allowed(url, expansion)
    filename = url.rstrip("/").rsplit("/", 1)[-1]
    dest = ROOT / "17lands" / "sets" / expansion / filename
    status, length = http_head(url)
    if status != 200:
        die(f"17lands file missing ({status}): {url}")
    if dest.exists() and dest.stat().st_size > 0 and length is not None and dest.stat().st_size == length:
        if filename.endswith(".gz"):
            subprocess.run(["gzip", "-t", str(dest)], check=True)
        print(f"skip {dest.relative_to(ROOT)}", flush=True)
    else:
        print(f"get  {expansion} {item['format']} {item['kind']}", flush=True)
        curl(url, dest)
        if filename.endswith(".gz"):
            subprocess.run(["gzip", "-t", str(dest)], check=True)
        if length is not None and dest.stat().st_size != length:
            die(f"size mismatch for {dest}: got {dest.stat().st_size}, expected {length}")
    return {
        "expansion": expansion,
        "format": item["format"],
        "kind": item["kind"],
        "updated": item.get("updated"),
        "url": url,
        "path": str(dest.relative_to(ROOT)),
        "bytes": dest.stat().st_size,
        "sha256": sha256(dest),
    }


def filter_card_list(src: Path, dest: Path) -> dict:
    import csv

    dest.parent.mkdir(parents=True, exist_ok=True)
    removed: dict[str, int] = {code: 0 for code in sorted(FORBIDDEN_17LANDS)}
    kept = 0
    with src.open(newline="") as infile, dest.open("w", newline="") as outfile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames or "expansion" not in reader.fieldnames:
            die(f"{src} has no expansion column")
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            expansion = (row.get("expansion") or "").strip()
            if expansion in FORBIDDEN_17LANDS:
                removed[expansion] += 1
                continue
            writer.writerow(row)
            kept += 1
    src.unlink()
    return {"kept_rows": kept, "removed_rows": removed}


def download_helpers(helpers: list[dict]) -> list[dict]:
    saved = []
    for helper in helpers:
        url = helper["url"]
        assert_url_allowed(url)
        filename = url.rstrip("/").rsplit("/", 1)[-1]
        if helper["name"] == "Cards":
            with tempfile.TemporaryDirectory(prefix="cards-raw-") as tmp:
                raw = Path(tmp) / "cards.csv"
                print("get  17lands cards.csv (will strip Star Trek rows)", flush=True)
                curl(url, raw)
                dest = ROOT / "17lands" / "cards" / "cards.csv"
                filtered = filter_card_list(raw, dest)
            saved.append(
                {
                    "name": helper["name"],
                    "url": url,
                    "path": str(dest.relative_to(ROOT)),
                    "bytes": dest.stat().st_size,
                    "sha256": sha256(dest),
                    "filtered": filtered,
                }
            )
            continue
        dest = ROOT / "17lands" / "cards" / filename
        if filename.endswith(".py") or filename == "dungeon.csv":
            dest = ROOT / "17lands" / "helper" / filename
        print(f"get  helper {helper['name']}", flush=True)
        curl(url, dest)
        saved.append(
            {
                "name": helper["name"],
                "url": url,
                "path": str(dest.relative_to(ROOT)),
                "bytes": dest.stat().st_size,
                "sha256": sha256(dest),
            }
        )
    return saved


def scryfall_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def download_scryfall() -> dict:
    bulk = scryfall_json("https://api.scryfall.com/bulk-data/default_cards")
    url = bulk.get("jsonl_download_uri") or bulk.get("download_uri")
    if not url:
        die("scryfall bulk data has no download URI")
    dest = ROOT / "scryfall" / "default-cards.jsonl.gz"
    print(f"get  scryfall default_cards {bulk.get('updated_at')}", flush=True)
    curl(url, dest)
    subprocess.run(["gzip", "-t", str(dest)], check=True)

    wanted = {code.lower() for code in HISTORICAL_SETS + [TARGET_SET] + SIDE_SETS}
    counts = {code: 0 for code in sorted(wanted)}
    handles = {}
    sets_dir = ROOT / "scryfall" / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    try:
        for code in wanted:
            handles[code] = gzip.open(sets_dir / f"{code}.jsonl.gz", "wt")
        with gzip.open(dest, "rt") as infile:
            for line in infile:
                if not line.strip():
                    continue
                card = json.loads(line)
                code = (card.get("set") or "").lower()
                if code in handles:
                    handles[code].write(line if line.endswith("\n") else line + "\n")
                    counts[code] += 1
    finally:
        for handle in handles.values():
            handle.close()

    guests = download_special_guests()
    set_objects = []
    for code in SET_OBJECT_CODES:
        set_objects.append(scryfall_json(f"https://api.scryfall.com/sets/{code}"))
        time.sleep(0.1)
    objects_path = ROOT / "scryfall" / "set_objects.json"
    objects_path.write_text(json.dumps(set_objects, indent=2) + "\n")
    return {
        "bulk_updated_at": bulk.get("updated_at"),
        "bulk_url": url,
        "bulk_path": str(dest.relative_to(ROOT)),
        "bulk_bytes": dest.stat().st_size,
        "bulk_sha256": sha256(dest),
        "per_set_counts": counts,
        "special_guests": guests,
    }


def download_special_guests() -> dict:
    """Play Boosters may include Special Guests released with Star Trek."""
    cards = []
    url = "https://api.scryfall.com/cards/search?q=" + urllib.parse.quote(SPECIAL_GUEST_QUERY)
    try:
        while url:
            page = scryfall_json(url)
            cards.extend(page.get("data") or [])
            url = page.get("next_page")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        cards = []
    cards = [
        card
        for card in cards
        if card.get("set") == "spg" and card.get("released_at") == SPECIAL_GUEST_RELEASED
    ]
    dest = ROOT / "scryfall" / "sets" / "trk-special-guests.jsonl"
    if cards:
        with dest.open("w") as handle:
            for card in cards:
                if card.get("set") != "spg":
                    die(f"unexpected special guest set {card.get('set')}")
                handle.write(json.dumps(card) + "\n")
        path = str(dest.relative_to(ROOT))
    else:
        if dest.exists():
            dest.unlink()
        path = None
    return {
        "query": SPECIAL_GUEST_QUERY,
        "released_at": SPECIAL_GUEST_RELEASED,
        "count": len(cards),
        "names": [card["name"] for card in cards],
        "path": path,
    }


def download_rules() -> dict:
    files = {
        "txt": "https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.txt",
        "pdf": "https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.pdf",
    }
    alt = {
        "txt": "https://media.wizards.com/2026/downloads/MagicCompRules 20260925.txt",
        "pdf": "https://media.wizards.com/2026/downloads/MagicCompRules 20260925.pdf",
    }
    saved = {}
    for kind, url in files.items():
        dest = ROOT / "rules" / f"MagicCompRules-20260925.{kind}"
        print(f"get  comprehensive rules {kind}", flush=True)
        try:
            curl(url, dest)
        except subprocess.CalledProcessError:
            curl(alt[kind], dest)
            url = alt[kind]
        if dest.stat().st_size < 1000:
            die(f"rules file too small: {dest}")
        saved[kind] = {
            "url": url,
            "path": str(dest.relative_to(ROOT)),
            "bytes": dest.stat().st_size,
            "sha256": sha256(dest),
        }
    text = (ROOT / "rules" / "MagicCompRules-20260925.txt").read_text(errors="replace")[:4000]
    if "Comprehensive Rules" not in text and "Magic: The Gathering" not in text:
        die("comprehensive rules text does not look like the rules document")
    return saved


def head_status(url: str) -> int:
    status, _length = http_head(url)
    return status


def remove_obsolete() -> list[str]:
    removed = []
    for path in OBSOLETE:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path))
        print(f"drop {path}", flush=True)
    return removed


def main() -> None:
    overlap = FORBIDDEN_17LANDS.intersection(HISTORICAL_SETS)
    if overlap:
        die(f"historical sets overlap forbidden codes: {sorted(overlap)}")
    if TARGET_SET in HISTORICAL_SETS:
        die("target set is listed as historical")

    allow = json.loads(ALLOWLIST.read_text())
    if allow["sets"] != HISTORICAL_SETS:
        die("allowlist set list does not match the downloader")
    for item in allow["files"]:
        assert_url_allowed(item["url"], item["expansion"])
        if item["expansion"] not in HISTORICAL_SETS:
            die(f"unexpected expansion in allowlist: {item['expansion']}")

    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "manifests").mkdir(exist_ok=True)

    print(f"downloading {len(allow['files'])} 17lands files", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download_one, item) for item in allow["files"]]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (row["expansion"], row["format"], row["kind"]))

    helpers = download_helpers(allow["helpers"])
    scryfall = download_scryfall()
    rules = download_rules()
    dropped = remove_obsolete()

    probe_url = (
        "https://17lands-public.s3.amazonaws.com/analysis_data/game_data/"
        "game_data_public.TRK.PremierDraft.csv.gz"
    )
    trk_probe = head_status(probe_url)
    if trk_probe == 200:
        die(f"Star Trek 17lands premier draft exists upstream and was not saved: {probe_url}")
    report = {
        "historical_sets": HISTORICAL_SETS,
        "target_set": {
            "code": TARGET_SET,
            "name": TARGET_NAME,
            "released_at": "2026-11-13",
            "seventeenlands_files": 0,
            "upstream_premier_game_http_status": trk_probe,
            "note": "HTTP status is metadata from a HEAD request. The body was not downloaded.",
        },
        "seventeenlands_files": results,
        "helpers": helpers,
        "scryfall": scryfall,
        "rules": rules,
        "removed_obsolete": dropped,
    }
    (ROOT / "manifests" / "download_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("download complete", flush=True)
    trk_cards = scryfall["per_set_counts"].get("trk", 0)
    if trk_cards < 250:
        die(f"expected a full Star Trek Scryfall extract, got {trk_cards}")
    sds_cards = scryfall["per_set_counts"].get("sds", 0)
    if sds_cards < 20:
        die(f"expected Stardates cards in the Scryfall extract, got {sds_cards}")
    for code in [item.lower() for item in HISTORICAL_SETS]:
        if scryfall["per_set_counts"].get(code, 0) < 50:
            die(f"scryfall extract for {code} looks empty")


if __name__ == "__main__":
    main()
