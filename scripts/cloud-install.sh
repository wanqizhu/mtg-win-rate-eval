#!/usr/bin/env bash
# Idempotent, offline-safe bootstrap. Dataset files already live on the snapshot.
set -euo pipefail

if [[ ! -d /opt/mtg-data/17lands || ! -d /opt/mtg-data/scryfall || ! -d /opt/mtg-data/rules ]]; then
  echo "missing /opt/mtg-data dataset; this environment is snapshot-backed and does not download 17lands" >&2
  exit 1
fi

if ! /opt/mtg-data/venv/bin/python -c "import pandas, numpy" >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  if ! python3 -c "import venv" >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-venv python3-pip
  fi
  python3 -m venv /opt/mtg-data/venv
  req="/opt/mtg-data/requirements.txt"
  if [[ -f /workspace/requirements.txt ]]; then
    req="/workspace/requirements.txt"
  fi
  /opt/mtg-data/venv/bin/pip install -r "$req"
fi

/opt/mtg-data/venv/bin/python /opt/mtg-data/scripts/verify_dataset.py
ln -sfn /opt/mtg-data /workspace/data
echo "install ok"
