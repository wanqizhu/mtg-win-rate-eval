#!/usr/bin/env bash
# Recreate the workspace link on every boot and refuse a Reality Fracture 17lands leak.
set -euo pipefail

if [[ ! -d /opt/mtg-data/17lands ]]; then
  echo "missing /opt/mtg-data/17lands" >&2
  exit 1
fi

ln -sfn /opt/mtg-data /workspace/data
python3 /opt/mtg-data/scripts/verify_dataset.py --fast
echo "mtg data ready at /opt/mtg-data and /workspace/data"
