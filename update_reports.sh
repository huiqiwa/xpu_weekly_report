#!/bin/bash
# Merge two report directories: newer report data overwrites older report data.
# The newer/older order is determined by the timestamp in directory names automatically.
# Usage: bash update_reports.sh <report_dir_1> <report_dir_2>
set -e

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <report_dir_1> <report_dir_2>"
  echo "  Merge two reports into the older one. Newer timestamps overwrite older."
  exit 1
fi

DIR1="$(realpath "$1")"
DIR2="$(realpath "$2")"

if [[ ! -d "$DIR1" ]]; then
  echo "[ERROR] Report dir not found: $DIR1"
  exit 1
fi
if [[ ! -d "$DIR2" ]]; then
  echo "[ERROR] Report dir not found: $DIR2"
  exit 1
fi

# Extract timestamps from directory names (e.g. reports_2026-04-20-10-59-57)
TS1=$(basename "$DIR1" | grep -oE '[0-9]{4}(-[0-9]{2}){5}' | tr -d '-')
TS2=$(basename "$DIR2" | grep -oE '[0-9]{4}(-[0-9]{2}){5}' | tr -d '-')

if [[ -z "$TS1" || -z "$TS2" ]]; then
  echo "[ERROR] Cannot extract timestamp from directory names."
  echo "  Expected format: reports_YYYY-MM-DD-HH-MM-SS"
  exit 1
fi

if [[ "$TS1" -le "$TS2" ]]; then
  OLD_DIR="$DIR1"
  NEW_DIR="$DIR2"
else
  OLD_DIR="$DIR2"
  NEW_DIR="$DIR1"
fi

echo "[INFO] Older report: $OLD_DIR"
echo "[INFO] Newer report: $NEW_DIR"
echo "[INFO] Merging into: $OLD_DIR"

echo "[INFO] Syncing csv, jsonl and txt files..."
rsync -a \
  --include='*/' \
  --include='*.csv' \
  --include='*.jsonl' \
  --include='*.txt' \
  --exclude='*' \
  "$NEW_DIR/" "$OLD_DIR/"

# Summary
echo ""
echo "=== Merge complete: $OLD_DIR ==="
echo "Synced files:"
find "$OLD_DIR" -type f \( -name '*.csv' -o -name '*.txt' -o -name '*.jsonl' \) -printf '  %P\n' 2>/dev/null | sort
