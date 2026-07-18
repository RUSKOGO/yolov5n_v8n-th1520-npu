#!/usr/bin/env bash
# Pack board runtime folder (copy to LicheePi).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist/check_run8}"
mkdir -p "$OUT"

need() { [[ -f "$1" ]] || { echo "missing: $1"; exit 1; }; }

need "$ROOT/libyolov8n.so"
PARAMS="$ROOT/hhb_out/model.params"
[[ -f "$PARAMS" ]] || PARAMS="$ROOT/model.params"
need "$PARAMS"
need "$ROOT/python/check_v8.py"
need "$ROOT/python/classes.txt"

cp -f "$ROOT/libyolov8n.so" "$OUT/"
cp -f "$PARAMS" "$OUT/model.params"
cp -f "$ROOT/python/check_v8.py" "$OUT/"
cp -f "$ROOT/python/classes.txt" "$OUT/"
cp -f "$ROOT/python/requirements.txt" "$OUT/"

echo "Board bundle → $OUT"
ls -lh "$OUT"
echo
echo "scp -r $OUT sipeed@<board-ip>:~/ruskogo/"
echo "On board: cd ~/ruskogo/check_run8 && python3 check_v8.py --source auto"
