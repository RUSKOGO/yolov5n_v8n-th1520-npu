#!/usr/bin/env bash
# One-command demo: YOLOv5n on TH1520 (run on the board).
set -euo pipefail
cd "$(dirname "$0")"

echo "YOLOv5n NPU demo — $(pwd)"
python3 -m pip install -q -r requirements.txt
exec python3 check.py --source auto "$@"
