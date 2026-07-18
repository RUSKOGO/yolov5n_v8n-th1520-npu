#!/usr/bin/env bash
# One-command demo: YOLOv8 PPE on TH1520 (run on the board).
set -euo pipefail
cd "$(dirname "$0")"

echo "YOLOv8 NPU demo — $(pwd)"
python3 -m pip install -q -r requirements.txt
exec python3 check_v8.py --source auto --names classes.txt "$@"
