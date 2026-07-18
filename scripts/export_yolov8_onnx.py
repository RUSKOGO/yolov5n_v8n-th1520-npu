#!/usr/bin/env python3
"""
Export YOLOv8n ONNX for TH1520 / HHB.

Goals (same lessons as YOLOv5n on VIP9000):
  - opset 12
  - no end-to-end NMS in graph
  - prefer raw detect heads if HHB chokes on DFL/Softmax
  - imgsz 640

Usage:
  pip install ultralytics onnx onnxslim
  python3 scripts/export_yolov8_onnx.py
  python3 scripts/export_yolov8_onnx.py --raw-heads   # if DFL breaks NPU
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", default="export/yolov8n_th1520.onnx")
    ap.add_argument(
        "--raw-heads",
        action="store_true",
        help="Try to keep pre-DFL / multi-head outputs (may need ultralytics patches)",
    )
    args = ap.parse_args()

    from ultralytics import YOLO

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    # end2end=False → classical (1, 84, 8400) after DFL inside graph
    # If VIP9000/HHB fails on Softmax(DFL), re-export with custom output cut
    # (see docs/YOLOV8_HHB.md) and use --raw-heads notes.
    path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        nms=False,
    )
    src = Path(path)
    if src.resolve() != out.resolve():
        out.write_bytes(src.read_bytes())
        print(f"Copied {src} → {out}")
    print("OK:", out.resolve())
    print(
        "Next: run HHB quantize/codegen for TH1520 on this ONNX, "
        "copy model.c/io.*/model.params into vendor/hhb_v8/"
    )
    if args.raw_heads:
        print(
            "NOTE: --raw-heads: Ultralytics default export still embeds DFL. "
            "For true raw heads, modify Detect.forward export path or use "
            "onnx-graphsurgeon to cut before dfl/softmax — see docs/YOLOV8_HHB.md"
        )


if __name__ == "__main__":
    main()
