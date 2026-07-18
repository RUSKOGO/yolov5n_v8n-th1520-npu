#!/usr/bin/env python3
"""
Export YOLOv8 Detect as THREE raw heads (no DFL in graph) + NPU-friendly C2f.

Each output: (1, 4*reg_max + nc, H, W)  e.g. nc=10 → (1, 74, 80/40/20)

Also rewrites C2f: cv1(x).chunk(2) → two half-Convs (cv1a/cv1b).
Without that, HHB emits CusStridedSlice on CPU → ~240 ms/frame.

Usage:
  python3 scripts/export_yolov8_raw_heads.py --weights ppe.pt --out yolov8n_raw.onnx
  # optional safety net if any Slice remains:
  python3 scripts/onnx_slice_to_conv.py --in yolov8n_raw.onnx --out yolov8n_raw_npu.onnx
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn


def _split_ultralytics_conv(conv_mod: nn.Module, c_half: int):
    """Split Conv(c1, 2*c) + BN into two Conv(c1, c) with half weights/BN."""
    old = conv_mod.conv
    assert old.weight.shape[0] == 2 * c_half, (
        f"expected out_ch={2 * c_half}, got {old.weight.shape[0]}"
    )

    def half(start: int, end: int) -> nn.Module:
        m = deepcopy(conv_mod)
        new_conv = nn.Conv2d(
            old.in_channels,
            c_half,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=old.bias is not None,
        )
        new_conv.weight.data.copy_(old.weight.data[start:end])
        if old.bias is not None:
            new_conv.bias.data.copy_(old.bias.data[start:end])
        m.conv = new_conv

        old_bn = conv_mod.bn
        new_bn = nn.BatchNorm2d(c_half)
        new_bn.weight.data.copy_(old_bn.weight.data[start:end])
        new_bn.bias.data.copy_(old_bn.bias.data[start:end])
        new_bn.running_mean.data.copy_(old_bn.running_mean.data[start:end])
        new_bn.running_var.data.copy_(old_bn.running_var.data[start:end])
        new_bn.eps = old_bn.eps
        new_bn.momentum = old_bn.momentum
        m.bn = new_bn
        return m

    return half(0, c_half), half(c_half, 2 * c_half)


def patch_c2f_no_chunk(root: nn.Module) -> int:
    """Replace C2f chunk(2) with cv1a/cv1b — no Slice in ONNX."""
    n = 0
    for m in root.modules():
        if m.__class__.__name__ != "C2f":
            continue
        c = int(m.c)
        cv1a, cv1b = _split_ultralytics_conv(m.cv1, c)
        m.cv1a = cv1a
        m.cv1b = cv1b
        # keep cv1 for state_dict compat but unused

        def forward_noslice(self, x, _c=c):
            y = [self.cv1a(x), self.cv1b(x)]
            y.extend(blk(y[-1]) for blk in self.m)
            return self.cv2(torch.cat(y, 1))

        m.forward = forward_noslice.__get__(m, m.__class__)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ppe.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", default="yolov8n_raw.onnx")
    args = ap.parse_args()

    from ultralytics import YOLO

    yolo = YOLO(args.weights)
    model = yolo.model.eval().cpu()

    n_c2f = patch_c2f_no_chunk(model)
    print(f"Patched C2f modules (no chunk/Slice): {n_c2f}")

    detect = model.model[-1]
    nl = detect.nl
    nc = detect.nc
    reg_max = getattr(detect, "reg_max", 16)
    ch = 4 * reg_max + nc
    print(f"Detect: nl={nl} nc={nc} reg_max={reg_max} → channels/head={ch}")
    names = getattr(yolo, "names", None) or getattr(model, "names", None)
    if names:
        print("class names:", names)

    # Force export-friendly flags, then bypass DFL path
    detect.export = True
    detect.format = "onnx"
    detect.dynamic = False

    def forward_raw(self, x):
        outs = []
        for i in range(self.nl):
            outs.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        return outs  # list of 3 tensors

    detect.forward = forward_raw.__get__(detect, detect.__class__)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, im):
            y = self.m(im)
            # ultralytics model returns tuple/list depending on version
            if isinstance(y, (list, tuple)):
                # may be (preds, feats) — take feats or preds
                if len(y) == 2 and isinstance(y[0], list):
                    return tuple(y[0])
                if len(y) >= 1 and isinstance(y[0], torch.Tensor) and y[0].dim() == 4:
                    return tuple(y[:nl])
                if isinstance(y[0], list):
                    return tuple(y[0])
                return tuple(y)
            return y

    wrapped = Wrap(model).eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)

    with torch.no_grad():
        sample = wrapped(dummy)
    if not isinstance(sample, (list, tuple)) or len(sample) != nl:
        raise RuntimeError(f"Unexpected raw export outputs: {type(sample)} {sample}")
    for i, t in enumerate(sample):
        print(f"  head[{i}]: {tuple(t.shape)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        str(out),
        opset_version=args.opset,
        input_names=["images"],
        output_names=[f"output{i}" for i in range(nl)],
        dynamic_axes=None,
    )
    print("Wrote", out.resolve())
    # Verify no Slice leaked into ONNX
    try:
        import onnx

        om = onnx.load(str(out))
        n_slice = sum(1 for n in om.graph.node if n.op_type == "Slice")
        n_split = sum(1 for n in om.graph.node if n.op_type == "Split")
        print(f"ONNX ops: Slice={n_slice} Split={n_split} (want both 0)")
        if n_slice or n_split:
            print("WARNING: run scripts/onnx_slice_to_conv.py and/or onnx_split_to_slice.py")
    except Exception as e:
        print("onnx check skipped:", e)

    if names:
        # write classes.txt for check_v8.py
        cls_path = out.with_name("classes.txt")
        if isinstance(names, dict):
            lines = [names[i] for i in range(len(names))]
        else:
            lines = list(names)
        cls_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("Wrote", cls_path.resolve())

    print("Next:")
    print(f"  # if Slice>0: python3 scripts/onnx_slice_to_conv.py --in {out.name} --out yolov8n_raw_npu.onnx")
    print(
        f'  hhb -D --model-file {out.name} --data-scale-div 255 --board th1520 '
        f'--input-name "images" '
        f'--output-name "output0;output1;output2" '
        f'--input-shape "1 3 {args.imgsz} {args.imgsz}" '
        f'--calibrate-dataset calib --quantization-scheme "int8_asym"'
    )
    print("  # after HHB: grep -c strided_slice hhb_out/model.c   # want 0")


if __name__ == "__main__":
    main()
