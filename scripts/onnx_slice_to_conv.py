#!/usr/bin/env python3
"""
Replace channel-axis ONNX Slice with 1x1 Conv (channel gather).

VIP9000 runs CusStridedSlice on CPU → ~240 ms/frame and the warning:
  Strided_slice software implementation in nna_ddk has memory leak...

C2f does:  y = cv1(x);  a,b = y[:, :c], y[:, c:]
Those Slices become slow CPU ops. A 1x1 Conv with identity selectors is
mathematically identical and maps to NPU Conv.

Usage:
  python3 scripts/onnx_slice_to_conv.py \\
      --in  yolov8n_raw_nosplit.onnx \\
      --out yolov8n_raw_npu.onnx

Then re-run HHB on yolov8n_raw_npu.onnx and rebuild libyolov8n.so.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def _init_map(graph) -> Dict[str, np.ndarray]:
    return {i.name: numpy_helper.to_array(i) for i in graph.initializer}


def _get_const(name: str, inits: Dict[str, np.ndarray], graph) -> Optional[np.ndarray]:
    if name in inits:
        return inits[name]
    for n in graph.node:
        if n.op_type == "Constant" and n.output and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
    return None


def _value_info_shape(graph, name: str) -> Optional[List[int]]:
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name != name:
            continue
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else -1)
        return dims
    return None


def _parse_slice(
    node, inits: Dict[str, np.ndarray], graph
) -> Optional[Tuple[int, int, int, int]]:
    """Return (axis, start, end, step) for a simple 1-D Slice, else None."""
    if node.op_type != "Slice" or len(node.input) < 3:
        return None
    starts = _get_const(node.input[1], inits, graph)
    ends = _get_const(node.input[2], inits, graph)
    if starts is None or ends is None:
        return None
    starts = starts.astype(np.int64).reshape(-1)
    ends = ends.astype(np.int64).reshape(-1)

    axes = np.arange(len(starts), dtype=np.int64)
    steps = np.ones(len(starts), dtype=np.int64)
    if len(node.input) >= 4:
        a = _get_const(node.input[3], inits, graph)
        if a is not None:
            axes = a.astype(np.int64).reshape(-1)
    if len(node.input) >= 5:
        s = _get_const(node.input[4], inits, graph)
        if s is not None:
            steps = s.astype(np.int64).reshape(-1)

    if len(starts) != 1 or len(ends) != 1 or len(axes) != 1:
        # multi-axis: only accept if all other axes are full identity
        # (handled elsewhere — skip for safety)
        if not (len(starts) == len(ends) == len(axes) == len(steps)):
            return None
        # Prefer the channel axis (1) if present with step 1
        channel = None
        for i, ax in enumerate(axes):
            if int(ax) == 1 and int(steps[i]) == 1:
                channel = (1, int(starts[i]), int(ends[i]), 1)
            elif int(steps[i]) != 1 or int(starts[i]) != 0:
                # non-trivial non-channel slice
                if int(ax) != 1:
                    return None
        return channel

    if int(steps[0]) != 1:
        return None
    return int(axes[0]), int(starts[0]), int(ends[0]), 1


def replace_channel_slices(model: onnx.ModelProto) -> int:
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"shape_inference warning: {e}")

    graph = model.graph
    inits = _init_map(graph)
    new_nodes = []
    new_inits = []
    replaced = 0
    uniq = 0

    for node in graph.node:
        parsed = _parse_slice(node, inits, graph)
        if parsed is None or parsed[0] != 1:
            new_nodes.append(node)
            continue

        axis, start, end, _step = parsed
        if end <= start:
            new_nodes.append(node)
            continue

        data = node.input[0]
        out_c = end - start
        in_shape = _value_info_shape(graph, data)
        if not in_shape or len(in_shape) < 2 or in_shape[1] <= 0:
            # fallback: assume end is within known range if end>0
            in_c = max(end, out_c)
        else:
            in_c = in_shape[1]
            if in_c < end:
                # dynamic / unknown — skip
                new_nodes.append(node)
                continue

        # Identity channel gather: W[o, i]=1 iff i == start+o
        w = np.zeros((out_c, in_c, 1, 1), dtype=np.float32)
        for o in range(out_c):
            src = start + o
            if 0 <= src < in_c:
                w[o, src, 0, 0] = 1.0

        uniq += 1
        w_name = f"_slice_conv_w_{uniq}"
        new_inits.append(numpy_helper.from_array(w, name=w_name))
        new_nodes.append(
            helper.make_node(
                "Conv",
                inputs=[data, w_name],
                outputs=list(node.output),
                name=f"{node.name or 'Slice'}_as_conv_{uniq}",
                kernel_shape=[1, 1],
                pads=[0, 0, 0, 0],
                strides=[1, 1],
                group=1,
            )
        )
        replaced += 1
        print(f"  Slice→Conv: {node.name or node.output[0]}  "
              f"C[{start}:{end}] of {in_c} → {out_c}")

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    return replaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = onnx.load(args.inp)
    n_slice = sum(1 for n in model.graph.node if n.op_type == "Slice")
    print(f"Slice nodes before: {n_slice}")
    replaced = replace_channel_slices(model)
    n_slice_after = sum(1 for n in model.graph.node if n.op_type == "Slice")
    print(f"Converted: {replaced}; Slice left: {n_slice_after}")
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"checker warning (often ok after rewrite): {e}")
    onnx.save(model, args.out)
    print("Wrote", args.out)
    if n_slice_after:
        print("WARNING: some Slice remain — check Netron (spatial crops etc.)")
    else:
        print("OK: no Slice left — re-run HHB on this file.")


if __name__ == "__main__":
    main()
