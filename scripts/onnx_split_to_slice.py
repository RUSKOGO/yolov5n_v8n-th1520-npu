#!/usr/bin/env python3
"""
Replace ONNX Split with equivalent Slice ops (VIP9000 / imgdnn often breaks on Split).

YOLOv8 C2f uses many Split nodes → HHB emits csinn_split → board crash:
  imgdnnNetworkCastOp_v2 / shl_pnna_create_split_internal
  dmlc::Error ... Expect number

Usage:
  python3 scripts/onnx_split_to_slice.py --in yolov8n_raw.onnx --out yolov8n_raw_nosplit.onnx
  # then re-run HHB on yolov8n_raw_nosplit.onnx
"""
from __future__ import annotations

import argparse
from copy import deepcopy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _const_i64(name: str, values):
    arr = np.asarray(values, dtype=np.int64)
    return numpy_helper.from_array(arr, name=name)


def replace_splits(model: onnx.ModelProto) -> int:
    graph = model.graph
    initializers = {i.name: i for i in graph.initializer}
    new_nodes = []
    new_inits = []
    replaced = 0
    uniq = 0

    for node in graph.node:
        if node.op_type != "Split":
            new_nodes.append(node)
            continue

        # axis attribute (default 0)
        axis = 0
        for a in node.attribute:
            if a.name == "axis":
                axis = int(a.i)

        # split sizes: attr "split" (opset<13) or 2nd input (opset>=13)
        sizes = None
        for a in node.attribute:
            if a.name == "split":
                sizes = list(a.ints)
        if sizes is None and len(node.input) >= 2 and node.input[1] in initializers:
            sizes = numpy_helper.to_array(initializers[node.input[1]]).astype(np.int64).tolist()

        if not sizes or len(sizes) != len(node.output):
            # cannot convert safely — keep Split
            new_nodes.append(node)
            continue

        data = node.input[0]
        starts = 0
        for out_name, length in zip(node.output, sizes):
            uniq += 1
            s_name = f"_slice_starts_{uniq}"
            e_name = f"_slice_ends_{uniq}"
            a_name = f"_slice_axes_{uniq}"
            t_name = f"_slice_steps_{uniq}"
            new_inits += [
                _const_i64(s_name, [starts]),
                _const_i64(e_name, [starts + int(length)]),
                _const_i64(a_name, [axis]),
                _const_i64(t_name, [1]),
            ]
            new_nodes.append(
                helper.make_node(
                    "Slice",
                    inputs=[data, s_name, e_name, a_name, t_name],
                    outputs=[out_name],
                    name=f"{node.name or 'Split'}_as_slice_{uniq}",
                )
            )
            starts += int(length)
        replaced += 1

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
    n_split = sum(1 for n in model.graph.node if n.op_type == "Split")
    print(f"Split nodes before: {n_split}")
    replaced = replace_splits(model)
    n_split_after = sum(1 for n in model.graph.node if n.op_type == "Split")
    print(f"Converted: {replaced}; Split left: {n_split_after}")
    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print("Wrote", args.out)
    if n_split_after:
        print("WARNING: some Split nodes remain (missing sizes) — inspect with Netron")


if __name__ == "__main__":
    main()
