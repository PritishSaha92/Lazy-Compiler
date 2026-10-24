"""Read-only structural analysis for scheduled Emmy IR; never a runtime policy.

The numeric feature vector contains no kernel/buffer names, reference outputs,
input data, seeds, or filenames. Static access-site bytes are not dynamic memory
traffic. Optional emission/assembly statistics remain separate from IR features.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Expr
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCopy, KernelOp, Smem, Sync, Tile, TreeHalve, WarpShuffle,
)
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Write
from ptx_backend import CompileError, dtype, extent, _SIZE


def structural_expansion_cost(body):
    """The emitter's structural unroll-cost convention, not machine instructions."""
    total = 0
    for stmt in body:
        if isinstance(stmt, Loop):
            factor = extent(stmt.axis.extent) if stmt.unroll else 1
            total += 1 + factor * structural_expansion_cost(stmt.body)
        elif isinstance(stmt, (StridedLoop, Tile)):
            total += 1 + structural_expansion_cost(stmt.body)
        elif isinstance(stmt, Cond):
            total += 1 + structural_expansion_cost(stmt.body) + structural_expansion_cost(stmt.else_body)
        else:
            total += 1
    return total


def analyze_graph(graph, launch):
    nodes = [n for n in graph.nodes.values() if isinstance(n.op, KernelOp)]
    if len(nodes) != 1:
        raise CompileError(f"analysis requires exactly one KernelOp, got {len(nodes)}")
    kernel = nodes[0]
    buffers = {}
    for node in graph.nodes.values():
        for tensor in node.outputs:
            buffers[tensor.name] = {
                "space": "global", "shape": [extent(d) for d in tensor.shape],
                "dtype": dtype(tensor.dtype), "element_bytes": _SIZE[dtype(tensor.dtype)],
            }
    statements, operations, expressions, declared_types = Counter(), Counter(), Counter(), Counter()
    features = Counter({
        "statement_count": 0, "expression_count": 0, "loop_count": 0,
        "max_loop_depth": 0, "plain_loop_count": 0, "strided_loop_count": 0,
        "hinted_loop_count": 0, "hinted_loop_work_max": 0,
        "loop_extent_max": 0, "loop_extent_sum": 0, "conditional_count": 0,
        "load_count": 0, "store_count": 0, "atomic_store_count": 0,
        "grouped_load_count": 0, "grouped_store_count": 0, "vector_width_max": 1,
        "accumulator_count": 0, "rescaled_accumulator_count": 0,
        "sync_count": 0, "warp_shuffle_count": 0, "tree_count": 0,
        "async_copy_count": 0, "async_copy_site_bytes": 0,
        "tile_count": 0, "tile_cells_max": 0, "auxiliary_threads_max": 0,
        "input_elements": 0, "output_elements": 0, "max_buffer_rank": 0,
        "shared_bytes_declared": 0, "shared_bytes_aligned": 0,
        "grid_blocks": math.prod(launch.get("grid", [1])),
        "block_threads": math.prod(launch.get("block", [1])),
        "launch_shared_bytes": int(launch.get("smem_bytes", 0)),
    })
    site_bytes = Counter({"global_read": 0, "global_write": 0, "shared_read": 0, "shared_write": 0})
    shared = {}
    loops, tiles, barriers = [], [], []

    def all_body(body):
        for stmt in body:
            yield stmt
            if isinstance(stmt, (Loop, StridedLoop, Tile)):
                yield from all_body(stmt.body)
            elif isinstance(stmt, Cond):
                yield from all_body(stmt.body)
                yield from all_body(stmt.else_body)

    # Shared declarations are storage declarations, not per-iteration allocations.
    for stmt in all_body(kernel.op.body):
        if isinstance(stmt, Smem):
            typ = dtype(stmt.dtype)
            desc = {"space": "shared", "shape": list(stmt.extents), "dtype": typ,
                    "element_bytes": _SIZE[typ], "alignment": max(stmt.align or _SIZE[typ], _SIZE[typ])}
            if stmt.name in shared and shared[stmt.name] != desc:
                raise CompileError(f"inconsistent shared declaration {stmt.name!r}")
            shared[stmt.name] = desc
    buffers.update(shared)
    offset = 0
    for desc in shared.values():
        size = math.prod(desc["shape"]) * desc["element_bytes"]
        features["shared_bytes_declared"] += size
        align = desc["alignment"]
        offset = (offset + align - 1) // align * align
        desc["aligned_offset"] = offset
        offset += size
    features["shared_bytes_aligned"] = offset

    def visit_expression(value):
        if isinstance(value, Expr):
            expressions[type(value).__name__] += 1
            features["expression_count"] += 1
            for field in fields(value):
                visit_expression(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit_expression(item)
        elif is_dataclass(value):
            # SelectBranch carries a selector expression but is not itself Expr.
            for field in fields(value):
                visit_expression(getattr(value, field.name))

    def visit_body(body, loop_depth=0):
        for stmt in body:
            statements[type(stmt).__name__] += 1
            features["statement_count"] += 1
            typ = getattr(stmt, "dtype", None) or getattr(stmt, "value_dtype", None)
            if typ is not None:
                declared_types[dtype(typ)] += 1
            for field in fields(stmt):
                if field.name not in {"body", "else_body", "combine_states", "axis", "axes", "op", "dtype", "value_dtype"}:
                    visit_expression(getattr(stmt, field.name))
            if isinstance(stmt, (Assign, Accum)):
                operations[stmt.op.name] += 1
            if isinstance(stmt, (Loop, StridedLoop)):
                features["loop_count"] += 1
                plain = isinstance(stmt, Loop)
                features["plain_loop_count" if plain else "strided_loop_count"] += 1
                depth = loop_depth + 1
                features["max_loop_depth"] = max(features["max_loop_depth"], depth)
                count = extent(stmt.axis.extent)
                features["loop_extent_max"] = max(features["loop_extent_max"], count)
                features["loop_extent_sum"] += count
                work = count * structural_expansion_cost(stmt.body) if plain else None
                if plain and stmt.unroll:
                    features["hinted_loop_count"] += 1
                    features["hinted_loop_work_max"] = max(features["hinted_loop_work_max"], work)
                loops.append({"node": type(stmt).__name__, "extent": count, "depth": depth,
                              "unroll_hint": bool(stmt.unroll), "seed": bool(stmt.seed),
                              "structural_expansion_cost": work,
                              "actual_trip_count": count if plain else None,
                              "trip_count_note": "static plain loop" if plain else "depends on start, captured end, and potentially live step"})
                visit_body(stmt.body, depth)
            elif isinstance(stmt, Tile):
                features["tile_count"] += 1
                cells = math.prod(extent(a.extent) for a in stmt.axes)
                features["tile_cells_max"] = max(features["tile_cells_max"], cells)
                features["auxiliary_threads_max"] = max(features["auxiliary_threads_max"], stmt.aux_threads)
                tiles.append({"axis_extents": [extent(a.extent) for a in stmt.axes], "logical_cells": cells,
                              "block_threads": stmt.block_threads, "auxiliary_threads": stmt.aux_threads,
                              "raster_group": stmt.raster_group, "raster_orientation": stmt.raster_orient})
                visit_body(stmt.body, loop_depth)
            elif isinstance(stmt, Cond):
                features["conditional_count"] += 1
                visit_body(stmt.body, loop_depth)
                visit_body(stmt.else_body, loop_depth)
            elif isinstance(stmt, (Load, Write)):
                load = isinstance(stmt, Load)
                name = stmt.input if load else stmt.output
                width = len(stmt.names if load else stmt.values)
                desc = buffers[name]
                features["load_count" if load else "store_count"] += 1
                features["vector_width_max"] = max(features["vector_width_max"], width)
                if width > 1:
                    features["grouped_load_count" if load else "grouped_store_count"] += 1
                if not load and stmt.atomic:
                    features["atomic_store_count"] += 1
                site_bytes[desc["space"] + ("_read" if load else "_write")] += width * desc["element_bytes"]
            elif isinstance(stmt, Accum):
                features["accumulator_count"] += 1
                features["rescaled_accumulator_count"] += int(stmt.base is not None)
            elif isinstance(stmt, Sync):
                features["sync_count"] += 1
                barriers.append({"warp": bool(stmt.warp), "id": stmt.barrier_id, "count": stmt.count})
            elif isinstance(stmt, (WarpShuffle, TreeHalve)):
                features["warp_shuffle_count" if isinstance(stmt, WarpShuffle) else "tree_count"] += 1
                visit_body(stmt.combine_states, loop_depth)
            elif isinstance(stmt, CpAsyncCopy):
                features["async_copy_count"] += 1
                features["async_copy_site_bytes"] += stmt.nbytes
                site_bytes["global_read"] += stmt.nbytes
                site_bytes["shared_write"] += stmt.nbytes

    visit_body(kernel.op.body)
    for name in kernel.inputs:
        features["input_elements"] += math.prod(buffers[name]["shape"])
    for tensor in kernel.outputs:
        features["output_elements"] += math.prod(buffers[tensor.name]["shape"])
    features["max_buffer_rank"] = max((len(d["shape"]) for d in buffers.values()), default=0)
    for typ, count in declared_types.items():
        features["dtype_" + typ + "_count"] = count
    return {
        "schema_version": 1, "features": dict(sorted(features.items())),
        "statement_counts": dict(sorted(statements.items())), "operation_counts": dict(sorted(operations.items())),
        "expression_counts": dict(sorted(expressions.items())), "declared_dtype_counts": dict(sorted(declared_types.items())),
        "buffers": buffers, "loops": loops, "tiles": tiles, "barriers": barriers,
        "static_memory_access_site_bytes": dict(site_bytes),
        "limits": [
            "Features are structural counts; neither names nor input/reference values are features.",
            "Access-site bytes count each written IR access once, including both conditional arms. They are not executed bytes, cache transactions, or DRAM measurements.",
            "Collective combine statements are counted once; their runtime communication/repetition is not expanded into instruction counts.",
            "Shared offsets use the current emitter's aligned allocation convention; they are not hardware occupancy.",
            "Access bounds, lane coalescing, arbitrary-expression alignment and dynamic trip counts are not proved by this report.",
        ],
    }


def analyze_file(path):
    path = Path(path)
    raw = path.read_bytes()
    graph = Graph.from_dict(json.loads(raw))
    nodes = [n for n in graph.nodes.values() if isinstance(n.op, KernelOp)]
    if len(nodes) != 1:
        raise CompileError("analysis requires exactly one KernelOp")
    metadata = path.with_name("meta.json").read_bytes()
    matches = [k for k in json.loads(metadata)["kernels"] if k["entry"] == nodes[0].op.name]
    if len(matches) != 1:
        raise CompileError("metadata must contain exactly one matching kernel")
    report = analyze_graph(graph, matches[0])
    report["input_sha256"] = {"ir": hashlib.sha256(raw).hexdigest(), "metadata": hashlib.sha256(metadata).hexdigest()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--emit", action="store_true", help="also measure own-emitter compile time, size and virtual registers")
    parser.add_argument("--assemble", action="store_true", help="also run ptxas on the emitted PTX; no GPU execution")
    parser.add_argument("--target", default="sm_89")
    args = parser.parse_args()
    result = analyze_file(args.ir)
    if args.emit or args.assemble:
        from ptx_backend import compile_file
        begin = time.perf_counter()
        ptx = compile_file(args.ir, target=args.target)
        result["emission"] = {"seconds": time.perf_counter() - begin, "ptx_bytes": len(ptx.encode("ascii")),
                              "ptx_sha256": hashlib.sha256(ptx.encode("ascii")).hexdigest(),
                              "virtual_register_declaration_slots": {kind: int(n) for kind, n in re.findall(r"\.reg \.\w+ %(\w+)<(\d+)>", ptx)},
                              "note": "virtual declaration slots include emitter numbering slack; not physical registers or occupancy"}
        if args.assemble:
            from tools.check_compiler import assemble
            result["assembly"] = assemble(ptx, args.target)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
