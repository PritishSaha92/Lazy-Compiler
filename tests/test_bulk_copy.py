"""Descriptor-free bulk-copy fixtures, CPU diagnostics and assembler tests.

These are manually constructed Emmy Kernel IR, not CUDA renderer output.
No test in this module launches a local GPU.  Export independent-reference
packages for the organizer runner with --export PATH.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from emmy.compiler.dtype import F32, F64, I32
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import (
    KernelOp, MbarrierArrive, MbarrierArriveExpectTx, MbarrierInit,
    MbarrierWait, Smem, Sync, Tile, TmaDescriptor, TmaLoad,
)
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Let, Load, Write
from ir_cases import axis, case, expression as ex, literal as lit, write_case
from ptx_backend import Compiler, CompileError, compile_file
from tools.check_compiler import assemble


def linear_reference(x):
    return [np.array(x, copy=True)]


def rectangle_reference(x):
    return [np.array(x[2:6, 4:16], copy=True)]


def rank5_reference(x):
    return [np.array(x[1:3, 1:2, 1:3, 1:3, 4:8], copy=True)]


OPEN = [
    {"id": "bulk_linear_multi_cta", "ref": linear_reference},
    {"id": "bulk_rectangle", "ref": rectangle_reference},
    {"id": "bulk_phase_reuse", "ref": linear_reference},
    {"id": "bulk_ring_slots", "ref": linear_reference},
    {"id": "bulk_rank5_crop", "ref": rank5_reference},
    {"id": "bulk_linear_f64", "ref": linear_reference},
]


def exact(value):
    value[1].update(rtol=0, atol=0, fixture_provenance={
        "construction": "Hand-built typed IR with independent NumPy slicing reference",
        "feature": "SM90 descriptor-free rectangular cp.async.bulk and mbarrier",
        "local_gpu_used": False,
    })
    return value


def cases():
    lane, batch, phase = Var("cell"), Var("batch"), Var("phase")
    leader = ex("==", Var("_tid"), lit(0))
    body = (
        Smem("scratch", (64,), "float", align=16),
        Smem("bar", (1,), "unsigned long long", align=8),
        TmaDescriptor("desc", "x", (192,), (64,)),
        Cond(leader, (MbarrierInit("bar"),)), Sync(),
        Cond(leader, (
            MbarrierArriveExpectTx("bar", 256),
            TmaLoad("scratch", (lit(0),), "desc", (ex("*", batch, lit(64)),), "bar"),
        )),
        MbarrierWait("bar", lit(0)),
        Load("value", "scratch", (lane,), dtype=F32),
        Write("out", (ex("+", ex("*", batch, lit(64)), lane),), "value", value_dtype=F32),
    )
    linear = exact(case("bulk_linear_multi_cta", (
        Tile((axis("batch", 3), axis("cell", 64)), body, block_threads=64),
    ), {"x": (192,)}, (192,), lambda x: x, block=64, grid=3))
    yield linear

    body = (
        Smem("scratch", (4, 12), "float", align=16),
        Smem("bar", (1,), "unsigned long long", align=8),
        TmaDescriptor("desc", "x", (8, 20), (4, 12)),
        Cond(leader, (MbarrierInit("bar"),)), Sync(),
        Cond(leader, (
            MbarrierArriveExpectTx("bar", 192),
            TmaLoad("scratch", (lit(0),), "desc", (lit(2), lit(4)), "bar"),
        )),
        MbarrierWait("bar", lit(0)),
        Cond(ex("<", lane, lit(48)), (
            Load("value", "scratch", (lane,), dtype=F32),
            Write("out", (lane,), "value", value_dtype=F32),
        )),
    )
    yield exact(case("bulk_rectangle", (
        Tile((axis("cell", 64),), body, block_threads=64),
    ), {"x": (8, 20)}, (4, 12), lambda x: x[2:6, 4:16], block=64, reverse_args=True))

    for slots, name in ((1, "bulk_phase_reuse"), (2, "bulk_ring_slots")):
        slot = ex("%", phase, lit(slots))
        parity = ex("%", ex("//", phase, lit(slots)), lit(2))
        iteration = (
            Cond(leader, (
                MbarrierArriveExpectTx("bar", 256, slot),
                TmaLoad("scratch", (ex("*", slot, lit(64)),), "desc", (phase, lit(0)), "bar", slot),
            )),
            Cond(ex("==", Var("_tid"), lit(1)), (MbarrierArrive("bar", slot),)),
            MbarrierWait("bar", parity, slot),
            Load("value", "scratch", (ex("+", ex("*", slot, lit(64)), lane),), dtype=F32),
            Write("out", (phase, lane), "value", value_dtype=F32),
            # Every consumer finishes using the buffer and waiting on this
            # generation before either barrier or destination is reused.
            Sync(),
        )
        body = (
            Smem("scratch", (slots * 64,), "float", align=16),
            Smem("bar", (slots,), "unsigned long long", align=8),
            TmaDescriptor("desc", "x", (4, 64), (1, 64)),
            Cond(leader, tuple(MbarrierInit("bar", 2, lit(i)) for i in range(slots))),
            Sync(), Loop(axis("phase", 4), iteration, unroll=False),
        )
        yield exact(case(name, (
            Tile((axis("cell", 64),), body, block_threads=64),
        ), {"x": (4, 64)}, (4, 64), lambda x: x, block=64))

    shape, box = (3, 2, 3, 4, 8), (2, 1, 2, 2, 4)
    body = (
        Smem("scratch", box, "float", align=16),
        Smem("bar", (1,), "unsigned long long", align=8),
        TmaDescriptor("desc", "x", shape, box),
        Cond(leader, (MbarrierInit("bar"),)), Sync(),
        Cond(leader, (
            MbarrierArriveExpectTx("bar", 128),
            TmaLoad("scratch", (lit(0),), "desc", tuple(lit(x) for x in (1, 1, 1, 1, 4)), "bar"),
        )),
        MbarrierWait("bar", lit(0)),
        Load("value", "scratch", (lane,), dtype=F32),
        Write("out", (lane,), "value", value_dtype=F32),
    )
    yield exact(case("bulk_rank5_crop", (
        Tile((axis("cell", 32),), body, block_threads=32),
    ), {"x": shape}, box, lambda x: rank5_reference(x)[0], block=32))

    # A genuine f64 ABI: nearly every random payload has bits lost by a silent
    # f32 round trip.  This observes the copy engine's byte semantics directly.
    wide = transformed(linear, Smem, lambda s: replace(s, dtype="double") if s.name == "scratch" else s)
    wide = transformed(wide, TmaDescriptor, lambda s: replace(s, dtype="double"))
    wide = transformed(wide, MbarrierArriveExpectTx, lambda s: replace(s, bytes_=512))
    wide = transformed(wide, Load, lambda s: replace(s, dtype=F64))
    wide = transformed(wide, Write, lambda s: replace(s, value_dtype=F64))
    graph, old_meta, _ = wide
    meta = json.loads(json.dumps(old_meta))
    meta["id"] = "bulk_linear_f64"
    meta["kernels"][0]["entry"] = "k_regression_bulk_linear_f64"
    for name, node in graph.nodes.items():
        operator = replace(node.op, name=meta["kernels"][0]["entry"]) if isinstance(node.op, KernelOp) else node.op
        graph.nodes[name] = replace(node, op=operator, outputs=tuple(replace(t, dtype=F64) for t in node.outputs))
    for argument in meta["kernels"][0]["args"]:
        argument["dtype"] = "f64"
    expected = np.random.default_rng(0).standard_normal((192,)).astype(np.float64)
    yield graph, meta, {"out": expected}


def transformed(value, kind, change):
    graph, meta, expected = value
    graph = type(graph).from_dict(graph.to_dict())

    def visit(body):
        result = []
        for statement in body:
            if isinstance(statement, (Tile, Loop)):
                statement = replace(statement, body=visit(statement.body))
            elif isinstance(statement, Cond):
                statement = replace(statement, body=visit(statement.body), else_body=visit(statement.else_body))
            if isinstance(statement, kind):
                statement = change(statement)
            result.append(statement)
        return tuple(result)

    for name, node in graph.nodes.items():
        if isinstance(node.op, KernelOp):
            graph.nodes[name] = replace(node, op=replace(node.op, body=visit(node.op.body)))
    return graph, meta, expected


def emit(value, target="sm_90"):
    graph, meta, _ = value
    return Compiler(graph, meta["kernels"][0], target=target).compile()


class BulkCopyTests(unittest.TestCase):
    def test_references_are_exact_copy_slices(self):
        for value, spec in zip(cases(), OPEN):
            _, meta, expected = value
            self.assertEqual(meta["id"], spec["id"])
            descriptor = next(a for a in meta["kernels"][0]["args"] if a["name"] == "x")
            kind = np.float64 if descriptor["dtype"] == "f64" else np.float32
            x = np.random.default_rng(descriptor["seed"]).standard_normal(descriptor["shape"]).astype(kind)
            np.testing.assert_array_equal(expected["out"], spec["ref"](x)[0])
            self.assertFalse(np.array_equal(expected["out"], np.zeros_like(expected["out"])))
            if kind == np.float64:
                self.assertGreater(np.count_nonzero(x != x.astype(np.float32).astype(np.float64)), 150)

    def test_cli_target_and_pointer_abi(self):
        with tempfile.TemporaryDirectory() as folder:
            for value in cases():
                ir = write_case(Path(folder) / value[1]["id"], value)
                ptx = compile_file(ir)
                self.assertIn(".target sm_90", ptx)
                self.assertEqual(ptx.count(".param .u64"), 2)
                self.assertIn("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes", ptx)
                self.assertIn("fence.proxy.async.shared::cta", ptx)
                self.assertIn("mbarrier.try_wait.parity.acquire.cta.shared::cta.b64", ptx)
                self.assertNotIn("cp.async.bulk.tensor", ptx)

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas required for instruction legality")
    def test_sm90_assembler(self):
        for value in cases():
            with self.subTest(case=value[1]["id"]):
                assemble(emit(value), "sm_90")

    def test_unsupported_architecture(self):
        with self.assertRaisesRegex(CompileError, "sm_90"):
            emit(next(cases()), "sm_89")

    def test_multidimensional_launch_is_rejected(self):
        for dimension, position in (("block", 1), ("grid", 2)):
            with self.subTest(dimension=dimension):
                graph, meta, expected = next(cases())
                meta["kernels"][0][dimension][position] = 2
                with self.assertRaisesRegex(CompileError, "one-dimensional"):
                    emit((graph, meta, expected))

    def test_axis_shadowing_does_not_reuse_stale_bounds(self):
        source = next(cases())
        changed = transformed(source, Tile, lambda s: replace(
            s, body=(Let("batch", lit(999), I32),) + s.body
        ))
        with self.assertRaisesRegex(CompileError, "provable"):
            emit(changed)

    def test_loop_axis_shadowing_is_rejected(self):
        source = list(cases())[2]
        changed = transformed(source, Loop, lambda s: replace(
            s, body=(Let("phase", lit(1000), I32),) + s.body
        ))
        with self.assertRaisesRegex(CompileError, "provable"):
            emit(changed)

    def test_reused_node_merges_context_ranges(self):
        from ptx_bulk import BulkLowering
        graph, meta, _ = list(cases())[1]
        compiler = Compiler(graph, meta["kernels"][0], target="sm_90")
        bulk = BulkLowering(compiler)
        descriptor = TmaDescriptor("desc", "x", (8, 20), (4, 12))
        shared_node = TmaLoad("scratch", (lit(0),), "desc", (Var("offset"), lit(4)), "bar")
        bulk.prepare((descriptor,
                      Loop(axis("offset", 8), (shared_node,)),
                      Loop(axis("offset", 1), (shared_node,))))
        self.assertEqual(bulk.ranges[id(shared_node)]["offset"], (0, 7))
        with self.assertRaisesRegex(CompileError, "provably in bounds"):
            bulk.handles(shared_node)

    def test_reject_invalid_boxes(self):
        source = list(cases())[1]
        variants = [
            (TmaDescriptor, lambda s: replace(s, swizzle="B128"), "NONE swizzle"),
            (TmaDescriptor, lambda s: replace(s, src_shape=(8, 24)), "shape must match"),
            (TmaDescriptor, lambda s: replace(s, box_extents=(4, 11)), "multiple of 16"),
            (TmaDescriptor, lambda s: replace(s, dtype="double"), "dtype must match"),
            (TmaLoad, lambda s: replace(s, coords=(lit(6), lit(4))), "provably in bounds"),
            (TmaLoad, lambda s: replace(s, coords=(lit(2), lit(5))), "16-byte aligned"),
            (TmaLoad, lambda s: replace(s, smem_index=(lit(4),)), "destination box"),
            (TmaLoad, lambda s: replace(s, smem_index=(lit(2**31-1), lit(1), lit(-(2**31-1)))), "overflow int32 during flattening"),
            (TmaLoad, lambda s: replace(s, coords=(Var("unknown"), lit(4))), "provable"),
            (TmaLoad, lambda s: replace(s, coords=(ex("+", lit(2**31-1), lit(3)), lit(4))), "overflow-free"),
            (MbarrierWait, lambda s: replace(s, phase=lit(2)), "phase parity"),
            (MbarrierInit, lambda s: replace(s, count=0), "initialization count"),
            (MbarrierArriveExpectTx, lambda s: replace(s, bytes_=2**20), "transaction byte"),
            (MbarrierWait, lambda s: replace(s, slot=lit(1)), "mbarrier slot"),
        ]
        for kind, change, message in variants:
            with self.subTest(diagnostic=message):
                with self.assertRaisesRegex(CompileError, message):
                    emit(transformed(source, kind, change))


def export(destination):
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for value in cases():
        ir = write_case(destination / value[1]["id"], value)
        ptx = compile_file(ir)
        artifact = ir.with_suffix(".ptx")
        artifact.write_text(ptx, encoding="ascii")
        entries.append({"id": value[1]["id"], "ptx_sha256": hashlib.sha256(ptx.encode("ascii")).hexdigest(),
                        "assembly": assemble(ptx, "sm_90"), "pass": True})
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in ("compile.py", "ptx_backend.py", "ptx_bulk.py", "tests/test_bulk_copy.py")}
    (destination / "manifest.json").write_text(json.dumps({
        "pass": True, "scope": "CPU reference fixtures and sm90 assembly; no GPU execution",
        "packages": entries, "sources_sha256": sources,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"packages": len(entries), "assembly_pass": True, "destination": str(destination)}))


if __name__ == "__main__":
    if "--export" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--export", type=Path, required=True)
        export(parser.parse_args().export)
    else:
        unittest.main()
