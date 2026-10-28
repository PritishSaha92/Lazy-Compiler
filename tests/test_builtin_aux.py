"""Logical GPU builtins and auxiliary-role guards with exact references.

These fixtures use typed Emmy0.3.11 nodes directly. They exercise existing
auxiliary Tile decoding and named barriers, not general warp-specialized
SetMaxNReg/TMA schedules. No local GPU is launched by this module.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]

from emmy.compiler.dtype import F32, I32
from emmy.compiler.ir.expr import Builtin, Var
from emmy.compiler.ir.kernel.ir import (
    MbarrierArriveExpectTx, MbarrierInit, MbarrierWait,
    Smem, Sync, Tile, TmaDescriptor, TmaLoad,
)
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.stmt.leaves import Let, Load, Write
from ir_cases import axis, case, expression as ex, literal as lit, write_case
from ptx_backend import Compiler, CompileError, compile_file
from tools.check_compiler import assemble, compile_ptx

PAIRS = [(f"{logical}.{axis}", f"{cuda}.{axis}")
         for logical, cuda in (("thread_idx", "threadIdx"), ("block_idx", "blockIdx"),
                               ("block_dim", "blockDim"), ("grid_dim", "gridDim"))
         for axis in "xyz"] + [("warp_size", "warpSize")]


def exact(value):
    value[1].update(rtol=0, atol=0, fixture_provenance={
        "construction": "Explicit typed Emmy 0.3.11 IR with independent NumPy reference",
        "feature": "logical GPU builtins and raw-thread auxiliary-band coordination",
        "local_gpu_used": False,
    })
    return value


def builtin_values_case(legacy=False):
    names = [pair[int(legacy)] for pair in PAIRS]
    # No Tile: directly exercise every dimension of a 3D launch. Emmy's Tile
    # schedule itself is linear; the builtin expression vocabulary is general.
    def b(name):
        logical = {a: z for a, z in PAIRS}
        return Builtin(logical[name] if legacy else name)
    block_id = ex("+", ex("*", ex("+", ex("*", b("block_idx.z"), b("grid_dim.y")),
                                           b("block_idx.y")), b("grid_dim.x")), b("block_idx.x"))
    thread_id = ex("+", ex("*", ex("+", ex("*", b("thread_idx.z"), b("block_dim.y")),
                                            b("thread_idx.y")), b("block_dim.x")), b("thread_idx.x"))
    flat = ex("+", ex("*", block_id, lit(16)), thread_id)
    body = [Let("physical", flat, I32)]
    for column, name in enumerate(names):
        body.extend((Let(f"value_{column}", Builtin(name), I32),
                     Write("out", (Var("physical"), lit(column)), f"value_{column}", value_dtype=I32)))
    rows = []
    for bz, by, bx in np.ndindex((2, 2, 2)):
        for tz, ty, tx in np.ndindex((2, 2, 4)):
            rows.append((tx, ty, tz, bx, by, bz, 4, 2, 2, 2, 2, 2, 32))
    result = exact(case("builtin_values_3d", tuple(body), {}, (128, 13),
                        lambda: np.asarray(rows, dtype=np.float32), block=16, grid=8))
    result[1]["kernels"][0].update(block=[4, 2, 2], grid=[2, 2, 2])
    return result


def auxiliary_raw_thread_case(legacy=False):
    raw = Builtin("threadIdx.x" if legacy else "thread_idx.x")
    batch = Var("batch")
    cell = Var("cell")
    producer_lane = ex("-", raw, lit(64))
    # The extra warp fills both halves of the row. Raw thread IDs distinguish
    # roles; the Tile's wrapped coordinate remains reserved for compute work.
    producers = (
        Load("first", "x", (batch, producer_lane), dtype=F32),
        Load("second", "x", (batch, ex("+", producer_lane, lit(32))), dtype=F32),
        Write("scratch", (producer_lane,), "first", value_dtype=F32),
        Write("scratch", (ex("+", producer_lane, lit(32)),), "second", value_dtype=F32),
    )
    consumers = (
        Load("held", "scratch", (cell,), dtype=F32),
        # Snapshot all 64 cells before anyone overwrites a reversed neighbor.
        Sync(barrier_id=1, count=64),
        Write("scratch", (ex("-", lit(63), cell),), "held", value_dtype=F32),
        Sync(barrier_id=1, count=64),
        Load("reversed", "scratch", (cell,), dtype=F32),
        Write("out", (batch, cell), "reversed", value_dtype=F32),
    )
    body = (Smem("scratch", (64,), "float", align=16),
            Cond(ex(">=", raw, lit(64)), producers), Sync(),
            Cond(ex("<", raw, lit(64)), consumers))
    return exact(case("auxiliary_raw_thread_named_barrier", (
        Tile((axis("batch", 3), axis("cell", 64)), body, block_threads=64, aux_threads=32),
    ), {"x": (3, 64)}, (3, 64), lambda x: x[:, ::-1], block=96, grid=3))


def bulk_builtin_case(legacy=False):
    block = Builtin("blockIdx.x" if legacy else "block_idx.x")
    leader = ex("==", Var("_tid"), lit(0))
    offset = ex("*", block, lit(64))
    body = (
        Smem("scratch", (64,), "float", align=16),
        Smem("bar", (1,), "unsigned long long", align=8),
        TmaDescriptor("desc", "x", (192,), (64,)),
        Cond(leader, (MbarrierInit("bar"),)), Sync(),
        Cond(leader, (MbarrierArriveExpectTx("bar", 256),
                      TmaLoad("scratch", (lit(0),), "desc", (offset,), "bar"))),
        MbarrierWait("bar", lit(0)),
        Load("value", "scratch", (Var("cell"),), dtype=F32),
        Write("out", (ex("+", offset, Var("cell")),), "value", value_dtype=F32),
    )
    return exact(case("bulk_builtin_block_coordinate", (
        Tile((axis("batch", 3), axis("cell", 64)), body, block_threads=64),
    ), {"x": (192,)}, (192,), lambda x: x, block=64, grid=3))


def builtin_unsigned_case(legacy=False):
    """CUDA geometry components are u32; warpSize is a signed int.

    Comparisons with -1 observe usual arithmetic conversions. Further columns
    distinguish unsigned subtraction, high-bit addition, division and shifts.
    Outputs are only zero/one, so the FP32 wire cannot hide an integer error.
    """
    names = [pair[int(legacy)] for pair in PAIRS]
    tid, cta = Builtin(names[0]), Builtin(names[3])
    physical = ex("+", ex("*", cta, lit(32)), tid)
    expressions = [ex("<", Builtin(name), lit(-1)) for name in names]
    expressions.extend((
        ex(">>", ex("-", tid, lit(1)), lit(31)),
        ex("<", ex("+", tid, lit(2**31 - 16)), lit(0)),
        ex("==", ex("/", ex("-", tid, lit(1)), lit(2)), lit(2**31 - 1)),
        ex(">", ex("-", cta, lit(1)), lit(0)),
    ))
    body = [Let("physical", physical, I32)]
    for column, expression in enumerate(expressions):
        body.extend((Let(f"result_{column}", expression, I32),
                     Write("out", (Var("physical"), lit(column)),
                           f"result_{column}", value_dtype=I32)))
    expected = [[1] * 12 + [0, int(thread == 0), 0, int(thread == 0), int(block != 1)]
                for block in range(3) for thread in range(32)]
    return exact(case("builtin_unsigned_arithmetic", tuple(body), {}, (96, 17),
                      lambda: np.asarray(expected, dtype=np.float32), block=32, grid=3))


def cases():
    yield builtin_values_case()
    yield auxiliary_raw_thread_case()
    yield bulk_builtin_case()
    yield builtin_unsigned_case()


def emit(value):
    graph, meta, _ = value
    target = "sm_90" if meta["id"].startswith("bulk_") else "sm_89"
    return Compiler(graph, meta["kernels"][0], target=target).compile()


class BuiltinAuxiliaryTests(unittest.TestCase):
    def test_logical_names_and_legacy_aliases_emit_identical_modules(self):
        for make in (builtin_values_case, auxiliary_raw_thread_case, bulk_builtin_case,
                     builtin_unsigned_case):
            with self.subTest(case=make.__name__):
                self.assertEqual(emit(make()), emit(make(legacy=True)))

    def test_serialized_fixtures_compile_through_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    ptx, _ = compile_ptx(ROOT / "compile.py", path)
                    self.assertEqual(ptx.replace("\r\n", "\n"), emit(value))
                    with np.load(path.parent / "expected.npz") as archive:
                        np.testing.assert_array_equal(archive["out"], value[2]["out"])

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas required for instruction legality")
    def test_native_assembler(self):
        for value in cases():
            with self.subTest(case=value[1]["id"]):
                target = "sm_90" if value[1]["id"].startswith("bulk_") else "sm_89"
                assemble(emit(value), target)

    def test_unknown_builtin_is_diagnosed(self):
        for spelling in ("thread_idx.w", "threadIdx.w", "not_a_builtin"):
            value = case("unknown_builtin", (
                Let("v", Builtin(spelling), I32),
                Write("out", (lit(0),), "v", value_dtype=I32),
            ), {}, (1,), lambda: np.zeros(1, dtype=np.float32), block=1)
            with self.subTest(spelling=spelling):
                with self.assertRaisesRegex(CompileError, "unknown builtin"):
                    emit(value)

    def test_bulk_coordinate_proof_covers_every_cta(self):
        # A fourth CTA would copy beyond the three-row source. Both spelling
        # conventions must reject it instead of assuming the tested grid size.
        for legacy in (False, True):
            value = bulk_builtin_case(legacy)
            value[1]["kernels"][0]["grid"][0] = 4
            with self.subTest(legacy=legacy):
                with self.assertRaisesRegex(CompileError, "not provably in bounds"):
                    emit(value)

    def test_bulk_dimension_and_warp_constants_have_precise_bounds(self):
        from ptx_bulk import BulkLowering
        for legacy in (False, True):
            graph, meta, _ = bulk_builtin_case(legacy)
            compiler = Compiler(graph, meta["kernels"][0], target="sm_90")
            bulk = BulkLowering(compiler)
            bulk.prepare(compiler.kernel.body)
            expected = ((0, 63), (0, 0), (0, 0), (0, 2), (0, 0), (0, 0),
                        (64, 64), (1, 1), (1, 1), (3, 3), (1, 1), (1, 1), (32, 32))
            for pair, limits in zip(PAIRS, expected, strict=True):
                with self.subTest(builtin=pair[int(legacy)]):
                    self.assertEqual(bulk._interval(Builtin(pair[int(legacy)]), {}), limits)

    def test_bulk_unsigned_wrap_is_not_treated_as_a_signed_interval(self):
        from ptx_bulk import BulkLowering
        for legacy in (False, True):
            graph, meta, _ = bulk_builtin_case(legacy)
            compiler = Compiler(graph, meta["kernels"][0], target="sm_90")
            bulk = BulkLowering(compiler)
            bulk.prepare(compiler.kernel.body)
            cta = Builtin("blockIdx.x" if legacy else "block_idx.x")
            underflow = ex("-", cta, lit(1))
            for expression in (underflow, ex("/", underflow, lit(2)),
                               ex("%", underflow, lit(2)),
                               ex("+", underflow, lit(1)),
                               ex("+", cta, lit(2**31 - 1))):
                with self.subTest(legacy=legacy, expression=expression):
                    with self.assertRaisesRegex(CompileError, "overflow-free"):
                        bulk._interval(expression, {})


def export(destination):
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for value in cases():
        ir = write_case(destination / value[1]["id"], value)
        ptx = compile_file(ir)
        artifact = ir.with_suffix(".ptx")
        artifact.write_text(ptx, encoding="ascii")
        target = "sm_90" if value[1]["id"].startswith("bulk_") else "sm_89"
        entries.append({"id": value[1]["id"], "target": target,
                        "ptx_sha256": hashlib.sha256(ptx.encode("ascii")).hexdigest(),
                        "assembly": assemble(ptx, target), "pass": True})
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in ("compile.py", "ptx_backend.py", "ptx_bulk.py", "tests/test_builtin_aux.py")}
    (destination / "manifest.json").write_text(json.dumps({
        "pass": True, "scope": "Independent exact-reference packages and assembly; no GPU execution",
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
