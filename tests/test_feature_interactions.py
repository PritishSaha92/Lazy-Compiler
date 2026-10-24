"""Joint bulk-copy/tensor-core pipelines with independent exact CPU references.

Three K=16 slabs reuse shared storage and one parity barrier while the FP32 C
fragment survives outside the loop. These tests assemble only; --export writes
ordinary packages for separate organizer-GPU execution, never a local GPU run.
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
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from emmy.compiler.dtype import F16, F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import (
    ELEM_COL, LdmatrixLoad, MbarrierArriveExpectTx, MbarrierInit,
    MbarrierWait, MmaSyncPtx, RegFragment, RegStore, Smem, Sync,
    Tile, TmaDescriptor, TmaLoad,
)
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Assign, Load
from ir_cases import axis, case, expression as ex, literal as lit, op, write_case
from ptx_backend import CompileError, compile_file
from tools.check_compiler import assemble, compile_ptx, input_array

SHAPE = (16, 8, 16)


def reference(a, b, bias=None):
    """Independent matrix formula after the actual input-storage conversion."""
    aq, bq = np.asarray(a, np.float16), np.asarray(b, np.float16)
    result = (aq.astype(np.float64) @ bq.astype(np.float64)).astype(np.float32)
    if bias is not None:
        result = np.maximum(result + np.asarray(bias, np.float32), np.float32(0)).T
    return [result]


OPEN = [
    {"id": "bulk_tensor_k48_f16", "ref": reference},
    {"id": "bulk_tensor_k48_bias_relu_transposed", "ref": reference},
]


def cases():
    for fused in (False, True):
        name = OPEN[int(fused)]["id"]
        phase = Var("slab")
        leader = ex("==", Var("_tid"), lit(0))
        offset = ex("*", phase, lit(16))
        iteration = (
            Cond(leader, (
                # Both boxes complete against the same barrier generation:
                # 16*16 half elements + 16*8 half elements = 768 bytes.
                MbarrierArriveExpectTx("bar", 768),
                TmaLoad("aslab", (lit(0),), "adesc", (lit(0), offset), "bar"),
                TmaLoad("bslab", (lit(0),), "bdesc", (offset, lit(0)), "bar"),
            )),
            MbarrierWait("bar", ex("%", phase, lit(2))),
            LdmatrixLoad("af", "aslab", (lit(0), lit(0)), "a", ldm=16),
            LdmatrixLoad("bf", "bslab", (lit(0), lit(0)), "b", ldm=8),
            MmaSyncPtx("cf", "af", "bf", SHAPE, ab_dtype="f16"),
            # Complete every consumer's loads and parity wait before the leader
            # can reuse either shared box or the barrier's next generation.
            Sync(),
        )
        epilogue = None
        if fused:
            epilogue = Lambda(params=("acc", ELEM_COL), body=(
                Load("bias_value", "bias", (Var(ELEM_COL),), dtype=F32),
                Assign("biased", op("add"), ("acc", "bias_value"), F32),
                Assign("activated", op("relu"), ("biased",), F32),
            ), results=("activated",))
        body = (
            Smem("aslab", (16, 16), "__half", align=16),
            Smem("bslab", (16, 8), "__half", align=16),
            Smem("bar", (1,), "unsigned long long", align=8),
            TmaDescriptor("adesc", "a", (16, 48), (16, 16), dtype="__half"),
            TmaDescriptor("bdesc", "b", (48, 8), (16, 8), dtype="__half"),
            RegFragment("af", "a", SHAPE, F16),
            RegFragment("bf", "b", SHAPE, F16),
            RegFragment("cf", "c", SHAPE, F32),
            Cond(leader, (MbarrierInit("bar", 1),)), Sync(),
            Loop(axis("slab", 3), iteration, unroll=False),
            RegStore("out", (lit(0), lit(0)), "cf", SHAPE, epilogue=epilogue,
                     row_dim=1 if fused else 0, col_dim=0 if fused else 1),
        )
        shapes = {"a": (16, 48), "b": (48, 8)}
        if fused:
            shapes["bias"] = (8,)
        graph, meta, _ = case(name, (Tile((axis("cell", 32),), body, block_threads=32),),
                             shapes, (8, 16) if fused else (16, 8),
                             lambda **inputs: reference(**inputs)[0], block=32,
                             reverse_args=fused)
        # Dyadic values make every product/sum exact in FP32. A zero tolerance
        # can therefore expose a skipped slab, wrong phase or lost C lifetime.
        rng = np.random.default_rng(20261024)
        arrays = {"a": (rng.integers(-12, 13, (16, 48)) / 8).astype(np.float16),
                  "b": (rng.integers(-12, 13, (48, 8)) / 8).astype(np.float16)}
        if fused:
            arrays["bias"] = np.arange(-4, 4, dtype=np.float32) / np.float32(4)
        for key, node in graph.nodes.items():
            graph.nodes[key] = replace(node, outputs=tuple(
                replace(tensor, dtype=F16) if tensor.name in ("a", "b") else tensor
                for tensor in node.outputs))
        for argument in meta["kernels"][0]["args"]:
            name = argument["name"]
            if name in arrays:
                argument.pop("seed", None)
                argument.update(dtype="f16" if name in ("a", "b") else "f32",
                                data=arrays[name].ravel().tolist())
        meta.update(rtol=0, atol=0, fixture_provenance={
            "construction": "Hand-built typed IR; quantized input bytes and independent NumPy matrix reference frozen before compilation",
            "feature": "SM90 bulk-copy + native shared ldmatrix + f16 tensor MMA across three reusable K slabs",
            "slabs": 3, "transaction_bytes_per_slab": 768,
            "fused_bias_relu_transposed_output": fused,
            "local_gpu_used": False,
        })
        yield graph, meta, {"out": reference(**arrays)[0]}


def export(destination):
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for value in cases():
        folder = destination / value[1]["id"]
        path = write_case(folder, value)
        records.append({"id": value[1]["id"], "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.is_file()
        }})
        print(path, flush=True)
    (destination / "manifest.json").write_text(json.dumps({
        "scope": "Joint feature input/reference packages only; export performs no GPU execution",
        "packages": records,
    }, indent=2) + "\n", encoding="utf-8")


class FeatureInteractionTests(unittest.TestCase):
    def test_explicit_f16_inputs_exact_oracle_and_all_slab_contributions(self):
        values = list(cases())
        self.assertEqual(len(values), 2)
        for graph, metadata, expected in values:
            with self.subTest(case=metadata["id"]):
                Graph.from_dict(graph.to_dict())
                arrays = {argument["name"]: input_array(argument, 0)
                          for argument in metadata["kernels"][0]["args"] if not argument.get("output")}
                self.assertEqual(arrays["a"].dtype, np.dtype("float16"))
                self.assertEqual(arrays["b"].dtype, np.dtype("float16"))
                np.testing.assert_array_equal(reference(**arrays)[0], expected["out"])
                a, b = arrays["a"].astype(np.float64), arrays["b"].astype(np.float64)
                slabs = [a[:, k:k+16] @ b[k:k+16, :] for k in (0, 16, 32)]
                full = a @ b
                np.testing.assert_array_equal(sum(slabs), full)
                self.assertFalse(np.array_equal(full, slabs[-1]))
                np.testing.assert_array_equal(full.astype(np.float32).astype(np.float64), full)
                self.assertTrue(np.any(expected["out"] != 0))
                self.assertEqual((metadata["rtol"], metadata["atol"]), (0, 0))

    def test_actual_cli_selects_sm90_and_keeps_both_instruction_families(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    ptx, _ = compile_ptx(ROOT / "compile.py", path)
                    self.assertIn(".target sm_90", ptx)
                    self.assertIn("cp.async.bulk.shared::cluster.global", ptx)
                    self.assertIn("mbarrier.arrive.expect_tx", ptx)
                    self.assertIn("mbarrier.try_wait.parity", ptx)
                    self.assertIn("ldmatrix.sync.aligned.m8n8.x4", ptx)
                    self.assertIn("ldmatrix.sync.aligned.m8n8.x2.trans", ptx)
                    self.assertIn("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", ptx)
                    with self.assertRaisesRegex(CompileError, "sm_90"):
                        compile_file(path, target="sm_89")

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas required; no GPU execution")
    def test_combined_pipelines_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    assemble(compile_file(path), "sm_90")


if __name__ == "__main__":
    if "--export" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--export", required=True, type=Path)
        export(parser.parse_args().export)
    else:
        unittest.main()
