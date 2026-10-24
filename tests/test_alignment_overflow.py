"""Fixed-width overflow must not manufacture vector-address alignment."""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy.compiler.dtype import F32, U32, U64
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Let, Load, Write
from ir_cases import axis, expression, literal, write_case
from ptx_backend import compile_file
from test_contract_edges import explicit_case
from tools.check_compiler import assemble, local_run


def alignment_overflow_cases():
    i = Var("i")
    for bits, integer_type, numpy_type, literal_type in (
        (32, U32, np.uint32, "unsigned int"),
        (64, U64, np.uint64, "unsigned long long"),
    ):
        # 6*x wraps twice and leaves 4; integer division by 3 yields the odd
        # index 1. Keeping the pre-wrap factor 6 would falsely infer evenness.
        wrapped_source = (2 * (1 << bits) + 4) // 6
        source = np.asarray([0, 1, wrapped_source, wrapped_source + 1], numpy_type)
        mask = (1 << bits) - 1
        offsets = np.asarray([((int(v) * 6) & mask) // 3 for v in source])
        x = np.asarray([1, 11, 111, 1111, 11111, 111111], np.float32)
        expected = np.column_stack((x[offsets], x[offsets + 1]))
        for grouped in (False, True):
            body = [
                Load("source", "indices", (i,), dtype=integer_type),
                Let(
                    "index",
                    expression(
                        "/",
                        expression("*", Var("source"), Literal(6, literal_type)),
                        Literal(3, literal_type),
                    ),
                    integer_type,
                ),
            ]
            if grouped:
                body.append(
                    Load(input="x", index=(Var("index"),), names=("a", "b"), dtype=F32)
                )
            else:
                body.extend(
                    (
                        Load(
                            "a",
                            "x",
                            (expression("+", Var("index"), literal(0)),),
                            dtype=F32,
                        ),
                        Load(
                            "b",
                            "x",
                            (expression("+", Var("index"), literal(1)),),
                            dtype=F32,
                        ),
                    )
                )
            body.append(
                Write("out", (i, literal(0)), values=("a", "b"), value_dtype=F32)
            )
            yield explicit_case(
                f"overflow_alignment_u{bits}_{'grouped' if grouped else 'scalar'}",
                (Tile((axis("i", 4),), tuple(body)),),
                {"indices": source, "x": x},
                expected,
                types={"indices": integer_type},
            )


class AlignmentOverflowTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_wrapping_address_arithmetic_assembles_without_false_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in alignment_overflow_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    ptx = compile_file(path)
                    # The third thread accesses x[1]; a v2 access is illegal
                    # there. This check prevents issuing a misaligned GPU load.
                    self.assertNotIn("ld.global.v2", ptx)
                    self.assertNotIn("ld.global.v4", ptx)
                    assemble(ptx, "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_wrapping_address_arithmetic_matches_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in alignment_overflow_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    ptx = compile_file(path)
                    self.assertNotIn("ld.global.v2", ptx)
                    self.assertNotIn("ld.global.v4", ptx)
                    kernel = value[1]["kernels"][0]
                    request = {
                        key: kernel[key]
                        for key in (
                            "entry",
                            "grid",
                            "block",
                            "smem_bytes",
                            "zero_outputs",
                            "args",
                        )
                    }
                    result = local_run({**request, "ptx": ptx, "adapt_target": True})
                    self.assertTrue(result.get("ok"), result)
                    out = result["outputs"]["out"]
                    np.testing.assert_array_equal(
                        np.asarray(out["values"]).reshape(out["shape"]), value[2]["out"]
                    )


if __name__ == "__main__":
    unittest.main()
