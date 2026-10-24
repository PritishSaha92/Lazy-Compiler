"""Address contract coverage, including Emmy's legacy partial-coordinate form."""

from dataclasses import replace
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Load, Write
from ir_cases import axis, expression, literal, write_case
from ptx_backend import Compiler, compile_file
from test_contract_edges import explicit_case
from tools.check_compiler import assemble, local_run


def index_layout_cases():
    i = Var("i")
    one = literal(1)
    x = np.arange(24, dtype=np.float32) * 3 - 17
    y = (np.arange(24, dtype=np.float32) * -2 + 11).reshape(2, 3, 4)
    z = (np.arange(24, dtype=np.float32) * 5 + 7).reshape(4, 6)
    body = (
        Load("mismatch_1d", "x", (i, one), dtype=F32),
        Load("mismatch_3d", "y", (i, one), dtype=F32),
        Load(
            "matched",
            "z",
            (expression("/", i, literal(6)), expression("%", i, literal(6))),
            dtype=F32,
        ),
        Load("empty", "x", (), dtype=F32),
        Write(
            "out",
            (i, literal(0)),
            values=("mismatch_1d", "mismatch_3d", "matched", "empty"),
            value_dtype=F32,
        ),
    )
    expected = np.column_stack(
        (x[1:18], y.ravel()[1:18], z.ravel()[:17], np.full(17, x[0]))
    )
    yield explicit_case(
        "index_rank_mismatch_loads",
        (Tile((axis("i", 17),), body),),
        {"x": x, "y": y, "z": z},
        expected,
    )

    # A two-index tuple addressing a rank-one output sums its coordinates.
    body = (
        Load("value", "x", (i,), dtype=F32),
        Write("out", (expression("-", i, one), one), "value", value_dtype=F32),
    )
    yield explicit_case(
        "index_rank_mismatch_write_1d",
        (Tile((axis("i", 17),), body),),
        {"x": x[:17]},
        x[:17].copy(),
    )

    # Three coordinates on a rank-two buffer also sum, rather than truncating
    # extra coordinates or pretending the declared shape has three axes.
    body = (
        Load("value", "x", (i,), dtype=F32),
        Write(
            "out",
            (expression("-", i, literal(3)), one, literal(2)),
            "value",
            value_dtype=F32,
        ),
    )
    yield explicit_case(
        "index_rank_mismatch_write_2d",
        (Tile((axis("i", 24),), body),),
        {"x": x},
        x.reshape(4, 6).copy(),
    )


class IndexLayoutTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_index_layout_assembles(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in index_layout_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    assemble(compile_file(path), "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_index_layout_matches_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in index_layout_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
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
                    result = local_run(
                        {**request, "ptx": compile_file(path), "adapt_target": True}
                    )
                    self.assertTrue(result.get("ok"), result)
                    out = result["outputs"]["out"]
                    actual = np.asarray(out["values"]).reshape(out["shape"])
                    np.testing.assert_array_equal(actual, value[2]["out"])

    def test_large_offsets_widen_before_summing(self):
        # Compile-time address test only: do not allocate a 16-GiB buffer on the
        # local GPU. All three address forms must retain the high index bits.
        graph, metadata, _ = next(index_layout_cases())
        for key, node in graph.nodes.items():
            graph.nodes[key] = replace(
                node,
                outputs=tuple(
                    replace(t, shape=(Dim(65536), Dim(65536))) if t.name == "x" else t
                    for t in node.outputs
                ),
            )
        compiler = Compiler(graph, metadata["kernels"][0])
        for indices, expected in (
            ((Literal(2**31, "long long"), literal(3), literal(4)), 2**31 + 7),
            ((literal(65535), literal(65535)), 2**32 - 1),
            ((Literal(2**32 - 1, "long long"),), 2**32 - 1),
            ((), 0),
        ):
            with self.subTest(indices=indices):
                result = compiler.flat_index("x", indices)
                self.assertEqual(result.type, "s64")
                self.assertEqual(result.constant, expected)


if __name__ == "__main__":
    unittest.main()
