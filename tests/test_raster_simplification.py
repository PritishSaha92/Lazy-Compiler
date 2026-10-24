"""Exact physical-order checks for constant and degenerate raster groups."""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Load, Write
from ir_cases import axis, case, write_case
from ptx_backend import compile_file
from tools.check_compiler import assemble, local_run


def raster_simplification_cases():
    """List traversal order independently of division/remainder lowering.

    The outer tile preserves the physical output slot; the inner tile chooses
    the logical source coordinate. Thus a wrong but bijective permutation does
    not cancel between reads and writes. Prefix batches and suffix lanes test
    both surrounding strides. Unique integer-valued inputs make each mapping
    error visible while requiring no floating-point tolerance.
    """
    configurations = (
        ("identity_m", 7, 1, 3, "m"),
        ("identity_n", 1, 7, 3, "n"),
        ("full_m", 6, 7, 3, "m"),
        ("full_n", 5, 6, 3, "n"),
        ("clamped_m", 5, 7, 20, "m"),
        ("clamped_n", 5, 7, 20, "n"),
        ("tail_m", 5, 7, 3, "m"),
        ("tail_n", 5, 7, 3, "n"),
    )
    batches, lanes = 2, 32
    for name, m_count, n_count, group, orientation in configurations:
        shape = (batches, m_count, n_count, lanes)
        size = int(np.prod(shape))
        inner = Tile(
            (
                axis("batch", batches),
                axis("m", m_count),
                axis("n", n_count),
                axis("lane_cell", lanes),
            ),
            (
                Load(
                    "value",
                    "x",
                    (Var("batch"), Var("m"), Var("n"), Var("lane_cell")),
                    dtype=F32,
                ),
                Write("out", (Var("physical"),), "value", value_dtype=F32),
            ),
            block_threads=lanes,
            raster_axes=("m", "n"),
            raster_group=group,
            raster_orient=orientation,
        )

        def reference(x):
            rows = []
            for batch in range(batches):
                if orientation == "m":
                    for start in range(0, m_count, group):
                        for n in range(n_count):
                            for m in range(start, min(start + group, m_count)):
                                rows.append(x[batch, m, n])
                else:
                    for start in range(0, n_count, group):
                        for m in range(m_count):
                            for n in range(start, min(start + group, n_count)):
                                rows.append(x[batch, m, n])
            return np.concatenate(rows)

        value = case(
            "raster_simplification_" + name,
            (Tile((axis("physical", size),), (inner,), block_threads=lanes),),
            {"x": shape},
            (size,),
            reference,
            block=lanes,
            grid=batches * m_count * n_count,
        )
        value[1]["rtol"] = value[1]["atol"] = 0
        data = np.arange(size, dtype=np.float32).reshape(shape)
        descriptor = value[1]["kernels"][0]["args"][0]
        descriptor.pop("seed")
        descriptor["data"] = data.reshape(-1).tolist()
        value[2]["out"] = reference(data)
        yield value


class RasterSimplificationTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_raster_shapes_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in raster_simplification_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    assemble(compile_file(path), "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_raster_shapes_match_exact_physical_order(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in raster_simplification_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    kernel = value[1]["kernels"][0]
                    request = {
                        key: kernel[key]
                        for key in (
                            "entry", "grid", "block", "smem_bytes", "zero_outputs", "args"
                        )
                    }
                    result = local_run(
                        {**request, "ptx": compile_file(path), "adapt_target": True}
                    )
                    self.assertTrue(result.get("ok"), result)
                    output = result["outputs"]["out"]
                    actual = np.asarray(output["values"]).reshape(output["shape"])
                    np.testing.assert_array_equal(actual, value[2]["out"])


if __name__ == "__main__":
    unittest.main()
