"""Independent numeric checks for less common Emmy 0.3.11 layout fields.

Native cp.async is assembled for sm_89. On the sm_75 development GPU, numeric
checks explicitly use the backend's synchronous-copy alternative. That checks
addressing and producer/consumer barriers, not native asynchronous completion.
"""

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
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    Smem,
    Sync,
    Tile,
)
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Load, Write
from ir_cases import axis, case, expression, literal, op, write_case
from ptx_backend import compile_file
from tools.check_compiler import assemble, compile_ptx, local_run


def exact(value):
    value[1]["rtol"] = value[1]["atol"] = 0
    return value


def raster_cases():
    """Observe launch order, so an incorrect but bijective raster cannot pass.

    An outer Tile retains the physical linear cell as ``physical``. The inner
    Tile binds the rasterized coordinates. Both cover the same complete grid;
    every thread writes one distinct physical output. The host reference lists
    groups in traversal order instead of reproducing the emitter's division and
    remainder formulas. The batch prefix and lane suffix check both outer and
    inner strides around the adjacent raster axes.
    """
    batch_count, m_count, n_count, lanes, group = 2, 5, 7, 32, 3
    size = batch_count * m_count * n_count * lanes
    for orientation in ("m", "n"):
        inner = Tile(
            (
                axis("batch", batch_count),
                axis("m", m_count),
                axis("n", n_count),
                axis("cell", lanes),
            ),
            (
                Load(
                    "value",
                    "x",
                    (Var("batch"), Var("m"), Var("n"), Var("cell")),
                    dtype=F32,
                ),
                Write("out", (Var("physical"),), "value", value_dtype=F32),
            ),
            block_threads=lanes,
            raster_axes=("m", "n"),
            raster_group=group,
            raster_orient=orientation,
        )

        def reference(x, orient=orientation):
            rows = []
            for batch in range(batch_count):
                if orient == "m":
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

        yield exact(
            case(
                "raster_rectangular_tail_" + orientation,
                (Tile((axis("physical", size),), (inner,), block_threads=lanes),),
                {"x": (batch_count, m_count, n_count, lanes)},
                (size,),
                reference,
                block=lanes,
                grid=batch_count * m_count * n_count,
            )
        )


def auxiliary_producer_case():
    """A producer warp fills the compute warp's reversed row across three CTAs.

    Producers use the raw warp/lane bindings and only the Tile's block axis.
    They do not read the aliased compute-cell coordinate. All 64 threads reach
    the CTA barrier, and only the compute warp writes global outputs.
    """
    producer = expression(">=", Var("warp"), literal(1))
    compute = expression("==", Var("warp"), literal(0))
    body = (
        Smem("scratch", (32,), "float", align=16),
        Cond(
            producer,
            (
                Load("loaded", "x", (Var("batch"), Var("lane")), dtype=F32),
                Write("scratch", (Var("lane"),), "loaded", value_dtype=F32),
            ),
        ),
        Sync(),
        Cond(
            compute,
            (
                Load(
                    "readback",
                    "scratch",
                    (expression("-", literal(31), Var("cell")),),
                    dtype=F32,
                ),
                Write(
                    "out", (Var("batch"), Var("cell")), "readback", value_dtype=F32
                ),
            ),
        ),
    )
    return exact(
        case(
            "auxiliary_producer_warp",
            (
                Tile(
                    (axis("batch", 3), axis("cell", 32)),
                    body,
                    block_threads=32,
                    aux_threads=32,
                ),
            ),
            {"x": (3, 32)},
            (3, 32),
            lambda x: x[:, ::-1],
            block=64,
            grid=3,
        )
    )


def copy_cases():
    """Read the physical shared layout, avoiding a self-cancelling round trip.

    The published 0.3.11 software permutation moves eight-element chunks by the
    row's masked bits; B32/B64/B128 all use element shift six. The override
    fixture doubles the row width and selects shift seven explicitly. All copied
    chunks are 16-byte aligned, destinations are unique, and the shared buffer
    is completely initialized before its ordinary, unswizzled loads.
    """
    for mode, columns, mask in (
        ("B32", 64, 1),
        ("B64", 64, 3),
        ("B128", 64, 7),
        ("B128@7", 128, 7),
    ):
        rows, threads, batch_count, width = 8, 32, 2, 4
        rows_per_stripe = threads * width // columns
        stripes = rows // rows_per_stripe
        lane_row = expression("//", Var("lane"), literal(columns // width))
        lane_col = expression(
            "*", expression("%", Var("lane"), literal(columns // width)), literal(width)
        )
        row_base = expression("*", Var("stripe"), literal(rows_per_stripe))
        row = expression("+", row_base, lane_row)
        names = tuple("v" + str(i) for i in range(width))
        body = (
            Smem("slab", (rows, columns), "float", align=16),
            Loop(
                axis("stripe", stripes),
                (
                    CpAsyncCopy(
                        "slab",
                        (row_base, literal(0)),
                        "x",
                        (Var("batch"), row, lane_col),
                        nbytes=16,
                        swizzle=mode,
                        lane_index=(lane_row, lane_col),
                        lane_rows=rows_per_stripe,
                    ),
                ),
                unroll=True,
            ),
            CpAsyncCommit(),
            # An empty group is legal and remains part of native group history.
            CpAsyncCommit(),
            CpAsyncWait(0),
            Sync(),
            Loop(
                axis("stripe", stripes),
                (
                    Load(input="slab", index=(row, lane_col), names=names, dtype=F32),
                    Write(
                        "out",
                        (Var("batch"), row, lane_col),
                        values=names,
                        value_dtype=F32,
                    ),
                ),
                unroll=True,
            ),
        )

        def reference(x, row_mask=mask):
            output = np.empty_like(x)
            for r in range(x.shape[1]):
                source_chunks = x[:, r].reshape(batch_count, -1, 8)
                target_chunks = output[:, r].reshape(batch_count, -1, 8)
                for chunk in range(source_chunks.shape[1]):
                    target_chunks[:, chunk ^ (r & row_mask)] = source_chunks[:, chunk]
            return output

        yield exact(
            case(
                "copy_swizzle_" + mode.replace("@", "_shift_"),
                (
                    Tile(
                        (axis("batch", batch_count), axis("cell", threads)),
                        body,
                        block_threads=threads,
                    ),
                ),
                {"x": (batch_count, rows, columns)},
                (batch_count, rows, columns),
                reference,
                block=threads,
                grid=batch_count,
            )
        )


def warp_sync_case():
    """A full warp rendezvous follows different loop trip counts in each lane.

    Each warp owns one shared row. Integer-valued FP32 inputs make all sums
    exact, and each thread then reads a neighbor's result. A transient active
    mask would promise weaker ordering than Emmy's default-mask __syncwarp().
    A numerical pass alone does not prove all scheduling interleavings safe.
    """
    row, cell, k = Var("warp_row"), Var("cell"), Var("k")
    count = expression(
        "*", expression("+", expression("%", cell, literal(8)), literal(1)), literal(17)
    )
    neighbor = expression("%", expression("+", cell, literal(1)), literal(32))
    body = (
        Smem("scratch", (2, 32), "float", align=16),
        StridedLoop(
            axis("k", 136),
            literal(0),
            literal(1),
            (
                Load("term", "x", (row, cell, k), dtype=F32),
                Accum("sum", "term", op("add"), F32),
            ),
            end=count,
        ),
        Write("scratch", (row, cell), "sum", value_dtype=F32),
        Sync(warp=True),
        Load("neighbor", "scratch", (row, neighbor), dtype=F32),
        Write("out", (row, cell), "neighbor", value_dtype=F32),
    )
    value = exact(
        case(
            "warp_sync_uneven_work",
            (Tile((axis("warp_row", 2), axis("cell", 32)), body, block_threads=64),),
            {"x": (2, 32, 136)},
            (2, 32),
            lambda x: np.zeros((2, 32), dtype=np.float32),
            block=64,
        )
    )
    x = (
        np.arange(2, dtype=np.float32)[:, None, None] * 100
        + np.arange(1, 33, dtype=np.float32)[None, :, None]
        + (np.arange(136, dtype=np.float32) % 3)[None, None, :]
    )
    sums = np.asarray(
        [[x[r, lane, : (lane % 8 + 1) * 17].sum() for lane in range(32)] for r in range(2)],
        dtype=np.float32,
    )
    value[1]["kernels"][0]["args"][0].pop("seed")
    value[1]["kernels"][0]["args"][0]["data"] = x.ravel().tolist()
    value[2]["out"] = np.roll(sums, -1, axis=1)
    return value


def layout_cases():
    yield from raster_cases()
    yield auxiliary_producer_case()
    yield from copy_cases()
    yield warp_sync_case()


class LayoutRegressionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_layouts_assemble_native_and_synchronous(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in layout_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    native, _ = compile_ptx(ROOT / "compile.py", path)
                    assemble(native, "sm_89")
                    assemble(compile_file(path, synchronous_copy=True), "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_layouts_match_independent_exact_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in layout_cases():
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
                        {
                            **request,
                            "ptx": compile_file(path, synchronous_copy=True),
                            "adapt_target": True,
                        }
                    )
                    self.assertTrue(result.get("ok"), result)
                    for name, expected in value[2].items():
                        output = result["outputs"][name]
                        actual = np.asarray(output["values"]).reshape(output["shape"])
                        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
