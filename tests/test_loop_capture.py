"""Loop-expression lifetimes: a hoisted end, captured start, and live step.

StridedLoop.end is evaluated once into <axis>_end. The step expression retains
the ordinary for-loop semantics and is reevaluated after each body execution.
These independent expectations use integer recurrences, not emitted PTX text.
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
from emmy.compiler.dtype import I32
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Init, Let, Write
from ir_cases import axis, expression, literal, op, write_case
from ptx_backend import compile_file
from test_contract_edges import explicit_case
from tools.check_compiler import assemble, local_run


def loop_capture_cases():
    i, k = Var("i"), Var("k")
    limits = np.arange(17, dtype=np.int32) % 6
    # Updating limit must not alter either the loop's stop value or the
    # k_end binding read inside the loop. Includes zero and one trip counts.
    body = (
        Init("limit", 0, dtype=I32),
        Init("steps", 0, dtype=I32),
        Init("bound_sum", 0, dtype=I32),
        Let("initial", expression("%", i, literal(6)), I32),
        Let("one", literal(1), I32),
        Let("minus_one", literal(-1), I32),
        Accum("limit", "initial", op("add"), I32),
        StridedLoop(
            axis("k", 5),
            literal(0),
            literal(1),
            (
                Accum("limit", "minus_one", op("add"), I32),
                Accum("steps", "one", op("add"), I32),
                Let("saved_end", Var("k_end"), I32),
                Accum("bound_sum", "saved_end", op("add"), I32),
            ),
            end=Var("limit"),
            seed=False,
        ),
        Write(
            "out",
            (i, literal(0)),
            values=("limit", "steps", "bound_sum"),
            value_dtype=I32,
        ),
    )
    yield explicit_case(
        "strided_bound_carrier_snapshot",
        (Tile((axis("i", 17),), body),),
        {},
        np.column_stack((np.zeros(17, dtype=np.int32), limits, limits * limits)),
        types={"out": I32},
    )

    # Each outer iteration captures the current limit anew. An unrolled outer
    # loop must have the same state lifetime as the retained outer loop.
    body = (
        Init("limit", 5, dtype=I32),
        Init("steps", 0, dtype=I32),
        Let("one", literal(1), I32),
        Let("minus_one", literal(-1), I32),
        Loop(
            axis("outer", 3),
            (
                StridedLoop(
                    axis("k", 5),
                    literal(0),
                    literal(1),
                    (
                        Accum("limit", "minus_one", op("add"), I32),
                        Accum("steps", "one", op("add"), I32),
                    ),
                    end=Var("limit"),
                    seed=False,
                ),
            ),
            unroll=True,
        ),
        Write("out", (i, literal(0)), values=("limit", "steps"), value_dtype=I32),
    )
    yield explicit_case(
        "nested_strided_bound_recapture",
        (Tile((axis("i", 9),), body),),
        {},
        np.tile(np.asarray([0, 5], np.int32), (9, 1)),
        types={"out": I32},
    )

    # Capture start before the loop, but reevaluate the step AFTER the body.
    # Indices 1, 3, 6, 10 result as stride changes through 2, 3, 4, 5.
    body = (
        Init("start", 1, dtype=I32),
        Init("stride", 1, dtype=I32),
        Init("steps", 0, dtype=I32),
        Init("index_sum", 0, dtype=I32),
        Let("one", literal(1), I32),
        Let("minus_one", literal(-1), I32),
        StridedLoop(
            axis("k", 12),
            Var("start"),
            Var("stride"),
            (
                Accum("start", "minus_one", op("add"), I32),
                Accum("stride", "one", op("add"), I32),
                Accum("steps", "one", op("add"), I32),
                Let("index", k, I32),
                Accum("index_sum", "index", op("add"), I32),
            ),
            seed=False,
        ),
        Write(
            "out",
            (i, literal(0)),
            values=("start", "stride", "steps", "index_sum"),
            value_dtype=I32,
        ),
    )
    yield explicit_case(
        "strided_start_capture_live_step",
        (Tile((axis("i", 9),), body),),
        {},
        np.tile(np.asarray([-3, 5, 4, 20], np.int32), (9, 1)),
        types={"out": I32},
    )


class LoopCaptureTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_loop_capture_assembles(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in loop_capture_cases():
                path = write_case(Path(directory) / value[1]["id"], value)
                for budget in (0, 1024):
                    with self.subTest(case=value[1]["id"], unroll_budget=budget):
                        assemble(compile_file(path, unroll_budget=budget), "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_loop_capture_matches_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in loop_capture_cases():
                path = write_case(Path(directory) / value[1]["id"], value)
                for budget in (0, 1024):
                    with self.subTest(case=value[1]["id"], unroll_budget=budget):
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
                            {
                                **request,
                                "ptx": compile_file(path, unroll_budget=budget),
                                "adapt_target": True,
                            }
                        )
                        self.assertTrue(result.get("ok"), result)
                        out = result["outputs"]["out"]
                        actual = np.asarray(out["values"]).reshape(out["shape"])
                        np.testing.assert_array_equal(actual, value[2]["out"])


if __name__ == "__main__":
    unittest.main()
