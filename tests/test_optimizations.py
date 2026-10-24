"""Legality and independent numerical checks for optional dot accumulation modes."""

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
from emmy.compiler.ir.expr import TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import KernelOp, Tile, WarpShuffle
from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import (
    Accum,
    Assign,
    Let,
    Load,
    Select,
    SelectBranch,
    Write,
)
from ir_cases import axis, case, expression, literal, op, write_case
from ptx_backend import Compiler, compile_file
from tools.check_compiler import assemble, compare_outputs, local_run


def with_data(value, x, y, expected):
    for arg in value[1]["kernels"][0]["args"]:
        if arg["name"] in ("x", "y"):
            arg.pop("seed", None)
            arg["data"] = (x if arg["name"] == "x" else y).ravel().tolist()
    value[1]["rtol"] = value[1]["atol"] = 0
    return value[0], value[1], {"out": np.asarray(expected, dtype=np.float32)}


def dot_cases():
    i, k = Var("i"), Var("k")
    for transposed in (False, True):
        index = (k, i) if transposed else (i, k)
        shape = (37, 7) if transposed else (7, 37)
        body = (
            Loop(
                axis("k", 37),
                (
                    Load("a", "x", index, dtype=F32),
                    Load("b", "y", index, dtype=F32),
                    Assign("product", op("multiply"), ("a", "b"), F32),
                    Accum("state", "product", op("add"), F32, ("k",)),
                ),
                unroll=True,
            ),
            Write("out", (i,), "state", value_dtype=F32),
        )
        yield case(
            "dot_transposed" if transposed else "dot_rows",
            (Tile((axis("i", 7),), body),),
            {"x": shape, "y": shape},
            (7,),
            lambda x, y, dim=0 if transposed else 1: (
                x.astype(np.float64) * y.astype(np.float64)
            ).sum(dim),
            block=32,
        )
    body = (
        Loop(
            axis("k", 3),
            (
                Load("a", "x", (i, k), dtype=F32),
                Load("b", "y", (i, k), dtype=F32),
                Assign("product", op("multiply"), ("a", "b"), F32),
                Accum("state", "product", op("add"), F32, ("k",)),
            ),
        ),
        Write("out", (i,), "state", value_dtype=F32),
    )
    value = case(
        "dot_cancellation",
        (Tile((axis("i", 1),), body),),
        {"x": (1, 3), "y": (1, 3)},
        (1,),
        lambda x, y: (x.astype(np.float64) * y.astype(np.float64)).sum(-1),
        block=32,
    )
    yield with_data(
        value, np.array([[1e8, 1, -1e8]], np.float32), np.ones((1, 3), np.float32), [1]
    )


def observed_dot_cases():
    i, k = Var("i"), Var("k")
    core = (
        Load("a", "x", (i, k), dtype=F32),
        Load("b", "y", (i, k), dtype=F32),
        Assign("product", op("multiply"), ("a", "b"), F32),
        Accum("state", "product", op("add"), F32, ("k",)),
    )
    observed = Select(
        "chosen", (SelectBranch("state", literal(1)), SelectBranch("zero", literal(1)))
    )
    body = (
        Let("zero", literal(0.0), F32),
        Loop(
            axis("k", 3),
            core + (observed, Write("out", (i, k), "chosen", value_dtype=F32)),
        ),
    )
    value = case(
        "observed_select_dot",
        (Tile((axis("i", 1),), body),),
        {"x": (1, 3), "y": (1, 3)},
        (1, 3),
        lambda x, y: np.cumsum(x * y, axis=-1, dtype=np.float32),
        block=32,
    )
    yield with_data(
        value,
        np.array([[1e8, 1, -1e8]], np.float32),
        np.ones((1, 3), np.float32),
        [[1e8, 1e8, 0]],
    )

    bound = TernaryExpr(
        expression("==", Var("state"), literal(0.0)), literal(1), literal(0)
    )
    inner = StridedLoop(
        axis("j", 1),
        literal(0),
        literal(1),
        (Write("out", (i, k), "one", value_dtype=F32),),
        end=bound,
    )
    body = (
        Let("zero", literal(0.0), F32),
        Let("one", literal(1.0), F32),
        Loop(
            axis("k", 3), core + (Write("out", (i, k), "zero", value_dtype=F32), inner)
        ),
    )
    value = case(
        "observed_loop_bound_dot",
        (Tile((axis("i", 1),), body),),
        {"x": (1, 3), "y": (1, 3)},
        (1, 3),
        lambda x, y: (np.cumsum(x * y, axis=-1, dtype=np.float32) == 0).astype(
            np.float32
        ),
        block=32,
    )
    yield with_data(
        value,
        np.array([[1e8, 1, -1e8]], np.float32),
        np.ones((1, 3), np.float32),
        [[0, 0, 1]],
    )


def continuation_cases():
    i, k = Var("i"), Var("k")
    dot = Loop(
        axis("k", 3),
        (
            Load("a", "x", (i, k), dtype=F32),
            Load("b", "y", (i, k), dtype=F32),
            Assign("product", op("multiply"), ("a", "b"), F32),
            Accum("state", "product", op("add"), F32, ("k",)),
        ),
    )
    body = (
        dot,
        Let("seven", literal(7.0), F32),
        Accum("state", "seven", op("add"), F32),
        Write("out", (i,), "state", value_dtype=F32),
    )
    value = case(
        "post_dot_accumulation",
        (Tile((axis("i", 1),), body),),
        {"x": (1, 3), "y": (1, 3)},
        (1,),
        lambda x, y: (x.astype(np.float64) * y.astype(np.float64)).sum(-1) + 7,
        block=32,
    )
    yield with_data(
        value, np.array([[1, 2, 3]], np.float32), np.ones((1, 3), np.float32), [13]
    )
    shuffle = WarpShuffle(
        ("state",),
        ("other",),
        (Assign("state", op("add"), ("state", "other"), F32),),
        2,
        F32,
    )
    more = Loop(
        axis("j", 1),
        (
            Load("a", "x", (i, literal(3)), dtype=F32),
            Load("b", "y", (i, literal(3)), dtype=F32),
            Assign("product", op("multiply"), ("a", "b"), F32),
            Accum("state", "product", op("add"), F32, ("j",)),
        ),
        seed=False,
    )
    body = (dot, shuffle, more, Write("out", (i,), "state", value_dtype=F32))
    value = case(
        "dot_shuffle_dot_continuation",
        (Tile((axis("i", 2),), body),),
        {"x": (2, 4), "y": (2, 4)},
        (2,),
        lambda x, y: (x[:, :3].astype(np.float64) * y[:, :3].astype(np.float64)).sum()
        + x[:, 3] * y[:, 3],
        block=2,
    )
    yield with_data(
        value,
        np.array([[1, 2, 3, 4], [1, 2, 3, 4]], np.float32),
        np.ones((2, 4), np.float32),
        [16, 16],
    )


class DotOptimizationTests(unittest.TestCase):
    def test_independent_dot_recognizer_accepts_layout_variants(self):
        for graph, meta, _ in dot_cases():
            with self.subTest(case=meta["id"]):
                body = next(
                    n.op.body
                    for n in graph.nodes.values()
                    if isinstance(n.op, KernelOp)
                )
                self.assertIn("state", Compiler.independent_dots(body))

    def test_independent_dot_recognizer_rejects_observed_carriers(self):
        for graph, meta, _ in observed_dot_cases():
            with self.subTest(case=meta["id"]):
                body = next(
                    n.op.body
                    for n in graph.nodes.values()
                    if isinstance(n.op, KernelOp)
                )
                self.assertNotIn("state", Compiler.independent_dots(body))

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_optional_dot_mode_assembles(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("accurate_dots", "compensated_dots"):
                for value in (
                    *dot_cases(),
                    *observed_dot_cases(),
                    *continuation_cases(),
                ):
                    with self.subTest(case=value[1]["id"], mode=mode):
                        path = write_case(Path(directory) / value[1]["id"], value)
                        assemble(
                            compile_file(path, **{mode: True}, unroll_budget=512),
                            "sm_89",
                        )

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_optional_dot_mode_numeric_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("accurate_dots", "compensated_dots"):
                for value in (
                    *dot_cases(),
                    *observed_dot_cases(),
                    *continuation_cases(),
                ):
                    with self.subTest(case=value[1]["id"], mode=mode):
                        path = write_case(Path(directory) / value[1]["id"], value)
                        ptx = compile_file(path, **{mode: True}, unroll_budget=512)
                        meta = value[1]
                        kernel = meta["kernels"][0]
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
                            {**request, "ptx": ptx, "adapt_target": True}
                        )
                        self.assertTrue(result.get("ok"), result)
                        produced = {
                            name: np.asarray(
                                output["values"], dtype=np.float32
                            ).reshape(output["shape"])
                            for name, output in result["outputs"].items()
                        }
                        comparisons = compare_outputs(
                            produced, value[2], meta["rtol"], meta["atol"]
                        )
                        self.assertTrue(
                            all(row["pass"] for row in comparisons), comparisons
                        )


if __name__ == "__main__":
    unittest.main()
