"""Counterexamples for compensated independent-dot accumulation.

The expected values come from exact integer identities, not emitted arithmetic.
These examples exercise product rounding, catastrophic cancellation, persistent
state across slab loops, scope reset, and transition to a cooperative reduction.
"""

import os
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Tile, WarpShuffle
from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import (
    Accum,
    Assign,
    Init,
    Let,
    Load,
    Select,
    SelectBranch,
    Write,
)
from ir_cases import axis, case, expression, literal, op, write_case
from ptx_backend import Compiler, compile_file
from tools.check_compiler import assemble, local_run


def dot_body(index):
    return (
        Load("a", "x", index, dtype=F32),
        Load("b", "y", index, dtype=F32),
        Assign("product", op("multiply"), ("a", "b"), F32),
        Accum("dot", "product", op("add"), F32),
    )


def explicit_case(name, body, x, y, expected):
    value = case(
        name,
        (Tile((axis("i", len(x)),), body),),
        {"x": x.shape, "y": y.shape},
        np.shape(expected),
        lambda **_: expected,
        block=8,
    )
    for arg in value[1]["kernels"][0]["args"]:
        if arg["name"] in ("x", "y"):
            arg.pop("seed")
            arg["data"] = (x if arg["name"] == "x" else y).ravel().tolist()
    value[1]["rtol"] = value[1]["atol"] = 0
    return value


def precision_cases():
    i, j, k = Var("i"), Var("j"), Var("k")
    # 4097**2 - 4096*4098 == 1, but separate float32 products sum to 0.
    # Other rows test an addition residual: 1e8 + n - 1e8 == n.
    x = np.tile(np.asarray([4097, -4096, 0, 0], dtype=np.float32), (8, 1))
    y = np.tile(np.asarray([4097, 4098, 1, 1], dtype=np.float32), (8, 1))
    x[4:] = [[1e8, n, -1e8, 0] for n in range(1, 5)]
    y[4:] = 1
    expected = np.asarray([1, 1, 1, 1, 1, 2, 3, 4], dtype=np.float32)
    body = (
        Loop(axis("k", 4), dot_body((i, k)), unroll=True),
        Write("out", (i,), "dot", value_dtype=F32),
    )
    yield explicit_case("dot_product_and_sum_residual", body, x, y, expected)

    # Corrections from the first slab must survive seed=False drain loops.
    x = np.asarray([[1e8, n, -1e8, 0] for n in range(1, 9)], dtype=np.float32)
    y = np.ones_like(x)
    offset = expression("+", expression("*", j, literal(2)), k)
    body = (
        Init("dot", 0.0, dtype=F32),
        Loop(
            axis("j", 2),
            (Loop(axis("k", 2), dot_body((i, offset)), unroll=True, seed=False),),
        ),
        Write("out", (i,), "dot", value_dtype=F32),
    )
    yield explicit_case(
        "dot_seed_false_slabs", body, x, y, np.arange(1, 9, dtype=np.float32)
    )

    # Every inner dot needs a new zero correction when the outer loop repeats.
    x = np.asarray(
        [[[1e8, n, -1e8], [1e8, 2 * n, -1e8]] for n in range(1, 9)], dtype=np.float32
    )
    y = np.ones_like(x)
    body = (
        Loop(
            axis("j", 2),
            (
                Loop(axis("k", 3), dot_body((i, j, k)), unroll=True),
                Accum("total", "dot", op("add"), F32),
            ),
        ),
        Write("out", (i,), "total", value_dtype=F32),
    )
    yield explicit_case(
        "dot_nested_reset", body, x, y, 3 * np.arange(1, 9, dtype=np.float32)
    )

    # Cooperative state must receive hi+lo once, then discard its old correction.
    x = np.asarray([[1e8, n, -1e8] for n in range(1, 9)], dtype=np.float32)
    y = np.ones_like(x)
    merge = (Assign("dot", op("add"), ("dot", "other"), F32),)
    body = (
        Loop(axis("k", 3), dot_body((i, k)), unroll=True),
        Let("before", Var("dot"), F32),
        WarpShuffle(("dot",), ("other",), merge, 8, F32),
        Write("out", (i, literal(0)), values=("before", "dot"), value_dtype=F32),
    )
    expected = np.column_stack((np.arange(1, 9), np.full(8, 36))).astype(np.float32)
    yield explicit_case("dot_shuffle_snapshot", body, x, y, expected)

    # Input width does not determine accumulator width: half inputs can feed an
    # explicitly float32 multiply and dot. Compensation still updates its state.
    x = np.asarray([[2048, n, -2048] for n in range(1, 9)], dtype=np.float32)
    y = np.ones_like(x)
    inner = (
        Load("a", "x", (i, k), dtype=F16),
        Load("b", "y", (i, k), dtype=F16),
        Assign("product", op("multiply"), ("a", "b"), F32),
        Accum("dot", "product", op("add"), F32),
    )
    body = (
        Loop(axis("k", 3), inner, unroll=True),
        Write("out", (i,), "dot", value_dtype=F32),
    )
    value = explicit_case(
        "dot_half_inputs_float_state", body, x, y, np.arange(1, 9, dtype=np.float32)
    )
    graph, metadata, _ = value
    for name, node in graph.nodes.items():
        graph.nodes[name] = replace(
            node,
            outputs=tuple(
                replace(tensor, dtype=F16) if tensor.name in ("x", "y") else tensor
                for tensor in node.outputs
            ),
        )
    for arg in metadata["kernels"][0]["args"]:
        if arg["name"] in ("x", "y"):
            arg["dtype"] = "f16"
    yield value

    # A standalone update can precede the independent seeded-false dot region.
    x = np.asarray([[1e8, n, -1e8] for n in range(1, 9)], dtype=np.float32)
    y = np.ones_like(x)
    body = (
        Init("dot", 0.0, dtype=F32),
        Let("initial", literal(5.0), F32),
        Accum("dot", "initial", op("add"), F32),
        Loop(axis("k", 3), dot_body((i, k)), unroll=True, seed=False),
        Write("out", (i,), "dot", value_dtype=F32),
    )
    yield explicit_case(
        "dot_standalone_initial_update", body, x, y, np.arange(6, 14, dtype=np.float32)
    )

    # A later explicit-base recurrence has to consume, then replace, the dot.
    body = (
        Loop(axis("k", 3), dot_body((i, k)), unroll=True),
        Let("half", literal(0.5), F32),
        Let("one", literal(1.0), F32),
        Let("two", literal(2.0), F32),
        Assign("scaled", op("multiply"), ("dot", "half"), F32),
        Assign("product", op("multiply"), ("one", "two"), F32),
        Accum("dot", "product", op("add"), F32, base="scaled"),
        Write("out", (i,), "dot", value_dtype=F32),
    )
    yield explicit_case(
        "dot_followed_by_explicit_base",
        body,
        x,
        y,
        np.arange(1, 9, dtype=np.float32) / 2 + 2,
    )


class PrecisionTests(unittest.TestCase):
    def test_dot_read_by_select_is_not_transformed(self):
        i, k = Var("i"), Var("k")
        selector = Select("selected", (SelectBranch("dot", literal(True)),))
        body = (Loop(axis("k", 4), dot_body((i, k)) + (selector,)),)
        self.assertNotIn("dot", Compiler.independent_dots(body))

    def test_dot_read_by_nested_loop_bound_is_not_transformed(self):
        i, k = Var("i"), Var("k")
        nested = StridedLoop(axis("j", 4), literal(0), literal(1), (), end=Var("dot"))
        body = (Loop(axis("k", 4), dot_body((i, k)) + (nested,)),)
        self.assertNotIn("dot", Compiler.independent_dots(body))

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_precision_cases_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in precision_cases():
                for budget in (0, 128):
                    with self.subTest(case=value[1]["id"], unroll_budget=budget):
                        path = write_case(Path(directory) / value[1]["id"], value)
                        ptx = compile_file(
                            path, compensated_dots=True, unroll_budget=budget
                        )
                        assemble(ptx, "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_precision_cases_match_exact_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in precision_cases():
                for budget in (0, 128):
                    with self.subTest(case=value[1]["id"], unroll_budget=budget):
                        path = write_case(Path(directory) / value[1]["id"], value)
                        ptx = compile_file(
                            path, compensated_dots=True, unroll_budget=budget
                        )
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
                            {**request, "ptx": ptx, "adapt_target": True}
                        )
                        self.assertTrue(result.get("ok"), result)
                        for name, expected in value[2].items():
                            out = result["outputs"][name]
                            actual = np.asarray(out["values"]).reshape(out["shape"])
                            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
