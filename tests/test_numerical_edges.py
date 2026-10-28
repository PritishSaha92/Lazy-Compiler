"""Independent edge values: fixed-width folding and floating-point classes.

Special-value tests explicitly check NaN/infinity classes and zero signs. This
does not change the workload evaluator's rule that NaNs fail numerical checks.
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
from emmy.compiler.dtype import F32, I16, I32, I64, U16, U32, U64
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Assign, Let, Load, Write
from ir_cases import axis, case, expression, literal, op, write_case
from ptx_backend import compile_file
from tools.check_compiler import assemble, local_run


def integer_constants():
    # Explicit mathematical answers, including unsigned wrap and signed narrowing.
    rows = (
        (-1, U32, "/", 2, 2147483647),
        (-1, U32, "%", 2, 1),
        (-1, U64, "/", 2, 9223372036854775807),
        (-1, U64, "%", 2, 1),
        (4294967297, U32, "+", 2, 3),
        (65537, U16, "+", 2, 3),
        (65535, I16, "/", 2, 0),
        (4294967295, I32, "%", 3, -1),
        (4294967295, U32, "+", 2, 1),
        (4294967295, U32, "*", 3, 4294967293),
    )
    body = []
    for index, (value, dt, operation, rhs, expected) in enumerate(rows):
        body.append(Load("raw", "x", (literal(index),), dtype=I64))
        for column, source in enumerate((Literal(value, I64), Var("raw"))):
            body.extend(
                (
                    Let("input", source, dt),
                    Let(
                        "answer",
                        expression(operation, Var("input"), Literal(rhs, dt)),
                        dt,
                    ),
                    Let(
                        "ok",
                        expression("==", Var("answer"), Literal(expected, dt)),
                        F32,
                    ),
                    Write(
                        "out", (literal(index), literal(column)), "ok", value_dtype=F32
                    ),
                )
            )
    graph, meta, expected = case(
        "fixed_width_constant_runtime",
        (Tile((axis("i", 1),), tuple(body)),),
        {"x": (len(rows),)},
        (len(rows), 2),
        lambda **_: np.ones((len(rows), 2), np.float32),
        block=32,
    )
    node = graph.nodes["x"]
    graph.nodes["x"] = replace(
        node, outputs=tuple(replace(t, dtype=I64) for t in node.outputs)
    )
    arg = meta["kernels"][0]["args"][0]
    arg.update(dtype="i64", data=[row[0] for row in rows])
    arg.pop("seed")
    return graph, meta, expected


def floating_edges():
    smallest = np.nextafter(np.float32(0), np.float32(1))
    tiny = np.finfo(np.float32).tiny
    special = np.asarray(
        [-np.inf, -4, -tiny, -smallest, -0.0, 0.0, smallest, tiny, 4, np.inf, np.nan],
        np.float32,
    )
    # References evaluate the same real operation in float64 then round once.
    # A nonzero tolerance covers hardware transcendental approximation, while
    # exact elementary operations additionally check signed zero and subnormals.
    rows = [
        ("copy", special, lambda x: x, 0.0, True),
        ("negative", special, np.negative, 0.0, True),
        ("abs", special, np.abs, 0.0, True),
        ("sqrt", special, np.sqrt, 0.0, True),
        ("reciprocal", special, lambda x: 1 / x, 0.0, True),
        (
            "rsqrt",
            special,
            lambda x: 1 / np.sqrt(x.astype(np.float32)).astype(np.float64),
            1e-7,
            True,
        ),
        (
            "exp2",
            np.asarray(
                [
                    -np.inf,
                    -150,
                    -149,
                    -148,
                    -126,
                    -1,
                    -0.0,
                    0,
                    1,
                    127,
                    128,
                    np.inf,
                    np.nan,
                ],
                np.float32,
            ),
            np.exp2,
            3e-7,
            False,
        ),
        (
            "exp",
            np.asarray(
                [
                    -np.inf,
                    -104,
                    -100,
                    -90,
                    -20,
                    -1,
                    -0.0,
                    0,
                    1,
                    20,
                    80,
                    88,
                    89,
                    np.inf,
                    np.nan,
                ],
                np.float32,
            ),
            np.exp,
            6e-6,
            False,
        ),
        ("log", special, np.log, 5e-7, False),
        ("log2", special, np.log2, 5e-7, False),
        (
            "tanh",
            np.concatenate(
                (
                    special,
                    np.linspace(-6, 6, 385, dtype=np.float32),
                    np.asarray([2.0**-n for n in range(2, 26)], np.float32),
                )
            ),
            np.tanh,
            5e-7,
            True,
        ),
        (
            "sin",
            np.asarray(
                [
                    -np.inf,
                    -6,
                    -1,
                    -tiny,
                    -smallest,
                    -0.0,
                    0,
                    smallest,
                    tiny,
                    1,
                    6,
                    np.inf,
                    np.nan,
                ],
                np.float32,
            ),
            np.sin,
            2e-6,
            True,
        ),
        (
            "cos",
            np.asarray([-np.inf, -6, -1, -0.0, 0, 1, 6, np.inf, np.nan], np.float32),
            np.cos,
            2e-6,
            False,
        ),
    ]
    for name, inputs, fn, relative, zero_sign in rows:
        with np.errstate(all="ignore"):
            reference = np.asarray(fn(inputs.astype(np.float64)), np.float32)
        i = Var("i")
        body = (
            Load("v", "x", (i,), dtype=F32),
            Assign("answer", op(name), ("v",), F32),
            Write("out", (i,), "answer", value_dtype=F32),
        )
        value = case(
            "edge_" + name,
            (Tile((axis("i", len(inputs)),), body),),
            {"x": inputs.shape},
            inputs.shape,
            lambda **_: reference,
            block=32,
            grid=(len(inputs) + 31) // 32,
        )
        arg = value[1]["kernels"][0]["args"][0]
        arg.pop("seed")
        arg["data"] = inputs.tolist()
        yield value, relative, zero_sign


class NumericalEdgeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_edge_programs_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in [integer_constants(), *(v for v, _, _ in floating_edges())]:
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    assemble(compile_file(path), "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_edges_match_independent_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value, relative, zero_sign in [
                (integer_constants(), 0, False),
                *floating_edges(),
            ]:
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    kernel = value[1]["kernels"][0]
                    request = {
                        k: kernel[k]
                        for k in (
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
                    output = result["outputs"]["out"]
                    actual = np.asarray(output["values"], np.float32).reshape(
                        output["shape"]
                    )
                    expected = value[2]["out"]
                    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                    np.testing.assert_array_equal(
                        np.isposinf(actual), np.isposinf(expected)
                    )
                    np.testing.assert_array_equal(
                        np.isneginf(actual), np.isneginf(expected)
                    )
                    finite = np.isfinite(expected)
                    # Approximate functions may differ by two subnormal ULPs;
                    # this bound cannot hide a flushed minimum normal value.
                    absolute = (
                        2 * float(np.nextafter(np.float32(0), np.float32(1)))
                        if relative
                        else 0
                    )
                    np.testing.assert_allclose(
                        actual[finite], expected[finite], rtol=relative, atol=absolute
                    )
                    if zero_sign:
                        zeros = finite & (expected == 0)
                        np.testing.assert_array_equal(
                            np.signbit(actual[zeros]), np.signbit(expected[zeros])
                        )


if __name__ == "__main__":
    unittest.main()
