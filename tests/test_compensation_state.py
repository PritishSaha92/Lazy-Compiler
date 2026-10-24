"""Dot compensation across nonfinite values, observations, resets and collectives."""

import math
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
from emmy.compiler.ir.kernel.ir import Tile, WarpShuffle
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write
from ir_cases import axis, case, literal, op, write_case
from ptx_backend import compile_file
from tools.check_compiler import assemble, local_run


def compensation_state_cases():
    rows, length = 16, 16
    x = np.zeros((rows, length), dtype=np.float32)
    y = np.ones_like(x)
    x[0, :3] = [1e20, 1, -1e20]
    # The rounded FP32 product loses the 2^-24 residual. An independently
    # computed exact binary product/sum must recover it through compensation.
    a = np.float32(1 + 2**-12)
    x[1, :2] = [a, -np.float32(a * a)]
    y[1, :2] = [a, 1]
    x[2] = 1
    x[3, ::2], x[3, 1::2] = 1, -1
    x[4, :8] = np.nextafter(np.float32(0), np.float32(1))
    x[5] = -0.0
    x[6, 0], x[7, 0] = np.inf, -np.inf
    x[8, :2] = [np.inf, -np.inf]
    x[9, 0] = np.nan
    y[10, 0] = np.inf  # 0 * infinity is NaN.
    x[11, 0], y[11, 0] = np.float32(3e38), 2
    x[12, :4] = np.float32(3e38) * np.asarray([1, 1, -1, -1], np.float32)
    x[13, :4] = -x[12, :4]
    x[14, :3] = [np.float32(3e38), np.float32(3e38), -np.inf]
    x[15] = np.arange(length, dtype=np.float32) - 8
    expected = np.empty(rows, np.float32)
    for row in [0, 1, 2, 3, 4, 5, 15]:
        expected[row] = np.float32(
            math.fsum(float(a) * float(b) for a, b in zip(x[row], y[row]))
        )
    # These expected classes intentionally describe the serial rounded high
    # state. A finite-input overflowing prefix is not repaired into a finite
    # mathematical sum; this test checks preservation of that existing rule.
    expected[6:15] = [
        np.inf,
        -np.inf,
        np.nan,
        np.nan,
        np.nan,
        np.inf,
        np.inf,
        -np.inf,
        np.nan,
    ]
    z = np.full((rows, 4), 3, dtype=np.float32)
    w = np.ones_like(z)
    row, k, q = Var("row"), Var("k"), Var("q")

    def dot_loop(xname="x", yname="y", count=length, iv=k, seed=True):
        return Loop(
            axis(iv.name, count),
            (
                Load("a", xname, (row, iv), dtype=F32),
                Load("b", yname, (row, iv), dtype=F32),
                Assign("product", op("multiply"), ("a", "b"), F32),
                Accum("dot", "product", op("add"), F32),
            ),
            unroll=True,
            seed=seed,
        )

    for mode in (
        "direct",
        "continuation",
        "init_reset",
        "base_reset",
        "reseed",
        "empty_reseed",
    ):
        body = [dot_loop()]
        data = {"x": x, "y": y}
        if mode == "direct":
            body += [Write("out", (row,), "dot", value_dtype=F32)]
            reference = expected
        else:
            body += [Assign("observed", op("copy"), ("dot",), F32)]
            data.update(z=z, w=w)
            if mode == "init_reset":
                body += [Init("dot", 2.0, dtype=F32)]
            elif mode == "base_reset":
                body += [
                    Let("replacement", literal(2.0), F32),
                    Let("zero", literal(0.0), F32),
                    Accum("dot", "replacement", op("add"), F32, base="zero"),
                ]
            seed = mode in ("reseed", "empty_reseed")
            body += [dot_loop("z", "w", 0 if mode == "empty_reseed" else 4, q, seed)]
            body += [
                Write(
                    "out",
                    (row * literal(2),),
                    values=("observed", "dot"),
                    value_dtype=F32,
                )
            ]
            if mode == "continuation":
                with np.errstate(all="ignore"):
                    final = expected + np.float32(12)
            else:
                final = np.full(
                    rows,
                    0 if mode == "empty_reseed" else 12 if mode == "reseed" else 14,
                    np.float32,
                )
            reference = np.stack((expected, final), axis=1).ravel()
        value = case(
            "compensation_state_" + mode,
            (Tile((axis("row", rows),), tuple(body)),),
            {key: array.shape for key, array in data.items()},
            reference.shape,
            lambda **_: reference,
            block=32,
        )
        for descriptor in value[1]["kernels"][0]["args"]:
            if descriptor["name"] in data:
                descriptor.pop("seed", None)
                descriptor["data"] = data[descriptor["name"]].ravel().tolist()
        value[1]["rtol"] = value[1]["atol"] = 0
        yield value

    xs = np.zeros_like(x)
    xs[0, 0], xs[8, 0] = np.inf, np.nan
    body = (
        dot_loop(),
        WarpShuffle(
            ("dot",),
            ("peer",),
            (Assign("dot", op("add"), ("dot", "peer"), F32),),
            8,
            F32,
        ),
        Write("out", (row,), "dot", value_dtype=F32),
    )
    reference = np.asarray([np.inf] * 8 + [np.nan] * 8, np.float32)
    value = case(
        "compensation_state_shuffle",
        (Tile((axis("row", rows),), body),),
        {"x": xs.shape, "y": y.shape},
        reference.shape,
        lambda **_: reference,
        block=32,
    )
    for descriptor in value[1]["kernels"][0]["args"]:
        if descriptor["name"] in ("x", "y"):
            descriptor.pop("seed", None)
            descriptor["data"] = (
                (xs if descriptor["name"] == "x" else np.ones_like(y)).ravel().tolist()
            )
    value[1]["rtol"] = value[1]["atol"] = 0
    yield value


class CompensationStateTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_compensation_states_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in compensation_state_cases():
                path = write_case(Path(directory) / value[1]["id"], value)
                for budget in (0, 1024):
                    with self.subTest(case=value[1]["id"], budget=budget):
                        assemble(
                            compile_file(
                                path, compensated_dots=True, unroll_budget=budget
                            ),
                            "sm_89",
                        )

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_compensation_states_match_exact_values_and_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in compensation_state_cases():
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
                for budget in (0, 1024):
                    with self.subTest(case=value[1]["id"], budget=budget):
                        ptx = compile_file(
                            path, compensated_dots=True, unroll_budget=budget
                        )
                        result = local_run(
                            {**request, "ptx": ptx, "adapt_target": True}
                        )
                        self.assertTrue(result.get("ok"), result)
                        output = result["outputs"]["out"]
                        actual = np.asarray(output["values"], np.float32).reshape(
                            output["shape"]
                        )
                        expected = value[2]["out"]
                        for classify in (np.isnan, np.isposinf, np.isneginf):
                            np.testing.assert_array_equal(
                                classify(actual), classify(expected)
                            )
                        finite = np.isfinite(expected)
                        np.testing.assert_array_equal(
                            actual[finite].view(np.uint32),
                            expected[finite].view(np.uint32),
                        )


if __name__ == "__main__":
    unittest.main()
