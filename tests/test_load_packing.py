"""Scalar-load packing preserves memory accesses, scope and index definitions."""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy.compiler.dtype import F32, I32
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.stmt.leaves import Assign, Let, Load, Write
from ir_cases import axis, case, expression, literal, op, write_case
from ptx_backend import compile_file
from tools.check_compiler import assemble, local_run


def explicit(name, body, data, expected, *, cells=37, block=64):
    value = case(
        "packing_" + name,
        (Tile((axis("i", cells),), tuple(body)),),
        {key: array.shape for key, array in data.items()},
        np.shape(expected),
        lambda **_: expected,
        block=block,
        grid=(cells + block - 1) // block,
    )
    value[1]["rtol"] = value[1]["atol"] = 0
    for descriptor in value[1]["kernels"][0]["args"]:
        if descriptor["name"] in data:
            descriptor.pop("seed", None)
            descriptor["data"] = data[descriptor["name"]].ravel().tolist()
    return value


def load_packing_cases():
    i, j = Var("i"), Var("j")
    n = 37
    for width in (2, 3, 4):
        for shift in (0, 1):
            x = np.arange(n * width + shift, dtype=np.float32) - 70
            base = expression("*", i, literal(width))
            if shift:
                base = expression("+", base, literal(shift))
            body = []
            for k in range(width):
                body += [
                    Load(f"a{k}", "x", (expression("+", base, literal(k)),), dtype=F32),
                    Assign(f"v{k}", op("negative"), (f"a{k}",), F32),
                ]
            body += [
                Write(
                    "out",
                    (expression("*", i, literal(width)),),
                    values=tuple(f"v{k}" for k in range(width)),
                    value_dtype=F32,
                )
            ]
            yield explicit(f"width{width}_shift{shift}", body, {"x": x}, -x[shift:])

    base = expression("*", i, literal(2))

    def offset(amount):
        return expression("+", base, literal(amount))

    x = np.arange(2 * n + 4, dtype=np.float32) + 2
    y = np.arange(n, dtype=np.float32) + 1000
    body = [
        Load("reuse", "y", (i,), dtype=F32),
        Load("a", "x", (offset(0),), dtype=F32),
        Assign("old", op("copy"), ("reuse",), F32),
        Load("reuse", "x", (offset(1),), dtype=F32),
        Write("out", (base,), values=("old", "reuse"), value_dtype=F32),
    ]
    yield explicit(
        "deferred_name_binding",
        body,
        {"x": x, "y": y},
        np.stack((y, x[1 : 2 * n : 2]), axis=1).ravel(),
    )

    body = [
        Let("j", base, I32),
        Load("a", "x", (expression("+", j, literal(0)),), dtype=F32),
        Let("j", expression("+", j, literal(2)), I32),
        Load("b", "x", (expression("+", j, literal(1)),), dtype=F32),
        Write("out", (base,), values=("a", "b"), value_dtype=F32),
    ]
    yield explicit(
        "redefined_index",
        body,
        {"x": x},
        np.stack((x[: 2 * n : 2], x[3 : 2 * n + 3 : 2]), axis=1).ravel(),
    )

    body[2:3] = [
        Let("delta", literal(2), I32),
        Assign("j", op("add"), ("j", "delta"), I32),
    ]
    yield explicit(
        "assign_redefines_index",
        body,
        {"x": x},
        np.stack((x[: 2 * n : 2], x[3 : 2 * n + 3 : 2]), axis=1).ravel(),
    )

    body = [
        Let("j", base, I32),
        Load("j", "x", (expression("+", j, literal(0)),), dtype=F32),
        Load("b", "x", (expression("+", j, literal(1)),), dtype=F32),
        Write("out", (base,), values=("j", "b"), value_dtype=F32),
    ]
    yield explicit(
        "load_redefines_index",
        body,
        {"x": x},
        np.stack((x[: 2 * n : 2], x[x[: 2 * n : 2].astype(int) + 1]), axis=1).ravel(),
    )

    for conditional in (False, True):
        mutation = Write("x", (offset(1),), "negative", value_dtype=F32)
        if conditional:
            mutation = Cond(expression("<", i, literal(n)), (mutation,))
        body = [
            Load("a", "x", (offset(0),), dtype=F32),
            Assign("negative", op("negative"), ("a",), F32),
            mutation,
            Load("b", "x", (offset(1),), dtype=F32),
            Write("out", (base,), values=("a", "b"), value_dtype=F32),
        ]
        yield explicit(
            "conditional_store_boundary" if conditional else "store_boundary",
            body,
            {"x": x},
            np.stack((x[: 2 * n : 2], -x[: 2 * n : 2]), axis=1).ravel(),
        )

    body = [
        Let("j", Literal(-0.25, "float"), F32),
        Load("a", "x", (expression("+", j, literal(0)),), dtype=F32),
        Load("b", "x", (expression("+", j, literal(1)),), dtype=F32),
        Write("out", (base,), values=("a", "b"), value_dtype=F32),
    ]
    yield explicit(
        "floating_index_declined",
        body,
        {"x": x},
        np.full(2 * n, x[0], dtype=np.float32),
    )

    # Load.dtype describes the stamped source type; it is not a conversion.
    # Inconsistent annotations are deliberately excluded from load packing.
    for types in ((I32, F32), (F32, I32)):
        body = [
            Load("a", "x", (offset(0),), dtype=types[0]),
            Load("b", "x", (offset(1),), dtype=types[1]),
            Write("out", (base,), values=("a", "b"), value_dtype=F32),
        ]
        yield explicit(
            "declared_types_" + "_".join(t.name for t in types),
            body,
            {"x": x},
            x[: 2 * n],
        )

    base4 = expression("*", i, literal(4))
    x4 = np.arange(4 * n, dtype=np.float32) - 80
    body = []
    for k in range(4):
        body += [
            Load("reuse", "x", (expression("+", base4, literal(k)),), dtype=F32),
            Assign(f"saved{k}", op("copy"), ("reuse",), F32),
        ]
    body += [
        Write(
            "out",
            (base4,),
            values=tuple(f"saved{k}" for k in range(4)),
            value_dtype=F32,
        )
    ]
    yield explicit("reused_pending_name", body, {"x": x4}, x4)

    for offsets in ((0, 2), (1, 0), (0, 0)):
        body = [
            Load("a", "x", (offset(offsets[0]),), dtype=F32),
            Load("b", "x", (offset(offsets[1]),), dtype=F32),
            Write("out", (base,), values=("a", "b"), value_dtype=F32),
        ]
        expected = np.stack(
            tuple(x[np.arange(n) * 2 + k] for k in offsets), axis=1
        ).ravel()
        yield explicit(
            "offsets_" + "_".join(map(str, offsets)), body, {"x": x}, expected
        )


class LoadPackingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_load_packing_cases_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in load_packing_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    assemble(compile_file(path), "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_load_packing_cases_match(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in load_packing_cases():
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
                    output = result["outputs"]["out"]
                    actual = np.asarray(output["values"]).reshape(output["shape"])
                    np.testing.assert_array_equal(actual, value[2]["out"])


if __name__ == "__main__":
    unittest.main()
