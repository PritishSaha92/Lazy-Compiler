"""Additional independently constructed semantic regressions.

These cases target IR fields absent from the open suite. Set EMMY_GPU_TESTS=1
to run them through the isolated CUDA driver worker as well as the assembler.
"""

import os
import math
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy.compiler.dtype import F16, F32, I32
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt.blocks import StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write
from ir_cases import axis, case, expression, literal, op, write_case
from tools.check_compiler import assemble, compare_outputs, compile_ptx, local_run


def semantic_cases():
    # A warp-indexed tree can have a distinct inner column for EVERY lane.
    # Restricting tree workers to lane zero silently drops 31 of 32 reductions.
    warp, lane = Var("warp"), Var("lane")
    combine = (Assign("acc", op("add"), ("acc", "other"), F32),)
    body = (
        Smem("scratch", (2, 32), "float"),
        Load("acc", "x", (warp, lane), dtype=F32),
        Write("scratch", (warp, lane), "acc", value_dtype=F32),
        Sync(),
        TreeHalve(
            ("scratch",),
            ("acc",),
            ("other",),
            combine,
            2,
            "warp",
            F32,
            inner=("lane", 32),
        ),
        Write("out", (warp, lane), "acc", value_dtype=F32),
    )
    yield case(
        "warp_tree_inner_lane",
        (Tile((axis("i", 64),), body),),
        {"x": (2, 32)},
        (2, 32),
        lambda x: np.broadcast_to(x.astype(np.float64).sum(axis=0), (2, 32)),
        block=64,
    )

    # The explicit end value must be available under the documented binding.
    i, k = Var("i"), Var("k")
    body = (
        StridedLoop(
            axis("k", 7),
            literal(0),
            literal(1),
            (
                Load("v", "x", (i, k), dtype=F32),
                Let("count", Var("k_end"), I32),
                Assign("scaled", op("divide"), ("v", "count"), F32),
                Accum("acc", "scaled", op("add"), F32),
            ),
            end=expression("+", expression("%", i, literal(7)), literal(1)),
        ),
        Write("out", (i,), "acc", value_dtype=F32),
    )
    yield case(
        "strided_end_binding",
        (Tile((axis("i", 9),), body),),
        {"x": (9, 7)},
        (9,),
        lambda x: np.asarray([row[: 1 + n % 7].mean() for n, row in enumerate(x)]),
        block=32,
    )

    # Subgroup membership survives uneven loop exits before the shuffle.
    combine = (Assign("acc", op("add"), ("acc", "other"), F32),)
    body = (
        StridedLoop(
            axis("k", 5),
            literal(0),
            literal(1),
            (Load("v", "x", (i, k), dtype=F32), Accum("acc", "v", op("add"), F32)),
            end=expression("+", expression("%", i, literal(5)), literal(1)),
        ),
        WarpShuffle(("acc",), ("other",), combine, 8, F32),
        Write("out", (i,), "acc", value_dtype=F32),
    )

    def uneven_reference(x):
        totals = np.asarray(
            [row[: 1 + n % 5].astype(np.float64).sum() for n, row in enumerate(x)]
        )
        return np.repeat(totals.reshape(5, 8).sum(axis=1), 8)

    yield case(
        "shuffle_uneven_loop_exit",
        (Tile((axis("i", 40),), body),),
        {"x": (40, 5)},
        (40,),
        uneven_reference,
        block=40,
    )

    body = (
        Load("acc", "x", (i,), dtype=F32),
        Let("old", Var("acc"), F32),
        WarpShuffle(("acc",), ("other",), combine, 8, F32),
        Write("out", (i, literal(0)), values=("old", "acc"), value_dtype=F32),
    )
    yield case(
        "shuffle_loaded_state_snapshot",
        (Tile((axis("i", 8),), body),),
        {"x": (8,)},
        (8, 2),
        lambda x: np.column_stack((x, np.full(8, x.astype(np.float64).sum()))),
        block=8,
    )

    for name, reference in (
        ("exp_fast", np.exp),
        ("bitwise_not", np.logical_not),
        ("erf", np.vectorize(math.erf)),
    ):
        body = (
            Load("v", "x", (i,), dtype=F32),
            Assign("result", op(name), ("v",), F32),
            Write("out", (i,), "result", value_dtype=F32),
        )
        value = case(
            "operator_" + name,
            (Tile((axis("i", 17),), body),),
            {"x": (17,)},
            (17,),
            lambda x, fn=reference: fn(x),
            block=32,
        )
        if name == "bitwise_not":
            values = np.asarray(
                [0, 1, -1, -0.0, 2, 0, -2, 3, 0, 4, -4, 0, 5, 0, -5, 6, 0],
                dtype=np.float32,
            )
            arg = value[1]["kernels"][0]["args"][0]
            arg.pop("seed")
            arg["data"] = values.tolist()
            value[2]["out"] = np.logical_not(values).astype(np.float32)
        yield value

    # pow(-x, 2) is defined and positive. A log2/exp2 lowering loses the sign
    # domain and returns NaN for every negative base.
    body = (
        Load("v", "x", (i,), dtype=F32),
        Let("two", Literal(2.0), F32),
        Assign("result", op("power"), ("v", "two"), F32),
        Write("out", (i,), "result", value_dtype=F32),
    )
    yield case(
        "power_signed_square",
        (Tile((axis("i", 17),), body),),
        {"x": (17,)},
        (17,),
        lambda x: np.power(x, np.float32(2.0)),
        block=32,
    )

    body = (
        Load("v", "x", (i,), dtype=F32),
        Let("three", Literal(3.0), F32),
        Assign("result", op("power"), ("v", "three"), F32),
        Write("out", (i,), "result", value_dtype=F32),
    )
    yield case(
        "power_signed_cube",
        (Tile((axis("i", 17),), body),),
        {"x": (17,)},
        (17,),
        lambda x: np.power(x, np.float32(3.0)),
        block=32,
    )

    # SSA copies must preserve their value before a mutable accumulator changes.
    body = (
        Init("state", 0.0, dtype=F32),
        Load("v", "x", (i,), dtype=F32),
        Accum("state", "v", op("add"), F32),
        Assign("old_assign", op("copy"), ("state",), F32),
        Let("old_let", Var("state"), F32),
        Accum("state", "v", op("add"), F32),
        Write(
            "out", (i, literal(0)), values=("old_assign", "old_let"), value_dtype=F32
        ),
    )
    yield case(
        "accumulator_snapshot",
        (Tile((axis("i", 17),), body),),
        {"x": (17,)},
        (17, 2),
        lambda x: np.column_stack((x, x)),
        block=32,
    )

    # PTX selects half payloads as b16, not an unsupported selp.f16.
    body = (
        Load("a", "x", (i,), dtype=F16),
        Load("b", "y", (i,), dtype=F16),
        Let("zero", Literal(0.0), F16),
        Assign("negative", op("less"), ("a", "zero"), F32),
        Assign("selected", op("where"), ("negative", "a", "b"), F16),
        Write("out", (i,), "selected", value_dtype=F16),
    )
    value = case(
        "half_where",
        (Tile((axis("i", 17),), body),),
        {"x": (17,), "y": (17,)},
        (17,),
        lambda x, y: np.where(
            x.astype(np.float16) < 0, x.astype(np.float16), y.astype(np.float16)
        ),
        block=32,
    )
    graph, meta, expected = value
    for name, node in graph.nodes.items():
        graph.nodes[name] = replace(
            node, outputs=tuple(replace(tensor, dtype=F16) for tensor in node.outputs)
        )
    for arg in meta["kernels"][0]["args"]:
        arg["dtype"] = "f16"
    yield graph, meta, expected

    # Integer remainder follows the divisor's sign, like NumPy's operator.
    data = np.asarray([-7, -3, -1, 0, 1, 3, 7], dtype=np.int32)
    body = (
        Load("v", "x", (i,), dtype=I32),
        Let("three", Literal(3, "int"), I32),
        Assign("r", op("remainder"), ("v", "three"), I32),
        Write("out", (i,), "r", value_dtype=I32),
    )
    graph, meta, _ = case(
        "signed_integer_remainder",
        (Tile((axis("i", 7),), body),),
        {"x": (7,)},
        (7,),
        lambda x: x,
        block=32,
    )
    for name, node in graph.nodes.items():
        graph.nodes[name] = replace(
            node, outputs=tuple(replace(tensor, dtype=I32) for tensor in node.outputs)
        )
    for arg in meta["kernels"][0]["args"]:
        arg["dtype"] = "i32"
        if arg["name"] == "x":
            arg.pop("seed")
            arg["data"] = data.tolist()
    yield graph, meta, {"out": np.remainder(data, 3)}


class SemanticRegressionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_semantic_fixtures_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in semantic_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    ptx, _ = compile_ptx(ROOT / "compile.py", path)
                    assemble(ptx, "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_semantic_fixtures_match_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in semantic_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    ptx, _ = compile_ptx(ROOT / "compile.py", path)
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
                    result = local_run({**request, "ptx": ptx, "adapt_target": True})
                    self.assertTrue(result.get("ok"), result)
                    actual = {
                        name: np.asarray(output["values"]).reshape(output["shape"])
                        for name, output in result["outputs"].items()
                    }
                    checks = compare_outputs(
                        actual, value[2], meta["rtol"], meta["atol"]
                    )
                    self.assertTrue(all(row["pass"] for row in checks), checks)


if __name__ == "__main__":
    unittest.main()
