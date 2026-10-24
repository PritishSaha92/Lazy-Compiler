"""Contract-edge regressions: counted cooperation, empty folds, integer Exprs.

All named barriers have complete warp participants and matching counts. Integer
expression expectations follow the installed BinaryExpr C operators; they are
deliberately distinct from NumPy floor_divide/remainder elementwise operations.
BinaryExpr.eval uses Python floor semantics, while its renderer maps / and //
to C division (the source assumes nonnegative shape operands). The signed
counterexample explicitly chooses compiled C semantics and records that gap.
"""

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
from emmy.compiler.dtype import F32, I32, I64, U32, U64
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile, TreeHalve
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write
from ir_cases import axis, case, expression, literal, op, write_case
from ptx_backend import compile_file
from tools.check_compiler import assemble, local_run


def explicit_case(name, body, inputs, expected, *, types=None, block=32):
    """Use concrete data and independently computed outputs, retaining the ABI."""
    types = types or {}
    value = case(
        name,
        body,
        {key: data.shape for key, data in inputs.items()},
        np.shape(expected),
        lambda **_: expected,
        block=block,
    )
    graph, metadata, _ = value
    for key, node in graph.nodes.items():
        graph.nodes[key] = replace(
            node,
            outputs=tuple(
                replace(tensor, dtype=types.get(tensor.name, F32))
                for tensor in node.outputs
            ),
        )
    for arg in metadata["kernels"][0]["args"]:
        arg["dtype"] = types.get(arg["name"], F32).name
        if arg["name"] in inputs:
            arg.pop("seed", None)
            arg["data"] = inputs[arg["name"]].ravel().tolist()
    metadata["rtol"] = metadata["atol"] = 0
    return graph, metadata, {"out": np.asarray(expected)}


def contract_edge_cases():
    i, j, k = Var("i"), Var("j"), Var("k")
    first_two_warps = expression("<", i, literal(64))
    # Exactly two of three CTA warps participate; a CTA barrier would deadlock.
    x = np.arange(96, dtype=np.float32) - 17
    body = (
        Smem("scratch", (64,), "float"),
        Load("value", "x", (i,), dtype=F32),
        Cond(
            first_two_warps,
            (
                Write("scratch", (i,), "value", value_dtype=F32),
                Sync(1, 64),
                Load("peer", "scratch", (expression("^", i, literal(32)),), dtype=F32),
                Write("out", (i,), "peer", value_dtype=F32),
            ),
            (Write("out", (i,), "value", value_dtype=F32),),
        ),
    )
    expected = x.copy()
    expected[:64] = x[np.arange(64) ^ 32]
    yield explicit_case(
        "named_barrier_partial_cta",
        (Tile((axis("i", 96),), body),),
        {"x": x},
        expected,
        block=96,
    )

    # Reusing one barrier generation requires read completion before overwrites.
    x = (np.arange(96 * 3 * 2, dtype=np.float32).reshape(96, 3, 2) % 29) - 11
    inner = (
        Load("value", "x", (i, j, k), dtype=F32),
        Write("scratch", (i,), "value", value_dtype=F32),
        Sync(7, 64),
        Load("peer", "scratch", (expression("^", i, literal(32)),), dtype=F32),
        Accum("total", "peer", op("add"), F32),
        Sync(7, 64),
    )
    body = (
        Smem("scratch", (64,), "float"),
        Cond(
            first_two_warps,
            (
                Init("total", 0.0, dtype=F32),
                Loop(
                    axis("j", 3),
                    (Loop(axis("k", 2), inner, unroll=True, seed=False),),
                    unroll=True,
                ),
                Write("out", (i,), "total", value_dtype=F32),
            ),
            (
                Load("value", "x", (i, literal(0), literal(0)), dtype=F32),
                Write("out", (i,), "value", value_dtype=F32),
            ),
        ),
    )
    expected = x[:, 0, 0].copy()
    expected[:64] = x[np.arange(64) ^ 32].sum(axis=(1, 2), dtype=np.float64)
    yield explicit_case(
        "named_barrier_nested_reuse",
        (Tile((axis("i", 96),), body),),
        {"x": x},
        expected,
        block=96,
    )

    # Every tree barrier must preserve the named subgroup count; all its lanes
    # then receive the shared root while a third warp executes a separate path.
    x = np.arange(96, dtype=np.float32) - 40
    merge = (Assign("state", op("add"), ("state", "other"), F32),)
    body = (
        Smem("scratch", (64,), "float"),
        Load("state", "x", (i,), dtype=F32),
        Cond(
            first_two_warps,
            (
                Write("scratch", (i,), "state", value_dtype=F32),
                Sync(3, 64),
                TreeHalve(
                    ("scratch",),
                    ("state",),
                    ("other",),
                    merge,
                    64,
                    "i",
                    F32,
                    barrier_id=3,
                    barrier_count=64,
                ),
                Write("out", (i,), "state", value_dtype=F32),
            ),
            (Write("out", (i,), "state", value_dtype=F32),),
        ),
    )
    expected = x.copy()
    expected[:64] = x[:64].sum(dtype=np.float64)
    yield explicit_case(
        "named_barrier_tree_broadcast",
        (Tile((axis("i", 96),), body),),
        {"x": x},
        expected,
        block=96,
    )

    # Empty seeded folds still expose the registry's identities to consumers.
    x = np.arange(7, dtype=np.float32) - 3
    folds = tuple(
        Accum(name, "value", op(operator), F32)
        for name, operator in (
            ("sum_state", "add"),
            ("product_state", "multiply"),
            ("maximum_state", "maximum"),
            ("minimum_state", "minimum"),
        )
    )
    names = tuple(fold.name for fold in folds)
    body = (
        Loop(axis("k", 0), (Load("value", "x", (i,), dtype=F32),) + folds, unroll=True),
        Write("out", (i, literal(0)), values=names, value_dtype=F32),
    )
    expected = np.tile(np.asarray([0, 1, -1e30, 1e30], np.float32), (7, 1))
    yield explicit_case(
        "loop_zero_seeded_identities",
        (Tile((axis("i", 7),), body),),
        {"x": x},
        expected,
    )

    # A zero-trip continuation leaves Init untouched; a one-trip continuation
    # then updates the same carrier. The saved old value must remain seven.
    update = (Accum("state", "value", op("add"), F32),)
    body = (
        Init("state", 7.0, dtype=F32),
        Load("value", "x", (i,), dtype=F32),
        Loop(axis("k", 0), update, unroll=True, seed=False),
        Let("empty", Var("state"), F32),
        Loop(axis("j", 1), update, unroll=True, seed=False),
        Write("out", (i, literal(0)), values=("empty", "state"), value_dtype=F32),
    )
    yield explicit_case(
        "loop_zero_single_seed_false",
        (Tile((axis("i", 7),), body),),
        {"x": x},
        np.column_stack((np.full(7, 7), 7 + x)),
    )

    # Alternate zero and single-trip loops across threads. Both explicit end
    # and an integer (rather than Expr) step are accepted by installed Emmy.
    x = np.arange(35, dtype=np.float32).reshape(7, 5) - 12
    start = expression("%", i, literal(3))
    end = expression("+", start, expression("%", i, literal(2)))
    body = (
        StridedLoop(
            axis("k", 5),
            start,
            2,
            (
                Load("value", "x", (i, k), dtype=F32),
                Accum("seeded", "value", op("add"), F32),
            ),
            end=end,
        ),
        Init("continued", 9.0, dtype=F32),
        StridedLoop(
            axis("j", 5),
            start,
            2,
            (
                Load("value", "x", (i, j), dtype=F32),
                Accum("continued", "value", op("add"), F32),
            ),
            end=end,
            seed=False,
        ),
        Write("out", (i, literal(0)), values=("seeded", "continued"), value_dtype=F32),
    )
    selected = np.where(np.arange(7) % 2, x[np.arange(7), np.arange(7) % 3], 0)
    yield explicit_case(
        "strided_zero_single_seed_modes",
        (Tile((axis("i", 7),), body),),
        {"x": x},
        np.column_stack((selected, 9 + selected)),
    )

    # The inner fold is empty, nonempty, then empty. Its final identity must
    # replace the previous iteration's inner value instead of leaking it.
    x = np.arange(21, dtype=np.float32).reshape(7, 3) - 8
    inner = StridedLoop(
        axis("k", 1),
        literal(0),
        literal(1),
        (
            Load("value", "x", (i, j), dtype=F32),
            Accum("inner", "value", op("add"), F32),
        ),
        end=expression("%", j, literal(2)),
    )
    body = (
        Loop(
            axis("j", 3), (inner, Accum("outer", "inner", op("add"), F32)), unroll=True
        ),
        Write("out", (i,), "outer", value_dtype=F32),
    )
    yield explicit_case(
        "nested_empty_inner_reset", (Tile((axis("i", 7),), body),), {"x": x}, x[:, 1]
    )

    # Let/BinaryExpr uses ordinary C arithmetic conversions. An unsigned type
    # wins an equal-width rank tie; a wider signed type can represent uint32.
    for signed_type, unsigned_type, bits in ((I32, U32, 32), (I64, U64, 64)):
        signed = np.asarray(
            [-1, -2, 0, 1, 2, (1 << (bits - 1)) - 1, -(1 << (bits - 1))],
            dtype=np.int32 if bits == 32 else np.int64,
        )
        unsigned = np.asarray(
            [1, (1 << bits) - 1, 0, 2, 1, 1 << (bits - 1), 1 << (bits - 1)],
            dtype=np.uint32 if bits == 32 else np.uint64,
        )
        inputs, types = {"x": signed, "y": unsigned}, {
            "x": signed_type,
            "y": unsigned_type,
        }
        instructions = [
            Load("a", "x", (i,), dtype=signed_type),
            Load("b", "y", (i,), dtype=unsigned_type),
            Let("lt", expression("<", Var("a"), Var("b")), F32),
            Let("gt", expression(">", Var("a"), Var("b")), F32),
        ]
        promoted = signed.astype(unsigned.dtype)
        columns, values = [(promoted < unsigned), (promoted > unsigned)], ("lt", "gt")
        if bits == 64:
            narrow = np.asarray([1, 1, 0, 2, 1, 0xFFFFFFFF, 1], dtype=np.uint32)
            inputs["z"], types["z"] = narrow, U32
            instructions.extend(
                (
                    Load("c", "z", (i,), dtype=U32),
                    Let("wide_lt", expression("<", Var("a"), Var("c")), F32),
                )
            )
            columns.append(signed < narrow.astype(np.int64))
            values += ("wide_lt",)
        instructions.append(
            Write("out", (i, literal(0)), values=values, value_dtype=F32)
        )
        yield explicit_case(
            f"integer_mixed_rank_{bits}",
            (Tile((axis("i", 7),), tuple(instructions)),),
            inputs,
            np.column_stack(columns).astype(np.float32),
            types=types,
        )

    # C integer / truncates toward zero and % keeps the dividend's sign.
    # Constant folding must agree with the same expression over a loaded value.
    x = np.asarray([-7, -6, -1, 0, 1, 6, 7], np.int32)
    body = (
        Load("value", "x", (i,), dtype=I32),
        Let("const_q", expression("/", literal(-7), literal(3)), I32),
        Let("const_r", expression("%", literal(-7), literal(3)), I32),
        Let("loaded_q", expression("/", Var("value"), literal(3)), I32),
        Let("loaded_r", expression("%", Var("value"), literal(3)), I32),
        Let("const_slashes", expression("//", literal(-7), literal(3)), I32),
        Let("loaded_slashes", expression("//", Var("value"), literal(3)), I32),
        Write(
            "out",
            (i, literal(0)),
            values=(
                "const_q",
                "const_r",
                "loaded_q",
                "loaded_r",
                "const_slashes",
                "loaded_slashes",
            ),
            value_dtype=I32,
        ),
    )
    q = np.asarray([int(int(item) / 3) for item in x], np.int32)
    expected = np.column_stack(
        (np.full(7, -2), np.full(7, -1), q, x - 3 * q, np.full(7, -2), q)
    ).astype(np.int32)
    yield explicit_case(
        "integer_constant_division_consistency",
        (Tile((axis("i", 7),), body),),
        {"x": x},
        expected,
        types={"x": I32, "out": I32},
    )


class ContractEdgeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_contract_edge_kernels_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in contract_edge_cases():
                for budget in (0, 1024):
                    with self.subTest(case=value[1]["id"], unroll_budget=budget):
                        path = write_case(Path(directory) / value[1]["id"], value)
                        assemble(compile_file(path, unroll_budget=budget), "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_contract_edge_kernels_match_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in contract_edge_cases():
                for budget in (0, 1024):
                    with self.subTest(case=value[1]["id"], unroll_budget=budget):
                        path = write_case(Path(directory) / value[1]["id"], value)
                        ptx = compile_file(path, unroll_budget=budget)
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
