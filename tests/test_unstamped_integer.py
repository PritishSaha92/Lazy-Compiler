"""Mixed integer Assign inference and C expression shift promotions.

All runner descriptors are signed i32/i64. Narrow and unsigned operand types
are explicit IR conversions, and all expected values come from fixed integer
examples. Shift counts remain inside the promoted left operand's width.
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy.compiler.dtype import F32, I16, I32, I64, U32, U64
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Assign, Let, Load, Write
from ir_cases import axis, expression, literal, op, write_case
from ptx_backend import CompileError, Compiler, Value, compile_file
from test_contract_edges import explicit_case
from tools.check_compiler import assemble, local_run


def unstamped_integer_cases():
    i = Var("i")
    for name, operation, x, y, answer in (
        ("unstamped_mixed_integer_add", "add", 1, 16777217, 16777218),
        ("unstamped_mixed_integer_and", "bitwise_and", 255, 16777470, 254),
        ("unstamped_mixed_integer_xor", "bitwise_xor", 255, 16777217, 16777470),
    ):
        body = (
            Load("raw_a", "x", (i,), dtype=I32),
            Let("a", Var("raw_a"), I16),
            Load("b", "y", (i,), dtype=I32),
            Assign("v", op(operation), ("a", "b")),
            Write("out", (i,), "v", value_dtype=I32),
        )
        yield explicit_case(
            name, (Tile((axis("i", 1),), body),),
            {"x": np.asarray([x], np.int32), "y": np.asarray([y], np.int32)},
            np.asarray([answer], np.int32),
            types={"x": I32, "y": I32, "out": I32},
        )


def shift_promotion_cases():
    examples = (
        ("shift_unsigned_left_wide_count", U32, I64, "<<",
         [-1, 2147483647, 1073741824, 1], [1, 1, 2, 31],
         [4294967294, 4294967294, 0, 2147483648]),
        ("shift_signed_right_unsigned_count", I32, U64, ">>",
         [-4, -2147483648, -1, 2147483647], [1, 31, 0, 2],
         [-2, -1, -1, 536870911]),
        ("shift_narrow_left_promotion", I16, U64, ">>",
         [-4, -32768, -1, 32767], [1, 15, 0, 3],
         [-2, -1, -1, 4095]),
    )
    for name, left_type, right_type, operator, inputs, counts, answers in examples:
        body = []
        for row, (x, count) in enumerate(zip(inputs, counts)):
            body.extend((
                Load("raw_a", "x", (literal(row),), dtype=I32),
                Let("a", Var("raw_a"), left_type),
                Load("raw_b", "n", (literal(row),), dtype=I64),
                Let("b", Var("raw_b"), right_type),
                Let("runtime", expression(operator, Var("a"), Var("b")), I64),
                Let("constant", expression(operator, Literal(x, left_type),
                                           Literal(count, right_type)), I64),
                Write("out", (literal(row), literal(0)),
                      values=("runtime", "constant"), value_dtype=I64),
            ))
        yield explicit_case(
            name, (Tile((axis("i", 1),), tuple(body)),),
            {"x": np.asarray(inputs, np.int32), "n": np.asarray(counts, np.int64)},
            np.repeat(np.asarray(answers, np.int64)[:, None], 2, axis=1),
            types={"x": I32, "n": I64, "out": I64},
        )


def stamped_control_case():
    # An explicit floating result intentionally retains floating rounding and
    # the historical logical-mask spelling. The inference fix must not alter it.
    body = (
        Let("a", Literal(1, I16), I16),
        Let("b", Literal(16777217, I32), I32),
        Assign("floating_add", op("add"), ("a", "b"), F32),
        Assign("integer_add", op("add"), ("a", "b"), I64),
        Let("mask", Literal(255, I16), I16),
        Let("bits", Literal(16777470, I32), I32),
        Assign("logical_and", op("bitwise_and"), ("mask", "bits"), F32),
        Assign("integer_and", op("bitwise_and"), ("mask", "bits"), I32),
        Write("out", (literal(0),),
              values=("floating_add", "integer_add", "logical_and", "integer_and"),
              value_dtype=I64),
    )
    return explicit_case(
        "stamped_integer_policy_controls", (Tile((axis("i", 1),), body),), {},
        np.asarray([16777216, 16777218, 1, 254], np.int64), types={"out": I64},
    )


def integer_inference_cases():
    yield from unstamped_integer_cases()
    yield from shift_promotion_cases()
    yield stamped_control_case()


def export(folder):
    return [write_case(Path(folder) / case[1]["id"], case)
            for case in integer_inference_cases()]


class UnstampedIntegerTests(unittest.TestCase):
    def compiler(self):
        graph, meta, _ = stamped_control_case()
        return Compiler(graph, meta["kernels"][0])

    def test_integer_rank_and_sign_without_float_fallback(self):
        for left, right, expected in (
            ("s16", "s32", "s32"), ("u16", "u32", "u32"),
            ("s16", "u16", "u16"), ("s32", "u16", "s32"),
            ("s64", "u32", "s64"), ("u64", "s32", "u64"),
            ("u8", "u16", "u16"), ("s16", "u8", "s16"),
        ):
            for operation in ("add", "subtract", "multiply", "bitwise_and",
                              "bitwise_or", "bitwise_xor", "left_shift", "right_shift"):
                with self.subTest(left=left, right=right, operation=operation):
                    args = [Value("a", left), Value("b", right)]
                    self.assertEqual(Compiler.assign_type(operation, args), expected)
                    self.assertEqual(Compiler.assign_type(operation, args[::-1]), expected)
        # These unstamped policies are intentionally unchanged by this fix.
        for operation, types, expected in (
            ("equal", ("s16", "s32"), "f32"),
            ("add", ("f32", "s32"), "f32"),
            ("add", ("f16", "f16"), "f16"),
            ("add", ("f64", "f64"), "f64"),
        ):
            self.assertEqual(Compiler.assign_type(operation, [Value("a", t) for t in types]), expected)

    def test_constant_shift_result_uses_promoted_left_type(self):
        c = self.compiler()
        for operator, left, right, expected_type, expected in (
            ("<<", c.literal(-1, "u32"), c.literal(1, "s64"), "u32", 4294967294),
            (">>", c.literal(-4, "s32"), c.literal(1, "u64"), "s32", -2),
            (">>", c.literal(-4, "s16"), c.literal(1, "u64"), "s32", -2),
        ):
            result = c.binary(operator, left, right)
            self.assertEqual((result.type, result.constant), (expected_type, expected))
        # A named operation with an explicit wide result retains that contract.
        wide = c.operation("left_shift", [c.literal(-1, "u32"), c.literal(1, "u64")], "u64")
        self.assertEqual((wide.type, wide.constant), ("u64", 8589934590))

    def test_shift_rejects_floating_operands(self):
        c = self.compiler()
        for operator in ("<<", ">>"):
            for args in ((c.literal(4, "s32"), c.literal(1, "f32")),
                         (c.literal(4, "f32"), c.literal(1, "s32"))):
                with self.assertRaisesRegex(CompileError, "requires integer operands"):
                    c.binary(operator, *args)

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_portable_integer_programs_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in export(directory):
                with self.subTest(case=path.parent.name):
                    assemble(compile_file(path), "sm_89")

    @unittest.skipUnless(os.environ.get("EMMY_GPU_TESTS") == "1",
                         "set EMMY_GPU_TESTS=1 for CUDA execution")
    def test_integer_programs_match_independent_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in integer_inference_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    kernel = value[1]["kernels"][0]
                    request = {key: kernel[key] for key in
                               ("entry", "grid", "block", "smem_bytes", "zero_outputs", "args")}
                    result = local_run({**request, "ptx": compile_file(path), "adapt_target": True})
                    self.assertTrue(result.get("ok"), result)
                    out = result["outputs"]["out"]
                    actual = np.asarray(out["values"], np.int64).reshape(out["shape"])
                    np.testing.assert_array_equal(actual, value[2]["out"])


if __name__ == "__main__":
    if "--export" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--export", type=Path, required=True)
        destination = parser.parse_args().export
        if destination.exists():
            parser.error(f"refusing to replace existing {destination}")
        paths = export(destination)
        print(f"Exported {len(paths)} independent-reference packages to {destination}")
    else:
        unittest.main()
