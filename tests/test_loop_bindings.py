"""Loop for-init types and scope, independently of operand types.

Pinned Emmy 0.3.11 ir/stmt/blocks.py:183 declares Loop's axis as int;
StridedLoop:376-379 declares int axis and, when explicit, int axis_end.
The latter binding can refer to the newly initialized axis. The increment is
ordinary compound assignment: evaluate its current operand, compute with the
usual promotions, then assign to the int carrier. We read these definitions;
the fixtures and compiler never invoke Emmy's CUDA renderer.

Floating starts/steps use finite in-range values. The integer-narrowing case
uses the backend's established two's-complement integer conversion contract.
The negative-end case is safe on the fixed compiler; do not execute its old
unsigned-comparison baseline (it can take billions of iterations).
"""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]
from emmy.compiler.dtype import F64, I32, I64
from emmy.compiler.ir.expr import Builtin, Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.blocks import StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Init, Let, Load, Write
from ir_cases import axis, expression, op, write_case
from test_contract_edges import explicit_case
from ptx_backend import Compiler, compile_file
from tools.check_compiler import assemble


def counted_loop(start, step, extent=6, end=None, extra=()):
    return StridedLoop(
        axis("j", extent), start, step,
        (*extra, Let("index", Var("j"), I32),
         Accum("count", "one", op("add"), I32),
         Accum("sum", "index", op("add"), I32)),
        end=end, seed=False,
    )


def initial_state():
    return (Let("one", Literal(1, I32), I32),
            Init("count", 0, dtype=I32), Init("sum", 0, dtype=I32))


def output():
    return Write("out", (Var("i"), Literal(0, I32)),
                 values=("count", "sum"), value_dtype=I32)


def loop_binding_cases():
    starts = np.asarray([2**32, 2**32 + 1, -2**32 + 1], np.int64)
    body = (*initial_state(), Load("start", "x", (Var("i"),), dtype=I64),
            counted_loop(Var("start"), Literal(1, I32), extent=3), output())
    yield explicit_case(
        "strided_i64_start_narrows", (Tile((axis("i", 3),), body),),
        {"x": starts}, np.asarray([[3, 3], [2, 3], [2, 3]], np.int32),
        types={"x": I64, "out": I32},
    )

    # After every iteration int += double truncates again: j sequences are
    # [0,1,2,3,4,5], [1,2,3,4,5], [2,4], not a floating induction sequence.
    starts = np.asarray([.75, 1.75, 2.75], np.float64)
    steps = np.asarray([1.25, 1.75, 2.5], np.float64)
    body = (*initial_state(), Load("start", "x", (Var("i"),), dtype=F64),
            Load("step", "y", (Var("i"),), dtype=F64),
            counted_loop(Var("start"), Var("step")), output())
    yield explicit_case(
        "strided_f64_start_and_step", (Tile((axis("i", 3),), body),),
        {"x": starts, "y": steps}, np.asarray([[6, 15], [5, 15], [2, 6]], np.int32),
        types={"x": F64, "y": F64, "out": I32},
    )

    # Canonical thread_idx.x is unsigned; its loop binding is still signed.
    # The old baseline converts end=-1 to UINT_MAX and must not be launched.
    body = (*initial_state(), counted_loop(Builtin("thread_idx.x"), Literal(1, I32),
                                          end=Literal(-1, I32)), output())
    value = explicit_case(
        "strided_unsigned_start_negative_end", (Tile((axis("i", 4),), body),),
        {}, np.zeros((4, 2), np.int32), types={"out": I32},
    )
    value[1]["fixture_provenance"] = {"unsafe_on_pre_fix_baseline": True,
        "reason": "Unsigned induction comparison can take billions of iterations."}
    yield value

    # Both conversion and snapshot matter: mutable limit falls by 1.0 per
    # iteration, while j_end must remain int(initial limit). Index sums alone
    # cannot expose a float-vs-int bound, so accumulate the visible j_end too.
    values = np.asarray([.9, 1.9, 4.9], np.float64)
    body = (*initial_state(), Init("limit", 0, dtype=F64),
            Init("bound_sum", 0, dtype=I32),
            Load("initial", "x", (Var("i"),), dtype=F64),
            Accum("limit", "initial", op("add"), F64),
            Let("minus_one", Literal(-1.0, F64), F64),
            counted_loop(Literal(0, I32), Literal(1, I32), end=Var("limit"), extra=(
                Accum("limit", "minus_one", op("add"), F64),
                Let("bound", Var("j_end"), I32),
                Accum("bound_sum", "bound", op("add"), I32))),
            Write("out", (Var("i"), Literal(0, I32)),
                  values=("count", "sum", "bound_sum"), value_dtype=I32))
    yield explicit_case(
        "strided_f64_end_snapshot", (Tile((axis("i", 3),), body),),
        {"x": values}, np.asarray([[0, 0, 0], [1, 0, 1], [4, 6, 16]], np.int32),
        types={"x": F64, "out": I32},
    )

    # int j=1, j_end=j+3. There is no outer j to accidentally resolve.
    body = (*initial_state(), counted_loop(Literal(1, I64), Literal(1, I32),
                end=expression("+", Var("j"), Literal(3, I32))), output())
    yield explicit_case(
        "strided_end_reads_initialized_axis", (Tile((axis("i", 1),), body),),
        {}, np.asarray([[3, 6]], np.int32), types={"out": I32},
    )

    # The live step becomes -.25, .75, 1.75, 2.75, 3.75 after each body.
    # Assigning int(j + step) produces j=2,1,1,2,4, then7 exits. Converting
    # step to int before addition would instead observe 2,2,2,3,5.
    body = (*initial_state(), Init("step", -1.25, dtype=F64),
            Let("increment", Literal(1.0, F64), F64),
            counted_loop(Literal(2, I32), Var("step"), extra=(
                Accum("step", "increment", op("add"), F64),)), output())
    yield explicit_case(
        "strided_live_fractional_step", (Tile((axis("i", 1),), body),),
        {}, np.asarray([[5, 10]], np.int32), types={"out": I32},
    )


def export(folder):
    return [write_case(Path(folder) / value[1]["id"], value)
            for value in loop_binding_cases()]


class LoopBindingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_packages_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in export(directory):
                for budget in (0, 1024):
                    with self.subTest(case=path.parent.name, unroll_budget=budget):
                        assemble(compile_file(path, unroll_budget=budget), "sm_89")

    def test_unsigned_start_uses_signed_comparison(self):
        value = list(loop_binding_cases())[2]
        with tempfile.TemporaryDirectory() as directory:
            source = compile_file(write_case(Path(directory), value))
        self.assertIn("setp.ge.s32", source)
        self.assertNotIn("setp.ge.u32", source)
        self.assertNotIn("4294967295", source)

    def test_large_implicit_extent_is_not_narrowed(self):
        # Static-only check: executing this loop would require over 2**31
        # iterations and reach signed overflow. Check the literal's promotion,
        # without suggesting that such runtime iteration spaces are supported.
        value = list(loop_binding_cases())[0]
        compiler = Compiler(value[0], value[1]["kernels"][0])
        compiler.loop(StridedLoop(axis("wide", 2**31), Literal(0, I32),
                                 Literal(1, I32), (), seed=False))
        source = "\n".join(compiler.lines)
        self.assertIn("setp.ge.s64", source)
        self.assertIn("2147483648", source)
        self.assertNotIn("-2147483648", source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path)
    args, rest = parser.parse_known_args()
    if args.export:
        for path in export(args.export):
            print(path)
    else:
        unittest.main(argv=[sys.argv[0], *rest])
