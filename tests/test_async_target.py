"""Architecture admission for ordinary asynchronous global-to-shared copies.

These tests construct scheduled IR directly and use CPU/ptxas checks only.
PTX 8.0 requires SM80 for copy, commit_group and wait_group separately.
An explicitly requested synchronous lowering remains available on older GPUs.
"""

from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, Sync, Tile
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Load, Write
from ir_cases import axis, case, expression as ex, literal as lit
from ptx_backend import Compiler, CompileError
from tools.check_compiler import assemble


def copy_case(body=None):
    index = Var("i")
    if body is None:
        body = (
            CpAsyncCopy("stage", (index,), "x", (index,), nbytes=4),
            CpAsyncCommit(),
            CpAsyncWait(0),
            Sync(),
            Load("value", "stage", (index,), dtype=F32),
            Write("out", (index,), "value", value_dtype=F32),
        )
    return case(
        "async_target",
        (Smem("stage", (32,), "float", align=16), Tile((axis("i", 32),), tuple(body))),
        {"x": (32,)}, (32,), lambda x: x.copy(), block=32,
    )


def emit(target, *, body=None, synchronous_copy=False):
    graph, meta, _ = copy_case(body)
    return Compiler(graph, meta["kernels"][0], target=target, synchronous_copy=synchronous_copy).compile()


def native_nodes():
    return (
        CpAsyncCopy("stage", (Var("i"),), "x", (Var("i"),), nbytes=4),
        CpAsyncCommit(),
        CpAsyncWait(0),
    )


class AsyncTargetTests(unittest.TestCase):
    def test_each_native_copy_node_rejects_pre_ampere(self):
        for target in ("sm_70", "sm_75"):
            for node in native_nodes():
                with self.subTest(target=target, node=type(node).__name__):
                    graph, meta, _ = copy_case((node,))
                    # Admission occurs during construction, before emission.
                    with self.assertRaisesRegex(CompileError, "cp.async.*sm_80"):
                        Compiler(graph, meta["kernels"][0], target=target)

    def test_nested_conditional_and_loop_are_validated(self):
        condition = ex("==", Var("i"), lit(0))
        for node in native_nodes():
            for branch in ("then", "else"):
                nested = Loop(axis("iteration", 2), (node,))
                conditional = Cond(condition, (nested,), ()) if branch == "then" else Cond(condition, (), (nested,))
                with self.subTest(node=type(node).__name__, branch=branch):
                    graph, meta, _ = copy_case((conditional,))
                    with self.assertRaisesRegex(CompileError, "cp.async.*sm_80"):
                        Compiler(graph, meta["kernels"][0], target="sm_75")

    def test_older_target_remains_available_without_native_copies(self):
        index = Var("i")
        text = emit("sm_75", body=(
            Load("value", "x", (index,), dtype=F32),
            Write("out", (index,), "value", value_dtype=F32),
        ))
        self.assertIn(".target sm_75", text)
        self.assertIn("ld.global.f32", text)
        self.assertNotIn("cp.async", text)

    def test_native_copies_compile_on_sm80_and_sm89(self):
        for target in ("sm_80", "sm_89"):
            with self.subTest(target=target):
                text = emit(target)
                self.assertIn(".target " + target, text)
                self.assertEqual(text.count("cp.async.ca.shared.global"), 1)
                self.assertEqual(text.count("cp.async.commit_group;"), 1)
                self.assertEqual(text.count("cp.async.wait_group 0;"), 1)
                self.assertLess(text.index("cp.async.wait_group 0;"), text.index("bar.sync 0;"))
                self.assertLess(text.index("bar.sync 0;"), text.index("ld.shared.f32"))

    def test_explicit_synchronous_fallback_preserves_copy_and_barrier(self):
        text = emit("sm_75", synchronous_copy=True)
        self.assertIn(".target sm_75", text)
        self.assertNotIn("cp.async", text)
        self.assertEqual(text.count("ld.global.b32"), 1)
        self.assertEqual(text.count("st.shared.b32"), 1)
        self.assertLess(text.index("st.shared.b32"), text.index("bar.sync 0;"))
        self.assertLess(text.index("bar.sync 0;"), text.index("ld.shared.f32"))

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas required for instruction legality")
    def test_supported_and_fallback_modules_assemble(self):
        for target, synchronous in (("sm_80", False), ("sm_89", False), ("sm_75", True)):
            with self.subTest(target=target, synchronous=synchronous):
                assemble(emit(target, synchronous_copy=synchronous), target)


if __name__ == "__main__":
    unittest.main()
