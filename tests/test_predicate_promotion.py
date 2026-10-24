"""C integer promotion for arithmetic on comparison-valued expressions."""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy.compiler.dtype import I32
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Let, Write
from ir_cases import axis, expression, literal, write_case
from ptx_backend import compile_file
from test_contract_edges import explicit_case
from tools.check_compiler import assemble, local_run


def predicate_promotion_cases():
    i = Var("i")
    a = expression("<", i, literal(4))
    b = expression(">=", i, literal(2))
    nonzero = expression(">=", i, literal(0))
    indices = np.arange(9, dtype=np.int32)
    av = (indices < 4).astype(np.int32)
    bv = (indices >= 2).astype(np.int32)
    expressions = (
        (expression("+", a, b), av + bv),
        (expression("-", a, b), av - bv),
        (expression("*", a, b), av * bv),
        (expression("&", a, b), av & bv),
        (expression("|", a, b), av | bv),
        (expression("^", a, b), av ^ bv),
        (expression("<<", a, b), av << bv),
        (expression(">>", a, b), av >> bv),
        (expression("/", a, nonzero), av),
        (expression("%", a, nonzero), np.zeros_like(av)),
        (expression("&&", a, b), av & bv),
        (expression("||", a, b), av | bv),
    )
    names = tuple(f"v{k}" for k in range(len(expressions)))
    body = tuple(
        Let(name, value, I32) for name, (value, _) in zip(names, expressions)
    ) + (Write("out", (i, literal(0)), values=names, value_dtype=I32),)
    yield explicit_case(
        "comparison_integer_promotion",
        (Tile((axis("i", 9),), body),),
        {},
        np.column_stack(tuple(expected for _, expected in expressions)),
        types={"out": I32},
    )


class PredicatePromotionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_predicate_arithmetic_assembles(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in predicate_promotion_cases():
                path = write_case(Path(directory) / value[1]["id"], value)
                assemble(compile_file(path), "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_predicate_arithmetic_matches_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in predicate_promotion_cases():
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
                out = result["outputs"]["out"]
                np.testing.assert_array_equal(
                    np.asarray(out["values"]).reshape(out["shape"]), value[2]["out"]
                )


if __name__ == "__main__":
    unittest.main()
