"""Typed GELU formulas with independent host math and exceptional-value checks.

Emmy 0.3.11 defines gelu through erf and gelu_tanh through a cubic and tanh.
These references use the standard-library transcendental functions, explicit
f32 arithmetic boundaries, and independent IEEE narrow-format rounding. The
numeric tolerances allow the backend's documented erf/tanh approximations and
one narrow-format rounding step. Exceptional outputs use exact classification
flags, with raw i32 input transport to preserve NaNs, infinities and -0.
No GPU is launched here. Export the eight runner packages with --export.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy.compiler.dtype import BF16, F16, F32, F64, I32
from emmy.compiler.ir.expr import FuncCallExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Assign, Let, Load, Write
from ir_cases import axis, literal, op, write_case
from ptx_backend import CompileError, compile_file
from test_contract_edges import explicit_case
from tools.check_compiler import assemble


def quantize(values, kind):
    """Host RNE conversion; NaN payload/sign is deliberately unspecified."""
    values = np.asarray(values, np.float32)
    with np.errstate(all="ignore"):
        if kind == F32:
            return values.copy()
        if kind == F16:
            return values.astype(np.float16).astype(np.float32)
        if kind == BF16:
            words = values.view(np.uint32)
            rounded = words + np.uint32(32767) + ((words >> 16) & np.uint32(1))
            result = ((rounded >> 16) << 16).view(np.float32)
            return np.where(np.isnan(values), np.float32(np.nan), result)
    raise ValueError(f"unsupported oracle type {kind}")


def gelu_reference(values, operation, kind=F32):
    """Formula oracle, independent of emitted erf/tanh polynomials/instructions."""
    x = quantize(values, kind)
    f = np.float32
    with np.errstate(all="ignore"):
        if operation == "gelu":
            scaled = x / f(math.sqrt(2.0))
            gate = np.asarray([math.erf(float(v)) for v in scaled], np.float32)
        elif operation == "gelu_tanh":
            cube = (x * x) * x
            scaled = f(math.sqrt(2.0 / math.pi)) * (x + f(0.044715) * cube)
            gate = np.asarray([math.tanh(float(v)) for v in scaled], np.float32)
        else:
            raise ValueError(operation)
        return quantize((f(0.5) * x) * (f(1.0) + gate), kind)


def finite_inputs():
    rng = np.random.default_rng(260029)
    # 273 cells exercise a partial final CTA. Both forms receive identical data.
    return np.concatenate((
        np.linspace(-9, 9, 193, dtype=np.float32),
        rng.uniform(-7, 7, 64).astype(np.float32),
        np.asarray([0, -0.0, 2**-24, -2**-24, 2**-14, -2**-14,
                    0.25, -0.25, 1, -1, 3, -3, 12, -12, 20, -20], np.float32),
    ))


def gelu_numeric_cases():
    i = Var("i")
    x = finite_inputs()
    for operation in ("gelu", "gelu_tanh"):
        for kind in (F32, F16, BF16):
            expected = gelu_reference(x, operation, kind)
            body = (
                Load("input_value", "x", (i,), dtype=F32),
                Let("typed_value", Var("input_value"), kind),
                Assign("assigned", op(operation), ("typed_value",), kind),
                Let("called", FuncCallExpr(operation, (Var("typed_value"),)), kind),
                Let("assigned_f32", Var("assigned"), F32),
                Let("called_f32", Var("called"), F32),
                Write("out", (i, literal(0)), "assigned_f32", value_dtype=F32),
                Write("out", (i, literal(1)), "called_f32", value_dtype=F32),
            )
            value = explicit_case(
                f"activation_{operation}_{kind.name}",
                (Tile((axis("i", len(x)),), body),), {"x": x},
                np.stack((expected, expected), axis=1), block=32,
            )
            value[1]["kernels"][0]["grid"] = [(len(x) + 31) // 32, 1, 1]
            value[1]["rtol"] = {F32: 1e-5, F16: 1e-3, BF16: 8e-3}[kind]
            value[1]["atol"] = 2e-6
            yield value


def exceptional_inputs():
    # Signalling/quiet NaNs, signed zero, subnormal boundaries, extremes and
    # values straddling the tanh primitive's near-zero branch.
    return np.asarray([
        0x00000000, 0x80000000, 0x7f800000, 0xff800000,
        0x7fc00000, 0xffc12345, 0x7f800001, 0xff800001,
        0x00000001, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff,
        0x3f800000, 0xbf800000, 0x33800000, 0xb3800000,
        0x3e7fffff, 0xbe7fffff, 0x3e800000, 0xbe800000,
        0x3e800001, 0xbe800001, 0x477fe000, 0xc77fe000,
        0x477ff000, 0xc77ff000, 0x7f7f0000, 0xff7f0000,
        0x3e9f0000, 0xbe9f0000, 0x3e9f8000, 0xbe9f8000,
        0x3ea00000,
    ], np.uint32).view(np.float32)


def classify(values):
    return np.stack((np.isnan(values), np.isposinf(values), np.isneginf(values),
                     values == 0, (values == 0) & np.signbit(values),
                     np.isfinite(values)), axis=1).astype(np.int32)


def gelu_exception_cases():
    i = Var("i")
    x = exceptional_inputs()
    for operation in ("gelu", "gelu_tanh"):
        body = [Load("input_bits", "x", (i,), dtype=I32),
                Assign("input_value", op("bitcast"), ("input_bits",), F32),
                Let("positive_inf", Literal(float("inf"), F32), F32),
                Let("negative_inf", Literal(float("-inf"), F32), F32),
                Let("zero", Literal(0.0, F32), F32),
                Let("integer_zero", Literal(0, I32), I32)]
        expected = []
        for column, kind in enumerate((F32, F16, BF16)):
            prefix = kind.name + "_"
            def name(suffix):
                return prefix + suffix
            body.extend((
                Let(name("input"), Var("input_value"), kind),
                Assign(name("activation"), op(operation), (name("input"),), kind),
                Let(name("value"), Var(name("activation")), F32),
                Assign(name("bits"), op("bitcast"), (name("value"),), I32),
                Assign(name("nan"), op("not_equal"), (name("value"), name("value")), I32),
                Assign(name("positive_inf"), op("equal"), (name("value"), "positive_inf"), I32),
                Assign(name("negative_inf"), op("equal"), (name("value"), "negative_inf"), I32),
                Assign(name("zero"), op("equal"), (name("value"), "zero"), I32),
                Assign(name("sign"), op("less"), (name("bits"), "integer_zero"), I32),
                Assign(name("negative_zero"), op("logical_and"), (name("zero"), name("sign")), I32),
                Assign(name("above"), op("greater"), (name("value"), "negative_inf"), I32),
                Assign(name("below"), op("less"), (name("value"), "positive_inf"), I32),
                Assign(name("finite"), op("logical_and"), (name("above"), name("below")), I32),
            ))
            for offset, suffix in enumerate(("nan", "positive_inf", "negative_inf", "zero", "negative_zero", "finite")):
                body.append(Write("out", (i, literal(column * 6 + offset)), name(suffix), value_dtype=I32))
            expected.append(classify(gelu_reference(x, operation, kind)))
        value = explicit_case(
            f"activation_{operation}_special", (Tile((axis("i", len(x)),), tuple(body)),),
            {"x": x.view(np.int32)}, np.concatenate(expected, axis=1),
            types={"x": I32, "out": I32}, block=32,
        )
        value[1]["kernels"][0]["grid"] = [(len(x) + 31) // 32, 1, 1]
        yield value


def gelu_cases():
    yield from gelu_numeric_cases()
    yield from gelu_exception_cases()


def export(folder):
    return [write_case(Path(folder) / value[1]["id"], value) for value in gelu_cases()]


class GeluTests(unittest.TestCase):
    def test_reference_values_specials_and_rounding(self):
        for operation in ("gelu", "gelu_tanh"):
            for kind in (F32, F16, BF16):
                x = np.asarray([0.0, -0.0, np.inf, -np.inf, np.nan, 1, -1], np.float32)
                result = gelu_reference(x, operation, kind)
                self.assertEqual(result[0], 0)
                self.assertFalse(np.signbit(result[0]))
                self.assertEqual(result[1], 0)
                self.assertTrue(np.signbit(result[1]))
                self.assertTrue(np.isposinf(result[2]))
                self.assertTrue(np.isnan(result[3:5]).all())
                # Both formulas give GELU(1) around .841 and GELU(-1) around -.159.
                self.assertAlmostEqual(float(result[5]), .8413, delta=.004)
                self.assertAlmostEqual(float(result[6]), -.1587, delta=.002)
        self.assertEqual(len(finite_inputs()), 273)
        self.assertEqual(len(exceptional_inputs()), 37)
        for _, metadata, expected in gelu_cases():
            # No exceptional JSON numbers are sent to the runner; i32 payloads
            # remain exact even when its numeric transport passes through f64.
            input_data = metadata["kernels"][0]["args"][0]["data"]
            json.dumps(input_data, allow_nan=False)
            if metadata["id"].endswith("special"):
                words = np.asarray(input_data, np.int32)
                np.testing.assert_array_equal(words.astype(np.float64).astype(np.int32), words)
                self.assertEqual((metadata["rtol"], metadata["atol"]), (0, 0))
                self.assertEqual(expected["out"].shape, (37, 18))

    def test_f64_is_rejected_instead_of_silently_narrowed(self):
        i = Var("i")
        with tempfile.TemporaryDirectory() as directory:
            for operation in ("gelu", "gelu_tanh"):
                body = (Load("value", "x", (i,), dtype=F64),
                        Assign("activation", op(operation), ("value",), F64),
                        Write("out", (i,), "activation", value_dtype=F64))
                value = explicit_case(
                    f"reject_{operation}_f64", (Tile((axis("i", 1),), body),),
                    {"x": np.asarray([1], np.float64)}, np.asarray([0], np.float64),
                    types={"x": F64, "out": F64}, block=32,
                )
                path = write_case(Path(directory) / value[1]["id"], value)
                with self.subTest(operation=operation), self.assertRaisesRegex(CompileError, "f64"):
                    compile_file(path)

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas required for GELU assembly")
    def test_all_eight_runtime_input_packages_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in export(directory):
                with self.subTest(case=path.parent.name):
                    assemble(compile_file(path), "sm_89")


if __name__ == "__main__":
    if "--export" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--export", type=Path, required=True)
        args = parser.parse_args()
        if args.export.exists():
            parser.error(f"refusing to replace existing {args.export}")
        paths = export(args.export)
        print(f"Exported {len(paths)} GELU packages to {args.export}")
    else:
        unittest.main()
