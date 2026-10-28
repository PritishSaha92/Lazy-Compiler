"""Scalar admission, instruction legality and independently rounded fixtures.

These tests never launch a GPU. Export numerical packages for the official
runner with: python tests/test_type_capabilities.py --export build/type-cases
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emmy.compiler.dtype import BF16, F16, F32, F64, I16, I32, I64, U16, U32, U64
from emmy.compiler.ir.expr import BinaryExpr, CastExpr, FuncCallExpr, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Assign, Let, Load, Write
from ir_cases import axis, case, literal, op, write_case
from ptx_backend import Compiler, CompileError, Value, _ELEMENTWISE_CAPABILITIES, compile_file
from tools.check_compiler import assemble


def quantize(values, kind):
    """Host rounding independent of the emitter; inputs are numeric f32."""
    values = np.asarray(values, np.float32)
    if kind == F16:
        return values.astype(np.float16).astype(np.float32)
    # Explicit nearest-even BF16 payload rounding. No Emmy conversion helper.
    words = values.view(np.uint32)
    rounded = words + np.uint32(32767) + ((words >> 16) & np.uint32(1))
    return ((rounded >> 16) << 16).view(np.float32)


def explicit_case(name, body, inputs, expected):
    expected = np.asarray(expected, np.float32)
    value = case(
        name, (Tile((axis("i", len(next(iter(inputs.values())))),), tuple(body)),),
        {key: np.shape(data) for key, data in inputs.items()}, expected.shape,
        lambda **_: expected, block=32,
    )
    _, metadata, _ = value
    for descriptor in metadata["kernels"][0]["args"]:
        if descriptor["name"] in inputs:
            descriptor.pop("seed", None)
            descriptor["data"] = np.asarray(inputs[descriptor["name"]], np.float32).tolist()
    metadata["rtol"] = metadata["atol"] = 0
    return value


def type_cases():
    """Numerical cases use f32 ABI storage, typed internal operations, exact oracles."""
    i = Var("i")
    x = np.asarray([1, 1.5, -2.25, 0, -0.0, 2**-14, 2**-24, 31.875], np.float32)
    y = np.asarray([3, 2, 0.5, -2, 1, 3, 0.25, -1.75], np.float32)
    for kind in (F16, BF16):
        a, b = quantize(x, kind), quantize(y, kind)
        for form in ("binary", "assign", "function"):
            body = [Load("xv", "x", (i,), dtype=F32), Load("yv", "y", (i,), dtype=F32),
                    Let("a", Var("xv"), kind), Let("b", Var("yv"), kind)]
            expected = []
            for column, (symbol, name, fn) in enumerate((
                ("+", "add", np.add), ("-", "subtract", np.subtract),
                ("*", "multiply", np.multiply), ("/", "divide", np.divide),
            )):
                if form == "assign":
                    body.append(Assign("r", op(name), ("a", "b"), kind))
                else:
                    expr = (BinaryExpr(symbol, Var("a"), Var("b")) if form == "binary"
                            else FuncCallExpr(name, (Var("a"), Var("b"))))
                    body.append(Let("r", expr, kind))
                body.append(Write("out", (i, literal(column)), "r", value_dtype=F32))
                expected.append(quantize(fn(a, b), kind))
            yield explicit_case(f"types_{kind.name}_{form}", body, {"x": x, "y": y}, np.stack(expected, 1))

        # Each nested binary node rounds. Computing the whole expression in
        # f32 would retain the small increment instead of returning zero.
        tiny = 2**(-11 if kind == F16 else -8)
        left = np.ones(8, np.float32)
        right = np.full(8, tiny, np.float32)
        body = [Load("xv", "x", (i,), dtype=F32), Load("yv", "y", (i,), dtype=F32),
                Let("a", Var("xv"), kind), Let("b", Var("yv"), kind),
                Let("r", BinaryExpr("-", BinaryExpr("+", Var("a"), Var("b")), Var("a")), F32),
                Write("out", (i,), "r", value_dtype=F32)]
        yield explicit_case(f"types_{kind.name}_nested_round", body, {"x": left, "y": right}, np.zeros(8))

        body = [Load("xv", "x", (i,), dtype=F32), Load("yv", "y", (i,), dtype=F32),
                Let("a", Var("xv"), kind), Let("b", Var("yv"), kind),
                Let("p", BinaryExpr("<", Var("a"), Var("b")), "bool"),
                Assign("selected", op("where"), ("p", "a", "b"), kind),
                Assign("not_a", op("logical_not"), ("a",), I32),
                Let("as_i", CastExpr(I32, Var("a")), I32),
                Let("as_d", CastExpr(F64, Var("a")), F64),
                Let("other_half", CastExpr(BF16 if kind == F16 else F16, Var("a")), BF16 if kind == F16 else F16)]
        names = ("p", "selected", "not_a", "as_i", "as_d", "other_half")
        body += [Write("out", (i, literal(c)), n, value_dtype=F32) for c, n in enumerate(names)]
        expected = np.stack((a < b, np.where(a < b, a, b), a == 0, np.trunc(a), a,
                             quantize(a, BF16 if kind == F16 else F16)), 1)
        yield explicit_case(f"types_{kind.name}_compare_cast", body, {"x": x, "y": y}, expected)

        # Input payload words are finite, exactly representable f32 numbers.
        # Reinterpret only inside the kernel, so JSON never carries NaN/Inf.
        words = np.asarray([0, 0x80000000, 0x7f800000, 0xff800000,
                            0x7fc00000, 0xffc00000, 0x3f800000, 0xbf800000,
                            0x33800000, 0x00010000], np.uint32)
        decoded = words.view(np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = quantize(decoded, kind)
        body = [Load("raw", "x", (i,), dtype=F32), Let("bits", Var("raw"), U32),
                Assign("decoded", op("bitcast"), ("bits",), F32),
                Let("a", Var("decoded"), kind), Let("zero", literal(0.0), kind),
                Let("is_nan", BinaryExpr("!=", Var("a"), Var("a")), "bool"),
                Assign("self_eq", op("equal"), ("a", "a"), F32),
                Assign("positive", op("greater"), ("a", "zero"), F32),
                Assign("negative", op("less"), ("a", "zero"), F32),
                Assign("is_zero", op("logical_not"), ("a",), F32),
                Let("truth", Var("a"), "bool")]
        body += [Write("out", (i, literal(c)), n, value_dtype=F32) for c, n in enumerate(
            ("is_nan", "self_eq", "positive", "negative", "is_zero", "truth"))]
        expected = np.stack((np.isnan(narrowed), narrowed == narrowed, narrowed > 0,
                             narrowed < 0, narrowed == 0, narrowed != 0), 1)
        yield explicit_case(f"types_{kind.name}_nonfinite_predicates", body,
                            {"x": words.astype(np.float32)}, expected)

    # Unsigned negation is modulo-width subtraction; abs is an identity.
    for kind, modulus in ((U32, 2**32), (U64, 2**64)):
        values = np.asarray([0, 1, 2, 17, 63, 255, 4095, 65535], np.float32)
        body = [Load("raw", "x", (i,), dtype=F32), Let("a", Var("raw"), kind),
                Assign("neg", op("negative"), ("a",), kind),
                Assign("abs", op("absolute"), ("a",), kind),
                Assign("not", op("logical_not"), ("a",), kind)]
        body += [Write("out", (i, literal(c)), n, value_dtype=F32) for c, n in enumerate(("neg", "abs", "not"))]
        expected = [[(-int(v)) % modulus, int(v), int(v == 0)] for v in values]
        yield explicit_case(f"types_{kind.name}_unary", body, {"x": values}, expected)

    values = np.asarray([-32768, -7, -3, -1, 0, 1, 17, 32767], np.float32)
    body = [Load("raw", "x", (i,), dtype=F32), Let("a", Var("raw"), I16),
            Let("zero", literal(0), I16),
            Assign("negative", op("less"), ("a", "zero"), I16),
            Assign("chosen", op("where"), ("negative", "a", "zero"), I16),
            Let("count", FuncCallExpr("bitwise_count", (Var("a"),)), I16),
            Assign("bits", op("bitcast"), ("a",), U16),
            Assign("roundtrip", op("bitcast"), ("bits",), I16)]
    body += [Write("out", (i, literal(c)), n, value_dtype=F32)
             for c, n in enumerate(("negative", "chosen", "count", "roundtrip"))]
    expected = [[v < 0, min(v, 0), abs(int(v)).bit_count(), v] for v in values]
    yield explicit_case("types_small_integer_select_bits", body, {"x": values}, expected)

    for kind, bits in ((I32, 32), (I64, 64)):
        values = np.asarray([-(2**(bits-1)), -7, -3, -1, 0, 1, 7, 2**(bits-2)], np.float32)
        body = [Load("raw", "x", (i,), dtype=F32), Let("a", Var("raw"), kind),
                Let("count", FuncCallExpr("bitwise_count", (Var("a"),)), kind),
                Write("out", (i,), "count", value_dtype=F32)]
        expected = [abs(int(v)).bit_count() for v in values]
        yield explicit_case(f"types_{kind.name}_signed_count", body, {"x": values}, expected)

    for kind in (F16, BF16):
        a = quantize(x, kind)
        payloads = (a.astype(np.float16).view(np.uint16) if kind == F16
                    else a.view(np.uint32) >> 16)
        body = [Load("raw", "x", (i,), dtype=F32), Let("a", Var("raw"), kind),
                Assign("bits", op("bitcast"), ("a",), U16),
                Assign("roundtrip", op("bitcast"), ("bits",), kind),
                Write("out", (i, literal(0)), "bits", value_dtype=F32),
                Write("out", (i, literal(1)), "roundtrip", value_dtype=F32)]
        yield explicit_case(f"types_{kind.name}_bitcast", body, {"x": x}, np.stack((payloads, a), 1))


def bare_compiler(target="sm_89"):
    """Exercise instruction selection without relying on a generated schedule."""
    compiler = object.__new__(Compiler)
    compiler.target = target
    compiler.lines, compiler.declarations = [], []
    compiler.counts, compiler.regtypes = defaultdict(int), {}
    return compiler


def instruction_module(compiler):
    declarations = [f".reg .{compiler.regtypes[p]} %{p}<{n+1}>;" for p, n in compiler.counts.items()]
    return (f".version 8.0\n.target {compiler.target}\n.address_size 64\n"
            ".visible .entry type_matrix() {\n" + "\n".join(declarations + compiler.lines) + "\nret;\n}\n")


class TypeCapabilitiesTests(unittest.TestCase):
    def test_positive_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in type_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    ptx = compile_file(path)
                    self.assertNotIn("div.rn.f16", ptx)
                    self.assertNotIn("div.rn.bf16", ptx)
                    self.assertNotIn("setp.lt.bf16", ptx)
                    if shutil.which("ptxas"):
                        assemble(ptx, "sm_89")

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas required for instruction legality matrix")
    def test_admitted_elementwise_matrix_assembles(self):
        # Every admitted result dtype/op pair has live stores, preventing dead
        # arithmetic from hiding operand/register legality in assembly checks.
        all_types = sorted(set().union(*(types for _, types in _ELEMENTWISE_CAPABILITIES.values())))
        for dt in all_types:
            with self.subTest(dtype=dt):
                compiler = bare_compiler()
                pointer = compiler.instruction("mov.u64", "u64", 0)
                for name, (arity, types) in _ELEMENTWISE_CAPABILITIES.items():
                    if dt not in types:
                        continue
                    args = [compiler.materialize(compiler.literal(k + 1, dt)) for k in range(arity)]
                    result = compiler.operation(name, args, dt)
                    result = compiler.convert(result, "u32") if result.type == "pred" else result
                    memtype = "b16" if result.type in ("f16", "bf16") else result.type
                    compiler.emit(f"st.global.{memtype} [{pointer}], {result};")
                assemble(instruction_module(compiler), "sm_89")

    def test_rejected_pairs_and_arity(self):
        for dt, name, arity in (("s32", "sqrt", 1), ("u64", "fma", 3),
                                ("s16", "floor", 1), ("f64", "exp", 1),
                                ("bf16", "bitwise_xor", 2), ("pred", "add", 2),
                                ("b32", "add", 2)):
            with self.subTest(dtype=dt, op=name):
                compiler = bare_compiler()
                args = [Value("%operand", dt)] * arity
                with self.assertRaisesRegex(CompileError, f"{name!r}.*{dt}"):
                    compiler.operation(name, args, dt)
        with self.assertRaisesRegex(CompileError, "expects 2 operands, got 1"):
            bare_compiler().operation("add", [Value("%f", "f32")], "f32")
        for symbol in ("%", "&", "|", "^", "<<", ">>"):
            with self.subTest(symbol=symbol):
                diagnostic = ("shift operator.*requires integer operands"
                              if symbol in ("<<", ">>") else "binary operator.*f32")
                with self.assertRaisesRegex(CompileError, diagnostic):
                    bare_compiler().binary(symbol, Value("%a", "f32"), Value("%b", "f32"))

    def test_conversion_admission(self):
        for source in ("f64", "s32", "u32", "s64", "u64"):
            with self.subTest(source=source):
                with self.assertRaisesRegex(CompileError, f"{source} -> bf16 requires sm_90"):
                    bare_compiler().convert(Value("%x", source), "bf16")
        with self.assertRaisesRegex(CompileError, "requires sm_80"):
            bare_compiler("sm_75").convert(Value("%x", "f32"), "bf16")
        for source, destination in (("b32", "f32"), ("f64", "b64"), ("b64", "s64")):
            with self.subTest(source=source, destination=destination):
                with self.assertRaisesRegex(CompileError, "use an explicit equal-width bitcast"):
                    bare_compiler().convert(Value("%x", source), destination)
        with self.assertRaisesRegex(CompileError, "comparison.*raw dtype b32"):
            bare_compiler().compare("lt", Value("%a", "b32"), Value("%b", "b32"))

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas required for conversion legality matrix")
    def test_conversion_and_bitcast_matrix_assembles(self):
        types = ("s8", "u8", "s16", "u16", "s32", "u32", "s64", "u64", "f16", "bf16", "f32", "f64", "pred")
        for target in ("sm_89", "sm_90"):
            compiler = bare_compiler(target)
            pointer = compiler.instruction("mov.u64", "u64", 0)
            for source in types:
                arg = compiler.materialize(compiler.literal(1, source))
                for destination in types:
                    if (target == "sm_89" and destination == "bf16"
                            and source in ("s32", "u32", "s64", "u64", "f64")):
                        continue
                    with self.subTest(target=target, source=source, destination=destination):
                        result = compiler.convert(arg, destination)
                        result = compiler.convert(result, "u32") if destination == "pred" else result
                        memtype = "b16" if result.type in ("f16", "bf16") else result.type
                        compiler.emit(f"st.global.{memtype} [{pointer}], {result};")
                if source == "pred":
                    continue
                for destination in types:
                    if destination == "pred":
                        continue
                    width = lambda t: 16 if t == "bf16" else int(t[1:])
                    if width(source) != width(destination):
                        continue
                    result = compiler.operation("bitcast", [arg], destination)
                    memtype = "b16" if result.type in ("f16", "bf16") else result.type
                    compiler.emit(f"st.global.{memtype} [{pointer}], {result};")
            assemble(instruction_module(compiler), target)

    def test_cli_rejection_is_diagnostic_only(self):
        value = explicit_case("types_invalid_sqrt", (
            Load("xv", "x", (Var("i"),), dtype=F32),
            Assign("r", op("sqrt"), ("xv",), I32),
            Write("out", (Var("i"),), "r", value_dtype=F32)), {"x": [4]}, [2])
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), value)
            result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(path)], capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"Assign: elementwise 'sqrt' is unsupported for result dtype s32", result.stderr)
        self.assertNotIn(b"Traceback", result.stderr)


if __name__ == "__main__":
    if "--export" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--export", type=Path, required=True)
        folder = parser.parse_args().export
        for value in type_cases():
            print(write_case(folder / value[1]["id"], value))
    else:
        unittest.main()
