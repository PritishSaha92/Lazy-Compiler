"""Exact floating floor-remainder cases with integer-only runner transport.

Inputs and outputs carry IEEE bits in signed i32 buffers, using two words per
f64. Every word survives a runner-side float64 JSON conversion exactly. This
preserves subnormals, signed zero and exceptional inputs. NaN outputs are
canonicalized before transport; their payload and sign are not specified.
No GPU is launched by these tests. Export packages with --export.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy.compiler.dtype import F32, F64, I32, U32, U64
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Assign, Let, Load, Write
from ir_cases import axis, literal, op, write_case
from ptx_backend import compile_file
from test_contract_edges import explicit_case
from tools.check_compiler import assemble


def remainder_inputs(dtype):
    """IEEE transitions, all signs, seeded raw bits, and quotient-rounding edges."""
    typ, word = (np.float32, np.uint32) if dtype == F32 else (np.float64, np.uint64)
    limits = np.finfo(typ)
    tiny = np.nextafter(typ(0), typ(1))
    values = [typ(0), typ(-0.0), tiny, -tiny, typ(limits.tiny), typ(-limits.tiny),
              typ(1), typ(-1), typ(3), typ(-3), typ(limits.max), typ(-limits.max),
              typ(np.inf), typ(-np.inf), typ(np.nan)]
    pairs = [(x, y) for x in values for y in values]
    pairs += [(typ(x), typ(y)) for x, y in (
        (1e8, 3), (1e20, 3), (-1e8, 3), (1e8, -3), (-1e8, -3),
        (5.3, 2.7), (-5.3, 2.7), (5.3, -2.7), (-5.3, -2.7),
        (5, 2), (-5, 2), (5, -2), (-5, -2), (1, 1), (1, -1),
    )]
    rng = np.random.default_rng(260028)
    raw = np.frombuffer(rng.bytes(512 * np.dtype(word).itemsize), dtype=word).view(typ)
    pairs += list(zip(raw[::2], raw[1::2]))
    for exponent in (-20, 0, 20, 60):
        multiple = typ(np.ldexp(3.0, exponent))
        for direction in (-np.inf, np.inf):
            for sign_x in (-1, 1):
                for sign_y in (-1, 1):
                    pairs.append((typ(sign_x * np.nextafter(multiple, typ(direction))),
                                  typ(sign_y * 3)))
    return np.asarray(pairs, dtype=typ).T.copy()


def canonical_bits(values):
    """Independent IEEE oracle: NumPy computes values, host view exposes bits."""
    values = np.asarray(values)
    signed, quiet_nan = ((np.int32, 0x7fc00000) if values.dtype == np.float32
                         else (np.int64, 0x7ff8000000000000))
    bits = values.view(signed).copy()
    bits[np.isnan(values)] = quiet_nan
    return bits


def word_pairs(values):
    """Low/high signed words, with order independent of host byte order."""
    bits = np.asarray(values).view(np.uint64)
    words = np.stack((bits.astype(np.uint32),
                      (bits >> np.uint64(32)).astype(np.uint32)), axis=-1)
    return words.view(np.int32)


def floating_remainder_word_cases():
    """F64 payloads never traverse the runner as potentially rounded i64 data."""
    i = Var("i")
    x, y = remainder_inputs(F64)
    with np.errstate(all="ignore"):
        expected = word_pairs(canonical_bits(np.remainder(x, y)))
    for operation in ("remainder", "mod"):
        body = [Let("word_shift", Literal(32, U32), U32)]
        for name in ("x", "y"):
            body.extend((
                Load(name + "_low", name, (i, literal(0)), dtype=I32),
                Load(name + "_high", name, (i, literal(1)), dtype=I32),
                Assign(name + "_low_u32", op("bitcast"), (name + "_low",), U32),
                Assign(name + "_high_u32", op("bitcast"), (name + "_high",), U32),
                Let(name + "_low_u64", Var(name + "_low_u32"), U64),
                Let(name + "_high_u64", Var(name + "_high_u32"), U64),
                Assign(name + "_upper", op("left_shift"),
                       (name + "_high_u64", "word_shift"), U64),
                Assign(name + "_bits", op("bitwise_or"),
                       (name + "_low_u64", name + "_upper"), U64),
                Assign(name + "_value", op("bitcast"), (name + "_bits",), F64),
            ))
        body.extend((
            Assign("r", op(operation), ("x_value", "y_value"), F64),
            Assign("raw_r", op("bitcast"), ("r",), U64),
            Assign("not_nan", op("equal"), ("r", "r"), I32),
            Let("nan_bits", Literal(0x7ff8000000000000, U64), U64),
            Assign("canonical", op("where"), ("not_nan", "raw_r", "nan_bits"), U64),
            Let("low_u32", Var("canonical"), U32),
            Assign("upper_u64", op("right_shift"), ("canonical", "word_shift"), U64),
            Let("high_u32", Var("upper_u64"), U32),
            Assign("low", op("bitcast"), ("low_u32",), I32),
            Assign("high", op("bitcast"), ("high_u32",), I32),
            Write("out", (i, literal(0)), "low", value_dtype=I32),
            Write("out", (i, literal(1)), "high", value_dtype=I32),
        ))
        value = explicit_case(
            f"floating_f64_{operation}_words",
            (Tile((axis("i", len(x)),), tuple(body)),),
            {"x": word_pairs(x), "y": word_pairs(y)}, expected,
            types={"x": I32, "y": I32, "out": I32}, block=32,
        )
        value[1]["kernels"][0]["grid"] = [(len(x) + 31) // 32, 1, 1]
        yield value


def floating_remainder_cases():
    i = Var("i")
    for floating, integer, signed, quiet_nan in (
        (F32, I32, np.int32, 0x7fc00000),
    ):
        x, y = remainder_inputs(floating)
        with np.errstate(all="ignore"):
            expected = canonical_bits(np.remainder(x, y))
        for operation in ("remainder", "mod"):
            body = (
                Load("raw_x", "x", (i,), dtype=integer),
                Load("raw_y", "y", (i,), dtype=integer),
                Assign("x_value", op("bitcast"), ("raw_x",), floating),
                Assign("y_value", op("bitcast"), ("raw_y",), floating),
                Assign("r", op(operation), ("x_value", "y_value"), floating),
                Assign("raw_r", op("bitcast"), ("r",), integer),
                Assign("not_nan", op("equal"), ("r", "r"), I32),
                Let("nan_bits", Literal(quiet_nan, integer), integer),
                Assign("canonical", op("where"), ("not_nan", "raw_r", "nan_bits"), integer),
                Write("out", (i,), "canonical", value_dtype=integer),
            )
            value = explicit_case(
                f"floating_{floating.name}_{operation}",
                (Tile((axis("i", len(x)),), body),),
                {"x": x.view(signed), "y": y.view(signed)}, expected,
                types={"x": integer, "y": integer, "out": integer},
                block=32,
            )
            value[1]["kernels"][0]["grid"] = [(len(x) + 31) // 32, 1, 1]
            yield value
    yield from floating_remainder_word_cases()


def export(folder, f64_words_only=False):
    cases = floating_remainder_word_cases() if f64_words_only else floating_remainder_cases()
    return [write_case(Path(folder) / value[1]["id"], value)
            for value in cases]


class FloatingRemainderTests(unittest.TestCase):
    def test_reference_edges_and_lossless_integer_transport(self):
        for floating, typ, signed, nan_bits in (
            (F32, np.float32, np.int32, 0x7fc00000),
            (F64, np.float64, np.int64, 0x7ff8000000000000),
        ):
            x, y = remainder_inputs(floating)
            self.assertEqual(len(x), 528)
            with np.errstate(all="ignore"):
                # Fixed mathematical counterexamples, independent of emitter code.
                expected = np.remainder(typ(1e8 if floating == F32 else 1e20), typ(3))
                self.assertEqual(expected, typ(1))
                signs = np.remainder(np.asarray([0, -0.0, 6, -6], dtype=typ), typ(-3))
                self.assertTrue(np.all(signs == 0))
                self.assertTrue(np.all(np.signbit(signs)))
                self.assertEqual(np.remainder(-typ(1), typ(np.inf)), typ(np.inf))
                special = np.remainder(np.asarray([1, np.inf, np.nan], dtype=typ), typ(0))
            np.testing.assert_array_equal(canonical_bits(special),
                                          np.full(3, nan_bits, dtype=signed))
            # The runner's numeric data model may convert JSON numbers through
            # float64. Every signed i32 word remains exact through that path.
            for source in (x, y):
                words = source.view(np.int32) if floating == F32 else word_pairs(source)
                wire = np.asarray(json.loads(json.dumps(words.tolist(), allow_nan=False)), dtype=np.float64)
                restored = wire.astype(np.int32)
                np.testing.assert_array_equal(restored, words)
                if floating == F64:
                    unsigned = restored.view(np.uint32).astype(np.uint64)
                    reconstructed = unsigned[:, 0] | (unsigned[:, 1] << np.uint64(32))
                    np.testing.assert_array_equal(reconstructed, source.view(np.uint64))

        # The original single-i64 transport loses this negative subnormal bit.
        self.assertNotEqual(int(np.float64(-(1 << 63) + 1)), -(1 << 63) + 1)

        for _, metadata, expected in floating_remainder_cases():
            self.assertEqual((metadata["rtol"], metadata["atol"]), (0, 0))
            kernel = metadata["kernels"][0]
            self.assertEqual(kernel["grid"], [17, 1, 1])
            self.assertEqual(expected["out"].dtype, np.dtype("int32"))
            np.testing.assert_array_equal(expected["out"].astype(np.float64).astype(np.int32),
                                          expected["out"])
            json.dumps(metadata, allow_nan=False)

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas required for remainder assembly")
    def test_all_four_runtime_input_packages_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in export(directory):
                with self.subTest(case=path.parent.name):
                    assemble(compile_file(path), "sm_89")


if __name__ == "__main__":
    if "--export" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--export", type=Path, required=True)
        parser.add_argument("--f64-words-only", action="store_true")
        args = parser.parse_args()
        destination = args.export
        if destination.exists():
            parser.error(f"refusing to replace existing {destination}")
        paths = export(destination, args.f64_words_only)
        print(f"Exported {len(paths)} integer-transport remainder packages to {destination}")
    else:
        unittest.main()

