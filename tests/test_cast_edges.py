"""Independent constant/runtime conversion checks at representational edges."""

from dataclasses import replace
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emmy.compiler.dtype import F32, F64, I32, I64, U32, U64
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Let, Load, Write
from ir_cases import axis, case, expression, literal, write_case
from ptx_backend import compile_file
from tools.check_compiler import assemble, local_run


def integer_float32_reference(number):
    """Round the integer's bits once, without an intermediate float64 round."""
    magnitude = abs(number)
    shift = max(0, magnitude.bit_length() - 24)
    quotient, remainder = divmod(magnitude, 1 << shift)
    if shift:
        half = 1 << (shift - 1)
        if remainder > half or (remainder == half and quotient & 1):
            quotient += 1
    result = math.ldexp(float(quotient), shift)
    return -result if number < 0 else result


def typed_case(name, source_dtype, inputs, body, expected):
    expected = np.asarray(expected, np.float32)
    value = case(
        name,
        (Tile((axis("worker", 1),), tuple(body)),),
        {"x": (len(inputs),)},
        expected.shape,
        lambda **_: expected,
        block=32,
    )
    graph, metadata, _ = value
    node = graph.nodes["x"]
    graph.nodes["x"] = replace(
        node, outputs=tuple(replace(t, dtype=source_dtype) for t in node.outputs)
    )
    descriptor = metadata["kernels"][0]["args"][0]
    descriptor.pop("seed")
    descriptor.update(dtype=source_dtype.name, data=list(inputs))
    metadata["rtol"] = metadata["atol"] = 0
    return value


def floating_destination(name, source_dtype, inputs, expected, literal_dtype=None):
    body = []
    for row, value in enumerate(inputs):
        body.append(Load("raw", "x", (literal(row),), dtype=source_dtype))
        for column, source in enumerate(
            (Literal(value, literal_dtype or source_dtype), Var("raw"))
        ):
            body.extend(
                (
                    Let("converted", source, F32),
                    Write(
                        "out", (literal(row), literal(column)), "converted",
                        value_dtype=F32,
                    ),
                )
            )
    return typed_case(
        name, source_dtype, inputs, body, np.repeat(np.asarray(expected)[:, None], 2, 1)
    )


def integer_destination(name, source_dtype, rows, signed, unsigned):
    """Compare integers before conversion to float, retaining all integer bits."""
    body = []
    for row, (value, signed_expected, unsigned_expected) in enumerate(rows):
        body.append(Load("raw", "x", (literal(row),), dtype=source_dtype))
        for source_index, source in enumerate(
            (Literal(value, source_dtype), Var("raw"))
        ):
            for destination_index, (destination, expected) in enumerate(
                ((signed, signed_expected), (unsigned, unsigned_expected))
            ):
                body.extend(
                    (
                        Let("converted", source, destination),
                        Let(
                            "matches",
                            expression("==", Var("converted"), Literal(expected, destination)),
                            F32,
                        ),
                        Write(
                            "out",
                            (literal(row), literal(2 * source_index + destination_index)),
                            "matches",
                            value_dtype=F32,
                        ),
                    )
                )
    return typed_case(
        name, source_dtype, [row[0] for row in rows], body,
        np.ones((len(rows), 4), np.float32),
    )


def cast_edge_cases():
    # Halfway between neighboring f32 numbers, offset by one exact integer.
    # Converting through Python float loses the offset and can round twice.
    signed_midpoint = (1 << 62) + (1 << 38)
    unsigned_midpoint = (1 << 63) + (1 << 39)
    for name, dt, values in (
        (
            "signed64_to_float32",
            I64,
            [
                -(1 << 63), (1 << 63) - 1,
                signed_midpoint - 1, signed_midpoint, signed_midpoint + 1,
                -signed_midpoint + 1, -signed_midpoint, -signed_midpoint - 1,
                (1 << 53) - 1, (1 << 53) + 1,
            ],
        ),
        (
            "unsigned64_to_float32",
            U64,
            [
                0, (1 << 64) - 1,
                unsigned_midpoint - 1, unsigned_midpoint, unsigned_midpoint + 1,
                (1 << 53) - 1, (1 << 53) + 1,
            ],
        ),
    ):
        yield floating_destination(
            name, dt, values, [integer_float32_reference(v) for v in values]
        )
        # A typed f32 literal may still be spelled with an exact integer in
        # JSON. It must round once just like the loaded integer conversion.
        yield floating_destination(
            "integer_spelled_" + name,
            dt,
            values,
            [integer_float32_reference(v) for v in values],
            literal_dtype=F32,
        )

    midpoint = 1.0 + 2**-24
    half_subnormal = 2**-150
    values = [
        -0.0, 0.0, math.nextafter(midpoint, 0), midpoint,
        math.nextafter(midpoint, math.inf),
        math.nextafter(half_subnormal, 0), half_subnormal,
        math.nextafter(half_subnormal, math.inf), -half_subnormal,
        1e300, -1e300, math.inf, -math.inf, math.nan,
    ]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        reference = np.asarray(values, np.float64).astype(np.float32)
    yield floating_destination("float64_to_float32", F64, values, reference)

    yield integer_destination(
        "float32_to_integer32", F32,
        [
            (-3.75, -3, 0), (3.75, 3, 3), (-0.0, 0, 0), (0.0, 0, 0),
            (2147483648.0, (1 << 31) - 1, 1 << 31),
            (4294967296.0, (1 << 31) - 1, (1 << 32) - 1),
            (-1e20, -(1 << 31), 0),
            (math.inf, (1 << 31) - 1, (1 << 32) - 1),
            (-math.inf, -(1 << 31), 0), (math.nan, 0, 0),
        ],
        I32, U32,
    )
    yield integer_destination(
        "float64_to_integer64", F64,
        [
            (-3.75, -3, 0), (3.75, 3, 3), (-0.0, 0, 0),
            (float(1 << 63), (1 << 63) - 1, 1 << 63),
            (float(1 << 64), (1 << 63) - 1, (1 << 64) - 1),
            (-1e300, -(1 << 63), 0),
            (math.inf, (1 << 63) - 1, (1 << 64) - 1),
            (-math.inf, -(1 << 63), 0),
        ],
        I64, U64,
    )


class CastEdgeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_constant_runtime_casts_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in cast_edge_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    assemble(compile_file(path), "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_constant_runtime_casts_match_independent_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in cast_edge_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    kernel = value[1]["kernels"][0]
                    request = {
                        key: kernel[key] for key in (
                            "entry", "grid", "block", "smem_bytes", "zero_outputs", "args"
                        )
                    }
                    result = local_run(
                        {**request, "ptx": compile_file(path), "adapt_target": True}
                    )
                    self.assertTrue(result.get("ok"), result)
                    raw = result["outputs"]["out"]
                    actual = np.asarray(raw["values"], np.float32).reshape(raw["shape"])
                    expected = value[2]["out"]
                    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                    valid = ~np.isnan(expected)
                    np.testing.assert_array_equal(actual[valid], expected[valid])
                    zeros = valid & (expected == 0)
                    np.testing.assert_array_equal(
                        np.signbit(actual[zeros]), np.signbit(expected[zeros])
                    )


if __name__ == "__main__":
    unittest.main()
