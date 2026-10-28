"""Reconstruct phase8 fixtures without importing or executing the compiler.

Remainder answers are also recomputed with exact rational arithmetic and an
independent ties-to-even IEEE encoder, without NumPy remainder/fmod. Loop
answers are recomputed using small host recurrences with bounded iteration.
"""
from fractions import Fraction
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import types

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = WORKSPACE / "emmy-hackathon-2026" / "Final Emmy"
HERE = Path(__file__).resolve().parent


def forbidden(*args, **kwargs):
    raise AssertionError("Compiler, process or network execution is forbidden in oracle reconstruction")


def block_execution():
    backend = types.ModuleType("ptx_backend")
    backend.Compiler = backend.compile_file = forbidden
    backend.CompileError = ValueError
    sys.modules["ptx_backend"] = backend
    harness = types.ModuleType("tools.check_compiler")
    harness.assemble = harness.local_run = harness.compile_ptx = forbidden
    sys.modules["tools.check_compiler"] = harness
    class BlockedPopen:
        def __init__(self, *args, **kwargs):
            forbidden()
    # Keep Popen class-shaped so asyncio's import-time subclass definition is
    # legal; constructing any process remains forbidden.
    subprocess.Popen = BlockedPopen
    subprocess.run = subprocess.call = forbidden
    subprocess.check_call = subprocess.check_output = forbidden
    os.system = socket.create_connection = socket.socket.connect = forbidden


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_word(word, width):
    return word - (1 << width) if word & (1 << (width - 1)) else word


def round_fraction_bits(value, width, negative):
    """One exact nearest-even rounding of a binary rational to IEEE bits."""
    precision, emin, emax = (24, -126, 127) if width == 32 else (53, -1022, 1023)
    sign = int(negative) << (width - 1)
    value = abs(value)
    if not value:
        return signed_word(sign, width)
    numerator, denominator = value.numerator, value.denominator
    assert denominator & (denominator - 1) == 0
    exponent = numerator.bit_length() - denominator.bit_length()
    exponent = max(exponent, emin)
    spacing = exponent - precision + 1
    if spacing >= 0:
        denominator <<= spacing
    else:
        numerator <<= -spacing
    integer, residual = divmod(numerator, denominator)
    if residual * 2 > denominator or (residual * 2 == denominator and integer & 1):
        integer += 1
    if integer == (1 << precision):
        integer >>= 1
        exponent += 1
    if exponent > emax:
        bits = ((1 << (width - precision)) - 1) << (precision - 1)
    elif integer < (1 << (precision - 1)):
        bits = integer
    else:
        bits = ((exponent - emin + 1) << (precision - 1)) | (integer - (1 << (precision - 1)))
    return signed_word(sign | bits, width)


def remainder_word(x, y, width):
    """Exact rational floor remainder plus NumPy's exceptional-value rules."""
    nan = 0x7fc00000 if width == 32 else 0x7ff8000000000000
    negative = math.copysign(1.0, y) < 0
    if not math.isfinite(x) or math.isnan(y) or y == 0:
        return nan
    if math.isinf(y):
        if x == 0:
            return signed_word(int(negative) << (width - 1), width)
        if (x < 0) != (y < 0):
            precision = 24 if width == 32 else 53
            inf = ((1 << (width - precision)) - 1) << (precision - 1)
            return signed_word((int(negative) << (width - 1)) | inf, width)
        return round_fraction_bits(Fraction.from_float(x), width, negative)
    a, b = Fraction.from_float(x), Fraction.from_float(y)
    residual = a - (a // b) * b
    assert abs(residual) < abs(b)
    return round_fraction_bits(residual, width, negative)


def loop_reference(identifier, inputs):
    def run(j, end, step):
        seen = []
        while j < end:
            assert len(seen) < 32, "independent loop reference failed to terminate"
            seen.append(j)
            j = int(j + step)
        return [len(seen), sum(seen)]
    if identifier == "strided_i64_start_narrows":
        return [run(((int(x) + 2**31) % 2**32) - 2**31, 3, 1) for x in inputs["x"]]
    if identifier == "strided_f64_start_and_step":
        return [run(int(x), 6, float(step)) for x, step in zip(inputs["x"], inputs["y"], strict=True)]
    if identifier == "strided_unsigned_start_negative_end":
        return [run(tid, -1, 1) for tid in range(4)]
    if identifier == "strided_f64_end_snapshot":
        rows = []
        for value in inputs["x"]:
            bound, limit, j, seen, total_bound = int(value), float(value), 0, [], 0
            while j < bound:
                assert len(seen) < 32
                limit -= 1.0
                seen.append(j)
                total_bound += bound
                j += 1
            rows.append([len(seen), sum(seen), total_bound])
        return rows
    if identifier == "strided_end_reads_initialized_axis":
        j = 1
        return [run(j, j + 3, 1)]
    if identifier == "strided_live_fractional_step":
        j, step, seen = 2, -1.25, []
        while j < 6:
            assert len(seen) < 32
            step += 1.0
            seen.append(j)
            j = int(j + step)
        assert seen == [2, 1, 1, 2, 4]
        return [[len(seen), sum(seen)]]
    raise AssertionError(identifier)


def main():
    sources = [ROOT / name for name in (
        "ptx_backend.py", "tests/test_loop_bindings.py", "tests/test_floating_remainder.py",
        "tests/ir_cases.py", "tests/test_contract_edges.py", "tools/check_compiler.py")]
    packages = sorted((ROOT / "examples/phase8").glob("*/*/meta.json"))
    assert len(packages) == 10
    paths = sources + [p for m in packages for p in m.parent.iterdir() if p.is_file()]
    before = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    block_execution()
    sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]
    loops = importlib.import_module("test_loop_bindings")
    remainders = importlib.import_module("test_floating_remainder")
    regenerated = list(loops.loop_binding_cases()) + list(remainders.floating_remainder_cases())
    by_name = {meta["id"]: (graph, meta, expected) for graph, meta, expected in regenerated}
    assert len(by_name) == 10
    rows = []
    for meta_path in packages:
        meta = json.loads(meta_path.read_text())
        identifier = meta["id"]
        graph, rebuilt_meta, rebuilt_expected = by_name[identifier]
        assert rebuilt_meta == meta, identifier
        ir_paths = list(meta_path.parent.glob("*.ir.json"))
        assert len(ir_paths) == 1
        rebuilt_ir = json.loads(json.dumps(graph.to_dict(), default=repr))
        assert rebuilt_ir == json.loads(ir_paths[0].read_text()), identifier
        with np.load(meta_path.with_name("expected.npz"), allow_pickle=False) as data:
            expected = data["out"].copy()
            assert data.files == ["out"]
        assert expected.dtype == rebuilt_expected["out"].dtype
        assert expected.shape == rebuilt_expected["out"].shape
        assert expected.tobytes() == rebuilt_expected["out"].tobytes()
        args = {item["name"]: item for item in meta["kernels"][0]["args"]}
        raw_inputs = {name: arg["data"] for name, arg in args.items() if "data" in arg}
        if identifier.startswith("floating_"):
            width = 32 if "f32" in identifier else 64
            signed, floating = (np.int32, np.float32) if width == 32 else (np.int64, np.float64)
            for arg in args.values():
                assert arg["dtype"] == ("i32" if width == 32 else "i64")
            for values in raw_inputs.values():
                assert all(type(value) is int and -(2**(width - 1)) <= value < 2**(width - 1) for value in values)
                assert json.loads(json.dumps(values, allow_nan=False)) == values
            x = np.asarray(raw_inputs["x"], dtype=signed).view(floating)
            y = np.asarray(raw_inputs["y"], dtype=signed).view(floating)
            independent = np.asarray([remainder_word(float(a), float(b), width) for a, b in zip(x, y, strict=True)], dtype=signed)
            oracle = "Exact Fraction floor quotient and residual, independent RNE IEEE encoding; explicit exceptional-value rules"
            assert len(x) == len(y) == 528
        else:
            independent = np.asarray(loop_reference(identifier, raw_inputs), dtype=np.int32)
            oracle = "Bounded Python integer/double recurrence independent of compiler and fixture reference arrays"
        assert independent.dtype == expected.dtype and independent.shape == expected.shape, identifier
        assert independent.tobytes() == expected.tobytes(), identifier
        assert (meta["rtol"], meta["atol"]) == (0, 0)
        rows.append({"id": identifier, "pass": True, "elements": int(expected.size),
                     "dtype": str(expected.dtype), "shape": list(expected.shape), "oracle": oracle,
                     "expected_array_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                     "fixture_ir_and_metadata_regenerated_exactly": True})
    after = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    assert before == after
    report = {"schema": "emmy-phase8-independent-oracle-audit-v1", "pass": True,
              "scope": "Ten new fixture packages reconstructed without compiler execution; independent rational/recurrence oracles match exact expected arrays",
              "compiler_process_network_execution_blocked": True, "gpu_used": False,
              "source_and_fixture_files_unchanged": True, "files_sha256": before,
              "packages": len(rows), "elements": sum(row["elements"] for row in rows), "rows": rows}
    output = HERE / "final-oracle-audit.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("rows", "files_sha256")}, indent=2))


if __name__ == "__main__":
    main()
