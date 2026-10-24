"""CPU-only checks for the release gate's byte-preserving input transport."""

from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from ir_cases import write_case
from test_alignment_overflow import alignment_overflow_cases
from tools import check_release as release
from tools.check_compiler import input_array


def metadata(arg):
    return {
        "id": "transport_test",
        "kernels": [{"entry": "test", "args": [arg]}],
    }


class UnsignedInputTransportTests(unittest.TestCase):
    def test_full_unsigned_range_survives_float64_wire_coercion(self):
        for dtype, values in (
            ("u32", [0, 1, 2**31, 2**32 - 1]),
            ("u64", [0, 1, 2**53 + 1, 2**63, 2**64 - 1, 6148914691236517206]),
        ):
            with self.subTest(dtype=dtype):
                arg = {
                    "name": "x",
                    "dtype": dtype,
                    "shape": [2, len(values) // 2],
                    "data": values,
                }
                original = metadata(arg)
                saved = deepcopy(original)
                updated, records = release.unsigned_word_metadata(original)
                wire = updated["kernels"][0]["args"][0]
                self.assertEqual(original, saved)
                self.assertEqual(wire["dtype"], "i32")
                wire["data"] = [float(value) for value in wire["data"]]
                self.assertEqual(
                    input_array(arg, 0).tobytes(), input_array(wire, 0).tobytes()
                )
                self.assertEqual(
                    records[0]["original_bytes_sha256"],
                    records[0]["transport_bytes_sha256"],
                )
                self.assertTrue(records[0]["float64_roundtrip_exact"])

    def test_only_unsigned_payload_descriptors_change(self):
        for _, original, _ in alignment_overflow_cases():
            updated, records = release.unsigned_word_metadata(original)
            self.assertEqual(len(records), 1)
            for old_kernel, new_kernel in zip(
                original["kernels"], updated["kernels"], strict=True
            ):
                for old_arg, new_arg in zip(
                    old_kernel["args"], new_kernel["args"], strict=True
                ):
                    if old_arg["dtype"] in ("u32", "u64"):
                        for field in ("dtype", "shape", "data"):
                            new_arg[field] = deepcopy(old_arg[field])
            self.assertEqual(updated, original)

    def test_unsupported_unsigned_contracts_are_explicit_errors(self):
        base = {"name": "x", "dtype": "u64", "shape": [1], "data": [1]}
        for changes in (
            {"output": True},
            {"scalar": True},
            {"value": 1},
            {"data": None},
        ):
            arg = {**base, **changes}
            if arg["data"] is None:
                del arg["data"]
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                release.unsigned_word_metadata(metadata(arg))
        with (
            patch.object(release.sys, "byteorder", "big"),
            self.assertRaises(ValueError),
        ):
            release.unsigned_word_metadata(metadata(base))

    def test_supported_inputs_need_no_copy_or_compiler(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            original = metadata(
                {"name": "x", "dtype": "f32", "shape": [1], "data": [1.0]}
            )
            (source / "meta.json").write_text(json.dumps(original))
            destination = Path(temporary) / "wire"
            with patch.object(
                release,
                "compile_ptx",
                side_effect=AssertionError("unexpected compilation"),
            ):
                result, record = release.prepare_unsigned_transport(
                    source, destination, ROOT / "compile.py"
                )
            self.assertEqual(result, source)
            self.assertIsNone(record)
            self.assertFalse(destination.exists())

    def test_changed_ptx_is_rejected_without_mutating_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = write_case(
                Path(temporary) / "source", next(alignment_overflow_cases())
            ).parent
            before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
            with patch.object(
                release,
                "compile_ptx",
                side_effect=[(".version 8.0\n", 0), (".version 8.1\n", 0)],
            ):
                with self.assertRaisesRegex(ValueError, "changed generated PTX"):
                    release.prepare_unsigned_transport(
                        source, Path(temporary) / "wire", ROOT / "compile.py"
                    )
            self.assertEqual(
                before,
                {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()},
            )

    def test_four_alignment_packages_keep_exact_ir_references_and_cli_ptx(self):
        compiler_hashes = release.compiler_fingerprints(ROOT / "compile.py")
        with tempfile.TemporaryDirectory() as temporary:
            for value in alignment_overflow_cases():
                with self.subTest(case=value[1]["id"]):
                    source = write_case(
                        Path(temporary) / "source" / value[1]["id"], value
                    ).parent
                    original_hashes = {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in source.iterdir()
                        if p.is_file()
                    }
                    destination, record = release.prepare_unsigned_transport(
                        source,
                        Path(temporary) / "wire" / source.name,
                        ROOT / "compile.py",
                    )
                    self.assertEqual(
                        original_hashes,
                        {
                            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in source.iterdir()
                            if p.is_file()
                        },
                    )
                    self.assertEqual(
                        (source / "expected.npz").read_bytes(),
                        (destination / "expected.npz").read_bytes(),
                    )
                    for kernel in record["kernels"]:
                        self.assertTrue(kernel["ptx_byte_identical"])
                        self.assertEqual(
                            kernel["original_ptx_sha256"],
                            kernel["transport_ptx_sha256"],
                        )
                    # A second invocation must preserve its existing evidence.
                    with self.assertRaises(FileExistsError):
                        release.prepare_unsigned_transport(
                            source, destination, ROOT / "compile.py"
                        )
        self.assertEqual(
            compiler_hashes, release.compiler_fingerprints(ROOT / "compile.py")
        )

    def test_release_plan_keeps_213_cases_and_97_logical_packages(self):
        def fake_remote(command, **kwargs):
            self.assertEqual(command[command.index("--mode") + 1], "remote")
            self.assertEqual(Path(command[3]), ROOT / "tools/check_compiler.py")
            folders = command[4 : command.index("--compiler")]
            repetitions = len(command[command.index("--seeds") + 1].split(",")) * len(
                command[command.index("--patterns") + 1].split(",")
            )
            report = Path(command[command.index("--report") + 1])
            report.write_text(
                json.dumps(
                    {
                        "results": [
                            {"pass": True, "cases": [{"pass": True}] * repetitions}
                            for _ in folders
                        ]
                    }
                )
            )
            return SimpleNamespace(returncode=0, stdout=b"CPU mock only\n", stderr=b"")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "mock-release"
            argv = [
                "check_release.py",
                "--compiler",
                str(ROOT / "compile.py"),
                "--report-dir",
                str(output),
            ]
            with (
                patch.object(release.sys, "argv", argv),
                patch.object(
                    release, "compile_ptx", return_value=(".version 8.0\n", 0)
                ),
                patch.object(
                    release.subprocess, "run", side_effect=fake_remote
                ) as remote,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(release.main(), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(remote.call_count, 11)
            self.assertEqual(summary["numerical_cases"], 213)
            self.assertEqual(summary["distinct_workload_packages"], 97)
            self.assertEqual(summary["transported_input_packages"], 4)
            self.assertTrue(summary["inputs_and_helpers_unchanged"])
            self.assertTrue(summary["compiler_unchanged"])
            self.assertEqual(
                len(list((output / "typed-inputs").glob("*/meta.json"))), 30
            )
            self.assertEqual(
                len(list((output / "typed-wire-inputs").glob("*/meta.json"))), 4
            )


if __name__ == "__main__":
    unittest.main()
