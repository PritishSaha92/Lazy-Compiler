"""Run with python -m unittest discover -s tests -v.

Set EMMY_GPU_TESTS=1 to run small semantic regressions on the local CUDA driver.
Assembly and semantic execution are separate checks: assembling is not evidence
that a kernel returns the correct answer.
"""

import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from subprocess import CompletedProcess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ir_cases import cases, write_case
from tools.check_compiler import assemble, compare_outputs, compile_ptx, local_run


class HarnessTests(unittest.TestCase):
    def test_assembler_diagnostics_preserve_both_streams(self):
        # The Windows ptxas installation reported invalid predicate arithmetic
        # on stdout, leaving the old stderr-only exception completely empty.
        for stdout, stderr in (
            ("ptxas: invalid predicate arithmetic", ""),
            ("", "ptxas: invalid entry"),
            ("ptxas: invalid operand", "assembly failed"),
        ):
            with self.subTest(stdout=stdout, stderr=stderr):
                result = CompletedProcess(["ptxas"], 1, stdout=stdout, stderr=stderr)
                with (
                    patch("tools.check_compiler.shutil.which", return_value="ptxas"),
                    patch("tools.check_compiler.subprocess.run", return_value=result),
                ):
                    with self.assertRaises(RuntimeError) as failure:
                        assemble(".version 8.0\n", "sm_89")
                for diagnostic in (stdout, stderr):
                    if diagnostic:
                        self.assertIn(diagnostic, str(failure.exception))

    def test_shape_mismatch_is_not_broadcast(self):
        with self.assertRaises(ValueError):
            compare_outputs({"x": np.ones((2, 1))}, {"x": np.ones((1, 2))}, 0, 0)

    def test_nan_does_not_pass(self):
        row = compare_outputs(
            {"x": np.array([np.nan])}, {"x": np.array([np.nan])}, 0, 0
        )[0]
        self.assertFalse(row["pass"])

    def test_relative_tolerance_uses_reference_magnitude(self):
        row = compare_outputs(
            {"x": np.array([111.0])}, {"x": np.array([100.0])}, 0.1, 0
        )[0]
        self.assertFalse(row["pass"])


class CompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "compile.py").exists():
            raise unittest.SkipTest("compile.py not yet present")

    def test_every_public_kernel_emits_ascii_ptx(self):
        for path in sorted((ROOT / "problems").glob("*/*.ir.json")):
            with self.subTest(kernel=path.parent.name):
                ptx, _ = compile_ptx(ROOT / "compile.py", path)
                raw = json.loads(path.read_text())
                entry = next(
                    node["op_fields"]["name"]
                    for node in raw["nodes"].values()
                    if node["op"] == "KernelOp"
                )
                self.assertRegex(ptx, r"\.entry\s+" + re.escape(entry) + r"\s*\(")

    def test_compile_without_reference_or_expression_metadata(self):
        source = ROOT / "problems/p00_scalar_add"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            meta = json.loads((source / "meta.json").read_text())
            meta.pop("code", None)
            (folder / "meta.json").write_text(json.dumps(meta))
            path = next(source.glob("*.ir.json"))
            shutil.copyfile(path, folder / path.name)
            ptx, _ = compile_ptx(ROOT / "compile.py", folder / path.name)
            self.assertIn(meta["kernels"][0]["entry"], ptx)

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_synthetic_kernels_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    ptx, _ = compile_ptx(ROOT / "compile.py", path)
                    assemble(ptx, "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_synthetic_kernels_match_independent_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    ptx, _ = compile_ptx(ROOT / "compile.py", path)
                    meta = value[1]
                    kernel = meta["kernels"][0]
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
                    result = local_run({**request, "ptx": ptx, "adapt_target": True})
                    self.assertTrue(result.get("ok"), result)
                    got = {
                        name: np.asarray(output["values"], dtype=np.float32).reshape(
                            output["shape"]
                        )
                        for name, output in result["outputs"].items()
                    }
                    checks = compare_outputs(got, value[2], meta["rtol"], meta["atol"])
                    self.assertTrue(all(check["pass"] for check in checks), checks)


if __name__ == "__main__":
    unittest.main()
