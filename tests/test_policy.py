"""Verify the real CLI's numerical-policy boundary on exact cancellation cases."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from ir_cases import write_case
from test_precision import precision_cases
from tools.check_compiler import local_run

ROOT = Path(__file__).resolve().parents[1]


class PolicyTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_cli_preserves_exact_and_selects_compensated_tolerant_contracts(self):
        # Exact mathematical identities in this fixture distinguish product and
        # addition rounding. Strict sequential FP32 produces zero for all rows;
        # Dot2 recovers the independently known small integer answers.
        value = next(precision_cases())
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), value)
            meta_path = path.with_name("meta.json")
            for tolerance in (None, 0.0, 1e-6):
                with self.subTest(tolerance=tolerance):
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    for key in ("rtol", "atol"):
                        if tolerance is None:
                            meta.pop(key, None)
                        else:
                            meta[key] = tolerance
                    meta_path.write_text(json.dumps(meta), encoding="utf-8")
                    emitted = subprocess.run(
                        [sys.executable, str(ROOT / "compile.py"), str(path)],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    self.assertEqual(emitted.stderr, "")
                    self.assertTrue(emitted.stdout.isascii())
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
                    result = local_run(
                        {**request, "ptx": emitted.stdout, "adapt_target": True}
                    )
                    self.assertTrue(result.get("ok"), result)
                    for name, mathematical in value[2].items():
                        out = result["outputs"][name]
                        actual = np.asarray(out["values"]).reshape(out["shape"])
                        expected = (
                            mathematical if tolerance else np.zeros_like(mathematical)
                        )
                        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
