"""Exercise the feature-example evaluator without network or GPU access.

The actual dtype-aware checker parses metadata, loads independent NPZ references,
constructs launch requests and compares outputs. Only compilation and HTTP are
mocked. These tests certify the evaluator's failure accounting, not GPU kernels.
"""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import check_feature_examples as driver


class FeatureRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="emmy-feature-runner-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt",
                     "tools/check_compiler.py", "tools/selector_study.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("mock source\n", encoding="utf-8")
        self.package = self.root / "example"
        self.package.mkdir()
        self.expected = np.array([1.0000000000000002, -3.5], dtype=np.float64)
        np.savez(self.package / "expected.npz", y=self.expected)
        metadata = {
            "id": "example", "rtol": 0, "atol": 0,
            "graph_inputs": ["x"], "graph_outputs": ["y"],
            "kernels": [{"entry": "k_example", "grid": [1, 1, 1], "block": [32, 1, 1],
                         "smem_bytes": 0, "zero_outputs": [], "arg_order": ["x", "y"],
                         "args": [{"name": "x", "dtype": "f64", "shape": [2], "data": self.expected.tolist()},
                                  {"name": "y", "dtype": "f64", "shape": [2], "output": True}]}],
        }
        (self.package / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        (self.package / "k_example.ir.json").write_text("{}\n", encoding="utf-8")
        self.destination = self.root / "report"

    def run_driver(self, *, mismatch=False, transport_error=False, changed=None):
        def fake_post(server, endpoint, payload, attempt):
            self.assertEqual(server, "http://mock.invalid")
            if transport_error:
                attempt({"exception": "mock connection dropped"})
                raise ConnectionError("mock connection dropped")
            if endpoint == "/compile":
                result = {"ok": True, "entry_points": ["k_example"], "cubin_id": "mock-cubin"}
            else:
                self.assertEqual(endpoint, "/run")
                self.assertEqual(payload["cubin_id"], "mock-cubin")
                self.assertEqual(payload["args"][0]["data"], self.expected.tolist())
                self.assertEqual(payload["warmup"], 0)
                self.assertEqual(payload["iters"], 1)
                values = self.expected.copy()
                if mismatch:
                    values[0] += 1
                result = {"ok": True, "outputs": {"y": {"dtype": "f64", "shape": [2], "values": values.tolist()}}}
                if changed is not None:
                    (self.root / changed).write_text("changed after execution\n", encoding="utf-8")
            attempt({"status_code": 200, "response": result})
            return result

        old_post = driver.checker.remote_post
        argv = ["check_feature_examples.py", str(self.package), "--report-dir", str(self.destination),
                "--compiler", str(self.root / "compile.py"), "--server", "http://mock.invalid"]
        with (patch.object(driver, "ROOT", self.root), patch.object(sys, "argv", argv),
              patch.object(driver.checker, "compile_ptx", return_value=(".version 8.0\n", 0.001)),
              patch.object(driver.transport, "audit_remote_post", side_effect=fake_post),
              patch.object(driver.checker, "local_run", side_effect=AssertionError("local GPU must not be called")),
              redirect_stdout(io.StringIO())):
            status = driver.main()
        self.assertIs(driver.checker.remote_post, old_post)
        report = json.loads((self.destination / "summary.json").read_text(encoding="utf-8"))
        for event in report["events"]:
            base = self.destination / "events" / f"{event['index']:04d}"
            self.assertEqual(hashlib.sha256(base.with_suffix(".request.json").read_bytes()).hexdigest(), event["request_sha256"])
            for attempt in event["http"]:
                self.assertEqual(hashlib.sha256((self.destination / attempt["path"]).read_bytes()).hexdigest(), attempt["sha256"])
            if event["status"] == "received":
                self.assertEqual(hashlib.sha256(base.with_suffix(".response.json").read_bytes()).hexdigest(), event["response_sha256"])
        return status, report

    def test_pass_preserves_f64_output_and_wire_evidence(self):
        status, report = self.run_driver()
        self.assertEqual(status, 0)
        self.assertTrue(report["pass"])
        self.assertEqual((report["planned_checks"], report["checks"], report["passed_checks"]), (1, 1, 1))
        self.assertEqual([event["endpoint"] for event in report["events"]], ["/compile", "/run"])
        raw = json.loads((self.destination / "events/0001.response.json").read_text())
        self.assertEqual(raw["outputs"]["y"]["values"][0], 1.0000000000000002)
        self.assertTrue(report["tracked_unchanged"])
        self.assertEqual(len(report["compiler_sha256"]), 5)

    def test_numerical_failure_remains_a_failed_execution(self):
        status, report = self.run_driver(mismatch=True)
        self.assertEqual(status, 1)
        self.assertFalse(report["pass"])
        self.assertEqual((report["checks"], report["passed_checks"]), (1, 0))
        case = report["jobs"][0]["result"]["cases"][0]
        self.assertFalse(case["pass"])
        self.assertEqual(case["outputs"][0]["mismatches"], 1)
        self.assertEqual([e["status"] for e in report["events"]], ["received", "received"])

    def test_transport_exception_cannot_be_counted_as_a_pass(self):
        status, report = self.run_driver(transport_error=True)
        self.assertEqual(status, 1)
        self.assertFalse(report["pass"])
        self.assertEqual((report["checks"], report["passed_checks"]), (0, 0))
        self.assertIn("ConnectionError", report["jobs"][0]["error"])
        self.assertEqual(report["events"][0]["status"], "transport_error")
        self.assertNotIn("response_sha256", report["events"][0])

    def test_extension_source_change_rejects_successful_outputs(self):
        status, report = self.run_driver(changed="ptx_bulk.py")
        self.assertEqual(status, 1)
        self.assertFalse(report["pass"])
        self.assertEqual(report["passed_checks"], 1)
        self.assertNotEqual(report["compiler_sha256"], report["compiler_sha256_after"])

    def test_fixture_change_rejects_successful_outputs(self):
        status, report = self.run_driver(changed="example/k_example.ir.json")
        self.assertEqual(status, 1)
        self.assertFalse(report["pass"])
        self.assertFalse(report["tracked_unchanged"])
        self.assertEqual(report["passed_checks"], 1)


if __name__ == "__main__":
    unittest.main()
