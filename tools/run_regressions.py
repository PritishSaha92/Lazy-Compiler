"""Run the local regression gate with source fingerprints and a durable transcript.

GPU mode requires the assembler and rejects skipped tests. Ordinary mode runs
without a GPU and records skips explicitly; it is not an execution gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fingerprints():
    files = [
        ROOT / "compile.py",
        ROOT / "ptx_backend.py",
        ROOT / "ptx_tensor.py",
        ROOT / "ptx_bulk.py",
        ROOT / "requirements.txt",
        *sorted((ROOT / "tests").glob("*.py")),
        *sorted((ROOT / "tools").glob("*.py")),
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


class Transcript:
    def __init__(self, file):
        self.file = file

    def write(self, text):
        self.file.write(text)
        sys.stderr.write(text)

    def flush(self):
        self.file.flush()
        sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "artifacts/validation/regressions.json"
    )
    args = parser.parse_args()
    if args.gpu and not shutil.which("ptxas"):
        parser.error("The GPU gate requires ptxas on PATH")
    os.environ["EMMY_GPU_TESTS"] = "1" if args.gpu else "0"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    before = fingerprints()
    start = time.perf_counter()
    with args.report.with_suffix(".log").open("w", encoding="utf-8") as file:
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(stream=Transcript(file), verbosity=2).run(
            suite
        )
    elapsed = time.perf_counter() - start
    after = fingerprints()
    # Generation/evaluation scripts can coexist in tests/ without being loaded
    # by unittest. Their hashes remain visible, but only code actually imported
    # by this gate (plus the subprocess entry points) determines stability.
    executed = {"compile.py", "ptx_backend.py", str(Path("tools/check_compiler.py"))}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            try:
                relative = str(Path(path).resolve().relative_to(ROOT))
            except ValueError:
                continue
            if relative in before:
                executed.add(relative)
    executed_unchanged = all(before.get(p) == after.get(p) for p in executed)
    passed = (
        result.wasSuccessful()
        and executed_unchanged
        and (not args.gpu or not result.skipped)
    )
    report = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "gpu_execution_requested": args.gpu,
        "tests_run": result.testsRun,
        "skipped": [{"test": str(t), "reason": reason} for t, reason in result.skipped],
        "failures": [
            {"test": str(t), "traceback": trace} for t, trace in result.failures
        ],
        "errors": [{"test": str(t), "traceback": trace} for t, trace in result.errors],
        "seconds": elapsed,
        "sources_before_sha256": before,
        "sources_after_sha256": after,
        "sources_unchanged": before == after,
        "executed_sources": sorted(executed),
        "executed_sources_unchanged": executed_unchanged,
        "pass": passed,
        "transcript": args.report.with_suffix(".log").name,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{'PASS' if passed else 'FAIL'}: {result.testsRun} methods, {len(result.skipped)} skipped; {args.report}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
