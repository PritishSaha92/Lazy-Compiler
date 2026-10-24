"""Run the unmodified organizer client once per public problem, retaining logs.

This is a correctness rerun, not a comparative performance benchmark. The
compiler and problem files are hashed before and after the serialized run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.report_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    problems = sorted(p.parent for p in (ROOT / "problems").glob("*/meta.json"))
    if len(problems) != 15:
        raise ValueError(f"Expected 15 public packages, found {len(problems)}")
    tracked = [ROOT / n for n in ("compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt", "submit.py")]
    tracked += [p for folder in problems for p in folder.iterdir()
                if p.name in ("meta.json", "expected.npz") or p.name.endswith(".ir.json")]
    before = {str(p.relative_to(ROOT)): digest(p) for p in tracked}
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Unmodified submit.py, default compiler, one canonical-input run per public problem",
        "python": sys.version, "source_and_input_sha256": before,
        "planned_problems": [p.name for p in problems], "rows": [], "pass": False,
    }
    report_path = output / "summary.json"
    for folder in problems:
        command = [sys.executable, "submit.py", folder.name, "--compiler", "compile.py"]
        start = time.monotonic()
        timeout = False
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=600)
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code = exc.stdout or b"", exc.stderr or b"", None
            timeout = True
        stdout_path = output / f"{folder.name}.stdout.txt"
        stderr_path = output / f"{folder.name}.stderr.txt"
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        text = stdout.decode("utf-8", errors="replace")
        import numpy as np
        with np.load(folder / "expected.npz", allow_pickle=False) as expected:
            names = sorted(expected.files)
        observed = {name: f"PASS  {name}:" in text for name in names}
        row = {"problem": folder.name, "command": command, "returncode": code,
               "timeout": timeout, "elapsed_seconds": time.monotonic() - start,
               "expected_outputs_observed": observed,
               "stdout": stdout_path.name, "stdout_sha256": digest(stdout_path),
               "stderr": stderr_path.name, "stderr_sha256": digest(stderr_path),
               "pass": code == 0 and bool(names) and all(observed.values())}
        report["rows"].append(row)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{folder.name}: {'PASS' if row['pass'] else 'FAIL'}", flush=True)
    after = {str(p.relative_to(ROOT)): digest(p) for p in tracked}
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                  tracked_unchanged=before == after,
                  source_and_input_sha256_after=after,
                  passed_problems=sum(r["pass"] for r in report["rows"]))
    report["pass"] = len(report["rows"]) == 15 and report["passed_problems"] == 15 and before == after
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "passed": report["passed_problems"],
                      "report": str(report_path)}), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
