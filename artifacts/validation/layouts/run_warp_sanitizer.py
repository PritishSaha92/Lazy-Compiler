"""Reproduce the layout fixture export and targeted warp-barrier sanitizer run."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_layouts import layout_cases
from ir_cases import write_case
from ptx_backend import compile_file
from tools.check_compiler import compare_outputs

folder = Path(__file__).resolve().parent
values = list(layout_cases())
sources = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (ROOT / "ptx_backend.py", ROOT / "compile.py", ROOT / "tests/test_layouts.py")
}
summary = {
    "description": "Targeted layout unittest run plus warp-barrier sanitizer executions",
    "source_sha256": sources,
    "unittest_command": "EMMY_GPU_TESTS=1 python -X utf8 -m unittest discover -s tests -p test_layouts.py -v",
    "unittest_log": "unittest.log",
    "distinct_programs": len(values),
    "numerical_cases": len(values),
    "assembly_checks": len(values) * 2,
    "assembly_target": "sm_89",
    "numerical_target": "sm_75, target adapted; synchronous-copy alternative",
    "native_async_numerical_execution": False,
    "cases": [value[1]["id"] for value in values],
    "sanitizer": [],
}
log = (folder / "unittest.log").read_text(encoding="utf-8")
if not log.rstrip().endswith("OK"):
    raise RuntimeError("The preceding targeted unittest run did not pass")

for value in values:
    path = write_case(folder / value[1]["id"], value)
    if value[1]["id"] != "warp_sync_uneven_work":
        continue
    kernel = value[1]["kernels"][0]
    request = {key: kernel[key] for key in (
        "entry", "grid", "block", "smem_bytes", "zero_outputs", "args"
    )}
    request.update(ptx=compile_file(path), adapt_target=True)
    request_path = folder / "warp_sync_uneven_work.request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    sanitizer = Path(
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/"
        "compute-sanitizer/compute-sanitizer.exe"
    )
    for tool in ("memcheck", "racecheck", "synccheck"):
        result_path = folder / f"warp_sync_uneven_work.{tool}.result.json"
        log_path = folder / f"warp_sync_uneven_work.{tool}.log"
        command = [
            str(sanitizer), "--tool", tool, "--target-processes", "all",
            "--error-exitcode", "99", "--log-file", str(log_path),
            sys.executable, str(ROOT / "tools/check_compiler.py"), "--worker",
            str(request_path), str(result_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        worker = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        actual = {
            name: np.asarray(output["values"]).reshape(output["shape"])
            for name, output in worker.get("outputs", {}).items()
        }
        comparisons = compare_outputs(actual, value[2], 0, 0)
        sanitizer_log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        record = {
            "tool": tool,
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "worker_ok": worker.get("ok", False),
            "comparisons": comparisons,
            "summary_lines": [line for line in sanitizer_log.splitlines() if "SUMMARY" in line],
        }
        summary["sanitizer"].append(record)
        (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(tool, result.returncode, record["worker_ok"], record["summary_lines"], flush=True)
        if result.returncode or not worker.get("ok") or not all(check["pass"] for check in comparisons):
            raise RuntimeError(f"{tool} failed: {record}")

(folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("Layout suite: 8 numeric cases, 16 assembly checks, 3 sanitizer executions passed.")
