#!/usr/bin/env python3
"""Instrument selected typed IR fixtures and verify their independent outputs.

Each tool/case runs in its own CUDA worker under Compute Sanitizer. Results,
requests, PTX, complete logs, and source hashes are retained in a unique run
directory. Local target adaptation requires an explicit command-line option.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from ir_cases import write_case
from test_contract_edges import contract_edge_cases
from test_load_packing import load_packing_cases
from tools.check_compiler import compare_outputs, compile_ptx, make_args

DEFAULT_CASES = (
    "named_barrier_partial_cta",
    "named_barrier_nested_reuse",
    "named_barrier_tree_broadcast",
)
PACKING_CASES = (
    "packing_width4_shift0",
    "packing_width4_shift1",
    "packing_store_boundary",
    "packing_conditional_store_boundary",
    "packing_deferred_name_binding",
)
TOOLS = ("memcheck", "racecheck", "synccheck")
SOURCE_FILES = (
    "compile.py",
    "ptx_backend.py",
    "tests/test_contract_edges.py",
    "tests/test_load_packing.py",
    "tests/ir_cases.py",
    "tools/check_compiler.py",
    "tools/check_sanitizer.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints() -> dict:
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def find_sanitizer(explicit: Path | None) -> Path:
    if explicit:
        candidate = explicit.resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(candidate)
    located = shutil.which("compute-sanitizer")
    if located:
        return Path(located).resolve()
    if os.name == "nt":
        roots = []
        if os.environ.get("CUDA_PATH"):
            roots.append(Path(os.environ["CUDA_PATH"]))
        roots.extend(
            sorted(
                Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("v*"),
                reverse=True,
            )
        )
        for root in roots:
            candidate = root / "compute-sanitizer/compute-sanitizer.exe"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("Compute Sanitizer not found; pass --sanitizer PATH")


def clean_summary(tool: str, log: str) -> tuple[bool, list[str]]:
    lines = [line for line in log.splitlines() if "SUMMARY" in line]
    pattern = (
        r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)"
        if tool == "racecheck"
        else r"ERROR SUMMARY: 0 errors"
    )
    # Require a recognized clean summary, not just a zero process status.
    return (
        bool(lines) and all(re.search(pattern + r"\s*$", line) for line in lines),
        lines,
    )


def text_output(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def run_one(sanitizer, tool, request_path, folder, expected, metadata, timeout):
    result_path = folder / f"{tool}.worker.json"
    log_path = folder / f"{tool}.log"
    command = [
        str(sanitizer),
        "--tool",
        tool,
        "--target-processes",
        "all",
        "--error-exitcode",
        "99",
        "--log-file",
        str(log_path),
        sys.executable,
        str(ROOT / "tools/check_compiler.py"),
        "--worker",
        str(request_path),
        str(result_path),
    ]
    started = time.perf_counter()
    row = {"case": metadata["id"], "tool": tool, "command": command, "pass": False}
    try:
        process = subprocess.run(
            command, capture_output=True, timeout=timeout, cwd=ROOT
        )
        row.update(
            returncode=process.returncode,
            stdout=text_output(process.stdout),
            stderr=text_output(process.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        row.update(
            returncode=None,
            timeout=True,
            stdout=text_output(exc.stdout),
            stderr=text_output(exc.stderr),
        )
    row["elapsed_seconds"] = time.perf_counter() - started
    (folder / f"{tool}.stdout.log").write_text(row["stdout"], encoding="utf-8")
    (folder / f"{tool}.stderr.log").write_text(row["stderr"], encoding="utf-8")
    log = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )
    row["sanitizer_clean"], row["summary_lines"] = clean_summary(tool, log)
    worker = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else {}
    )
    row["worker"] = worker
    row["comparisons"] = []
    if worker.get("ok"):
        try:
            if any(
                output.get("truncated") for output in worker.get("outputs", {}).values()
            ):
                raise ValueError("worker output is truncated")
            actual = {
                name: np.asarray(output["values"]).reshape(output["shape"])
                for name, output in worker.get("outputs", {}).items()
            }
            row["comparisons"] = compare_outputs(
                actual, expected, metadata["rtol"], metadata["atol"]
            )
            row["pass"] = (
                row["returncode"] == 0
                and row["sanitizer_clean"]
                and bool(row["comparisons"])
                and all(check["pass"] for check in row["comparisons"])
            )
        except Exception as exc:
            row["comparison_error"] = type(exc).__name__ + ": " + str(exc)
    row["artifacts"] = {
        path.name: {"path": str(path), "sha256": digest(path)}
        for path in (
            request_path,
            result_path,
            log_path,
            folder / f"{tool}.stdout.log",
            folder / f"{tool}.stderr.log",
        )
        if path.exists()
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("contract-edges", "load-packing"), default="contract-edges"
    )
    parser.add_argument(
        "--only", action="append", help="Fixture name; repeat or comma-separate"
    )
    parser.add_argument("--tools", nargs="+", choices=TOOLS, default=list(TOOLS))
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--sanitizer", type=Path)
    parser.add_argument("--adapt-local-target", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/validation/contract-edge-sanitizer",
    )
    args = parser.parse_args()
    factory = (
        contract_edge_cases if args.suite == "contract-edges" else load_packing_cases
    )
    defaults = DEFAULT_CASES if args.suite == "contract-edges" else PACKING_CASES
    values = {value[1]["id"]: value for value in factory()}
    if args.list_cases:
        print("\n".join(values))
        return 0
    names = (
        list(dict.fromkeys(name for item in args.only for name in item.split(",")))
        if args.only
        else list(defaults)
    )
    missing = set(names) - values.keys()
    if missing:
        parser.error("unknown fixtures: " + ", ".join(sorted(missing)))
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    sanitizer = find_sanitizer(args.sanitizer)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output)).resolve()
    hashes = fingerprints()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "source_sha256": hashes,
        "emmy_version": importlib.metadata.version("emmy-ml"),
        "python": sys.version,
        "python_executable": sys.executable,
        "sanitizer": str(sanitizer),
        "sanitizer_sha256": digest(sanitizer),
        "run_directory": str(run_dir),
        "selected_cases": names,
        "tools": list(dict.fromkeys(args.tools)),
        "adapt_local_target": args.adapt_local_target,
        "cases": [],
        "executions": [],
        "ok": False,
        "limitations": [
            "Instrumentation executes one local launch per case/tool; timings include instrumentation and are not performance measurements.",
            "Target adaptation changes only the PTX target directive and is recorded by the CUDA worker.",
            "These fixtures do not contain cp.async; this suite does not certify native asynchronous copies or unseen schedules.",
        ],
    }

    def save():
        summary["instrumented_executions"] = len(summary["executions"])
        summary["instrumented_numerical_passes"] = sum(
            row["worker"].get("ok", False)
            and bool(row["comparisons"])
            and all(check["pass"] for check in row["comparisons"])
            for row in summary["executions"]
        )
        summary["fully_passing_executions"] = sum(
            row["pass"] for row in summary["executions"]
        )
        payload = json.dumps(summary, indent=2)
        (run_dir / "summary.json").write_text(payload, encoding="utf-8")
        (output / "summary.json").write_text(payload, encoding="utf-8")

    try:
        for name in names:
            value = values[name]
            metadata, expected = value[1], value[2]
            folder = run_dir / name
            ir = write_case(folder, value)
            ptx, compile_seconds = compile_ptx(ROOT / "compile.py", ir)
            ptx_path = folder / "emitted.ptx"
            ptx_path.write_bytes(ptx.encode("ascii"))
            kernel = metadata["kernels"][0]
            request = {
                key: kernel[key]
                for key in ("entry", "grid", "block", "smem_bytes", "zero_outputs")
            }
            request.update(
                ptx=ptx,
                args=make_args(kernel, {}, 0, "normal"),
                adapt_target=args.adapt_local_target,
                warmup=0,
                iters=1,
            )
            request_path = folder / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            summary["cases"].append(
                {
                    "id": name,
                    "ir_sha256": digest(ir),
                    "ptx_sha256": digest(ptx_path),
                    "compile_seconds": compile_seconds,
                    "expected_sha256": digest(folder / "expected.npz"),
                }
            )
            for tool in summary["tools"]:
                if fingerprints() != hashes:
                    raise RuntimeError("source files changed during instrumentation")
                row = run_one(
                    sanitizer,
                    tool,
                    request_path,
                    folder,
                    expected,
                    metadata,
                    args.timeout,
                )
                if row["worker"].get("target_adapted"):
                    loaded = re.sub(
                        r"(\.target\s+)sm_\d+",
                        lambda match: match[1] + row["worker"]["arch"],
                        ptx,
                        count=1,
                    )
                else:
                    loaded = ptx
                row["loaded_ptx_sha256"] = hashlib.sha256(
                    loaded.encode("ascii")
                ).hexdigest()
                summary["executions"].append(row)
                save()
                print(f"{'PASS' if row['pass'] else 'FAIL'} {name} {tool}", flush=True)
        summary["source_unchanged_after_check"] = fingerprints() == hashes
        summary["ok"] = summary["source_unchanged_after_check"] and all(
            row["pass"] for row in summary["executions"]
        )
    except Exception as exc:
        summary["error"] = type(exc).__name__ + ": " + str(exc)
        print(summary["error"], file=sys.stderr)
    finally:
        save()
    print("Report: " + str(output / "summary.json"))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
