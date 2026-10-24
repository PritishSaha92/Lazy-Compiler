"""Snapshot a candidate, then run the frozen 54-case correctness gate.

The optional strict policy uses the SAME candidate backend with compensation
disabled. It is an ablation, not a reconstruction of a historical source hash.
This script performs no interleaved timing study or optimization selection.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/validation/holdout-2026-10-23"
STRICT_WRAPPER = '''#!/usr/bin/env python3
"""Strict arithmetic ablation using the adjacent frozen candidate backend."""
import sys
from ptx_backend import CompileError, compile_file

if len(sys.argv) != 2:
    print("usage: python compile_strict.py kernel.ir.json", file=sys.stderr)
    raise SystemExit(2)
try:
    ptx = compile_file(sys.argv[1], compensated_dots=False, accurate_dots=False,
                       fuse_accumulations=False)
except (CompileError, ValueError, OSError, KeyError, TypeError) as exc:
    print(f"compile_strict.py: {exc}", file=sys.stderr)
    raise SystemExit(1)
sys.stdout.write(ptx)
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_corpus():
    subprocess.run(
        [sys.executable, str(ROOT / "tests/freeze_holdout.py"), "--verify"], check=True
    )
    return json.loads((CORPUS / "manifest.json").read_text())


def check_snapshot(folder):
    snapshot = json.loads((folder / "candidate-source.json").read_text())
    for name, fingerprint in snapshot["files_sha256"].items():
        if digest(folder / name) != fingerprint:
            raise ValueError(f"Frozen candidate changed: {name}")
    if snapshot["holdout_manifest_sha256"] != digest(CORPUS / "manifest.json"):
        raise ValueError(
            "Held-out manifest differs from the candidate's recorded split"
        )
    return snapshot


def prepare(args):
    check_corpus()
    folder = args.snapshot.resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Candidate snapshot must use an empty directory")
    files = {
        name: (ROOT / name).read_bytes() for name in ("compile.py", "ptx_backend.py")
    }
    backend_hash = hashlib.sha256(files["ptx_backend.py"]).hexdigest()
    if backend_hash != args.expected_backend_sha:
        raise ValueError(
            f"Candidate backend is {backend_hash}, not the expected frozen hash"
        )
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    (folder / "compile_strict.py").write_text(STRICT_WRAPPER, encoding="utf-8")
    for name, data in files.items():
        if (ROOT / name).read_bytes() != data:
            raise ValueError(
                "Source changed during snapshot creation; do not evaluate this snapshot"
            )
    record = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_manifest_sha256": digest(CORPUS / "manifest.json"),
        "files_sha256": {
            name: digest(folder / name) for name in (*files, "compile_strict.py")
        },
        "policies": {
            "default": "Unmodified compile.py from the frozen candidate",
            "strict": "Same candidate backend; compensated_dots=False, accurate_dots=False, fuse_accumulations=False; other defaults unchanged",
        },
        "historical_baseline_available": False,
        "comparison_scope": "Optional strict run is a numerical-policy ablation, not the unavailable historical 3e68 source revision.",
        "compiler_evaluated_during_prepare": False,
    }
    (folder / "candidate-source.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Frozen candidate snapshot: {folder}")
    print(f"Backend SHA-256: {backend_hash}")
    return 0


def run(args):
    manifest = check_corpus()
    folder = args.snapshot.resolve()
    snapshot = check_snapshot(folder)
    output = args.report_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty report directory; preserve previous attempts")
    output.mkdir(parents=True, exist_ok=True)
    # The manifest, not an unrestricted directory glob, determines membership.
    problems = [CORPUS / item["id"] for item in manifest["workloads"]]
    assert len(problems) == 18 and all((p / "meta.json").is_file() for p in problems)
    reports = []
    env = dict(os.environ, PYTHONUTF8="1")
    for policy in args.policies:
        compiler = folder / (
            "compile.py" if policy == "default" else "compile_strict.py"
        )
        for name, seeds, pattern, per_problem in (
            ("normal", "37,89", "normal", 2),
            ("wide", "137", "wide", 1),
        ):
            check_snapshot(folder)
            report = output / f"{policy}-{name}.json"
            command = [
                sys.executable,
                "-u",
                str(ROOT / "tools/check_compiler.py"),
                "--mode",
                args.mode,
                "--compiler",
                str(compiler),
                "--references",
                str(ROOT / "tests/expressions_holdout.py"),
                "--seeds",
                seeds,
                "--patterns",
                pattern,
                "--warmup",
                "1",
                "--iters",
                "3",
                "--report",
                str(report),
            ]
            command.extend(str(p) for p in problems)
            with report.with_suffix(".log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    env=env,
                    bufsize=1,
                )
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
            check_snapshot(folder)
            record = {
                "policy": policy,
                "pattern": pattern,
                "report": report.name,
                "command": command,
                "returncode": status,
                "expected_numerical_cases": len(problems) * per_problem,
            }
            if report.exists():
                data = json.loads(report.read_text())
                cases = [
                    case for row in data["results"] for case in row.get("cases", [])
                ]
                record.update(
                    completed_numerical_cases=len(cases),
                    passed_numerical_cases=sum(
                        case.get("pass") is True for case in cases
                    ),
                    failed_workloads=[
                        row["problem"] for row in data["results"] if not row.get("pass")
                    ],
                    report_sha256=digest(report),
                )
            reports.append(record)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "scope": "Correctness gate; timings are incidental and not an interleaved performance comparison.",
                        "candidate_source": snapshot,
                        "reports": reports,
                        "planned_gates": len(args.policies) * 2,
                        "complete": len(reports) == len(args.policies) * 2,
                        "pass": len(reports) == len(args.policies) * 2
                        and all(
                            r["returncode"] == 0
                            and r.get("passed_numerical_cases")
                            == r["expected_numerical_cases"]
                            for r in reports
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
    check_corpus()
    return (
        0
        if all(
            r["returncode"] == 0
            and r.get("passed_numerical_cases") == r["expected_numerical_cases"]
            for r in reports
        )
        else 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    preparation = commands.add_parser(
        "prepare", help="Copy source bytes; no compiler invocation or GPU work"
    )
    preparation.add_argument("--snapshot", type=Path, required=True)
    preparation.add_argument("--expected-backend-sha", required=True)
    evaluation = commands.add_parser(
        "run", help="Run the authorized held-out correctness gate serially"
    )
    evaluation.add_argument("--snapshot", type=Path, required=True)
    evaluation.add_argument("--report-dir", type=Path, required=True)
    evaluation.add_argument("--mode", choices=("remote",), default="remote")
    evaluation.add_argument(
        "--policies", choices=("default", "strict"), nargs="+", default=["default"]
    )
    args = parser.parse_args()
    return prepare(args) if args.action == "prepare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
