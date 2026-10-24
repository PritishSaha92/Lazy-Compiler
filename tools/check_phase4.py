"""Serialize the new typed and interaction gates on the official runner.

This development tool builds independent fixture packages and snapshots variants.
It never launches a local GPU or changes the submission's default configuration.
"""

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from ir_cases import write_case
from test_type_capabilities import type_cases
from test_interactions import interaction_cases
from tools.check_compiler import SERVER, compiler_fingerprints


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--server", default=SERVER)
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    output = args.report_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    typed = [write_case(output / "inputs" / value[1]["id"], value).parent for value in type_cases()]
    interaction = [write_case(output / "inputs" / value[1]["id"], value).parent for value in interaction_cases()]
    if len(typed) != 19 or len(interaction) != 4:
        raise ValueError("Fixture roster changed; revise and review the explicit gate inventory")
    variants = {}
    for name, options in (("rolled", "unroll_budget=0"),
                          ("sync-unrolled", "synchronous_copy=True"),
                          ("sync-rolled", "synchronous_copy=True, unroll_budget=0")):
        destination = output / "variants" / name
        destination.mkdir(parents=True)
        for module in ("ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py"):
            if compiler.with_name(module).exists():
                shutil.copy2(compiler.with_name(module), destination / module)
        (destination / "compile.py").write_text(
            "import sys\nfrom ptx_backend import compile_file\n"
            f"sys.stdout.write(compile_file(sys.argv[1], {options}))\n", encoding="ascii")
        variants[name] = destination / "compile.py"
    async_cases = [p for p in interaction if "async_two_slots" in p.name]
    jobs = [("default", compiler, typed + interaction),
            ("rolled-interactions", variants["rolled"], interaction),
            ("sync-unrolled-async", variants["sync-unrolled"], async_cases),
            ("sync-rolled-async", variants["sync-rolled"], async_cases)]
    tracked = [Path(__file__), ROOT / "tests/test_type_capabilities.py", ROOT / "tests/test_interactions.py",
               ROOT / "tests/ir_cases.py", ROOT / "tools/check_compiler.py"]
    tracked += [p for p in output.rglob("*") if p.is_file()]
    hashes = {str(p): digest(p) for p in tracked}
    before = compiler_fingerprints(compiler)
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(), "pass": False,
               "planned_default_checks": 23, "planned_variant_checks": 6,
               "distinct_packages": 23, "sources_before_sha256": before,
               "inputs_helpers_variants_sha256": hashes, "jobs": [],
               "scope": "Development regressions, independent explicit-data oracles; no local GPU. Variants preserve launch metadata and arithmetic order."}
    summary_file = output / "summary.json"

    def save():
        summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    save()
    for label, cli, paths in jobs:
        report = output / (label + ".json")
        command = [sys.executable, "-X", "utf8", str(ROOT / "tools/check_compiler.py"),
                   *map(str, paths), "--compiler", str(cli), "--mode", "remote",
                   "--server", args.server, "--seeds", "0", "--warmup", "1", "--iters", "3",
                   "--report", str(report)]
        row = {"name": label, "command": command, "pass": False,
               "compiler_sha256": compiler_fingerprints(cli)}
        summary["jobs"].append(row)
        save()
        process = subprocess.run(command, cwd=ROOT, capture_output=True)
        (output / (label + ".log")).write_bytes(process.stdout + process.stderr)
        print((process.stdout + process.stderr).decode("utf-8", errors="replace"), end="", flush=True)
        row["returncode"] = process.returncode
        if report.exists():
            results = json.loads(report.read_text(encoding="utf-8"))["results"]
            cases = [c for result in results for c in result.get("cases", [])]
            row.update(report=report.name, report_sha256=digest(report), numerical_checks=len(cases),
                       numerical_passes=sum(c["pass"] for c in cases))
            row["pass"] = (process.returncode == 0 and len(results) == len(paths)
                           and all(r["pass"] and not r.get("skipped") for r in results)
                           and len(cases) == len(paths) and all(c["pass"] for c in cases))
        save()
    summary["sources_after_sha256"] = compiler_fingerprints(compiler)
    summary["tracked_sources_unchanged"] = hashes == {p: digest(Path(p)) for p in hashes}
    summary["pass"] = (before == summary["sources_after_sha256"] and summary["tracked_sources_unchanged"]
                       and all(row["pass"] for row in summary["jobs"]))
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"{'PASS' if summary['pass'] else 'FAIL'}: {summary_file}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
