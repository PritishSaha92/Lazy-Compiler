"""Compare current CLI output with PTX hashes in retained execution reports.

This is translation identity evidence, not a substitute for exercising a changed
code path. Run targeted tests for newly supported or corrected IR constructs too.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.check_compiler import compile_ptx, compiler_fingerprints  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts/validation/final-translation-identity.json",
    )
    args = parser.parse_args()
    compiler = ROOT / "compile.py"
    before = compiler_fingerprints(compiler)
    results = []
    sources = []
    for report_name, problem_root in (
        ("final-public.json", ROOT / "problems"),
        ("final-fresh.json", ROOT / "artifacts/validation/stress"),
    ):
        source = ROOT / "artifacts/validation" / report_name
        previous = json.loads(source.read_text())
        assert all(row.get("pass") is True for row in previous["results"])
        sources.append(
            {
                "report": report_name,
                "sha256": sha256(source),
                "compiler_sources_sha256": previous[
                    "report_time_compiler_sources_sha256"
                ],
            }
        )
        for problem in previous["results"]:
            for kernel in problem["kernels"]:
                folder = problem_root / problem["problem"]
                path = folder / (kernel["entry"] + ".ir.json")
                ptx, elapsed = compile_ptx(compiler, path)
                digest = hashlib.sha256(ptx.encode("ascii")).hexdigest()
                results.append(
                    {
                        "problem": problem["problem"],
                        "entry": kernel["entry"],
                        "ir_sha256": sha256(path),
                        "meta_sha256": sha256(folder / "meta.json"),
                        "previous_ptx_sha256": kernel["ptx_sha256"],
                        "current_ptx_sha256": digest,
                        "compile_seconds": elapsed,
                        "identical": digest == kernel["ptx_sha256"],
                    }
                )
                print(
                    problem["problem"],
                    "IDENTICAL" if results[-1]["identical"] else "CHANGED",
                )
    after = compiler_fingerprints(compiler)
    assert before == after, "Compiler changed while identity checks were running"
    # Public value stress must have exercised exactly the same translations.
    stress_path = ROOT / "artifacts/validation/final-public-stress.json"
    stress = json.loads(stress_path.read_text())
    public_hashes = {
        (row["problem"], row["entry"]): row["previous_ptx_sha256"] for row in results
    }
    for problem in stress["results"]:
        assert problem.get("pass") is True
        for kernel in problem["kernels"]:
            assert (
                public_hashes[problem["problem"], kernel["entry"]]
                == kernel["ptx_sha256"]
            )
    sources.append(
        {
            "report": stress_path.name,
            "sha256": sha256(stress_path),
            "compiler_sources_sha256": stress["report_time_compiler_sources_sha256"],
        }
    )
    report = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "current_compiler_sources_sha256": after,
        "source_reports": sources,
        "results": results,
        "identical_count": sum(row["identical"] for row in results),
        "checked_count": len(results),
        "pass": all(row["identical"] for row in results),
        "scope": "Byte-identical PTX preserves recorded execution evidence for these inputs only; changed IR paths require independent tests.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
