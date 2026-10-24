"""Compare emission with a frozen prior version of this same Python compiler.

This offline regression check is not GPU execution and does not compile through
another compiler backend. Both versions independently emit their own PTX.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptx_backend import compile_file
from tools.check_compiler import compiler_fingerprints


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path, required=True)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("refusing to overwrite an earlier comparison")
    paths = sorted({p.resolve() for folder in args.roots for p in folder.rglob("*.ir.json")})
    if not paths:
        parser.error("no IR inputs")
    before = compiler_fingerprints(ROOT / "compile.py")
    input_hashes = {str(p): sha(p) for ir in paths for p in (ir, ir.with_name("meta.json"))}
    rows = []
    with tempfile.TemporaryDirectory(prefix="emmy-prior-emitter-") as folder:
        source = Path(folder) / "prior_backend.py"
        with zipfile.ZipFile(args.baseline_zip) as archive:
            source.write_bytes(archive.read("ptx_backend.py"))
        baseline_hash = sha(source)
        spec = importlib.util.spec_from_file_location("prior_emmy_backend", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for path in paths:
            row = {"ir": str(path.relative_to(ROOT)), "pass": False}
            try:
                old, new = module.compile_file(path).encode("ascii"), compile_file(path).encode("ascii")
                row.update(baseline_ptx_sha256=hashlib.sha256(old).hexdigest(),
                           candidate_ptx_sha256=hashlib.sha256(new).hexdigest(),
                           baseline_bytes=len(old), candidate_bytes=len(new),
                           **{"pass": old == new})
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    after = compiler_fingerprints(ROOT / "compile.py")
    unchanged = input_hashes == {p: sha(Path(p)) for p in input_hashes}
    report = {"pass": all(r["pass"] for r in rows) and before == after and unchanged,
              "scope": "PTX byte identity only; no additional numerical executions",
              "baseline_zip_sha256": sha(args.baseline_zip), "baseline_backend_sha256": baseline_hash,
              "compiler_sources_before_sha256": before, "compiler_sources_after_sha256": after,
              "input_sha256": input_hashes, "inputs_unchanged": unchanged,
              "kernels": len(rows), "identical": sum(r["pass"] for r in rows), "rows": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pass", "kernels", "identical", "inputs_unchanged")}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
