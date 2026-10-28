"""Compare current PTX with retained phase-6 hashes, without importing old code.

This is an offline identity audit, not a new numerical execution. Scalar hashes
were taken from compile_file(...).encode("ascii"). Feature hashes were taken
from Windows CLI output: their retained requests contain CRLF line endings.
Both representations are checked explicitly; no whitespace other than CRLF is
normalized. A prior numerical result can be carried forward only for an
identical program and identical inputs, including its original limitations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
PHASE6 = ROOT / "artifacts/validation/phase6"
RUNTIME = ("compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def recorded_path(value: str) -> Path:
    """Rebase historical Windows absolute paths to this extracted checkout."""
    value = value.replace("\\", "/")
    marker = "/emmy-hackathon-2026/"
    if marker in value:
        value = value.split(marker, 1)[1]
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)  # Reject evidence paths outside the checkout.
    return path


def snapshot(paths: set[Path]) -> dict[str, str]:
    return {relative(p): sha_bytes(p.read_bytes()) for p in sorted(paths)}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(rows: list[dict]) -> dict:
    return {
        "kernels": len(rows),
        "identical": sum(r.get("identical", False) for r in rows),
        "changed": sum(not r["identical"] for r in rows if "identical" in r),
        "errors": sum("error" in r for r in rows),
        "prior_evidence_reusable": sum(r.get("prior_evidence_reusable", False) for r in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("compatibility.json"))
    args = parser.parse_args()
    if args.report.exists():
        parser.error("refusing to overwrite an earlier audit")
    started = datetime.now(timezone.utc).isoformat()
    scalar_report = PHASE6 / "ptx-compatibility.json"
    feature_report = PHASE6 / "feature-runner/summary.json"
    scalar = load(scalar_report)
    feature = load(feature_report)
    if not scalar["pass"] or not feature["pass"]:
        raise ValueError("the recorded phase-6 evidence is not successful")
    expected_inputs = {recorded_path(p): h for p, h in scalar["input_sha256"].items()}
    # PTX outputs and harness sources are not the numerical inputs. The feature
    # source of truth is the retained request, never a mutable neighboring .ptx.
    for p, h in feature["tracked_sha256"].items():
        path = recorded_path(p)
        if path.name == "meta.json" or path.name.endswith((".ir.json", ".npz")):
            previous = expected_inputs.setdefault(path, h)
            if previous != h:
                raise ValueError(f"conflicting baseline input hashes: {path}")
    compile_events = [e for e in feature["events"] if e["endpoint"] == "/compile"]
    requests = {
        e["index"]: feature_report.parent / f"events/{e['index']:04d}.request.json"
        for e in compile_events
    }
    source_paths = {ROOT / name for name in RUNTIME} | {Path(__file__).resolve()}
    evidence_paths = {scalar_report, feature_report} | set(requests.values())
    source_before = snapshot(source_paths)
    evidence_before = snapshot(evidence_paths)
    input_before = snapshot(set(expected_inputs))
    input_comparison = {
        relative(p): {"baseline_sha256": h, "current_sha256": input_before[relative(p)],
                      "identical": h == input_before[relative(p)]}
        for p, h in sorted(expected_inputs.items())
    }

    def same_inputs(paths: list[Path]) -> bool:
        return all(input_comparison[relative(p)]["identical"] for p in paths)

    # Load only this checkout's implementation, after fingerprinting its files.
    sys.path.insert(0, str(ROOT))
    from ptx_backend import compile_file

    scalar_rows = []
    for old in scalar["rows"]:
        path = recorded_path(old["ir"])
        row = {"ir": relative(path), "baseline_ptx_sha256": old["candidate_ptx_sha256"],
               "baseline_bytes": old["candidate_bytes"],
               "inputs_identical": same_inputs([path, path.with_name("meta.json")])}
        try:
            emitted = compile_file(path).encode("ascii")
            digest = sha_bytes(emitted)
            identical = digest == old["candidate_ptx_sha256"] and len(emitted) == old["candidate_bytes"]
            row.update(candidate_ptx_sha256=digest, candidate_bytes=len(emitted), identical=identical,
                       prior_evidence_reusable=identical and row["inputs_identical"])
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        scalar_rows.append(row)

    feature_rows = []
    used_events = set()
    for job_index, job in enumerate(feature["jobs"]):
        package = recorded_path(job["package"])
        package_inputs = [p for p in expected_inputs if p.parent == package]
        for kernel in job["result"]["kernels"]:
            entry = kernel["entry"]
            row = {"package": relative(package), "entry": entry,
                   "inputs_identical": same_inputs(package_inputs),
                   "baseline_wire_ptx_sha256": kernel["ptx_sha256"],
                   "baseline_wire_bytes": kernel["ptx_bytes"]}
            try:
                matching = []
                for event in compile_events:
                    if event["job"] != job_index:
                        continue
                    request_path = requests[event["index"]]
                    if evidence_before[relative(request_path)] != event["request_sha256"]:
                        raise ValueError("retained compile request hash differs from its summary")
                    request = load(request_path)
                    if request["endpoint"] != "/compile":
                        raise ValueError("retained request is not a compile request")
                    wire_text = request["payload"]["ptx"]
                    if re.search(r"\.entry\s+" + re.escape(entry) + r"\s*\(", wire_text):
                        matching.append((event, request_path, wire_text))
                if len(matching) != 1:
                    raise ValueError(f"expected one retained request for {entry}, found {len(matching)}")
                event, request_path, wire_text = matching[0]
                if event["index"] in used_events:
                    raise ValueError("one retained compile request matched multiple kernels")
                used_events.add(event["index"])
                wire = wire_text.encode("ascii")
                if sha_bytes(wire) != kernel["ptx_sha256"] or len(wire) != kernel["ptx_bytes"]:
                    raise ValueError("retained PTX differs from the recorded runner PTX hash/length")
                old_lf = wire_text.replace("\r\n", "\n")
                if "\r" in old_lf:
                    raise ValueError("retained PTX contains a bare carriage return")
                style = "CRLF" if "\r\n" in wire_text else "LF"
                if style == "CRLF" and old_lf.replace("\n", "\r\n") != wire_text:
                    raise ValueError("retained PTX has mixed line endings")
                path = package / f"{entry}.ir.json"
                if not path.is_file():
                    raise ValueError(f"missing unambiguous IR file for {entry}")
                current = compile_file(path)
                if "\r" in current:
                    raise ValueError("compile_file no longer emits LF text")
                current_wire = current.replace("\n", "\r\n") if style == "CRLF" else current
                canonical_identical = current.encode("ascii") == old_lf.encode("ascii")
                wire_identical = current_wire.encode("ascii") == wire
                if canonical_identical != wire_identical:
                    raise ValueError("canonical and reconstructed-wire comparisons disagree")
                row.update(
                    ir=relative(path), request=relative(request_path), baseline_line_endings=style,
                    baseline_canonical_ptx_sha256=sha_bytes(old_lf.encode("ascii")),
                    candidate_canonical_ptx_sha256=sha_bytes(current.encode("ascii")),
                    candidate_wire_ptx_sha256=sha_bytes(current_wire.encode("ascii")),
                    candidate_wire_bytes=len(current_wire.encode("ascii")),
                    identical=wire_identical,
                    prior_evidence_reusable=wire_identical and row["inputs_identical"],
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            feature_rows.append(row)

    source_after = snapshot(source_paths)
    evidence_after = snapshot(evidence_paths)
    input_after = snapshot(set(expected_inputs))
    sources_unchanged = source_before == source_after
    evidence_unchanged = evidence_before == evidence_after
    inputs_unchanged = input_before == input_after
    errors = sum("error" in r for r in scalar_rows + feature_rows)
    baseline_inputs_identical = all(r["identical"] for r in input_comparison.values())
    complete = len(scalar_rows) == 120 and len(feature_rows) == 29 and len(used_events) == len(compile_events)
    report = {
        "pass": not errors and complete and sources_unchanged and evidence_unchanged and inputs_unchanged and baseline_inputs_identical,
        "scope": "Offline PTX identity audit only; no new GPU executions or numerical claims. Changed PTX cannot reuse prior execution evidence. Prior failures remain failures.",
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "all_ptx_identical": all(r.get("identical", False) for r in scalar_rows + feature_rows),
        "expected_coverage_complete": complete,
        "compiler_and_audit_sources_before_sha256": source_before,
        "compiler_and_audit_sources_after_sha256": source_after,
        "source_evidence_before_sha256": evidence_before,
        "source_evidence_after_sha256": evidence_after,
        "inputs_before_sha256": input_before, "inputs_after_sha256": input_after,
        "sources_unchanged": sources_unchanged, "evidence_unchanged": evidence_unchanged,
        "inputs_unchanged": inputs_unchanged, "baseline_inputs_identical": baseline_inputs_identical,
        "baseline_input_comparison": input_comparison,
        "scalar": summarize(scalar_rows), "features": summarize(feature_rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "report": str(args.report),
                      "scalar": {k: v for k, v in report["scalar"].items() if k != "rows"},
                      "features": {k: v for k, v in report["features"].items() if k != "rows"},
                      "sources_unchanged": sources_unchanged, "inputs_unchanged": inputs_unchanged,
                      "baseline_inputs_identical": baseline_inputs_identical}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
