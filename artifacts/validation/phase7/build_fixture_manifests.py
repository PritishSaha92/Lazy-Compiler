"""Freeze the two phase7 fixture rosters and bytes without changing prior evidence.

The new roster comes from its completed runner report. The retained feature
roster comes from the completed phase6 report and its original manifest. These
are evidence snapshots, not a claim that the manifests predate the executions.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).resolve().parent
RUNTIME = ("compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}


def build(name, source_report, prior_manifest=None):
    output = FOLDER / name
    if output.exists():
        raise FileExistsError(f"Refusing to replace frozen manifest: {output}")
    report = json.loads(source_report.read_text(encoding="utf-8"))
    if "finished_utc" not in report:
        raise ValueError("Roster source must be a completed report")
    packages = [Path(p).resolve() for p in report["planned_packages"]]
    if not packages or len(set(packages)) != len(packages):
        raise ValueError("Roster must contain distinct packages")
    if not all(p.is_relative_to(ROOT / "examples") for p in packages):
        raise ValueError("Package is outside repository examples")
    files = {
        p.relative_to(ROOT).as_posix(): digest(p)
        for package in packages for p in sorted(package.rglob("*")) if p.is_file()
    }
    for package in packages:
        metadata = json.loads((package / "meta.json").read_text(encoding="utf-8"))
        required = [package / "meta.json", package / "expected.npz"]
        required.extend(package / (k["entry"] + ".ir.json") for k in metadata["kernels"])
        if not all(p.relative_to(ROOT).as_posix() in files for p in required):
            raise ValueError(f"Incomplete fixture: {package}")
    data = {
        "schema": "emmy-phase7-feature-inputs-v1",
        "scope": "Immutable fixture-file snapshot and selected roster bound to the phase7 compiler; created after roster-source execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "compiler_sha256": {name: digest(ROOT / name) for name in RUNTIME},
        "package_count": len(packages),
        "packages": [p.relative_to(ROOT).as_posix() for p in packages],
        "files_sha256": files,
        "provenance": {"roster_source_report": record(source_report)},
    }
    if prior_manifest is not None:
        prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
        if prior["packages"] != data["packages"]:
            raise ValueError("Prior manifest and report rosters disagree")
        for name, expected in prior["files_sha256"].items():
            path = "examples/" + name
            if files.get(path) != expected:
                raise ValueError(f"Prior fixture bytes changed: {path}")
        data["provenance"]["original_feature_manifest"] = record(prior_manifest)
        data["original_fixture_files_unchanged"] = True
        data["original_fixture_file_count"] = len(prior["files_sha256"])
    else:
        source = {Path(p).name: h for p, h in report["compiler_sha256"].items()}
        if data["compiler_sha256"] != source:
            raise ValueError("New-feature report identifies another compiler")
        for path, expected in report["tracked_sha256"].items():
            path = Path(path).resolve()
            if path.is_relative_to(ROOT / "examples") and files.get(path.relative_to(ROOT).as_posix()) != expected:
                raise ValueError(f"New fixture differs from runner tracked bytes: {path}")
    with output.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    print(json.dumps(record(output) | {"packages": len(packages), "files": len(files)}))


if __name__ == "__main__":
    build("new-feature-manifest.json", FOLDER / "new-feature-runner/summary.json")
    build("feature-manifest.json", ROOT / "artifacts/validation/phase6/feature-runner/summary.json",
          ROOT / "examples/research-feature-manifest.json")
