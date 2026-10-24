"""Create a deterministic runtime submission after source-matched gates pass.

No installation, assembly, GPU execution or network access is performed. The
extracted package is checked with the real CLI and the public p00 input.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt")
HASH_FIELDS = (
    "source_sha256", "sources_sha256", "compiler_sources_sha256",
    "sources_before_sha256", "sources_after_sha256",
    "compiler_sources_before_sha256", "compiler_sources_after_sha256",
    "compiler_sha256",
)


class PackagingError(ValueError):
    """A release precondition failed; no ambiguous evidence is promoted."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def source_name(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).name


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PackagingError(f"ambiguous gate JSON repeats key {key!r}")
        result[key] = value
    return result


def check_status(report, path="$", *, top=True):
    if not isinstance(report, dict):
        raise PackagingError("gate report must be a JSON object")
    flags = {key: report[key] for key in ("pass", "ok") if key in report}
    if top and not flags:
        raise PackagingError("gate requires literal top-level pass=true or ok=true")
    for key, value in flags.items():
        if value is not True:
            raise PackagingError(f"gate {path}.{key} must be the boolean true, got {value!r}")
    if top and "status" in report:
        status = report["status"]
        if not isinstance(status, str) or status.lower() not in ("pass", "passed", "ok", "success"):
            raise PackagingError(f"ambiguous or unsuccessful gate status {status!r}")
    if top:
        for key in ("error", "exception"):
            if report.get(key) not in (None, "", False):
                raise PackagingError(f"gate has a nonempty {key}")
        for key in ("returncode", "exit_code"):
            if key in report and (type(report[key]) is not int or report[key] != 0):
                raise PackagingError(f"gate {key} is not zero")
    # A successful summary cannot contain an explicitly failed subordinate
    # pass/ok flag. Expected-error diagnostics have pass=true and may have a
    # nonzero child return code; only top-level return codes are checked above.
    for key, value in report.items():
        if isinstance(value, dict):
            check_status(value, f"{path}.{key}", top=False)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    check_status(item, f"{path}.{key}[{index}]", top=False)
    return flags


def read_evidence(path: Path, sources: dict[str, str]):
    raw = path.read_bytes()
    try:
        report = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_object)
    except (ValueError, UnicodeError) as exc:
        raise PackagingError(f"invalid gate JSON: {path}: {exc}") from exc
    flags = check_status(report)
    matched = {}
    identities = {}
    for field in HASH_FIELDS:
        if field not in report:
            continue
        hashes = report[field]
        if not isinstance(hashes, dict) or not hashes:
            raise PackagingError(f"{path}: {field} must be a nonempty source-hash mapping")
        identities[field] = {}
        for original, expected in hashes.items():
            if not isinstance(original, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                raise PackagingError(f"{path}: malformed SHA-256 identity in {field}")
            name = source_name(original)
            if name not in sources:
                continue  # CPU reports also identify tests and development tools.
            expected = expected.lower()
            if expected != sources[name]:
                raise PackagingError(f"{path}: {field} identifies a different {name}")
            identities[field][original] = expected
            matched[name] = expected
    if not matched:
        raise PackagingError(f"{path}: no recognized production source hashes bind this passing gate")
    # Nested job compiler hashes may identify explicitly tested development
    # variants. Production identity is the summary's top-level source mapping.
    return {
        "original_path": display_path(path), "sha256": digest(raw),
        "status_flags": flags, "matched_sources_sha256": matched,
        "production_identity_fields": identities,
    }, raw


def make_zip(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(SOURCE_FILES):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, files[name])
    return output.getvalue()


AUDIT_GUARD = r'''
import atexit, json, sys
from pathlib import Path
audit_file = Path(__AUDIT_FILE__)
blocked = []
def reject(event, args):
    blocked.append({"event": event, "detail": repr(args)[:300]})
    raise PermissionError("packaging verification forbids " + event)
def audit(event, args):
    if event.startswith("socket.") or event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"):
        reject(event, args)
    if event == "ctypes.dlopen" and any(word in str(args).lower() for word in ("nvcuda", "libcuda", "cudart", "nvrtc", "nvvm")):
        reject(event, args)
    if event == "import" and args and str(args[0]).split(".")[0] in ("torch", "cupy", "pycuda", "triton", "tvm", "llvmlite"):
        reject(event, args)
sys.addaudithook(audit)
@atexit.register
def finish():
    audit_file.write_text(json.dumps({"guard_enabled": True, "blocked_events": blocked}), encoding="utf-8")
'''


def verify_cli(archive_bytes: bytes, files: dict[str, bytes]):
    case_dir = ROOT / "problems/p00_scalar_add"
    irs = sorted(case_dir.glob("*.ir.json"))
    if len(irs) != 1:
        raise PackagingError("p00 must contain exactly one actual Kernel IR input")
    ir = irs[0]
    metadata_path = case_dir / "meta.json"
    inputs = {ir.name: ir.read_bytes(), "meta.json": metadata_path.read_bytes()}
    metadata = json.loads(inputs["meta.json"])
    entry = metadata["kernels"][0]["entry"]
    with tempfile.TemporaryDirectory(prefix="emmy-release-verify-") as temporary:
        stage = Path(temporary)
        package, case, guard = stage / "package", stage / "case", stage / "guard"
        for folder in (package, case, guard):
            folder.mkdir()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            if archive.namelist() != sorted(SOURCE_FILES) or archive.testzip() is not None:
                raise PackagingError("ZIP member inventory or CRC verification failed")
            for name in SOURCE_FILES:
                data = archive.read(name)
                if data != files[name]:
                    raise PackagingError(f"ZIP payload differs for {name}")
                (package / name).write_bytes(data)
        for name, data in inputs.items():
            (case / name).write_bytes(data)
        audit_path = stage / "audit.json"
        (guard / "sitecustomize.py").write_text(
            AUDIT_GUARD.replace("__AUDIT_FILE__", repr(str(audit_path))), encoding="utf-8"
        )
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")}
        env["PYTHONPATH"] = str(guard)
        command = [sys.executable, "-B", "compile.py", str(case / ir.name)]
        result = subprocess.run(command, cwd=package, env=env, capture_output=True, timeout=120)
        if result.returncode != 0 or result.stderr:
            raise PackagingError("extracted CLI failed or emitted stderr: " + result.stderr.decode("utf-8", errors="replace"))
        try:
            ptx = result.stdout.decode("ascii")
        except UnicodeError as exc:
            raise PackagingError("extracted CLI emitted non-ASCII stdout") from exc
        if not ptx.startswith(".version ") or not re.search(r"\.visible\s+\.entry\s+" + re.escape(entry) + r"\s*\(", ptx):
            raise PackagingError("extracted CLI did not emit the expected PTX module")
        if not audit_path.is_file():
            raise PackagingError("CLI verification audit guard did not finish")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit != {"guard_enabled": True, "blocked_events": []}:
            raise PackagingError("CLI attempted a forbidden network, GPU-library or subprocess operation")
        if sorted(p.name for p in package.iterdir()) != sorted(SOURCE_FILES):
            raise PackagingError("CLI changed the extracted package inventory")
        for name, data in files.items():
            if (package / name).read_bytes() != data:
                raise PackagingError(f"CLI changed extracted source {name}")
    return {
        "pass": True, "argv": ["python", "-B", "compile.py", "<temporary-case>/" + ir.name],
        "input_path": display_path(ir), "ir_sha256": digest(inputs[ir.name]),
        "metadata_sha256": digest(inputs["meta.json"]), "entry": entry,
        "ptx_sha256": digest(result.stdout), "ptx_bytes": len(result.stdout),
        "extracted_source_sha256": {name: digest(data) for name, data in files.items()},
        "audit": audit,
        "scope": "Real extracted CLI on installed dependencies; no installation, assembly or GPU execution. Python audit guards are regression checks, not an OS sandbox.",
    }


def immutable_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise PackagingError(f"refusing to overwrite a different retained artifact: {path}")


def package_release(release_dir: Path, evidence_paths: list[Path], update_latest=False):
    if not evidence_paths:
        raise PackagingError("at least one passing evidence report is required")
    if len(set(evidence_paths)) != len(evidence_paths):
        raise PackagingError("duplicate evidence paths")
    files = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
    sources = {name: digest(data) for name, data in files.items()}
    evidence, covered = [], set()
    for path in sorted(evidence_paths, key=str):
        record, raw = read_evidence(path, sources)
        evidence.append((path, record, raw))
        covered.update(record["matched_sources_sha256"])
    if covered != set(SOURCE_FILES):
        raise PackagingError("passing evidence does not collectively identify all runtime submission files: missing " + ", ".join(sorted(set(SOURCE_FILES) - covered)))
    archive = make_zip(files)
    verification = verify_cli(archive, files)
    # Close the time-of-check interval before publishing immutable artifacts.
    if any((ROOT / name).read_bytes() != data for name, data in files.items()):
        raise PackagingError("submission sources changed during packaging")
    if any(path.read_bytes() != raw for path, _, raw in evidence):
        raise PackagingError("a gate report changed during packaging")
    source_id = digest(json_bytes(sources))
    zip_name = f"submission-{source_id[:20]}.zip"
    records = []
    for _, record, raw in evidence:
        retained = f"evidence/{record['sha256']}.json"
        records.append({**record, "retained_path": retained})
    manifest = {
        "schema": "emmy-submission-package-v1", "pass": True,
        "source_identity_sha256": source_id, "source_sha256": sources,
        "zip": {"path": zip_name, "sha256": digest(archive), "bytes": len(archive), "members": sorted(SOURCE_FILES), "compression": "stored", "member_timestamp": "1980-01-01T00:00:00", "member_mode": "100644"},
        "evidence": records, "extracted_cli_verification": verification,
        "packager_sha256": digest(Path(__file__).read_bytes()),
        "python_version": sys.version.split()[0],
    }
    manifest_data = json_bytes(manifest)
    manifest_sha = digest(manifest_data)
    manifest_name = f"submission-{source_id[:20]}-{manifest_sha[:20]}.manifest.json"
    release_dir.mkdir(parents=True, exist_ok=True)
    immutable_write(release_dir / zip_name, archive)
    for (_, _, raw), record in zip(evidence, records, strict=True):
        immutable_write(release_dir / record["retained_path"], raw)
    immutable_write(release_dir / manifest_name, manifest_data)
    immutable_write(release_dir / (manifest_name + ".sha256"), f"{manifest_sha}  {manifest_name}\n".encode("ascii"))
    latest = {"manifest": manifest_name, "manifest_sha256": manifest_sha, "zip": zip_name, "zip_sha256": digest(archive)}
    if update_latest:
        with tempfile.NamedTemporaryFile(dir=release_dir, prefix=".latest-", suffix=".json", delete=False) as temporary:
            temporary.write(json_bytes(latest))
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, release_dir / "latest.json")
    return latest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, nargs="+", action="extend", required=True)
    parser.add_argument("--update-latest", action="store_true", help="atomically update latest.json; immutable versions remain retained")
    args = parser.parse_args(argv)
    try:
        result = package_release(args.release_dir.resolve(), [p.resolve() for p in args.evidence], args.update_latest)
    except (PackagingError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"package_submission.py: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
