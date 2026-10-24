"""Build a deterministic source bundle for manual review, without executing code.

The runtime ZIP remains a separate artifact. This bundle includes examples,
tests, source documentation and the small retained fixtures those tests need.
Explicit --evidence files are copied unchanged, including failed or historical
reports; packaging a report is not an assertion that its checks passed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = ("compile.py", "ptx_backend.py", "requirements.txt")
OPTIONAL_RUNTIME_FILES = ("ptx_tensor.py", "ptx_bulk.py")
REQUIRED_FILES = (*RUNTIME_FILES, "README.md", "SOLUTION.md", "SUBMISSION.md")
OPTIONAL_FILES = ("submit.py", "TASK.md", "MVP.md", "LICENSE", "LICENSE.md", "LICENSE.txt")
SOURCE_TREES = ("docs", "tests", "tools", "examples", "problems")
# These are inputs, not the large request/response and binary-output archives.
# Keep their original paths so existing reproduction commands continue to work.
FIXTURE_PATHS = (
    "artifacts/validation/stress",
    "artifacts/validation/layouts",
    "artifacts/validation/holdout-2026-10-23",
    "artifacts/validation/phase3/schedules",
    "artifacts/validation/phase3/transport",
    "artifacts/validation/phase3/products",
    "artifacts/validation/2026-10-24-extra/packages",
    "artifacts/validation/2026-10-24-extra/expressions_inline.py",
)
ALLOWED_SUFFIXES = frozenset((
    ".py", ".md", ".json", ".txt", ".npz", ".npy", ".ptx", ".cu",
    ".csv", ".sha256", ".tex", ".pdf", ".toml", ".log",
))
SKIP_DIRECTORIES = frozenset(("__pycache__", "node_modules"))
MANIFEST_MEMBER = "REVIEW_MANIFEST.json"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


class ReviewPackagingError(ValueError):
    """A requested member cannot safely or reproducibly enter the bundle."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def relative_name(value: str | Path) -> str:
    """Accept explicit repository-relative names on Windows and POSIX."""
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if (not text or path.is_absolute() or path == PurePosixPath(".")
            or any(part in ("..", ".") or ":" in part for part in path.parts)):
        raise ReviewPackagingError(f"expected a repository-relative path: {text!r}")
    if any(part.startswith(".") or part in SKIP_DIRECTORIES for part in path.parts):
        raise ReviewPackagingError(f"hidden/cache paths are excluded: {text!r}")
    return path.as_posix()


def checked_path(root: Path, name: str) -> Path:
    """Reject links/junctions even when their current target is inside root."""
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ReviewPackagingError(f"links and junctions are excluded: {name}")
    if not path.resolve().is_relative_to(root):
        raise ReviewPackagingError(f"path escapes repository: {name}")
    return path


def collect_files(root: Path, *, includes=(), evidence=(), max_bytes=DEFAULT_MAX_BYTES):
    """Snapshot an explicit allowlist; never discover arbitrary repository files."""
    root = root.resolve()
    if max_bytes <= 0:
        raise ReviewPackagingError("max_bytes must be positive")
    files: dict[str, bytes] = {}
    spellings: dict[str, str] = {}
    total = 0

    def add_file(name: str):
        nonlocal total
        if name == MANIFEST_MEMBER:
            raise ReviewPackagingError(f"{MANIFEST_MEMBER} is reserved for the generated inventory")
        path = checked_path(root, name)
        if not path.is_file():
            raise ReviewPackagingError(f"required file is missing: {name}")
        if path.suffix.lower() not in ALLOWED_SUFFIXES and name not in OPTIONAL_FILES:
            raise ReviewPackagingError(f"file type is not allowlisted: {name}")
        previous = spellings.setdefault(name.casefold(), name)
        if previous != name:
            raise ReviewPackagingError(f"case-insensitive member collision: {previous}, {name}")
        if name in files:
            return
        if total + path.stat().st_size > max_bytes:
            raise ReviewPackagingError(f"bundle exceeds {max_bytes} source bytes at {name}; select smaller evidence or raise --max-bytes")
        data = path.read_bytes()
        total += len(data)
        if total > max_bytes:
            raise ReviewPackagingError("source grew past the bundle byte limit while reading")
        files[name] = data

    def add_path(value):
        name = relative_name(value)
        path = checked_path(root, name)
        if path.is_file():
            add_file(name)
        elif path.is_dir():
            # os.walk does not follow symlinks; explicitly reject visible links
            # and junctions instead of silently dropping an expected input.
            for current, directories, filenames in os.walk(path, followlinks=False):
                directories[:] = sorted(d for d in directories if not d.startswith(".") and d not in SKIP_DIRECTORIES)
                for directory in directories:
                    checked_path(root, (Path(current) / directory).relative_to(root).as_posix())
                for filename in sorted(filenames):
                    if filename.startswith("."):
                        continue
                    child = Path(current) / filename
                    if child.suffix.lower() in ALLOWED_SUFFIXES:
                        add_file(child.relative_to(root).as_posix())
        else:
            raise ReviewPackagingError(f"required path is missing: {name}")

    for name in REQUIRED_FILES:
        add_file(name)
    optional_runtime = []
    for name in (*OPTIONAL_RUNTIME_FILES, *OPTIONAL_FILES):
        if (root / name).exists():
            add_file(name)
            if name in OPTIONAL_RUNTIME_FILES:
                optional_runtime.append(name)
    for name in (*SOURCE_TREES, *FIXTURE_PATHS, *includes):
        add_path(name)

    evidence_records = []
    for value in sorted(set(map(str, evidence))):
        name = relative_name(value)
        add_file(name)
        try:
            report = json.loads(files[name].decode("utf-8-sig"))
        except (ValueError, UnicodeError) as exc:
            raise ReviewPackagingError(f"evidence is not valid JSON: {name}") from exc
        if not isinstance(report, dict):
            raise ReviewPackagingError(f"evidence must be a JSON object: {name}")
        evidence_records.append({
            "path": name, "sha256": digest(files[name]),
            "reported_status": {key: report[key] for key in ("pass", "ok", "status") if key in report},
        })
    manifest = {
        "schema": "emmy-review-bundle-contents-v1",
        "scope": "Source, examples, tests and selected exact evidence. Inclusion does not certify correctness; historical and failed reports retain their original statuses.",
        "inventory_scope": "Every archive member except this generated REVIEW_MANIFEST.json; the outer manifest separately hashes the complete archive.",
        "runtime_sha256": {name: digest(files[name]) for name in (*RUNTIME_FILES, *optional_runtime)},
        "files": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in sorted(files.items())},
        "evidence": evidence_records,
        "selection": {"source_trees": list(SOURCE_TREES), "fixture_paths": list(FIXTURE_PATHS), "explicit_includes": sorted(set(relative_name(v) for v in includes))},
        "total_source_bytes": total,
    }
    return files, manifest


def make_zip(files: dict[str, bytes], manifest_data: bytes) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted({**files, MANIFEST_MEMBER: manifest_data}.items()):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            member.compress_type = zipfile.ZIP_STORED
            archive.writestr(member, data)
    return stream.getvalue()


def verify_archive(data: bytes, files: dict[str, bytes], manifest_data: bytes):
    expected = {**files, MANIFEST_MEMBER: manifest_data}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if archive.namelist() != sorted(expected) or archive.testzip() is not None:
            raise ReviewPackagingError("archive inventory or CRC check failed")
        for name, original in expected.items():
            if archive.read(name) != original:
                raise ReviewPackagingError(f"archive payload changed: {name}")


def immutable_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ReviewPackagingError(f"refusing to overwrite a different retained artifact: {path.name}")


def package_review(root: Path, release_dir: Path, *, includes=(), evidence=(), update_latest=False, max_bytes=DEFAULT_MAX_BYTES):
    root = root.resolve()
    release_dir = release_dir.resolve()
    if not release_dir.is_relative_to(root):
        raise ReviewPackagingError("release directory must be inside the repository")
    files, contents = collect_files(root, includes=includes, evidence=evidence, max_bytes=max_bytes)
    contents_data = json_bytes(contents)
    contents_id = digest(contents_data)
    archive = make_zip(files, contents_data)
    verify_archive(archive, files, contents_data)
    # Close the source snapshot interval before publishing the bundle.
    for name, original in files.items():
        if checked_path(root, name).read_bytes() != original:
            raise ReviewPackagingError(f"source changed during packaging: {name}")
    archive_name = f"review-{contents_id[:20]}.zip"
    manifest_name = f"review-{contents_id[:20]}.manifest.json"
    outer = {
        "schema": "emmy-review-bundle-v1", "archive_integrity_pass": True,
        "scope": "Archive member/CRC/source identity verification only; no installation, compilation, test execution, network access or GPU use.",
        "zip": {"path": archive_name, "sha256": digest(archive), "bytes": len(archive), "members": len(files) + 1},
        "contents_manifest": {"member": MANIFEST_MEMBER, "sha256": contents_id},
        "runtime_sha256": contents["runtime_sha256"],
        "evidence": contents["evidence"],
    }
    outer_data = json_bytes(outer)
    immutable_write(release_dir / archive_name, archive)
    immutable_write(release_dir / manifest_name, outer_data)
    latest = {"zip": archive_name, "zip_sha256": digest(archive), "manifest": manifest_name, "manifest_sha256": digest(outer_data)}
    if update_latest:
        with tempfile.NamedTemporaryFile(dir=release_dir, prefix=".review-latest-", suffix=".json", delete=False) as temporary:
            temporary.write(json_bytes(latest))
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, release_dir / "latest.json")
    return latest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--include", nargs="+", action="extend", default=[], help="Additional allowlisted repository-relative files/directories; no wildcards")
    parser.add_argument("--evidence", nargs="+", action="extend", default=[], help="Exact repository-relative JSON report files; failed/historical reports remain explicitly labelled")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--update-latest", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = package_review(ROOT, args.release_dir, includes=args.include, evidence=args.evidence, update_latest=args.update_latest, max_bytes=args.max_bytes)
    except (ReviewPackagingError, OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"package_review.py: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
