"""Verify the curated Git checkout and reproduce its retained CLI checks.

This checks local files and runs the Python compiler. It does not contact a
runner, execute GPU kernels, or repeat the historical unit-test suites. Their
counts are reported separately from the fresh CLI checks performed here.
Retained CLI hashes use Windows CRLF; uniform LF output is compared after only
that line-ending conversion, with its native hash and comparison mode retained.
The original archive remains historical provenance. A version-2 checkout binds
its current runtime and CLI expectations to separate, inventoried validation
reports; it does not claim that a changed runtime matches the old archive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PHASE7 = ROOT / "artifacts/validation/phase7"
RUNTIME = {"compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt"}
PRIMARY_PAGES = (
    "README.md", "SOLUTION.md", "SUBMISSION.md", "docs/FEATURES.md",
    "examples/README.md", "docs/PHASE6.md", "docs/PHASE7.md", "docs/RESEARCH_REVIEW.md",
    "docs/PACKAGING.md", "docs/CHECKOUT.md", "docs/COMPILER.md", "docs/FINAL_CHECK_2026_10_24.md", "docs/PHASE5.md",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative_path(name: str) -> Path:
    value = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or value.is_absolute() or ".." in value.parts:
        raise ValueError(f"unsafe checkout-relative path: {name!r}")
    path = ROOT.joinpath(*value.parts)
    path.resolve().relative_to(ROOT)
    for parent in (path, *path.parents):
        if parent == ROOT:
            break
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError(f"linked checkout path: {name}")
    return path


def file_record(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": digest(data), "bytes": len(data)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=ROOT / "artifacts/validation/checkout-current/verification.json")
    args = parser.parse_args()
    destination = args.report.resolve()
    if destination.exists():
        parser.error("refusing to overwrite an earlier checkout verification")
    report = {
        "schema": "emmy-git-checkout-verification-v1", "pass": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Current checkout inventory, retained archive provenance, emitted PTX identity with explicit LF/CRLF handling, and primary documentation links; no new GPU execution.",
        "local_gpu_used": False, "network_used": False, "fresh_unit_tests_run": 0,
    }
    before = {}
    inventory_before = None
    manifest_path = ROOT / "REVIEW_MANIFEST.json"
    try:
        inventory_before = digest(manifest_path.read_bytes())
        manifest = read_json(manifest_path)
        assert manifest["schema"] in ("emmy-git-checkout-v1", "emmy-git-checkout-v2"), "unexpected checkout manifest schema"
        current_validation = manifest.get("current_validation")
        if manifest["schema"] == "emmy-git-checkout-v2":
            assert current_validation, "version-2 checkout requires current_validation"
        files = manifest["files"]
        assert isinstance(files, dict) and files and "REVIEW_MANIFEST.json" not in files
        excluded_paths = tuple(p.rstrip("/") for p in manifest["excluded_paths"])
        excluded_names = set(manifest["excluded_directory_names"])

        def excluded(name: str) -> bool:
            return (any(part in excluded_names for part in PurePosixPath(name).parts)
                    or any(name == p or name.startswith(p + "/") for p in excluded_paths))

        assert excluded("REVIEW_MANIFEST.json"), "self-hashing manifest must be explicitly excluded"
        if destination.is_relative_to(ROOT):
            assert excluded(destination.relative_to(ROOT).as_posix()), "report destination must be outside the tracked inventory"
        for name, expected in files.items():
            assert not excluded(name), f"inventory contains an excluded path: {name}"
            path = relative_path(name)
            assert path.is_file(), f"missing inventory file: {name}"
            actual = file_record(path)
            before[name] = actual
            assert actual == expected, f"inventory hash or length mismatch: {name}"
        found = set()
        for current, directories, filenames in os.walk(ROOT, followlinks=False):
            directories[:] = [name for name in directories
                              if not excluded((Path(current) / name).relative_to(ROOT).as_posix())]
            for name in directories:
                relative_path((Path(current) / name).relative_to(ROOT).as_posix())
            for name in filenames:
                relative = (Path(current) / name).relative_to(ROOT).as_posix()
                if not excluded(relative):
                    relative_path(relative)
                    found.add(relative)
        assert found == set(files), f"unlisted or missing files: added={sorted(found - set(files))}, missing={sorted(set(files) - found)}"
        report["inventory"] = {"manifest_sha256": inventory_before, "verified_files": len(files),
                               "excluded_paths": list(excluded_paths),
                               "excluded_directory_names": sorted(excluded_names)}

        historical_release = read_json(PHASE7 / "release-summary.json")
        retained = read_json(PHASE7 / "review-bundle-verification.json")
        assert historical_release["pass"] and retained["pass"], "retained release/verification gate failed"
        historical_runtime = historical_release["source_sha256"]
        assert set(historical_runtime) == RUNTIME
        assert historical_runtime == retained["runtime_source_sha256"]

        release, cli_reference = historical_release, retained
        if current_validation is not None:
            assert isinstance(current_validation, dict), "current_validation must be an object"
            references = {}
            for kind in ("release", "cli"):
                name = current_validation[kind + "_path"]
                assert name in files, f"current {kind} reference is not inventoried: {name}"
                path = relative_path(name)
                assert digest(path.read_bytes()) == current_validation[kind + "_sha256"], f"current {kind} reference hash mismatch"
                references[kind] = read_json(path)
                assert references[kind]["pass"], f"current {kind} reference gate failed"
            release, cli_reference = references["release"], references["cli"]
            report["current_validation"] = current_validation
        runtime = {name: before[name]["sha256"] for name in sorted(RUNTIME)}
        assert set(manifest["runtime_sha256"]) == RUNTIME
        assert runtime == manifest["runtime_sha256"] == release["source_sha256"] == cli_reference["runtime_source_sha256"]
        origin = manifest["origin"]
        archive_inventory_path = relative_path(origin["archive_inventory_path"])
        assert origin["archive_inventory_path"] in files
        assert digest(archive_inventory_path.read_bytes()) == origin["archive_inventory_sha256"]
        archive_inventory = read_json(archive_inventory_path)
        assert archive_inventory["schema"] == "emmy-review-bundle-contents-v1"
        assert archive_inventory["runtime_sha256"] == historical_runtime
        assert origin["archive_sha256"] == retained["archive"]["sha256"]
        assert len(archive_inventory["files"]) + 1 == retained["archive"]["verified_members"]
        original_files = archive_inventory["files"]
        report["origin"] = origin
        report["archive_to_checkout_changes"] = {
            "added": sorted(set(files) - set(original_files)),
            "removed": sorted(set(original_files) - set(files)),
            "modified": sorted(name for name in set(files) & set(original_files)
                               if files[name] != original_files[name]),
        }
        report["runtime_sha256"] = runtime
        report["historical_archive_runtime_sha256"] = historical_runtime
        # These are historical gates, not fresh tests executed by this command.
        report["retained_full_cpu_gate"] = release["cpu"]
        report["retained_original_release_cpu_gate"] = historical_release["cpu"]
        report["retained_archive_focused_cpu_gate"] = retained["focused_test_totals"]

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(EMMY_GPU_TESTS="0", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
        origin_code = ("import json,ptx_backend,ptx_tensor,ptx_bulk; "
                       "print(json.dumps({m.__name__:m.__file__ for m in "
                       "(ptx_backend,ptx_tensor,ptx_bulk)}))")
        module_result = subprocess.run([sys.executable, "-B", "-c", origin_code], cwd=ROOT,
                                       env=env, capture_output=True, timeout=120)
        assert module_result.returncode == 0, module_result.stderr.decode("utf-8", "replace")
        module_origins = json.loads(module_result.stdout)
        assert all(Path(path).resolve().parent == ROOT for path in module_origins.values())
        report["module_origins"] = module_origins
        checks = cli_reference["cli_checks"]
        assert len(checks) == 20 and len({row["ir"] for row in checks}) == 20
        report["fresh_cli_checks"] = []
        for expected in checks:
            ir_name = expected["ir"]
            assert expected["pass"] and ir_name in files
            ir = relative_path(ir_name)
            assert ir.with_name("meta.json").relative_to(ROOT).as_posix() in files
            result = subprocess.run([sys.executable, "-B", "compile.py", ir_name], cwd=ROOT,
                                    env=env, capture_output=True, timeout=120)
            row = {"ir": ir_name, "returncode": result.returncode,
                   "ptx_sha256": digest(result.stdout), "ptx_bytes": len(result.stdout),
                   "expected_ptx_sha256": expected["ptx_sha256"],
                   "expected_ptx_bytes": expected["ptx_bytes"],
                   "stderr": result.stderr.decode("utf-8", "replace")}
            canonical_lf = result.stdout.replace(b"\r\n", b"\n")
            uniform_endings = (b"\r" not in canonical_lf
                               and (b"\r\n" not in result.stdout
                                    or canonical_lf.replace(b"\n", b"\r\n") == result.stdout))
            windows_bytes = canonical_lf.replace(b"\n", b"\r\n")
            row["normalized_ptx_sha256"] = digest(windows_bytes)
            row["normalized_ptx_bytes"] = len(windows_bytes)
            raw_match = (row["ptx_sha256"] == expected["ptx_sha256"]
                         and row["ptx_bytes"] == expected["ptx_bytes"])
            normalized_match = (uniform_endings
                                and row["normalized_ptx_sha256"] == expected["ptx_sha256"]
                                and row["normalized_ptx_bytes"] == expected["ptx_bytes"])
            row["comparison_mode"] = "raw" if raw_match else "lf_to_crlf" if normalized_match else "mismatch"
            row["pass"] = result.returncode == 0 and not result.stderr and (raw_match or normalized_match)
            report["fresh_cli_checks"].append(row)
            assert row["pass"], f"CLI differs from retained PTX beyond permitted line endings: {ir_name}"
            print(f"CLI verified: {ir_name}", flush=True)
        report["fresh_cli_totals"] = {"checks": len(checks), "passed": len(checks),
                                      "raw_matches": sum(row["comparison_mode"] == "raw" for row in report["fresh_cli_checks"]),
                                      "line_ending_normalized_matches": sum(row["comparison_mode"] == "lf_to_crlf" for row in report["fresh_cli_checks"]),
                                      "comparison": "Native stdout compared first; uniform LF may be converted to retained Windows CRLF, with every other byte preserved"}

        links, distribution_links = [], []
        primary_pages = PRIMARY_PAGES + (("docs/PHASE8.md",)
                                        if manifest["schema"] == "emmy-git-checkout-v2" else ())
        for name in primary_pages:
            assert name in files, f"missing primary reviewer page: {name}"
            page = relative_path(name)
            for raw in re.findall(r"\]\(([^)]+)\)", page.read_text(encoding="utf-8-sig")):
                link = raw.strip().strip("<>").split("#", 1)[0]
                if not link or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", link):
                    continue
                target = (page.parent / unquote(link)).resolve()
                assert target.is_relative_to(ROOT), f"relative link escapes checkout: {name} -> {link}"
                if target.exists():
                    links.append({"page": name, "link": link})
                elif target.suffix.lower() == ".zip" and target.is_relative_to(ROOT / "artifacts/release"):
                    distribution_links.append({"page": name, "link": link,
                                               "reason": "Distribution ZIP intentionally kept outside the Git source checkout"})
                else:
                    raise AssertionError(f"broken primary reviewer link: {name} -> {link}")
        report["primary_links"] = {"pages": list(primary_pages), "verified": len(links),
                                   "external_distribution_zips": distribution_links}
        report["pass"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        after = {}
        for name in before:
            try:
                after[name] = file_record(relative_path(name))
            except (OSError, ValueError) as exc:
                after[name] = {"error": f"{type(exc).__name__}: {exc}"}
        report["sources_unchanged"] = bool(before) and before == after
        try:
            report["manifest_unchanged"] = inventory_before is not None and digest(manifest_path.read_bytes()) == inventory_before
        except OSError:
            report["manifest_unchanged"] = False
        report["source_sha256_before"] = {name: value["sha256"] for name, value in before.items()}
        report["source_sha256_after"] = {name: value.get("sha256") for name, value in after.items()}
        report["pass"] = report["pass"] and report["sources_unchanged"] and report["manifest_unchanged"]
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
        print(json.dumps({"pass": report["pass"], "report": str(destination),
                          "sources_unchanged": report["sources_unchanged"],
                          "fresh_cli_totals": report.get("fresh_cli_totals"),
                          "error": report.get("error")}, indent=2), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
