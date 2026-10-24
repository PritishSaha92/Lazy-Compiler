"""Run packaged examples on the organizer GPU and retain every wire exchange.

Inputs and independent expected.npz files are frozen before execution. This is
a correctness gate; one timed iteration supplies no performance ranking.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import check_compiler as checker
from tools import selector_study as transport


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="+", type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--compiler", type=Path, default=ROOT / "compile.py")
    parser.add_argument("--server", default=checker.SERVER)
    args = parser.parse_args()
    folders = [p.resolve() for p in args.packages]
    if len(set(folders)) != len(folders):
        parser.error("duplicate packages are not additional coverage")
    for folder in folders:
        for name in ("meta.json", "expected.npz"):
            if not (folder / name).is_file():
                parser.error(f"missing {folder / name}")
    destination = args.report_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "events").mkdir()
    compiler = args.compiler.resolve()
    sources = [Path(__file__), ROOT / "tools/check_compiler.py", ROOT / "tools/selector_study.py"]
    sources += [p for folder in folders for p in folder.iterdir() if p.is_file()]
    tracked = {str(p): digest(p) for p in sources}
    before = checker.compiler_fingerprints(compiler)
    report = {
        "scope": "Canonical explicit-data examples on organizer GPU; no local GPU or timing claim",
        "started_utc": datetime.now(timezone.utc).isoformat(), "pass": False,
        "compiler_sha256": before, "tracked_sha256": tracked,
        "planned_packages": [str(p) for p in folders], "planned_checks": len(folders),
        "events": [], "jobs": [],
    }
    path = destination / "summary.json"

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def remote_post(server, endpoint, payload):
        number = len(report["events"])
        base = destination / "events" / f"{number:04d}"
        request = base.with_suffix(".request.json")
        request.write_text(json.dumps({"endpoint": endpoint, "payload": payload}, indent=2) + "\n", encoding="utf-8")
        event = {"index": number, "endpoint": endpoint, "job": len(report["jobs"]) - 1,
                 "request_sha256": digest(request), "status": "pending", "http": []}
        report["events"].append(event)
        save()

        def attempt(value):
            item = base.with_name(base.name + f".http-{len(event['http']):02d}.json")
            item.write_text(json.dumps(transport.diagnostic_json(value), indent=2) + "\n", encoding="utf-8")
            event["http"].append({"path": item.relative_to(destination).as_posix(), "sha256": digest(item)})
            save()

        try:
            result = transport.audit_remote_post(server, endpoint, payload, attempt)
            response = base.with_suffix(".response.json")
            response.write_text(json.dumps(transport.diagnostic_json(result), indent=2) + "\n", encoding="utf-8")
            event.update(status="received", response_sha256=digest(response))
            save()
            return result
        except Exception as exc:
            event.update(status="transport_error", error=f"{type(exc).__name__}: {exc}")
            save()
            raise

    original_post = checker.remote_post
    checker.remote_post = remote_post
    save()
    try:
        for folder in folders:
            job = {"id": folder.name, "package": str(folder), "pass": False}
            report["jobs"].append(job)
            save()
            config = SimpleNamespace(compiler=str(compiler), mode="remote", server=args.server,
                                     seeds="0", patterns="normal", warmup=0, iters=1,
                                     adapt_local_target=False, arch="sm_90")
            try:
                result = checker.check_problem(folder, config, {})
                job["result"] = result
                job["pass"] = result["pass"] and len(result["cases"]) == 1
            except Exception as exc:
                job["error"] = f"{type(exc).__name__}: {exc}"
            save()
            print(folder.name, "PASS" if job["pass"] else "FAIL", job.get("error", ""), flush=True)
    finally:
        checker.remote_post = original_post
    checks = [case for job in report["jobs"] for case in job.get("result", {}).get("cases", [])]
    after = checker.compiler_fingerprints(compiler)
    report.update(checks=len(checks), passed_checks=sum(c["pass"] for c in checks),
                  compiler_sha256_after=after,
                  tracked_unchanged=all(digest(Path(p)) == h for p, h in tracked.items()),
                  finished_utc=datetime.now(timezone.utc).isoformat())
    report["pass"] = (len(checks) == len(folders) and all(j["pass"] for j in report["jobs"])
                      and report["tracked_unchanged"] and before == after)
    save()
    print(json.dumps({k: report[k] for k in ("pass", "checks", "passed_checks", "tracked_unchanged")}), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
