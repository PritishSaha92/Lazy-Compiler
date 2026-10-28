"""Offline, independent replay audit of a manifest-bound feature runner execution.

Reads exact recorded HTTP exchanges and input packages; never contacts a runner,
assembles instructions or launches a GPU. Real compile.py subprocesses reproduce
the submitted PTX bytes using the frozen current sources. Numerical failures,
if present, remain failures independently of evidence-integrity acceptance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ("compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt")
DTYPES = {"f16": np.dtype("float16"), "f32": np.dtype("float32"), "f64": np.dtype("float64"),
          "i32": np.dtype("int32"), "i64": np.dtype("int64")}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_sources(mapping):
    names = {Path(name).name: value for name, value in mapping.items()}
    require(len(names) == len(mapping) == len(RUNTIME), "compiler mapping is missing/duplicating runtime members")
    require(set(names) == set(RUNTIME), "compiler member roster differs from five-file runtime")
    return names


def independently_compare(actual, expected, rtol, atol, name):
    require(actual.dtype == expected.dtype, f"{name}: output/reference dtype differs")
    require(actual.shape == expected.shape, f"{name}: output/reference shape differs")
    if actual.dtype.kind in "iu" and rtol == atol == 0:
        # np.isclose promotes large integers to float64 and can hide +/-1 errors.
        good = actual == expected
        delta = np.asarray(np.abs(actual.astype(object) - expected.astype(object)), dtype=np.float64)
    else:
        good = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
        delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    finite = delta[np.isfinite(delta)]
    row = {"name": name, "pass": bool(good.all()), "elements": int(actual.size),
           "mismatches": int(np.count_nonzero(~good)),
           "max_abs_error": float(finite.max()) if finite.size else None}
    if not row["pass"]:
        index = tuple(np.argwhere(~good)[0]) if actual.shape else ()
        row["first_failure"] = {"index": list(map(int, index)), "actual": actual[index].item(), "expected": expected[index].item()}
    return row


def audit(folder, manifest_path):
    folder = folder.resolve()
    manifest_path = manifest_path.resolve()
    summary_path = folder / "summary.json"
    summary, manifest = read(summary_path), read(manifest_path)
    require("finished_utc" in summary, "refusing to audit an unfinished runner report")
    output = folder / "integrity-audit.json"
    require(not output.exists(), "refusing to overwrite an existing independent audit")
    source = {name: digest(ROOT / name) for name in RUNTIME}
    result = {
        "schema": "emmy-phase9-feature-wire-audit-v1",
        "summary": summary_path.relative_to(ROOT).as_posix(),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "integer_comparison": "Exact typed integer equality; JSON floats admitted only when integral, in-range and strictly below 2**53 in magnitude",
        "scope": "Independent offline raw HTTP/ABI/typed-output replay and real CLI PTX reproduction; no network, assembly or GPU execution",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_pass": False, "numerical_gate_pass": False,
        "original_report_pass": summary.get("pass"),
        "summary_sha256": digest(summary_path), "feature_manifest_sha256": digest(manifest_path),
        "compiler_sha256": source, "auditor_sha256": digest(Path(__file__)),
        "jobs": [], "events_verified": 0, "http_attempts_verified": 0,
    }
    retained_hashes = {summary_path: result["summary_sha256"], manifest_path: result["feature_manifest_sha256"]}
    try:
        require(normalized_sources(summary["compiler_sha256"]) == source, "runner source-before hashes differ from current compiler")
        require(normalized_sources(summary["compiler_sha256_after"]) == source, "runner source-after hashes differ from current compiler")
        require(normalized_sources(manifest["compiler_sha256"]) == source, "feature manifest identifies another compiler")
        require(summary["tracked_unchanged"] is True, "runner observed changed tracked sources")
        for name, expected in summary["tracked_sha256"].items():
            path = Path(name)
            require(digest(path) == expected, f"changed runner-tracked file: {path.name}")
            retained_hashes[path] = expected
        for name, expected in manifest["files_sha256"].items():
            path = ROOT / name
            require(path.resolve().is_relative_to(ROOT / "examples"), "feature manifest path escapes examples")
            require(digest(path) == expected, f"changed exported feature file: {name}")
            retained_hashes[path] = expected
        for record in manifest.get("provenance", {}).values():
            path = ROOT / record["path"]
            require(path.resolve().is_relative_to(ROOT), "provenance path escapes repository")
            require(digest(path) == record["sha256"], "changed manifest provenance source")
            retained_hashes[path] = record["sha256"]
        roster = [(ROOT / name).resolve() for name in manifest["packages"]]
        require(roster and len(roster) == len(set(roster)) == manifest["package_count"], "feature manifest package count/uniqueness differs")
        require(all(p.is_relative_to(ROOT / "examples") for p in roster), "package path escapes examples")
        require(set(manifest["files_sha256"]) == {p.relative_to(ROOT).as_posix() for package in roster for p in package.rglob("*") if p.is_file()}, "package file roster differs from manifest")
        require([Path(p).resolve() for p in summary["planned_packages"]] == roster, "planned runner roster differs from feature manifest")
        require(summary["planned_checks"] == len(roster), "planned check count differs from roster")
        require([Path(j["package"]).resolve() for j in summary["jobs"]] == roster, "attempted job roster differs from manifest")

        decoded = {}
        evidence_files = set()
        for number, event in enumerate(summary["events"]):
            require(event["index"] == number, "event index/order changed")
            require(0 <= event["job"] < len(roster), "event references nonexistent job")
            base = folder / "events" / f"{number:04d}"
            request_path = base.with_suffix(".request.json")
            require(digest(request_path) == event["request_sha256"], "request hash mismatch")
            request = read(request_path)
            require(request["endpoint"] == event["endpoint"], "request endpoint mismatch")
            retained_hashes[request_path] = event["request_sha256"]
            evidence_files.add(request_path.resolve())
            attempts = []
            for attempt_number, record in enumerate(event["http"]):
                path = folder / record["path"]
                expected_name = base.with_name(base.name + f".http-{attempt_number:02d}.json")
                require(path.resolve() == expected_name.resolve(), "HTTP attempt path/order differs")
                require(digest(path) == record["sha256"], "HTTP attempt hash mismatch")
                attempts.append(read(path))
                retained_hashes[path] = record["sha256"]
                evidence_files.add(path.resolve())
                result["http_attempts_verified"] += 1
            require(attempts, "wire event has no retained HTTP attempt")
            response = None
            if event["status"] == "received":
                response_path = base.with_suffix(".response.json")
                require(digest(response_path) == event["response_sha256"], "response hash mismatch")
                response = read(response_path)
                require(attempts[-1].get("http_status") == 200, "received response has no final HTTP 200")
                require(attempts[-1].get("body") == response, "response differs from final HTTP body")
                require(response.get("stage") != "busy", "terminal response is still busy")
                retained_hashes[response_path] = event["response_sha256"]
                evidence_files.add(response_path.resolve())
            else:
                require(event["status"] == "transport_error" and event.get("error"), "event is unfinished or has unknown status")
            decoded[number] = (request, response)
            result["events_verified"] += 1
        require({p.resolve() for p in (folder / "events").iterdir() if p.is_file()} == evidence_files,
                "unreferenced or missing wire evidence files")

        for job_number, (package, job) in enumerate(zip(roster, summary["jobs"])):
            metadata = read(package / "meta.json")
            require(len(metadata["kernels"]) == 1, "this fixed audit requires one kernel per example")
            kernel = metadata["kernels"][0]
            require(job["id"] == package.name, "job ID does not match selected package")
            events = [event for event in summary["events"] if event["job"] == job_number]
            require(events and events[0]["endpoint"] == "/compile", "job has no leading compile event")
            require([e["endpoint"] for e in events] in (["/compile"], ["/compile", "/run"]), "unexpected per-job request order/count")
            compile_request, compile_response = decoded[events[0]["index"]]
            ptx_bytes = compile_request["payload"]["ptx"].encode("ascii")
            require(set(compile_request["payload"]) == {"ptx"}, "unexpected compile request keys")
            ir = package / (kernel["entry"] + ".ir.json")
            rerun = subprocess.run([sys.executable, "-B", str(ROOT / "compile.py"), str(ir)], capture_output=True, timeout=90)
            require(rerun.returncode == 0 and not rerun.stderr, f"current CLI failed for {package.name}")
            require(rerun.stdout == ptx_bytes, f"current CLI PTX differs from submitted bytes: {package.name}")
            ptx_sha = hashlib.sha256(ptx_bytes).hexdigest()
            row = {"id": job["id"], "package": package.relative_to(ROOT).as_posix(), "ptx_sha256": ptx_sha,
                   "cli_byte_identical": True, "numerical_pass": False, "output_comparisons": []}
            result["jobs"].append(row)
            if compile_response is None or not compile_response.get("ok") or len(events) == 1:
                require(job["pass"] is False and "error" in job, "failed compile/transport was not retained as failed job")
                row["original_error"] = job["error"]
                continue
            require(kernel["entry"] in compile_response["entry_points"], "compiled entry differs from metadata")
            run_request, run_response = decoded[events[1]["index"]]
            payload = run_request["payload"]
            require(payload["cubin_id"] == compile_response["cubin_id"], "run cubin differs from compile response")
            for key in ("entry", "grid", "block", "smem_bytes", "zero_outputs"):
                require(payload[key] == kernel[key], f"run launch contract changed: {key}")
            require(payload["warmup"] == 0 and payload["iters"] == 1, "timing/launch repetition contract changed")
            args_by_name = {argument["name"]: argument for argument in kernel["args"]}
            require(len(args_by_name) == len(kernel["args"]), "duplicate metadata argument")
            require(set(kernel["arg_order"]) == set(args_by_name), "metadata ABI argument roster mismatch")
            arguments = [args_by_name[name] for name in kernel["arg_order"]]
            require(payload["args"] == arguments, "actual launch argument bytes/descriptors differ from canonical metadata")
            require(payload["max_return"] == max(4096, sum(math.prod(a["shape"]) for a in arguments if a.get("output"))), "output return limit differs")
            row["inputs"] = []
            for argument in arguments:
                if argument.get("output"):
                    continue
                dtype = DTYPES[argument["dtype"]]
                shape = tuple(argument["shape"])
                if "data" in argument:
                    array = np.asarray(argument["data"], dtype=dtype).reshape(shape)
                else:
                    require("seed" in argument, "unrecognized canonical input construction")
                    array = np.random.default_rng(argument["seed"]).standard_normal(shape).astype(dtype)
                row["inputs"].append({"name": argument["name"], "dtype": argument["dtype"], "shape": list(shape),
                                      "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest()})
            if run_response is None or not run_response.get("ok"):
                require(job["pass"] is False and "error" in job, "failed run/transport was not retained as failed job")
                row["original_error"] = job["error"]
                continue
            for key in ("entry", "grid", "block", "smem_bytes"):
                require(run_response[key] == kernel[key], f"response launch echo differs: {key}")
            output_args = {a["name"]: a for a in arguments if a.get("output")}
            require(set(run_response["outputs"]) == set(output_args), "response output roster differs from ABI")
            with np.load(package / "expected.npz", allow_pickle=False) as expected:
                require(set(expected.files) == set(metadata["graph_outputs"]) == set(output_args), "reference/graph/output roster differs")
                for name in expected.files:
                    raw = run_response["outputs"][name]
                    argument = output_args[name]
                    require(raw["dtype"] == argument["dtype"], f"{name}: returned dtype differs from ABI")
                    require(raw["shape"] == argument["shape"], f"{name}: returned shape differs from ABI")
                    require(not raw.get("truncated"), f"{name}: truncated output")
                    dtype = DTYPES[raw["dtype"]]
                    if dtype.kind in "iu":
                        bounds = np.iinfo(dtype)
                        wire_values = np.asarray(raw["values"], dtype=object).flat
                        # This runner serializes integer values as JSON floats.
                        # Accept only integral in-range floats strictly inside
                        # the binary64 exact-integer range. Large i64 outputs
                        # must use word transport, because lost low bits cannot
                        # be reconstructed from a rounded response.
                        require(all((type(v) is int and bounds.min <= v <= bounds.max)
                                    or (type(v) is float and math.isfinite(v)
                                        and v.is_integer() and abs(v) < 2**53
                                        and bounds.min <= v <= bounds.max)
                                    for v in wire_values),
                                f"{name}: integer wire values cannot certify exact bits")
                    actual = np.asarray(raw["values"], dtype=dtype).reshape(raw["shape"])
                    row["output_comparisons"].append(independently_compare(actual, expected[name], metadata["rtol"], metadata["atol"], name))
            row["numerical_pass"] = all(c["pass"] for c in row["output_comparisons"])
            recorded = job["result"]
            require(recorded["problem"] == metadata["id"] and recorded["mode"] == "remote", "job result identity changed")
            require(len(recorded["kernels"]) == len(recorded["cases"]) == 1, "result kernel/case count differs")
            compiled = recorded["kernels"][0]
            require(compiled["entry"] == kernel["entry"] and compiled["cubin_id"] == compile_response["cubin_id"], "kernel record/cubin mismatch")
            require(compiled["ptx_sha256"] == ptx_sha and compiled["ptx_bytes"] == len(ptx_bytes), "kernel record PTX identity mismatch")
            require(normalized_sources(compiled["compiler_sources_sha256"]) == source, "kernel record compiler identity mismatch")
            observed = recorded["cases"][0]
            require(observed["seed_offset"] == 0 and observed["pattern"] == "normal", "case input construction changed")
            require(observed["runs"] == [{k: v for k, v in run_response.items() if k != "outputs"}], "case launch result differs from raw response")
            require(observed["outputs"] == row["output_comparisons"], "independent comparison differs from recorded comparison")
            require(observed["pass"] == recorded["pass"] == job["pass"] == row["numerical_pass"], "numerical failure was relabelled")

        checks = sum(bool(job["output_comparisons"]) for job in result["jobs"])
        passed = sum(job["numerical_pass"] for job in result["jobs"])
        require(summary["checks"] == checks and summary["passed_checks"] == passed, "summary execution/pass count differs from replay")
        numerical_gate = checks == len(roster) and passed == len(roster)
        require(summary["pass"] == numerical_gate, "original aggregate status contradicts retained executions")
        require(all(digest(path) == expected for path, expected in retained_hashes.items()), "audited sources/evidence changed during offline replay")
        require({name: digest(ROOT / name) for name in RUNTIME} == source, "runtime sources changed during CLI replay")
        result.update(integrity_pass=True, numerical_gate_pass=numerical_gate,
                      planned_checks=len(roster), checks=checks, passed_checks=passed,
                      failed_checks=checks - passed, exact_selected_roster=True,
                      files_verified=len(retained_hashes), complete_replay=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("integrity_pass", "numerical_gate_pass", "events_verified", "http_attempts_verified")}
                     | {"report": str(output.relative_to(ROOT)), "error": result.get("error")}, indent=2))
    return 0 if result["integrity_pass"] and result["numerical_gate_pass"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(audit(args.report_dir, args.manifest))
