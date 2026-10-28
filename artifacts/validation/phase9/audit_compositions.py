"""Offline evidence replay for the fixed 8-package, 24-input composition study.

No network, GPU, ptxas, or execution of third-party code. The selected own
compile.py is rerun to reproduce submitted PTX. Input construction and result
decoding are implemented here rather than calling the runner/checker helpers.
Mathematical references are the frozen original expressions_phase9.py specs;
replaying them is an evidence audit, not another independent math oracle.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

RUNTIME = ("compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt")
DTYPES = {"f16": np.dtype("float16"), "f32": np.dtype("float32"),
          "f64": np.dtype("float64"), "i32": np.dtype("int32"), "i64": np.dtype("int64")}
IDS = ("phase9_gelu_erf_tail", "phase9_gelu_tanh_matrix", "phase9_gelu_erf_half",
       "phase9_gelu_tanh_half", "phase9_bias_gelu_gemm", "phase9_half_gelu_gemm",
       "phase9_gelu_row_dot", "phase9_gelu_attention")
CASES = ((0, "normal"), (17, "normal"), (101, "wide"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def diagnostic(value):
    """Canonical comparison of retained JSON, including diagnostic NaNs."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: diagnostic(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic(item) for item in value]
    return value


def restore(value):
    if isinstance(value, dict):
        if set(value) == {"invalid_nonfinite_float"}:
            require(value["invalid_nonfinite_float"] in ("nan", "inf", "-inf"),
                    "invalid diagnostic floating marker")
            return float(value["invalid_nonfinite_float"])
        return {key: restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore(item) for item in value]
    return value


def source_paths(mapping):
    result = {Path(name).resolve(): value for name, value in mapping.items()}
    require(len(result) == len(mapping) == len(RUNTIME), "runtime source roster count differs")
    return result


def independent_input(argument, offset, pattern):
    dtype = DTYPES[argument["dtype"]]
    shape = tuple(argument.get("shape") or [1])
    if "data" in argument:
        values = np.asarray(argument["data"], dtype=dtype)
        require(values.size == math.prod(shape), "input data count differs from shape")
        return values.reshape(shape)
    require("seed" in argument, "generated input has neither explicit data nor a seed")
    require(type(argument["seed"]) is int, "input seed is not an integer")
    values = np.random.default_rng(argument["seed"] + offset).standard_normal(shape)
    require(pattern in ("normal", "wide"), "unexpected input pattern")
    if pattern == "wide":
        values *= 8
    return values.astype(dtype)


def independent_arguments(kernel, offset, pattern):
    by_name = {arg["name"]: arg for arg in kernel["args"]}
    require(len(by_name) == len(kernel["args"]), "duplicate metadata argument name")
    require(len(kernel["arg_order"]) == len(set(kernel["arg_order"])), "duplicate ABI argument")
    require(set(by_name) == set(kernel["arg_order"]), "ABI argument roster differs from metadata")
    result, inputs = [], {}
    for name in kernel["arg_order"]:
        arg = dict(by_name[name])
        require(not arg.get("intermediate"), "fixed single-kernel study unexpectedly has intermediates")
        arg.pop("intermediate", None)
        if not arg.get("output") and not arg.get("scalar"):
            array = independent_input(arg, offset, pattern)
            inputs[name] = array
            if pattern == "wide":
                arg["data"] = array.ravel().tolist()
                arg.pop("seed", None)
            elif "seed" in arg:
                arg["seed"] += offset
        result.append(arg)
    return result, inputs


def compare(actual, expected, rtol, atol, name):
    require(actual.dtype == expected.dtype, f"{name}: output/reference dtype differs")
    require(actual.shape == expected.shape, f"{name}: output/reference shape differs")
    with np.errstate(all="ignore"):
        if actual.dtype.kind in "iu" and rtol == atol == 0:
            good = actual.astype(object) == expected.astype(object)
            delta = np.asarray(np.abs(actual.astype(object) - expected.astype(object)), np.float64)
        else:
            good = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
            delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    finite = delta[np.isfinite(delta)]
    row = {"name": name, "pass": bool(good.all()), "elements": int(actual.size),
           "mismatches": int(np.count_nonzero(~good)),
           "max_abs_error": float(finite.max()) if finite.size else None}
    if not row["pass"]:
        index = tuple(np.argwhere(~good)[0]) if actual.shape else ()
        row["first_failure"] = {"index": list(map(int, index)),
                                "actual": actual[index].item(), "expected": expected[index].item()}
    return row


def decode_output(raw, argument):
    name = argument["name"]
    require(raw["dtype"] == argument["dtype"], f"{name}: returned dtype differs from ABI")
    require(raw["shape"] == argument["shape"], f"{name}: returned shape differs from ABI")
    require(not raw.get("truncated"), f"{name}: output is truncated")
    dtype = DTYPES[raw["dtype"]]
    values = np.asarray(raw["values"], dtype=object)
    require(values.size == math.prod(raw["shape"]), f"{name}: output element count differs")
    require(all(type(v) in (int, float) for v in values.flat), f"{name}: nonnumeric wire output")
    if dtype.kind in "iu":
        bounds = np.iinfo(dtype)
        require(all((type(v) is int and bounds.min <= v <= bounds.max)
                    or (type(v) is float and math.isfinite(v) and v.is_integer()
                        and abs(v) < 2**53 and bounds.min <= v <= bounds.max)
                    for v in values.flat), f"{name}: integer wire values cannot certify exact bits")
    return np.asarray(raw["values"], dtype=dtype).reshape(raw["shape"])


def audit(folder, root, output):
    folder, root, output = folder.resolve(), root.resolve(), output.resolve()
    require(not output.exists(), "refusing to overwrite an existing audit")
    result = {"schema": "emmy-phase9-composition-wire-audit-v1",
              "scope": "Offline raw HTTP/ABI/typed-output replay and own CLI byte reproduction; no network, assembly or GPU",
              "oracle_scope": "Replays frozen own expression references; this is an evidence audit, not an independent mathematical oracle",
              "audited_utc": datetime.now(timezone.utc).isoformat(),
              "integrity_pass": False, "numerical_gate_pass": False,
              "auditor_sha256": digest(Path(__file__)), "jobs": [],
              "events_verified": 0, "http_attempts_verified": 0, "cli_replays": 0}
    retained = {}

    def retain(path, expected=None):
        path = path.resolve()
        observed = digest(path)
        require(expected is None or observed == expected, f"file hash mismatch: {path}")
        require(path not in retained or retained[path] == observed, f"file changed during audit: {path}")
        retained[path] = observed
        return observed

    try:
        require(folder.is_relative_to(root / "artifacts/validation/phase9"), "report directory escapes phase9 evidence")
        here = root / "artifacts/validation/phase9"
        summary_path = folder / "summary.json"
        summary = read(summary_path)
        result.update(summary=str(summary_path), summary_sha256=retain(summary_path),
                      original_report_pass=summary.get("pass"))
        require(summary.get("schema") == "emmy-phase9-composition-runner-v1", "unexpected report schema")
        require("finished_utc" in summary, "runner report is unfinished")
        require(summary["local_gpu_used"] is False, "report unexpectedly claims local GPU use")
        protocol_path = here / "composition-protocol.json"
        generation_path = here / "composition-generation.json"
        sanity_path = here / "composition-reference-sanity.json"
        protocol, generation, sanity = map(read, (protocol_path, generation_path, sanity_path))
        for path in (protocol_path, generation_path, sanity_path):
            retain(path)
        require(generation["pass"] is True and sanity["pass"] is True, "generation/reference sanity gate was not passed")
        require(sanity["checks"] == sanity["passed_checks"] == 24, "CPU reference sanity count differs")
        require(generation["planned_packages"] == generation["generated_packages"] == 8, "generated package count differs")
        require([item["id"] for item in protocol["planned"]] == list(IDS), "protocol workload roster differs")
        require(protocol["input_cases"] == [list(case) for case in CASES], "protocol input-case roster differs")
        require([(row["id"], row["seed"], row["pattern"]) for row in sanity["rows"]]
                == [(name, seed, pattern) for name in IDS for seed, pattern in CASES], "CPU sanity roster differs")
        require(all(row["pass"] for row in sanity["rows"]), "CPU reference sanity retained a failure")
        expression = root / "tests/expressions_phase9.py"
        retain(expression, protocol["expression_sha256"])
        require(generation["expression_sha256"] == sanity["expression_sha256"] == digest(expression), "expression hashes differ")
        spec = importlib.util.spec_from_file_location("phase9_audit_own_references", expression)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        specs = {item["id"]: item for item in module.OPEN}
        require(len(specs) == len(module.OPEN) == 8 and tuple(specs) == IDS, "own reference roster differs")
        require([{k: v for k, v in item.items() if k != "ref"} for item in module.OPEN]
                == protocol["planned"], "own expression definitions/tolerances differ from frozen protocol")

        compiler = Path(summary["compiler"]).resolve()
        require(compiler in ((root / "compile.py").resolve(), (here / "baseline/compile.py").resolve()),
                "report compiler is outside the two authorized own compiler locations")
        sources = {compiler.parent / name: digest(compiler.parent / name) for name in RUNTIME}
        require(source_paths(summary["compiler_sha256"]) == sources, "compiler source-before mapping differs")
        require(source_paths(summary["compiler_sha256_after"]) == sources, "compiler source-after mapping differs")
        for path, value in sources.items():
            retain(path, value)
        result.update(compiler=str(compiler), compiler_sha256={str(p): h for p, h in sources.items()})
        require(summary["tracked_unchanged"] is True, "runner observed changed tracked evidence")
        roster = [root / "examples/phase9/generated" / name for name in IDS]
        exported = {root / name: value for name, value in generation["files_sha256"].items()}
        require(set(exported) == {p for package in roster for p in package.rglob("*") if p.is_file()},
                "generated file manifest differs from actual package files")
        tracked_required = set(exported) | {here / "run_compositions.py", root / "tools/check_compiler.py",
            root / "tools/selector_study.py", expression, protocol_path, generation_path, sanity_path}
        tracked = {Path(name).resolve(): value for name, value in summary["tracked_sha256"].items()}
        require(set(tracked) == tracked_required, "runner tracked-file roster differs")
        for path, value in tracked.items():
            require(path.is_relative_to(root), "tracked path escapes own repository")
            retain(path, value)
        for path, value in exported.items():
            retain(path, value)
        require([Path(p).resolve() for p in summary["planned_packages"]] == roster, "planned package order differs")
        require(summary["planned_cases"] == protocol["input_cases"], "runner planned input cases differ")
        require(summary["planned_checks"] == 24, "planned job count differs")
        jobs = [(p, seed, pattern) for p in roster for seed, pattern in CASES]
        require([(Path(j["package"]).resolve(), j["seed"], j["pattern"]) for j in summary["jobs"]] == jobs,
                "attempted job roster/count/order differs from protocol")

        decoded, event_files = {}, set()
        previous_job = -1
        for number, event in enumerate(summary["events"]):
            require(event["index"] == number, "event index/order differs")
            require(type(event["job"]) is int and previous_job <= event["job"] < 24, "event job ownership/order differs")
            previous_job = event["job"]
            base = folder / "events" / f"{number:04d}"
            request_path = base.with_suffix(".request.json")
            retain(request_path, event["request_sha256"]); event_files.add(request_path)
            request = read(request_path)
            require(set(request) == {"endpoint", "payload"} and request["endpoint"] == event["endpoint"], "request endpoint/wrapper differs")
            attempts = []
            for attempt_number, record in enumerate(event["http"]):
                path = (folder / record["path"]).resolve()
                require(path == base.with_name(base.name + f".http-{attempt_number:02d}.json"), "HTTP path/order differs")
                retain(path, record["sha256"]); event_files.add(path)
                attempt = read(path)
                require(isinstance(attempt.get("seconds"), (int, float)) and attempt["seconds"] >= 0, "invalid HTTP duration")
                require("started" in attempt and "finished" in attempt, "HTTP attempt lacks timestamps")
                attempts.append(attempt)
                result["http_attempts_verified"] += 1
            require(attempts, "event has no recorded HTTP attempts")
            response = None
            if event["status"] == "received":
                path = base.with_suffix(".response.json")
                retain(path, event["response_sha256"]); event_files.add(path)
                response = read(path)
                require(attempts[-1].get("http_status") == 200, "received response lacks final HTTP 200")
                require(diagnostic(attempts[-1].get("body")) == diagnostic(response), "response differs from final HTTP body")
                require(response.get("stage") != "busy", "received response is still busy")
                response = restore(response)
            else:
                require(event["status"] == "transport_error" and event.get("error"), "event is unfinished/unknown")
            decoded[number] = (request, response)
            result["events_verified"] += 1
        require({p.resolve() for p in (folder / "events").iterdir() if p.is_file()} == event_files,
                "unreferenced/missing wire evidence files")

        cli_cache = {}
        for job_number, ((package, seed, pattern), job) in enumerate(zip(jobs, summary["jobs"])):
            metadata = read(package / "meta.json")
            own_spec = specs[package.name]
            require(metadata["id"] == package.name and metadata["code"] == own_spec["code"], "metadata workload identity differs")
            require((metadata["rtol"], metadata["atol"]) == (own_spec["rtol"], own_spec["atol"]), "metadata tolerances differ from frozen spec")
            require(len(metadata["kernels"]) == 1, "fixed study requires exactly one kernel per package")
            kernel = metadata["kernels"][0]
            arguments, inputs = independent_arguments(kernel, seed, pattern)
            require(set(metadata["graph_inputs"]) <= set(inputs), "missing mathematical input in kernel ABI")
            with np.errstate(all="ignore"):
                references = own_spec["ref"](*(inputs[name] for name in metadata["graph_inputs"]))
            require(len(references) == len(metadata["graph_outputs"]), "reference output count differs")
            expected = dict(zip(metadata["graph_outputs"], map(np.asarray, references)))
            if seed == 0 and pattern == "normal":
                with np.load(package / "expected.npz", allow_pickle=False) as archive:
                    require(set(archive.files) == set(expected), "baseline archive output roster differs")
                    for name, array in expected.items():
                        require(array.dtype == archive[name].dtype and np.array_equal(array, archive[name]),
                                "baseline expected archive differs from recomputed own reference")
            row = {"id": package.name, "seed": seed, "pattern": pattern,
                   "numerical_pass": False, "output_comparisons": [],
                   "rtol": metadata["rtol"], "atol": metadata["atol"],
                   "inputs": [{"name": name, "dtype": str(array.dtype), "shape": list(array.shape),
                               "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
                              for name, array in inputs.items()],
                   "references": [{"name": name, "dtype": str(array.dtype), "shape": list(array.shape),
                                   "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
                                  for name, array in expected.items()]}
            result["jobs"].append(row)
            ir = package / (kernel["entry"] + ".ir.json")
            if ir not in cli_cache:
                cli_cache[ir] = subprocess.run([sys.executable, "-B", str(compiler), str(ir)], capture_output=True, timeout=90)
                result["cli_replays"] += 1
            cli = cli_cache[ir]
            events = [event for event in summary["events"] if event["job"] == job_number]
            require([event["endpoint"] for event in events] in ([], ["/compile"], ["/compile", "/run"]), "unexpected per-job endpoint roster")
            if not events:
                require(cli.returncode != 0 and job["pass"] is False and job.get("error"), "missing compile event lacks reproduced CLI failure")
                row["original_error"] = job["error"]
                continue
            require(cli.returncode == 0 and not cli.stderr, f"selected CLI failed for {package.name}")
            compile_request, compile_response = decoded[events[0]["index"]]
            require(set(compile_request["payload"]) == {"ptx"}, "unexpected compile payload keys")
            ptx = compile_request["payload"]["ptx"].encode("ascii")
            require(ptx == cli.stdout, f"submitted PTX differs from selected compiler: {package.name}")
            ptx_sha = hashlib.sha256(ptx).hexdigest()
            row.update(cli_byte_identical=True, ptx_sha256=ptx_sha)
            if compile_response is None or not compile_response.get("ok"):
                require(len(events) == 1 and job["pass"] is False and job.get("error"), "failed compile/transport was not retained")
                row["original_error"] = job["error"]
                continue
            require(kernel["entry"] in compile_response["entry_points"], "compiled entry differs from metadata")
            require(len(events) == 2, "successful assembly has no run request")
            run_request, response = decoded[events[1]["index"]]
            expected_request = {key: kernel[key] for key in ("entry", "grid", "block", "smem_bytes", "zero_outputs")}
            expected_request.update(args=arguments, warmup=0, iters=1,
                max_return=max(4096, sum(math.prod(arg["shape"]) for arg in arguments if arg.get("output"))),
                cubin_id=compile_response["cubin_id"])
            require(run_request["payload"] == expected_request, "run request differs from exact reconstructed ABI/input contract")
            if response is None or not response.get("ok"):
                require(job["pass"] is False and job.get("error"), "failed run/transport was not retained")
                row["original_error"] = job["error"]
                continue
            for key in ("entry", "grid", "block", "smem_bytes"):
                require(response[key] == kernel[key], f"run response echo differs: {key}")
            require(response["iters"] == 1, "run response iteration count differs")
            output_args = {arg["name"]: arg for arg in arguments if arg.get("output")}
            require(set(response["outputs"]) == set(output_args) == set(expected), "output/reference/ABI roster differs")
            for name, reference in expected.items():
                actual = decode_output(response["outputs"][name], output_args[name])
                row["output_comparisons"].append(compare(actual, reference, metadata["rtol"], metadata["atol"], name))
            row["numerical_pass"] = all(item["pass"] for item in row["output_comparisons"])
            recorded = job["result"]
            require(recorded["problem"] == metadata["id"] and recorded["mode"] == "remote", "recorded workload identity differs")
            require(len(recorded["kernels"]) == len(recorded["cases"]) == 1, "recorded kernel/case count differs")
            compiled, observed = recorded["kernels"][0], recorded["cases"][0]
            require(compiled["entry"] == kernel["entry"] and compiled["cubin_id"] == compile_response["cubin_id"], "kernel record/cubin identity differs")
            require(compiled["ptx_sha256"] == ptx_sha and compiled["ptx_bytes"] == len(ptx), "kernel PTX record differs")
            require(source_paths(compiled["compiler_sources_sha256"]) == sources, "kernel compiler identity differs")
            require(compiled.get("registers") == compile_response.get("registers", {}).get(kernel["entry"]), "recorded register count differs")
            require(observed["seed_offset"] == seed and observed["pattern"] == pattern, "recorded input case differs")
            require(diagnostic(observed["runs"]) == diagnostic([{key: value for key, value in response.items() if key != "outputs"}]), "recorded launch result differs from response")
            require(diagnostic(observed["outputs"]) == diagnostic(row["output_comparisons"]), "recorded numerical metrics differ from typed replay")
            require(observed["pass"] == recorded["pass"] == job["pass"] == row["numerical_pass"], "numerical failure/status was relabelled")

        completed = sum(bool(row["output_comparisons"]) for row in result["jobs"])
        passed = sum(row["numerical_pass"] for row in result["jobs"])
        require(summary["checks"] == len(jobs) and summary["passed_checks"] == passed, "summary attempted/pass counts differ")
        gate = completed == passed == len(jobs)
        require(summary["pass"] == gate, "aggregate numerical status contradicts retained evidence")
        require(all(digest(path) == value for path, value in retained.items()), "sources/evidence changed during replay")
        result.update(integrity_pass=True, numerical_gate_pass=gate, planned_checks=24,
                      attempted_checks=24, completed_checks=completed, passed_checks=passed,
                      failed_checks=24 - passed, files_verified=len(retained),
                      exact_selected_roster=True, complete_replay=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(diagnostic(result), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("integrity_pass", "numerical_gate_pass", "events_verified", "http_attempts_verified")}
                     | {"report": str(output), "error": result.get("error")}, indent=2))
    return 0 if result["integrity_pass"] and result["numerical_gate_pass"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise SystemExit(audit(args.report_dir, args.root, args.output or args.report_dir / "integrity-audit.json"))
