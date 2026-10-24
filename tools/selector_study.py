"""Frozen CPU preparation and serialized remote evaluation of an unroll pilot.

Development only. Production compilation never imports this module. ``prepare``
uses CPU tracing/reference/assembly work; ``run`` is a separate, explicit remote
operation. ``fit`` reads train+validation only; ``evaluate`` opens final results
only after a hashed policy choice exists. Learning is offline reward replay.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_compiler import (DTYPES, SERVER, assemble, compare_outputs,
                                 input_array, load_specs, make_args, remote_post)

ACTIONS = (0, 256, 1024)
BASELINE = 1024
SPLITS = ("train", "validation", "final")
SEED = 20261023
INPUT_CASES = ((0, "normal"), (17, "normal"), (101, "wide"))
PROTOCOL = {
    "version": 1, "seed": SEED, "actions": list(ACTIONS), "baseline": BASELINE,
    "action_scope": "compile_file(ir, unroll_budget=action); every other default unchanged",
    "global_expansion_cap": 4096, "target": "sm_89",
    "numerical_flags": "Frozen backend defaults; no FMA, precision, compensation or reassociation override",
    "launch_and_synchronization": "Exact generated metadata and scheduled computation preserved",
    "workloads": 30, "families": 15, "shapes_per_family": 2,
    "splits": {"train": 10, "validation": 10, "final": 10},
    "correctness": [{"seed": s, "pattern": p} for s, p in INPUT_CASES],
    "timing": {"rounds": 6, "iterations": 20, "warmup": 3,
               "order": "All six permutations for three unique PTX modules; AB/BA alternating for two",
               "inputs": "Frozen explicit input values, seed0 normal; check every returned output"},
    "feedback": {"logical_observations_per_training_workload": 2,
                 "baseline_observations": 1, "alternative_observations": 1,
                 "tree_logger": "seeded balanced alternatives0/256",
                 "bandit": "LinUCB alpha0.5 ridge1; one adaptive alternative0/256",
                 "meaning": "Offline replay of aggregate labels measured once; no on-policy RL"},
    "tree": {"max_depth": 2, "minimum_leaf_workloads": 2,
             "minimum_action_support": 2},
    "heuristic": "hinted_loop_work_max<=256 ->1024; <=1024 ->256; otherwise0; missing ->1024",
    "uncertainty": {"bootstrap_draws": 4000, "confidence": 0.95,
                    "sampling": "expression families, then paired timing rounds within workload"},
    "promotion": {"minimum_geomean_speedup": 1.03, "lower_ci_strictly_above": 1.0,
                  "maximum_workload_slowdown": 1.05, "minimum_distinct_workloads": 6,
                  "minimum_distinct_families": 2, "all_candidate_outputs_correct": True,
                  "validation_selects_before_final": True,
                  "fallback": "fixed1024; no automatic production change"},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def diagnostic_json(value):
    """Keep invalid numeric responses inspectable without accepting them as data."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json(item) for item in value]
    return value


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_splits(specs):
    seen = {}
    counts = {split: 0 for split in SPLITS}
    categories = {split: set() for split in SPLITS}
    for item in specs:
        split = item["split"]
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split}")
        for key in ("family", "schedule_family"):
            group = (key, item[key])
            if group in seen and seen[group] != split:
                raise ValueError(f"Family crosses split: {group}")
            seen[group] = split
        counts[split] += 1
        categories[split].add(item["category"])
    if counts != PROTOCOL["splits"] or any(len(v) != 5 for v in categories.values()):
        raise ValueError(f"Require ten workloads/five categories per split: {counts}")
    if len({x["family"] for x in specs}) != 15:
        raise ValueError("Require15 distinct expression families")


def check_reference(spec, seed, pattern):
    """Independent CPU PyTorch sanity check of our authored NumPy oracle."""
    import torch
    torch.set_num_threads(1)
    inputs = []

    class CpuTorch:
        def randn(self, *shape):
            value = np.asarray(np.random.default_rng(seed + len(inputs)).standard_normal(shape)
                               * (8 if pattern == "wide" else 1), dtype=np.float32)
            inputs.append(value)
            return torch.from_numpy(value.copy())

        def __getattr__(self, name):
            return getattr(torch, name)

    tree = ast.parse(spec["code"])
    if not isinstance(tree.body[-1], ast.Expr):
        raise ValueError("Expression must end in its result")
    namespace = {"torch": CpuTorch()}
    exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), "<own-selector-spec>", "exec"), namespace)
    actual = eval(compile(ast.Expression(tree.body[-1].value), "<own-selector-result>", "eval"), namespace)
    expected = np.asarray(spec["ref"](*inputs)[0])
    actual = actual.detach().cpu().numpy()
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite reference-domain output")
    checks = compare_outputs({"out": actual}, {"out": expected}, spec["rtol"], spec["atol"])
    if spec["family"] == "sigmoid_product" and np.allclose(np.zeros_like(expected), expected,
                                                          rtol=spec["rtol"], atol=spec["atol"]):
        raise ValueError("Product oracle must reject an all-zero implementation")
    return {"seed": seed, "pattern": pattern, "pass": all(x["pass"] for x in checks), "checks": checks}


def spec_signature(spec):
    return {key: spec[key] for key in ("id", "code", "knobs", "rtol", "atol")}


def record_package_provenance(folder, spec, source):
    """Call only after this exact spec was generated by our recorded CPU command."""
    write_json(folder / "selector-provenance.json", {
        "spec": spec_signature(spec), "source": str(Path(source).resolve()),
        "source_sha256": sha(source), "generator_sha256": sha(ROOT / "tools/generate.py"),
        "worker_sha256": sha(ROOT / "tests/generate_stress.py"), "recorded": stamp(),
        "artifacts": {p.name: sha(p) for p in sorted(folder.iterdir())
                      if p.is_file() and p.name != "selector-provenance.json"}})


def verify_package(folder, spec):
    meta = read_json(folder / "meta.json")
    if any(meta.get(key) != spec[key] for key in ("code", "rtol", "atol")):
        raise ValueError("Existing package expression/tolerance differs from frozen spec")
    provenance = read_json(folder / "selector-provenance.json")
    if provenance["spec"] != spec_signature(spec):
        raise ValueError("Existing package generation knobs/spec differ from frozen spec")
    for key, path in (("source_sha256", provenance["source"]),
                      ("generator_sha256", ROOT / "tools/generate.py"),
                      ("worker_sha256", ROOT / "tests/generate_stress.py")):
        if provenance[key] != sha(path):
            raise ValueError("Package generation source/provenance changed")
    for name, digest in provenance["artifacts"].items():
        if sha(folder / name) != digest:
            raise ValueError("Generated package artifact changed")
    return meta


def frozen_inputs(meta, kernel, spec, seed, pattern):
    args = make_args(kernel, {}, seed, pattern)
    definitions = {a["name"]: a for a in kernel["args"]}
    arrays = {}
    for arg in args:
        if arg.get("intermediate"):
            raise ValueError("Single-kernel pilot cannot accept intermediate inputs")
        if not arg.get("output"):
            array = input_array(definitions[arg["name"]], seed, pattern)
            arrays[arg["name"]] = array
            arg["data"] = array.ravel().tolist()
            arg.pop("seed", None)
    reference = spec["ref"](*[arrays[n] for n in meta["graph_inputs"]])
    expected = {}
    for name, value in zip(meta["graph_outputs"], reference, strict=True):
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite independent oracle")
        expected[name] = {"shape": list(array.shape), "dtype": str(array.dtype), "values": array.ravel().tolist()}
    return {"seed": seed, "pattern": pattern, "args": args, "expected": expected,
            "input_sha256": {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in arrays.items()}}


def audit_remote_post(server, endpoint, payload, record_attempt):
    """Retain the existing wrapper's bounded retries as individual HTTP attempts.

    The runner is used serially in this process, so the temporary requests hook
    cannot intercept another study thread. It is restored even on exceptions.
    """
    import requests
    original_post = requests.post

    def logged_post(*args, **kwargs):
        start = time.perf_counter()
        entry = {"started": stamp()}
        try:
            response = original_post(*args, **kwargs)
            entry["http_status"] = response.status_code
            try:
                entry["body"] = response.json()
            except ValueError:
                entry["body_text"] = response.text
            return response
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry.update(finished=stamp(), seconds=time.perf_counter() - start)
            record_attempt(entry)

    with patch.object(requests, "post", logged_post):
        return remote_post(server, endpoint, payload)


def prepare(args):
    out = args.out.resolve()
    if (out / "seal.json").exists():
        raise ValueError("Study already sealed; use a new directory for a new protocol")
    out.mkdir(parents=True, exist_ok=True)
    specs_path = ROOT / "tests/expressions_selector.py"
    specs = list(load_specs(specs_path).values())
    check_splits(specs)
    draft = {"protocol": PROTOCOL, "spec_sha256": sha(specs_path),
             "roster": [{k: s[k] for k in ("id", "family", "schedule_family", "split", "category", "code", "knobs", "rtol", "atol")} for s in specs]}
    draft_path = out / "draft.json"
    if draft_path.exists() and read_json(draft_path) != draft:
        if not args.repair_note:
            raise ValueError("Changed unsealed draft requires --repair-note; never repair after timing")
        repairs = read_json(out / "generation-repairs.json") if (out / "generation-repairs.json").exists() else []
        repairs.append({"time": stamp(), "previous": read_json(draft_path), "note": args.repair_note})
        write_json(out / "generation-repairs.json", repairs)
    write_json(draft_path, draft)
    frozen = out / "frozen"
    frozen.mkdir(exist_ok=True)
    shutil.copy2(args.backend, frozen / "ptx_backend.py")
    shutil.copy2(specs_path, frozen / "expressions_selector.py")
    shutil.copy2(ROOT / "tools/analyze_ir.py", frozen / "analyze_ir.py")
    backend = load_module(frozen / "ptx_backend.py", "selector_frozen_backend")
    previous_backend = sys.modules.get("ptx_backend")
    sys.modules["ptx_backend"] = backend
    try:
        analyzer = load_module(frozen / "analyze_ir.py", "selector_frozen_analysis")
    finally:
        if previous_backend is None:
            del sys.modules["ptx_backend"]
        else:
            sys.modules["ptx_backend"] = previous_backend
    analyze_file = analyzer.analyze_file
    evidence = {str(p.resolve()): sha(p) for p in [Path(__file__), ROOT / "tools/check_compiler.py",
                ROOT / "tools/analyze_ir.py", ROOT / "tools/generate.py", ROOT / "tests/generate_stress.py",
                specs_path, args.backend]}
    distributions = {}
    for package in ("emmy-ml", "numpy", "torch"):
        try:
            distributions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            distributions[package] = "distribution-name-unavailable"
    if distributions["emmy-ml"] != "0.3.11":
        raise ValueError("This frozen study requires emmy-ml==0.3.11")
    report = {"started": stamp(), "cpu_only": True, "rows": [], "pass": False,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "packages": distributions}, "cpu_assembly_requested": not args.skip_assembly,
              "analysis_backend": "frozen analyzer bound to the same frozen backend used for emission"}
    write_json(out / "preparation.json", report)
    for spec in specs:
        row = {"id": spec["id"], "family": spec["family"], "split": spec["split"],
               "category": spec["category"], "pass": False}
        report["rows"].append(row)
        try:
            row["oracle_checks"] = [check_reference(spec, seed, pattern) for seed, pattern in INPUT_CASES]
            if not all(x["pass"] for x in row["oracle_checks"]):
                raise ValueError("CPU oracle disagrees with CPU PyTorch")
            folder = out / "corpus" / spec["id"]
            regenerate = args.generate
            try:
                verify_package(folder, spec)
            except (OSError, ValueError, KeyError) as exc:
                regenerate = True
                row["generation_reason"] = str(exc)
            if regenerate:
                result = subprocess.run([sys.executable, str(ROOT / "tests/generate_stress.py"),
                        "--expressions", str(frozen / "expressions_selector.py"), "--out", str(out / "corpus"),
                        "--only", spec["id"]], capture_output=True, text=True, encoding="utf-8", timeout=1000)
                row["generation"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
                if result.returncode:
                    raise ValueError("Emmy generation failed; retained diagnostics")
                record_package_provenance(folder, spec, frozen / "expressions_selector.py")
            meta = verify_package(folder, spec)
            if len(meta["kernels"]) != 1:
                raise ValueError("Workload is not single-kernel eligible")
            kernel = meta["kernels"][0]
            ir = folder / (kernel["entry"] + ".ir.json")
            analysis = analyze_file(ir)
            features = analysis["features"]
            if not features or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in features.values()):
                raise ValueError("Features must be finite structural numbers")
            row.update(features=features, meta=str((folder / "meta.json").relative_to(out)), ir=str(ir.relative_to(out)))
            write_json(out / "analysis" / (spec["id"] + ".json"), analysis)
            row["inputs"] = []
            for seed, pattern in INPUT_CASES:
                snapshot = frozen_inputs(meta, kernel, spec, seed, pattern)
                path = out / "inputs" / f"{spec['id']}-{seed}-{pattern}.json"
                write_json(path, snapshot)
                row["inputs"].append(str(path.relative_to(out)))
                if seed == 0 and pattern == "normal":
                    with np.load(folder / "expected.npz", allow_pickle=False) as generated:
                        for name, value in snapshot["expected"].items():
                            expected = np.asarray(value["values"], dtype=value["dtype"]).reshape(value["shape"])
                            if not np.array_equal(generated[name], expected):
                                raise ValueError("Packaged baseline reference differs from frozen oracle")
            row["actions"] = {}
            hashes = {}
            for action in ACTIONS:
                start = time.perf_counter()
                ptx = backend.compile_file(ir, unroll_budget=action)
                elapsed = time.perf_counter() - start
                encoded = ptx.encode("ascii")
                digest = hashlib.sha256(encoded).hexdigest()
                path = out / "ptx" / (digest + ".ptx")
                path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    path.write_bytes(encoded)
                data = {"ptx": str(path.relative_to(out)), "ptx_sha256": digest,
                        "ptx_bytes": len(encoded), "compile_seconds": elapsed}
                if digest not in hashes and not args.skip_assembly:
                    data["cpu_assembly"] = assemble(ptx, "sm_89")
                row["actions"][str(action)] = data
                hashes.setdefault(digest, []).append(action)
            row["unique_ptx"] = len(hashes)
            row["pass"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "preparation.json", report)
        print(spec["id"], "PASS" if row["pass"] else row.get("error"), flush=True)
    report["finished"] = stamp()
    report["pass"] = len(report["rows"]) == 30 and all(x["pass"] for x in report["rows"])
    report["source_evidence_unchanged"] = all(sha(p) == value for p, value in evidence.items())
    report["pass"] &= report["source_evidence_unchanged"]
    write_json(out / "preparation.json", report)
    if not report["pass"]:
        return 1
    artifacts = {}
    for directory in ("corpus", "inputs", "ptx", "analysis", "frozen"):
        for path in sorted((out / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                artifacts[str(path.relative_to(out))] = sha(path)
    write_json(out / "protocol.json", PROTOCOL)
    artifacts["protocol.json"] = sha(out / "protocol.json")
    artifacts["draft.json"] = sha(draft_path)
    artifacts["preparation.json"] = sha(out / "preparation.json")
    write_json(out / "seal.json", {"sealed": stamp(), "artifacts": artifacts,
               "source_evidence": evidence, "rows": report["rows"],
               "no_gpu_timings_observed_before_seal": True})
    return 0


def verify_seal(out):
    seal = read_json(out / "seal.json")
    for relative, expected in seal["artifacts"].items():
        if sha(out / relative) != expected:
            raise ValueError(f"Frozen artifact changed: {relative}")
    for path, expected in seal["source_evidence"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    return seal


def balanced_orders(labels):
    labels = list(labels)
    if len(labels) == 3:
        return [list(order) for order in itertools.permutations(labels)]
    if len(labels) == 2:
        return [labels[:] if i % 2 == 0 else labels[::-1] for i in range(6)]
    if len(labels) == 1:
        return [labels[:] for _ in range(6)]
    raise ValueError("Require one to three unique actions")


def validate_response(response, snapshot, kernel, meta):
    if not response.get("ok"):
        return {"pass": False, "error": diagnostic_json(response)}
    timing = float(response["median_us"])
    if not math.isfinite(timing) or timing <= 0:
        raise ValueError("Nonpositive or nonfinite runner timing")
    expected = {n: np.asarray(v["values"], dtype=v["dtype"]).reshape(v["shape"])
                for n, v in snapshot["expected"].items()}
    if set(response["outputs"]) != set(expected):
        raise ValueError("Unexpected or missing returned outputs")
    dtypes = {a["name"]: DTYPES[a["dtype"]] for a in kernel["args"]}
    actual = {}
    for name, output in response["outputs"].items():
        if output.get("truncated"):
            raise ValueError("Truncated returned output")
        value = np.asarray(output["values"], dtype=dtypes[name]).reshape(output["shape"])
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite returned output")
        actual[name] = value
    checks = compare_outputs(actual, expected, meta["rtol"], meta["atol"])
    return {"pass": bool(checks) and all(x["pass"] for x in checks),
            "median_us": timing, "outputs": checks}


def run_remote(args):
    if not args.runner_turn:
        raise ValueError("Explicit --runner-turn records the authorized exclusive runner slot")
    out = args.out.resolve()
    seal = verify_seal(out)
    if args.split == "final":
        selection = read_json(out / "selection.json")
        if selection["seal_sha256"] != sha(out / "seal.json"):
            raise ValueError("Final evaluation requires a policy selected on this exact study")
        if selection["models_sha256"] != sha(out / "models.json"):
            raise ValueError("Frozen models changed after validation selection")
        if selection["validation_sha256"] != sha(out / "results-validation.json"):
            raise ValueError("Validation evidence changed after policy selection")
    destination = out / f"results-{args.split}.json"
    if destination.exists():
        raise ValueError("Result file already exists; preserve failures rather than silently rerun")
    report = {"started": stamp(), "split": args.split, "server": args.server,
              "runner_turn": args.runner_turn, "seal_sha256": sha(out / "seal.json"),
              "selection_sha256": sha(out / "selection.json") if args.split == "final" else None,
              "events": [], "rows": [], "pass": False}
    lock = out / "runner.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(args.runner_turn)
    try:
        def request(endpoint, payload):
            number = len(report["events"])
            base = out / "events" / args.split / f"{number:04d}"
            write_json(base.with_suffix(".request.json"), {"endpoint": endpoint, "payload": payload})
            event = {"number": number, "workload": workload["id"], "endpoint": endpoint, "request_sha256": sha(base.with_suffix(".request.json")),
                     "started": stamp(), "status": "pending", "http_attempts": []}
            report["events"].append(event)
            write_json(destination, report)
            start = time.perf_counter()
            try:
                def record_attempt(attempt):
                    index = len(event["http_attempts"])
                    path = base.with_name(base.name + f".http-{index:02d}.json")
                    write_json(path, diagnostic_json(attempt))
                    event["http_attempts"].append({"path": str(path.relative_to(out)), "sha256": sha(path),
                                                   "http_status": attempt.get("http_status"),
                                                   "error": attempt.get("error"), "seconds": attempt["seconds"]})
                    write_json(destination, report)

                response = audit_remote_post(args.server, endpoint, payload, record_attempt)
                write_json(base.with_suffix(".response.json"), diagnostic_json(response))
                event.update(status="received", response_sha256=sha(base.with_suffix(".response.json")))
                return response
            except Exception as exc:
                event.update(status="transport_error", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                event.update(seconds=time.perf_counter() - start, finished=stamp())
                write_json(destination, report)

        for workload in [r for r in seal["rows"] if r["split"] == args.split]:
            row = {"id": workload["id"], "family": workload["family"], "variants": {}, "pass": False}
            report["rows"].append(row)
            try:
                meta = read_json(out / workload["meta"])
                kernel = meta["kernels"][0]
                grouped = {}
                for action in (BASELINE, 0, 256):
                    digest = workload["actions"][str(action)]["ptx_sha256"]
                    grouped.setdefault(digest, []).append(action)
                aliases, cubins = {}, {}
                for digest, actions in grouped.items():
                    canonical = str(actions[0])
                    for action in actions:
                        aliases[str(action)] = canonical
                    data = workload["actions"][canonical]
                    assembly = request("/compile", {"ptx": (out / data["ptx"]).read_text(encoding="ascii")})
                    variant = {"aliases": actions, "assembly": diagnostic_json(assembly), "assembly_event": len(report["events"]) - 1,
                               "correctness": [], "timings": [], "pass": False}
                    row["variants"][canonical] = variant
                    if assembly.get("ok") and kernel["entry"] in assembly.get("entry_points", []):
                        cubins[canonical] = assembly["cubin_id"]
                row["aliases"] = aliases

                def execute(label, snapshot, warmup, iterations):
                    payload = {k: kernel[k] for k in ("entry", "grid", "block", "smem_bytes", "zero_outputs")}
                    payload.update(cubin_id=cubins[label], args=snapshot["args"], warmup=warmup,
                                   iters=iterations, max_return=2**22)
                    number = len(report["events"])
                    response = request("/run", payload)
                    checked = validate_response(response, snapshot, kernel, meta)
                    checked["event"] = number
                    return checked

                for relative in workload["inputs"]:
                    snapshot = read_json(out / relative)
                    for label in cubins:
                        result = execute(label, snapshot, 1, 1)
                        result.update(seed=snapshot["seed"], pattern=snapshot["pattern"])
                        row["variants"][label]["correctness"].append(result)
                        write_json(destination, report)
                qualified = len(cubins) == len(grouped) and all(
                    len(v["correctness"]) == 3 and all(c["pass"] for c in v["correctness"])
                    for v in row["variants"].values())
                if qualified:
                    snapshot = read_json(out / workload["inputs"][0])
                    for round_index, order in enumerate(balanced_orders(cubins)):
                        for position, label in enumerate(order):
                            result = execute(label, snapshot, 3, 20)
                            result.update(round=round_index, position=position)
                            row["variants"][label]["timings"].append(result)
                            write_json(destination, report)
                for variant in row["variants"].values():
                    variant["pass"] = qualified and len(variant["timings"]) == 6 and all(t["pass"] for t in variant["timings"])
                    if variant["pass"]:
                        variant["median_us"] = float(np.median([t["median_us"] for t in variant["timings"]]))
                row["pass"] = bool(row["variants"]) and all(v["pass"] for v in row["variants"].values())
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            write_json(destination, report)
            print(row["id"], "PASS" if row["pass"] else "FAIL", flush=True)
        verify_seal(out)
        report.update(finished=stamp(), http_attempt_count=sum(len(e["http_attempts"]) for e in report["events"]),
                      **{"pass": len(report["rows"]) == 10 and all(r["pass"] for r in report["rows"])})
        write_json(destination, report)
        return 0 if report["pass"] else 1
    finally:
        lock.unlink()


def timings(row, action):
    variant = row["variants"][row["aliases"][str(action)]]
    if not variant["pass"]:
        raise ValueError("Failed candidate cannot supply a reward")
    entries = sorted(variant["timings"], key=lambda v: v["round"])
    if [v["round"] for v in entries] != list(range(6)) or not all(v["pass"] for v in entries):
        raise ValueError("Incomplete/duplicate timing rounds")
    values = np.asarray([v["median_us"] for v in entries], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Invalid timing label")
    return values


def reward(row, action):
    return float(np.median(np.log(timings(row, BASELINE) / timings(row, action))))


def feature_matrix(rows, transform=None):
    if transform is None:
        keys = sorted({key for row in rows for key in row["features"]})
    else:
        keys = transform["keys"]
    raw = np.asarray([[row["features"].get(key, 0) for key in keys] for row in rows], dtype=float)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("Invalid structural features")
    raw = np.sign(raw) * np.log1p(np.abs(raw))
    if transform is None:
        scale = raw.std(axis=0)
        scale[scale == 0] = 1
        transform = {"keys": keys, "mean": raw.mean(axis=0).tolist(), "scale": scale.tolist()}
    return (raw - np.asarray(transform["mean"])) / np.asarray(transform["scale"]), transform


def best_action(scores):
    # Deterministic conservative tie: current baseline wins equal predictions.
    return max((BASELINE, 256, 0), key=lambda action: scores.get(action, -math.inf))


def fit_tree(x, observations, seed=SEED):
    rng = np.random.default_rng(seed)
    feature_order = rng.permutation(x.shape[1]).tolist()

    def node(indices, depth):
        selected = [o for o in observations if o["index"] in indices]
        means, counts = {BASELINE: 0.0}, {}
        loss = 0.0
        for action in ACTIONS:
            values = [o["reward"] for o in selected if o["action"] == action]
            counts[action] = len(values)
            if values:
                average = float(np.mean(values))
                loss += sum((value - average) ** 2 for value in values)
                if len(values) >= 2 or action == BASELINE:
                    means[action] = average
        leaf = {"action": best_action(means), "means": {str(k): v for k, v in means.items()},
                "support": {str(k): v for k, v in counts.items()}, "workloads": len(indices)}
        if depth >= 2 or len(indices) < 4:
            return leaf, loss
        best = None
        for feature in feature_order:
            values = sorted(set(float(x[i, feature]) for i in indices))
            for a, b in zip(values, values[1:]):
                threshold = (a + b) / 2
                left = [i for i in indices if x[i, feature] <= threshold]
                right = [i for i in indices if x[i, feature] > threshold]
                if min(len(left), len(right)) < 2:
                    continue
                left_node, left_loss = node(left, depth + 1)
                right_node, right_loss = node(right, depth + 1)
                candidate_loss = left_loss + right_loss
                if candidate_loss < loss - 1e-12 and (best is None or candidate_loss < best[0] - 1e-12):
                    best = (candidate_loss, {"feature": feature, "threshold": threshold,
                                            "left": left_node, "right": right_node})
        return (best[1], best[0]) if best else (leaf, loss)

    return node(list(range(len(x))), 0)[0]


def tree_predict(tree, x):
    while "feature" in tree:
        tree = tree["left"] if x[tree["feature"]] <= tree["threshold"] else tree["right"]
    return int(tree["action"])


def train_learners(workloads, measured, seed=SEED):
    x, transform = feature_matrix(workloads)
    measured = {row["id"]: row for row in measured}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(workloads)).tolist()
    observations = {"tree": [], "bandit": []}
    alternatives = np.asarray([0, 256] * ((len(workloads) + 1) // 2))[:len(workloads)]
    rng.shuffle(alternatives)
    dimension = x.shape[1] + 1
    matrices = {action: np.eye(dimension) for action in (0, 256)}
    targets = {action: np.zeros(dimension) for action in (0, 256)}

    def observe(learner, index, action):
        workload = workloads[index]
        row = measured[workload["id"]]
        value = reward(row, action)
        observations[learner].append({"index": index, "id": workload["id"], "action": action,
                                     "reward": value, "physical_label": row["aliases"][str(action)]})
        return value

    for step, index in enumerate(order):
        observe("tree", index, BASELINE)
        observe("tree", index, int(alternatives[step]))
        observe("bandit", index, BASELINE)
        context = np.append(x[index], 1.0)
        scores = {}
        for action in (0, 256):
            inverse_context = np.linalg.solve(matrices[action], context)
            theta = np.linalg.solve(matrices[action], targets[action])
            scores[action] = float(theta @ context + 0.5 * np.sqrt(context @ inverse_context))
        maximum = max(scores.values())
        tied = [a for a, value in scores.items() if abs(value - maximum) <= 1e-12]
        action = int(rng.choice(tied))
        value = observe("bandit", index, action)
        matrices[action] += np.outer(context, context)
        targets[action] += value * context
    models = {
        "transform": transform,
        "tree": fit_tree(x, observations["tree"], seed),
        "bandit": {str(action): np.linalg.solve(matrices[action], targets[action]).tolist() for action in (0, 256)},
        "feedback": {name: {"logical_observations": len(rows),
                             "distinct_physical_labels": len({(r["id"], r["physical_label"]) for r in rows}),
                             "observations": rows} for name, rows in observations.items()},
        "training_order": [workloads[i]["id"] for i in order],
        "feedback_scope": "Offline aggregate-label replay; full grid measured once, only recorded observations exposed per learner",
    }
    if len(observations["tree"]) != len(observations["bandit"]) or len(observations["tree"]) != 2 * len(workloads):
        raise AssertionError("Unequal feedback budget")
    return models


def choose(model, workload, models):
    if model == "fixed1024":
        return BASELINE
    if model == "heuristic":
        cost = workload["features"].get("hinted_loop_work_max")
        return BASELINE if cost is None or cost <= 256 else (256 if cost <= 1024 else 0)
    context, _ = feature_matrix([workload], models["transform"])
    if model == "tree":
        return tree_predict(models["tree"], context[0])
    if model == "bandit":
        vector = np.append(context[0], 1.0)
        return best_action({BASELINE: 0.0, **{int(a): float(np.asarray(theta) @ vector) for a, theta in models["bandit"].items()}})
    raise ValueError(f"Unknown policy {model}")


def summarize(workloads, measured, decisions, seed=SEED, draws=4000):
    by_id = {r["id"]: r for r in measured}
    entries, families = [], {}
    for workload in workloads:
        row = by_id[workload["id"]]
        action = decisions[workload["id"]]
        ratios = np.log(timings(row, BASELINE) / timings(row, action))
        entry = {"id": workload["id"], "family": workload["family"], "action": action,
                 "paired_log_speedups": ratios.tolist(), "speedup": float(np.exp(np.median(ratios))),
                 "distinct_choices": workload["unique_ptx"]}
        families.setdefault(workload["family"], []).append(len(entries))
        entries.append(entry)
    if not entries:
        raise ValueError("Empty evaluation split")
    rng = np.random.default_rng(seed)
    family_keys = sorted(families)
    samples = []
    for _ in range(draws):
        values = []
        for name in rng.choice(family_keys, len(family_keys), replace=True):
            for index in families[name]:
                logs = entries[index]["paired_log_speedups"]
                values.append(float(np.median(rng.choice(logs, len(logs), replace=True))))
        samples.append(float(np.exp(np.mean(values))))
    different = [entry for entry in entries if entry["distinct_choices"] > 1]
    return {"workloads": len(entries), "families": len(families), "rows": entries,
            "geomean_speedup": float(np.exp(np.mean([np.log(e["speedup"]) for e in entries]))),
            "family_bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
            "worst_workload_speedup": min(e["speedup"] for e in entries),
            "distinct_workloads": len(different), "distinct_families": len({e["family"] for e in different})}


def promotion_gate(summary, all_correct=True):
    reasons = []
    if not all_correct:
        reasons.append("candidate_correctness_or_evidence_failure")
    if summary["distinct_workloads"] < 6 or summary["distinct_families"] < 2:
        reasons.append("too_few_nonidentical_choices")
    if summary["geomean_speedup"] < 1.03:
        reasons.append("gain_below_three_percent")
    if summary["family_bootstrap_95_ci"][0] <= 1.0:
        reasons.append("paired_uncertainty_includes_no_gain")
    if summary["worst_workload_speedup"] < 1 / 1.05:
        reasons.append("workload_regression_exceeds_five_percent")
    return {"pass": not reasons, "reasons": reasons}


def verify_measurement_evidence(out, split, workloads, result):
    """Rebuild correctness and timing records from hashed transport evidence."""
    events = result["events"]
    used = set()
    if [e["number"] for e in events] != list(range(len(events))):
        raise ValueError("Duplicate/missing transport event")
    payloads = {}
    for event in events:
        number = event["number"]
        base = out / "events" / split / f"{number:04d}"
        if event["status"] != "received":
            raise ValueError("Unresolved transport event")
        for suffix, key in ((".request.json", "request_sha256"), (".response.json", "response_sha256")):
            if sha(base.with_suffix(suffix)) != event[key]:
                raise ValueError("Transport artifact hash mismatch")
        request = read_json(base.with_suffix(".request.json"))
        response = read_json(base.with_suffix(".response.json"))
        if request["endpoint"] != event["endpoint"] or not event["http_attempts"]:
            raise ValueError("Missing/mismatched HTTP attempt evidence")
        for index, recorded in enumerate(event["http_attempts"]):
            path = base.with_name(base.name + f".http-{index:02d}.json")
            if str(path.relative_to(out)) != recorded["path"] or sha(path) != recorded["sha256"]:
                raise ValueError("HTTP attempt artifact hash mismatch")
            attempt = read_json(path)
            if any(attempt.get(key) != recorded.get(key) for key in ("http_status", "error", "seconds")):
                raise ValueError("HTTP attempt summary differs from evidence")
        if attempt.get("body") != response or not 200 <= attempt.get("http_status", 0) < 300:
            raise ValueError("Returned response differs from final HTTP attempt")
        payloads[number] = (event, request["payload"], response)
    if result["http_attempt_count"] != sum(len(e["http_attempts"]) for e in events):
        raise ValueError("HTTP attempt count mismatch")

    def consume(number, workload, endpoint, expected_payload):
        if number in used or number not in payloads:
            raise ValueError("Reused or missing response event")
        used.add(number)
        event, payload, response = payloads[number]
        if event["workload"] != workload or event["endpoint"] != endpoint or payload != expected_payload:
            raise ValueError("Response event belongs to different workload/action/input/launch")
        return response

    by_id = {w["id"]: w for w in workloads}
    for row in result["rows"]:
        workload = by_id[row["id"]]
        meta = read_json(out / workload["meta"])
        kernel = meta["kernels"][0]
        groups = {}
        for action in (BASELINE, 0, 256):
            digest = workload["actions"][str(action)]["ptx_sha256"]
            groups.setdefault(digest, []).append(action)
        aliases = {str(action): str(actions[0]) for actions in groups.values() for action in actions}
        if row["aliases"] != aliases or set(row["variants"]) != set(aliases.values()):
            raise ValueError("Measured PTX aliases differ from sealed actions")
        if row["family"] != workload["family"] or not row["pass"]:
            raise ValueError("Failed/mismatched measured workload")
        orders = balanced_orders(str(actions[0]) for actions in groups.values())
        snapshots = [read_json(out / path) for path in workload["inputs"]]
        if [(s["seed"], s["pattern"]) for s in snapshots] != list(INPUT_CASES):
            raise ValueError("Incomplete frozen input cases")
        for digest, actions in groups.items():
            label = str(actions[0])
            variant = row["variants"][label]
            ptx = (out / workload["actions"][label]["ptx"]).read_text(encoding="ascii")
            if hashlib.sha256(ptx.encode("ascii")).hexdigest() != digest:
                raise ValueError("PTX hash disagrees with sealed action")
            assembly = consume(variant["assembly_event"], row["id"], "/compile", {"ptx": ptx})
            if (variant["aliases"] != actions or variant["assembly"] != assembly or not assembly.get("ok")
                    or kernel["entry"] not in assembly.get("entry_points", [])):
                raise ValueError("Assembly/entry evidence mismatch")
            if len(variant["correctness"]) != 3 or len(variant["timings"]) != 6 or not variant["pass"]:
                raise ValueError("Incomplete correctness/timing evidence")
            for index, check in enumerate(variant["correctness"] + variant["timings"]):
                correctness = index < 3
                snapshot = snapshots[index] if correctness else snapshots[0]
                payload = {key: kernel[key] for key in ("entry", "grid", "block", "smem_bytes", "zero_outputs")}
                payload.update(cubin_id=assembly["cubin_id"], args=snapshot["args"],
                               warmup=1 if correctness else 3, iters=1 if correctness else 20,
                               max_return=2**22)
                response = consume(check["event"], row["id"], "/run", payload)
                expected = validate_response(response, snapshot, kernel, meta)
                expected["event"] = check["event"]
                if correctness:
                    expected.update(seed=snapshot["seed"], pattern=snapshot["pattern"])
                else:
                    round_index = index - 3
                    expected.update(round=round_index, position=orders[round_index].index(label))
                if check != expected or not expected["pass"]:
                    raise ValueError("Recorded correctness/timing differs from returned output evidence")
            if variant["median_us"] != float(np.median(timings(row, int(label)))):
                raise ValueError("Aggregate timing differs from returned rounds")
        labels = [str(actions[0]) for actions in groups.values()]
        chronology = [row["variants"][label]["assembly_event"] for label in labels]
        chronology += [row["variants"][label]["correctness"][index]["event"]
                       for index in range(3) for label in labels]
        chronology += [row["variants"][label]["timings"][index]["event"]
                       for index, order in enumerate(orders) for label in order]
        if chronology != sorted(chronology):
            raise ValueError("Response events do not follow balanced chronological measurement order")
    if used != set(payloads):
        raise ValueError("Unreferenced response events")


def checked_results(out, split, workloads):
    result = read_json(out / f"results-{split}.json")
    if result["seal_sha256"] != sha(out / "seal.json") or result["split"] != split:
        raise ValueError("Measurement belongs to another study/split")
    if not result["pass"] or {r["id"] for r in result["rows"]} != {w["id"] for w in workloads}:
        raise ValueError("Missing or failed candidates: no policy promotion permitted")
    if len(result["rows"]) != len(workloads):
        raise ValueError("Duplicate measured workload")
    for row in result["rows"]:
        for action in ACTIONS:
            timings(row, action)
    verify_measurement_evidence(out, split, workloads, result)
    return result


def fit(args):
    out = args.out.resolve()
    seal = verify_seal(out)
    if (out / "results-final.json").exists():
        raise ValueError("Final results already opened; training/selection is permanently closed")
    if (out / "selection.json").exists():
        raise ValueError("Selection already frozen")
    train = [r for r in seal["rows"] if r["split"] == "train"]
    validation = [r for r in seal["rows"] if r["split"] == "validation"]
    train_results = checked_results(out, "train", train)
    validation_results = checked_results(out, "validation", validation)
    models = train_learners(train, train_results["rows"])
    models.update(seal_sha256=sha(out / "seal.json"), training_sha256=sha(out / "results-train.json"))
    summaries = {}
    for name in ("fixed1024", "heuristic", "tree", "bandit"):
        decisions = {r["id"]: choose(name, r, models) for r in validation}
        summary = summarize(validation, validation_results["rows"], decisions)
        summary["gate"] = promotion_gate(summary)
        summaries[name] = summary
    eligible = [name for name in ("heuristic", "tree", "bandit") if summaries[name]["gate"]["pass"]]
    selected = max(eligible, key=lambda name: summaries[name]["geomean_speedup"]) if eligible else "fixed1024"
    write_json(out / "models.json", models)
    write_json(out / "selection.json", {"selected": selected, "selected_utc": stamp(),
               "seal_sha256": sha(out / "seal.json"), "models_sha256": sha(out / "models.json"),
               "validation_sha256": sha(out / "results-validation.json"),
               "training_sha256": sha(out / "results-train.json"), "validation": summaries,
               "reason": "validation gate and highest qualifying geometric mean" if eligible else "inconclusive_or_no_qualified_gain_keep_fixed1024",
               "final_results_opened": False})
    print("Frozen validation choice:", selected)
    return 0


def evaluate(args):
    out = args.out.resolve()
    seal = verify_seal(out)
    selection = read_json(out / "selection.json")
    models = read_json(out / "models.json")
    for key, path in (("models_sha256", "models.json"), ("validation_sha256", "results-validation.json"),
                      ("training_sha256", "results-train.json"), ("seal_sha256", "seal.json")):
        if selection[key] != sha(out / path):
            raise ValueError("Selected policy/evidence changed before final evaluation")
    final = [r for r in seal["rows"] if r["split"] == "final"]
    result = checked_results(out, "final", final)
    if result["selection_sha256"] != sha(out / "selection.json"):
        raise ValueError("Final measurements were not collected after this policy freeze")
    summaries = {}
    for name in ("fixed1024", "heuristic", "tree", "bandit"):
        summaries[name] = summarize(final, result["rows"], {r["id"]: choose(name, r, models) for r in final})
    measured = {r["id"]: r for r in result["rows"]}
    oracle = {r["id"]: max((BASELINE, 256, 0), key=lambda a: reward(measured[r["id"]], a)) for r in final}
    summaries["oracle_ceiling_not_deployable"] = summarize(final, result["rows"], oracle)
    selected = selection["selected"]
    gate = promotion_gate(summaries[selected])
    if selected == "fixed1024":
        gate = {"pass": False, "reasons": ["validation_retained_fixed1024"]}
    report = {"evaluated": stamp(), "selected_before_final": selected, "final": summaries,
              "promotion_gate": gate, "production_modified": False,
              "conclusion": "eligible_for_separate_review_not_automatic_promotion" if gate["pass"] else "keep_fixed1024",
              "source": {p: sha(out / p) for p in ("seal.json", "selection.json", "models.json", "results-final.json")},
              "limitations": "Ten workloads/five final families; unlocked clocks, six returned-output-checked timing rounds; offline bandit replay only, no hidden-test or RL-superiority claim"}
    write_json(out / "evaluation.json", report)
    print(report["conclusion"])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "fit", "evaluate"))
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/validation/phase4/selector")
    parser.add_argument("--backend", type=Path, default=ROOT / "artifacts/validation/phase4/baseline/ptx_backend.py")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--skip-assembly", action="store_true", help="Record CPU syntax checking as omitted")
    parser.add_argument("--repair-note", default="")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--runner-turn", default="", help="Label of the explicitly authorized exclusive remote-runner turn")
    args = parser.parse_args()
    try:
        return {"prepare": prepare, "run": run_remote, "fit": fit, "evaluate": evaluate}[args.command](args)
    except Exception as exc:
        print(f"selector_study: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
