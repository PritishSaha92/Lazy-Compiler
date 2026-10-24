"""Compare two frozen compiler CLIs with serialized, interleaved GPU measurements.

Every correctness and timed response checks independent reference outputs. Source,
IR, metadata and emitted PTX hashes tie the results to concrete artifacts. This
development tool is never imported or invoked by the submitted compiler.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_compiler import (
    DTYPES,
    SERVER,
    compare_outputs,
    compile_ptx,
    compiler_fingerprints,
    expected_outputs,
    load_specs,
    make_args,
    remote_post,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", nargs="+")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--references", type=Path, action="append", default=[])
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 20 or not 1 <= args.iters <= 200:
        parser.error("Require 1 <= rounds <= 20 and 1 <= iters <= 200")
    compilers = {
        "baseline": args.baseline.resolve(),
        "candidate": args.candidate.resolve(),
    }
    specs = load_specs()
    for reference in args.references:
        specs.update(load_specs(reference))
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "server": args.server,
        "rounds": args.rounds,
        "iters": args.iters,
        "warmup": 3,
        "source_sha256_before": {
            label: compiler_fingerprints(path) for label, path in compilers.items()
        },
        "reference_sha256": {
            str(p.resolve()): digest(p)
            for p in [ROOT / "tools/expressions_open.py", *args.references]
        },
        "harness_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [Path(__file__).resolve(), ROOT / "tools/check_compiler.py"]
        },
        "results": [],
        "pass": False,
        "limits": "Unlocked runner clocks; timings are indicative. Outputs are checked once per /run response, not after each repeated launch within that response. No hidden-test claim.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    artifacts = args.report.with_name(args.report.stem + "-artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    for name in args.problems:
        folder = Path(name)
        if not (folder / "meta.json").exists():
            folder = ROOT / "problems" / name
        folder = folder.resolve()
        result = {"problem": folder.name, "variants": {}, "pass": False}
        report["results"].append(result)
        try:
            meta = json.loads((folder / "meta.json").read_text())
            if len(meta["kernels"]) != 1:
                raise ValueError("Paired timing requires a single-kernel workload")
            kernel = meta["kernels"][0]
            ir = folder / (kernel["entry"] + ".ir.json")
            result["inputs_sha256"] = {
                str(p): digest(p)
                for p in [ir, folder / "meta.json", folder / "expected.npz"]
            }
            cubins = {}
            output_types = {a["name"]: DTYPES[a["dtype"]] for a in kernel["args"]}
            for label, compiler in compilers.items():
                row = {"correctness": [], "timings": [], "pass": False}
                result["variants"][label] = row
                ptx, seconds = compile_ptx(compiler, ir)
                ptx_path = artifacts / f"{folder.name}-{label}.ptx"
                ptx_path.write_bytes(ptx.encode("ascii"))
                row.update(
                    compile_seconds=seconds,
                    ptx_bytes=len(ptx),
                    ptx_path=str(ptx_path.resolve()),
                    ptx_sha256=digest(ptx_path),
                )
                assembly = remote_post(args.server, "/compile", {"ptx": ptx})
                row["assembly"] = assembly
                if not assembly.get("ok"):
                    raise RuntimeError(f"{label} assembly failed: {assembly}")
                cubins[label] = assembly["cubin_id"]
                save()

            def execute(label, seed, pattern, warmup, iters, reference):
                request = {
                    k: kernel[k]
                    for k in ("entry", "grid", "block", "smem_bytes", "zero_outputs")
                }
                request.update(
                    cubin_id=cubins[label],
                    args=make_args(kernel, {}, seed, pattern),
                    warmup=warmup,
                    iters=iters,
                    max_return=2**22,
                )
                response = remote_post(args.server, "/run", request)
                entry = {"seed": seed, "pattern": pattern, "pass": False}
                if response.get("ok"):
                    timing = float(response["median_us"])
                    if not math.isfinite(timing) or timing <= 0:
                        raise ValueError(f"Invalid runner median_us: {timing}")
                    if any(o.get("truncated") for o in response["outputs"].values()):
                        raise ValueError("Runner truncated an output")
                    actual = {
                        n: np.asarray(o["values"], dtype=output_types[n]).reshape(
                            o["shape"]
                        )
                        for n, o in response["outputs"].items()
                    }
                    checks = compare_outputs(
                        actual, reference, meta["rtol"], meta["atol"]
                    )
                    entry.update(
                        outputs=checks,
                        median_us=timing,
                        **{"pass": bool(checks) and all(c["pass"] for c in checks)},
                    )
                else:
                    entry["error"] = response
                return entry

            for seed, pattern in ((0, "normal"), (17, "normal"), (101, "wide")):
                reference = expected_outputs(folder, meta, seed, pattern, specs)
                for label in compilers:
                    entry = execute(label, seed, pattern, 1, 1, reference)
                    result["variants"][label]["correctness"].append(entry)
                    save()
            qualified = all(
                len(row["correctness"]) == 3
                and all(c["pass"] for c in row["correctness"])
                for row in result["variants"].values()
            )
            if qualified:
                reference = expected_outputs(folder, meta, 0, "normal", specs)
                for index in range(args.rounds):
                    labels = list(compilers)
                    if index % 2:
                        labels.reverse()
                    for position, label in enumerate(labels):
                        entry = execute(label, 0, "normal", 3, args.iters, reference)
                        entry.update(round=index, position=position)
                        result["variants"][label]["timings"].append(entry)
                        save()
            for row in result["variants"].values():
                timings = row["timings"]
                row["pass"] = (
                    qualified
                    and len(timings) == args.rounds
                    and all(t["pass"] for t in timings)
                )
                if row["pass"]:
                    values = [t["median_us"] for t in timings]
                    row.update(
                        median_us=float(np.median(values)),
                        min_us=min(values),
                        max_us=max(values),
                    )
            result["pass"] = all(row["pass"] for row in result["variants"].values())
            result["identical_ptx"] = (
                len({row["ptx_sha256"] for row in result["variants"].values()}) == 1
            )
            if result["pass"]:
                baseline = result["variants"]["baseline"]["median_us"]
                candidate = result["variants"]["candidate"]["median_us"]
                result["speedup"] = baseline / candidate
                result["paired_round_speedups"] = [
                    before["median_us"] / after["median_us"]
                    for before, after in zip(
                        result["variants"]["baseline"]["timings"],
                        result["variants"]["candidate"]["timings"],
                        strict=True,
                    )
                ]
                result["median_paired_speedup"] = float(
                    np.median(result["paired_round_speedups"])
                )
                print(
                    f"PASS {folder.name}: {baseline:.3f} -> {candidate:.3f} us; {result['speedup']:.3f}x; identical_ptx={result['identical_ptx']}",
                    flush=True,
                )
            else:
                print(
                    f"FAIL {folder.name}: see retained correctness/timing entries",
                    flush=True,
                )
        except Exception as exc:
            result["pass"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"FAIL {folder.name}: {result['error']}", flush=True)
        save()
    report["source_sha256_after"] = {
        label: compiler_fingerprints(path) for label, path in compilers.items()
    }
    report["sources_unchanged"] = (
        report["source_sha256_before"] == report["source_sha256_after"]
    )
    evidence = {**report["reference_sha256"], **report["harness_sha256"]}
    for result in report["results"]:
        evidence.update(result.get("inputs_sha256", {}))
    report["evidence_sha256_after"] = {
        name: digest(Path(name) if Path(name).is_absolute() else ROOT / name)
        for name in evidence
    }
    report["evidence_unchanged"] = evidence == report["evidence_sha256_after"]
    report["pass"] = (
        bool(report["results"])
        and report["sources_unchanged"]
        and report["evidence_unchanged"]
        and all(r["pass"] for r in report["results"])
    )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
