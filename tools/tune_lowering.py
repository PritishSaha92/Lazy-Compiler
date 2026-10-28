"""Compare a small, explicit lowering grid on the hosted runner.

Every candidate is checked against independent references before its timings
are considered. Variants are interleaved and no choice is learned from filenames.
This development tool never runs during submission compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptx_backend import compile_file
from tools.check_compiler import (
    SERVER,
    compare_outputs,
    expected_outputs,
    load_specs,
    make_args,
    remote_post,
)


CONFIGURATIONS = {
    "strict_loops": {"unroll_budget": 0, "compensated_dots": False},
    "strict_unroll": {"unroll_budget": 1024, "compensated_dots": False},
    "fma_loops": {"unroll_budget": 0, "fuse_accumulations": True},
    "fma_unroll": {"unroll_budget": 1024, "fuse_accumulations": True},
    "accurate_loops": {"unroll_budget": 0, "accurate_dots": True},
    "accurate_unroll": {"unroll_budget": 1024, "accurate_dots": True},
    "compensated_loops": {"unroll_budget": 0, "compensated_dots": True},
    "compensated_unroll": {"unroll_budget": 1024, "compensated_dots": True},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", nargs="+")
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--configurations", default=",".join(CONFIGURATIONS))
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts/validation/lowering-search.json",
    )
    args = parser.parse_args()
    configurations = {
        name: CONFIGURATIONS[name] for name in args.configurations.split(",")
    }
    specs = load_specs()
    report = {
        "configurations": configurations,
        "rounds": args.rounds,
        "compiler_sha256": hashlib.sha256(
            (ROOT / "ptx_backend.py").read_bytes()
        ).hexdigest(),
        "results": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name in args.problems:
        folder = ROOT / "problems" / name
        meta = json.loads((folder / "meta.json").read_text())
        if len(meta["kernels"]) != 1:
            raise ValueError("This timing comparison requires a single-kernel problem")
        kernel = meta["kernels"][0]
        ir = folder / (kernel["entry"] + ".ir.json")
        candidates = {}
        for label, config in configurations.items():
            start = time.perf_counter()
            ptx = compile_file(ir, **config)
            compile_seconds = time.perf_counter() - start
            assembly = remote_post(args.server, "/compile", {"ptx": ptx})
            row = {
                "problem": name,
                "configuration": label,
                "compile_seconds": compile_seconds,
                "ptx_bytes": len(ptx),
                "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                "registers": assembly.get("registers"),
                "assembly": assembly.get("ok", False),
                "correctness": [],
                "timings_us": [],
            }
            report["results"].append(row)
            if not assembly.get("ok"):
                row["error"] = assembly
                continue
            candidates[label] = (assembly["cubin_id"], row)
        # Two fixed seeds and one cancellation-heavy distribution. These are
        # development inputs, not a claim of held-out evaluation.
        for seed, pattern in ((0, "normal"), (17, "normal"), (101, "wide")):
            expected = expected_outputs(folder, meta, seed, pattern, specs)
            for label, (cubin, row) in candidates.items():
                request = {
                    key: kernel[key]
                    for key in ("entry", "grid", "block", "smem_bytes", "zero_outputs")
                }
                request.update(
                    cubin_id=cubin,
                    args=make_args(kernel, {}, seed, pattern),
                    warmup=1,
                    iters=1,
                    max_return=2**22,
                )
                result = remote_post(args.server, "/run", request)
                if result.get("ok"):
                    actual = {
                        n: np.asarray(o["values"], dtype=np.float32).reshape(o["shape"])
                        for n, o in result["outputs"].items()
                    }
                    checks = compare_outputs(
                        actual, expected, meta["rtol"], meta["atol"]
                    )
                    row["correctness"].append(
                        {
                            "seed": seed,
                            "pattern": pattern,
                            "outputs": checks,
                            "pass": all(c["pass"] for c in checks),
                        }
                    )
                else:
                    row["correctness"].append(
                        {
                            "seed": seed,
                            "pattern": pattern,
                            "pass": False,
                            "error": result,
                        }
                    )
                save()
        for round_index in range(args.rounds):
            labels = list(candidates)
            if round_index % 2:
                labels.reverse()
            for label in labels:
                cubin, row = candidates[label]
                if not all(case["pass"] for case in row["correctness"]):
                    continue
                request = {
                    key: kernel[key]
                    for key in ("entry", "grid", "block", "smem_bytes", "zero_outputs")
                }
                request.update(
                    cubin_id=cubin,
                    args=make_args(kernel, {}, 0, "normal"),
                    warmup=3,
                    iters=args.iters,
                    max_return=2**22,
                )
                result = remote_post(args.server, "/run", request)
                if result.get("ok"):
                    row["timings_us"].append(result["median_us"])
                else:
                    row["error"] = result
                save()
        for label, (_, row) in candidates.items():
            passed = all(case["pass"] for case in row["correctness"])
            timing = np.median(row["timings_us"]) if row["timings_us"] else None
            print(
                f"{name} {label}: correct={passed} median_us={timing} PTX={row['ptx_bytes']}",
                flush=True,
            )
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
