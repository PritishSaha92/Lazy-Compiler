"""Validate independent references on CPU without invoking the PTX compiler.

This checks test-data construction, not backend correctness or performance.
Replacing only randn feeds identical deterministic arrays into ordinary CPU
PyTorch and the independently implemented NumPy reference.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class CpuTorch:
    def __init__(self, seed_offset, scale):
        self.seed_offset = seed_offset
        self.scale = scale
        self.inputs = []

    def randn(self, *shape):
        values = np.asarray(
            np.random.default_rng(self.seed_offset + len(self.inputs)).standard_normal(
                shape
            )
            * self.scale,
            dtype=np.float32,
        )
        self.inputs.append(values)
        return torch.from_numpy(values.copy())

    def __getattr__(self, name):
        return getattr(torch, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expressions", type=Path, default=ROOT / "tests/expressions_holdout.py"
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "holdout_reference_specs", args.expressions
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    rows = []
    for workload in module.OPEN:
        for seed, pattern in ((37, "normal"), (89, "normal"), (137, "wide")):
            proxy = CpuTorch(seed, 8 if pattern == "wide" else 1)
            namespace = {"torch": proxy}
            program = ast.parse(workload["code"])
            assert isinstance(program.body[-1], ast.Expr)
            prefix = ast.Module(body=program.body[:-1], type_ignores=[])
            exec(compile(prefix, "<reference-input>", "exec"), namespace)
            value = eval(
                compile(
                    ast.Expression(program.body[-1].value), "<reference-output>", "eval"
                ),
                namespace,
            )
            actual = value.detach().numpy()
            expected = np.asarray(workload["ref"](*proxy.inputs)[0])
            equal_shape = actual.shape == expected.shape
            matches = np.isclose(
                actual,
                expected,
                rtol=workload["rtol"],
                atol=workload["atol"],
                equal_nan=False,
            )
            rows.append(
                {
                    "id": workload["id"],
                    "seed_offset": seed,
                    "pattern": pattern,
                    "shape": list(actual.shape),
                    "pass": bool(equal_shape and matches.all()),
                    "mismatches": int((~matches).sum()),
                    "max_abs_error": float(
                        np.max(np.abs(actual.astype(np.float64) - expected))
                    ),
                }
            )
            print(workload["id"], seed, pattern, "PASS" if rows[-1]["pass"] else "FAIL")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "CPU reference sanity only; no backend invocation, PTX assembly, or GPU execution.",
                "expression_sha256": hashlib.sha256(
                    args.expressions.read_bytes()
                ).hexdigest(),
                "results": rows,
                "pass": all(row["pass"] for row in rows),
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if all(row["pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
