"""Read-only numerical audit of the three recorded half-GELU runner failures.

No compiler import, GPU call or reference/tolerance edit. math.erf and NumPy's
IEEE half conversion form the independent typed-stage oracle.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "emmy-hackathon-2026/Final Emmy"
RUN = REPO / "artifacts/validation/phase9/compositions-baseline"
CASE = REPO / "examples/phase9/generated/phase9_gelu_erf_half"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def half(value):
    return np.asarray(value, dtype=np.float16)


def as_float(value):
    return np.asarray(value, dtype=np.float64)


def erf(value):
    return np.vectorize(math.erf, otypes=[np.float64])(as_float(value))


def mismatch_count(actual, expected, rtol, atol):
    return int(np.count_nonzero(~np.isclose(as_float(actual), as_float(expected), rtol=rtol, atol=atol)))


def bits(value):
    return half(value).view(np.uint16)


def main():
    summary = json.loads((RUN / "summary.json").read_text())
    metadata = json.loads((CASE / "meta.json").read_text())
    report = {"schema": "emmy-half-gelu-stage-audit-v1", "local_gpu_used": False,
              "runner_replayed": False, "reference_or_tolerance_changed": False,
              "rtol": metadata["rtol"], "atol": metadata["atol"], "cases": [],
              "source_files": {str(CASE / name): digest(CASE / name) for name in
                  ("meta.json", "k_gelu_pointwise.ir.txt", "k_gelu_pointwise.ref.cu")}}
    for job_index, job in enumerate(summary["jobs"]):
        if Path(job["package"]).name != "phase9_gelu_erf_half":
            continue
        event = next(e for e in summary["events"] if e["job"] == job_index and e["endpoint"] == "/run")
        stem = RUN / "events" / f'{event["index"]:04d}'
        request_path = stem.with_suffix(".request.json")
        response_path = stem.with_suffix(".response.json")
        request = json.loads(request_path.read_text())["payload"]
        response = json.loads(response_path.read_text())
        arguments = {arg["name"]: arg for arg in request["args"]}
        def input_value(name):
            arg = arguments[name]
            shape = tuple(arg["shape"])
            # The runner uses default_rng(seed).standard_normal for a seeded
            # input; wide inputs were serialized as explicit values by client.
            values = arg["data"] if "data" in arg else np.random.default_rng(arg["seed"]).standard_normal(shape)
            return half(values).reshape(shape)
        x = input_value("x")
        inv = input_value("gelu_inv_sqrt2")
        one = input_value("gelu_one")
        h = input_value("gelu_half")
        actual = half(response["outputs"]["gelu"]["values"]).reshape(x.shape)
        # Each typed Assign in the generated IR returns f16.
        v0 = half(as_float(inv) * as_float(x))
        v1 = half(erf(v0))
        v2 = half(as_float(one) + as_float(v1))
        v3 = half(as_float(h) * as_float(x))
        v4 = half(as_float(v2) * as_float(v3))
        mathematical = half(0.5 * as_float(x) * (1 + erf(as_float(x) / math.sqrt(2))))
        # Diagnostic only: model the inline .ref.cu expression's FP32 erff,
        # mixed addition and final multiplication, retaining its half products.
        inline = half((np.float32(one) + np.float32(erf(v0))) * np.float32(v3))
        tolerated = np.isclose(as_float(actual), as_float(mathematical),
                               rtol=metadata["rtol"], atol=metadata["atol"])
        failures = np.argwhere(~tolerated)
        examples = []
        for index in failures[:8]:
            key = tuple(index)
            examples.append({"index": index.tolist(), "x": float(x[key]),
                             "v0": float(v0[key]), "v1": float(v1[key]),
                             "v2": float(v2[key]), "v3": float(v3[key]),
                             "stage_oracle": float(v4[key]), "actual": float(actual[key]),
                             "mathematical": float(mathematical[key]), "inline_model": float(inline[key])})
        record = {"seed": job["seed"], "pattern": job["pattern"], "elements": int(x.size),
                  "request_sha256": digest(request_path), "response_sha256": digest(response_path),
                  "actual_vs_stage_bit_mismatches": int(np.count_nonzero(bits(actual) != bits(v4))),
                  "actual_vs_stage_tolerance_mismatches": mismatch_count(actual, v4, metadata["rtol"], metadata["atol"]),
                  "actual_vs_mathematical_tolerance_mismatches": int(len(failures)),
                  "stage_vs_mathematical_tolerance_mismatches": mismatch_count(v4, mathematical, metadata["rtol"], metadata["atol"]),
                  "inline_vs_mathematical_tolerance_mismatches": mismatch_count(inline, mathematical, metadata["rtol"], metadata["atol"]),
                  "actual_vs_stage_max_abs_error": float(np.max(np.abs(as_float(actual)-as_float(v4)))),
                  "first_failures": examples}
        if job["seed"] == 0 and job["pattern"] == "normal":
            with np.load(CASE / "expected.npz", allow_pickle=False) as archive:
                record["math_oracle_vs_frozen_expected_bit_mismatches"] = int(np.count_nonzero(bits(mathematical) != bits(archive["gelu"])))
        report["cases"].append(record)
    out = Path(__file__).with_name("half-gelu-stage-audit.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
