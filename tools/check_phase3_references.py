"""Freeze fresh schedule probes after CPU reference and traced-input checks.

This utility invokes no PTX backend, assembler, GPU, or network. Missing
frontend packages are recorded explicitly and excluded from executable counts.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from freeze_holdout import check_traced_inputs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/validation/phase3"
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    folder = args.out.resolve()
    destination = folder / "schedule-manifest.json"
    if args.verify:
        manifest = json.loads(destination.read_text())
        for name, fingerprint in manifest["files_sha256"].items():
            assert digest(folder / name) == fingerprint, f"Changed package: {name}"
        for name, fingerprint in manifest["sources_sha256"].items():
            assert digest(ROOT / name) == fingerprint, f"Changed source: {name}"
        print(f"Verified {len(manifest['workloads'])} frozen schedule packages")
        return 0
    if destination.exists():
        raise SystemExit("Already frozen; use --verify and retain evaluation history.")
    source = ROOT / "tests/expressions_phase3.py"
    sanity = json.loads((folder / "reference-sanity.json").read_text())
    assert sanity["pass"] and sanity["expression_sha256"] == digest(source)
    spec = importlib.util.spec_from_file_location("phase3_expressions", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records, missing, files = [], [], {}
    for workload in module.OPEN:
        package = folder / "schedules" / workload["id"]
        if not (package / "meta.json").exists():
            missing.append(workload["id"])
            continue
        metadata = json.loads((package / "meta.json").read_text())
        check_traced_inputs(workload, metadata)
        assert (metadata["rtol"], metadata["atol"]) == (
            workload["rtol"],
            workload["atol"],
        )
        descriptors = {
            arg["name"]: arg for kernel in metadata["kernels"] for arg in kernel["args"]
        }
        inputs = [
            np.random.default_rng(index)
            .standard_normal(descriptors[name]["shape"])
            .astype(np.float32)
            for index, name in enumerate(metadata["graph_inputs"])
        ]
        expected = workload["ref"](*inputs)
        with np.load(package / "expected.npz") as stored:
            for name, value in zip(metadata["graph_outputs"], expected, strict=True):
                np.testing.assert_array_equal(stored[name], value)
        for kernel in metadata["kernels"]:
            assert (package / (kernel["entry"] + ".ir.json")).is_file()
        for path in sorted(package.iterdir()):
            if path.is_file():
                files[path.relative_to(folder).as_posix()] = digest(path)
        records.append(
            {
                "id": workload["id"],
                "category": workload["category"],
                "family": workload["family"],
                "kernels": len(metadata["kernels"]),
                "rtol": workload["rtol"],
                "atol": workload["atol"],
            }
        )
    assert records, "No generated packages to freeze"
    for path in (
        folder / "reference-sanity.json",
        *sorted(folder.glob("generation*.log")),
    ):
        files[path.relative_to(folder).as_posix()] = digest(path)
    sources = [
        "tests/expressions_phase3.py",
        "tests/check_holdout_references.py",
        "tests/generate_stress.py",
        "tools/generate.py",
        "tools/check_phase3_references.py",
    ]
    manifest = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU checked and packaged before PTX evaluation; not organizer hidden tests.",
        "compiler_evaluated_before_freeze": False,
        "planned_workloads": len(module.OPEN),
        "workloads": records,
        "missing_frontend_packages": missing,
        "cpu_reference_checks": len(sanity["results"]),
        "sources_sha256": {name: digest(ROOT / name) for name in sources},
        "files_sha256": files,
    }
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Frozen {len(records)} valid packages; {len(missing)} missing frontend packages"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
