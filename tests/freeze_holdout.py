"""Seal or verify a generated workload split before compiler policy evaluation."""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_traced_inputs(entry, meta):
    """Detect frontend setup intermediates that accidentally become inputs."""
    program = ast.parse(entry["code"])
    calls = []

    class RandomCalls(ast.NodeVisitor):
        def visit_Call(self, node):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "randn"
            ):
                calls.append([ast.literal_eval(arg) for arg in node.args])
            self.generic_visit(node)

    RandomCalls().visit(program)
    names = [
        statement.targets[0].id
        for statement in program.body[:-1]
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "randn"
    ]
    assert len(meta["graph_inputs"]) == len(calls), f"Input-count drift: {entry['id']}"
    if names:
        assert names == meta["graph_inputs"], (
            f"Named-input drift: {entry['id']}: {meta['graph_inputs']} instead of {names}"
        )
    tensors = {arg["name"]: arg for kernel in meta["kernels"] for arg in kernel["args"]}
    assert [tensors[name]["shape"] for name in meta["graph_inputs"]] == calls, (
        f"Input-shape drift: {entry['id']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/validation/holdout-2026-10-23"
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    folder = args.out.resolve()
    manifest_path = folder / "manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        assert (folder / "MANIFEST.sha256").read_text().split()[0] == digest(
            manifest_path
        )
        for filename, fingerprint in manifest["files_sha256"].items():
            assert digest(folder / filename) == fingerprint, (
                f"Changed frozen file: {filename}"
            )
        for filename, fingerprint in manifest["test_sources_sha256"].items():
            assert digest(ROOT / filename) == fingerprint, (
                f"Changed test source: {filename}"
            )
        print(
            f"Verified frozen manifest: {manifest['workload_count']} workloads, {manifest['kernel_count']} kernels"
        )
        return 0
    if manifest_path.exists():
        raise SystemExit(
            "Split already frozen; use --verify. Do not rewrite evaluation history."
        )
    protocol_path = folder / "protocol-source-freeze.json"
    protocol = json.loads(protocol_path.read_text())
    expression_path = ROOT / "tests/expressions_holdout.py"
    expected_source = protocol["source_sha256"]["tests/expressions_holdout.py"]
    repairs_path = folder / "generation-repairs.json"
    if repairs_path.exists():
        for repair in json.loads(repairs_path.read_text())["repairs"]:
            assert repair["before_expression_sha256"] == expected_source
            assert repair["generation_only"] and not repair["compiler_evaluated"]
            expected_source = repair["after_expression_sha256"]
    assert digest(expression_path) == expected_source, (
        "Unrecorded expression source change"
    )
    assert json.loads((folder / "generation.json").read_text())["returncode"] == 0
    sanity = json.loads((folder / "reference-sanity.json").read_text())
    assert sanity["pass"] is True
    assert sanity["expression_sha256"] == expected_source
    spec = importlib.util.spec_from_file_location(
        "holdout_freeze_specs", expression_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.OPEN) == protocol["planned_workloads"]
    assert len({entry["id"] for entry in module.OPEN}) == len(module.OPEN)
    records = []
    files = {}
    for entry in module.OPEN:
        problem = folder / entry["id"]
        meta = json.loads((problem / "meta.json").read_text())
        assert meta["id"] == entry["id"]
        assert (meta["rtol"], meta["atol"]) == (entry["rtol"], entry["atol"])
        check_traced_inputs(entry, meta)
        for kernel in meta["kernels"]:
            assert (problem / (kernel["entry"] + ".ir.json")).is_file()
        for path in sorted(problem.iterdir()):
            if path.is_file():
                files[path.relative_to(folder).as_posix()] = digest(path)
        records.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "family": entry["family"],
                "kernels": len(meta["kernels"]),
                "entries": [kernel["entry"] for kernel in meta["kernels"]],
                "rtol": entry["rtol"],
                "atol": entry["atol"],
                "graph_inputs": meta["graph_inputs"],
                "graph_outputs": meta["graph_outputs"],
            }
        )
    for name in (
        "protocol-source-freeze.json",
        "generation.json",
        "generation.log",
        "reference-sanity.json",
        "README.md",
    ):
        files[name] = digest(folder / name)
    for path in sorted(folder.glob("*attempt*.json")):
        files[path.name] = digest(path)
    for path in sorted(folder.glob("*attempt*.log")):
        files[path.name] = digest(path)
    if repairs_path.exists():
        files[repairs_path.name] = digest(repairs_path)
    sources = [
        "tests/expressions_holdout.py",
        "tests/generate_stress.py",
        "tests/check_holdout_references.py",
        "tests/freeze_holdout.py",
        "tools/generate.py",
    ]
    manifest = {
        "manifest_version": 1,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "split_name": "held-out-2026-10-23",
        "compiler_evaluated_before_freeze": False,
        "policy_tuning_on_this_split": False,
        "baseline_compiler_sources_sha256": {
            name: protocol["source_sha256"][name]
            for name in ("compile.py", "ptx_backend.py")
        },
        "generation_target": protocol["target"],
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("emmy-ml", "numpy", "torch")
        },
        "workload_count": len(records),
        "kernel_count": sum(row["kernels"] for row in records),
        "category_counts": dict(Counter(row["category"] for row in records)),
        "correctness_cases": protocol["correctness_cases"],
        "workloads": records,
        "files_sha256": files,
        "test_sources_sha256": {name: digest(ROOT / name) for name in sources},
        "limitations": [
            "Synthetic public-semantic workloads are not the organizer hidden set.",
            "References were checked on CPU only; no PTX correctness or performance is implied by generation success.",
            "Any policy update using these evaluation results must disclose this split as development data.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "MANIFEST.sha256").write_text(digest(manifest_path) + "  manifest.json\n")
    print(
        f"FROZEN: {len(records)} workloads, {manifest['kernel_count']} kernels; manifest SHA-256 {digest(manifest_path)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
