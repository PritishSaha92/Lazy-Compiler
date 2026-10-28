"""Reconstruct phase7 expected values while compiler/process calls are blocked.

This is an offline reference-construction check, not a GPU correctness run or
a proof of the compiler over arbitrary inputs. The output report is write-once.
"""

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]

import test_builtin_aux as builtins
import test_unstamped_integer as integers
import test_predicate_promotion as predicate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check():
    output = FOLDER / "oracle-audit.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace retained oracle evidence: {output}")
    source_names = (
        "compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt",
        "tests/test_builtin_aux.py", "tests/test_unstamped_integer.py",
        "tests/test_predicate_promotion.py", "tests/ir_cases.py", "tests/test_contract_edges.py",
        "artifacts/validation/phase7/check_oracles.py",
    )
    sources = {name: sha(ROOT / name) for name in source_names}
    manifests = [FOLDER / name for name in ("new-feature-manifest.json", "changed-scalar-manifest.json")]
    files, packages = {}, []
    for path in manifests:
        manifest = read(path)
        packages.extend(ROOT / name for name in manifest["packages"])
        files.update(manifest["files_sha256"])
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    assert len(packages) == len(set(packages)) == 12
    assert all(sha(ROOT / name) == expected for name, expected in files.items())
    blocked_calls = []

    def forbidden(*args, **kwargs):
        blocked_calls.append("compiler or subprocess execution attempted")
        raise AssertionError(blocked_calls[-1])

    result = {
        "schema": "emmy-phase7-independent-oracle-audit-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pass": False,
        "scope": "Reconstruct 11 new counterexample references and one retained changed-scalar reference; compare exact stored arrays and serialized IR/metadata",
        "method": "During fixture construction, Compiler.compile, compile_file (including imported module aliases), subprocess.run and subprocess.Popen raise on any call. Expected arrays come from fixed integer values, NumPy arithmetic, explicit launch enumeration, or input identity/reversal.",
        "source_sha256": sources,
        "input_sha256": files,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "blocked_calls": blocked_calls,
        "gpu_execution": False,
        "cases": [],
        "limitations": [
            "These are developer-authored targeted fixtures, not organizer hidden tests or arbitrary-input proofs.",
            "Auxiliary execution covers explicit 64-compute/32-producer, three-CTA named-barrier IR; it does not establish general warp-specialized scheduling support.",
            "Oracle construction intentionally preserves the existing explicitly stamped floating/logical-mask operation policy.",
            "This check does not validate remote GPU outputs; separate source-bound wire audits do that.",
        ],
    }
    try:
        with ExitStack() as stack:
            for target in ("ptx_backend.Compiler.compile", "ptx_backend.compile_file",
                           "subprocess.run", "subprocess.Popen"):
                stack.enter_context(patch(target, forbidden))
            for module in (builtins, integers, predicate):
                stack.enter_context(patch.object(module, "compile_file", forbidden))
            cases = list(builtins.cases()) + list(integers.integer_inference_cases()) + list(predicate.predicate_promotion_cases())
            assert len(cases) == 12
            assert {meta["id"] for _, meta, _ in cases} == {path.name for path in packages}
            for graph, metadata, expected in cases:
                package = next(p for p in packages if p.name == metadata["id"])
                assert read(package / "meta.json") == metadata, package.name
                ir = package / (metadata["kernels"][0]["entry"] + ".ir.json")
                serialized = json.loads(json.dumps(graph.to_dict(), default=repr))
                assert read(ir) == serialized, package.name
                comparisons = []
                with np.load(package / "expected.npz", allow_pickle=False) as stored:
                    assert set(stored.files) == set(expected)
                    for name, array in expected.items():
                        assert stored[name].dtype == array.dtype and stored[name].shape == array.shape
                        assert np.array_equal(stored[name], array), (package.name, name)
                        comparisons.append({"name": name, "dtype": str(array.dtype), "shape": list(array.shape),
                                            "elements": int(array.size), "exact_equal": True,
                                            "array_bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest()})
                result["cases"].append({"id": metadata["id"], "package": package.relative_to(ROOT).as_posix(),
                                        "metadata_identical": True, "serialized_ir_identical": True,
                                        "outputs": comparisons})
        assert not blocked_calls
        assert all(sha(ROOT / name) == expected for name, expected in sources.items())
        assert all(sha(ROOT / name) == expected for name, expected in files.items())
        result.update(pass_=True, checks=len(result["cases"]), passed_checks=len(result["cases"]),
                      sources_unchanged=True, inputs_unchanged=True)
        result["pass"] = result.pop("pass_")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with output.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps({"pass": result["pass"], "checks": result.get("checks", 0),
                      "error": result.get("error"), "report": output.relative_to(ROOT).as_posix()}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(check())
