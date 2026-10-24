"""Export the reviewable optional-feature corpus without using a GPU.

Typed fixtures have independent CPU references. The original generated tensor
case is copied byte-for-byte, including its mathematical tolerance and oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_tensor_cores import export as tensor_export
from test_bulk_copy import export as bulk_export
from test_feature_interactions import export as combined_export
from tools.check_compiler import compiler_fingerprints, compile_ptx, assemble


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples")
    args = parser.parse_args()
    output = args.output.resolve()
    for name in ("tensor", "bulk", "combined", "generated_tensor", "feature-manifest.json"):
        if (output / name).exists():
            parser.error(f"refusing to replace existing {output / name}")
    before = compiler_fingerprints(ROOT / "compile.py")
    tensor_export(output / "tensor")
    bulk_export(output / "bulk")
    combined_export(output / "combined")
    original = ROOT / "artifacts/validation/2026-10-24-extra/packages/oct24_m02_f16_ragged"
    destination = output / "generated_tensor"
    destination.mkdir()
    hashes = {}
    for path in sorted(original.iterdir()):
        if path.is_file() and (path.name in ("meta.json", "expected.npz") or path.name.endswith(".ir.json")):
            shutil.copy2(path, destination / path.name)
            assert path.read_bytes() == (destination / path.name).read_bytes()
            hashes[path.name] = sha(path)
    generated_assembly = []
    for ir in destination.glob("*.ir.json"):
        ptx, _ = compile_ptx(ROOT / "compile.py", ir)
        generated_assembly.append(assemble(ptx, "sm_89"))
        ir.with_suffix(".ptx").write_bytes(ptx.encode("ascii"))
    (destination / "provenance.json").write_text(json.dumps({
        "original_package": original.relative_to(ROOT).as_posix(),
        "original_files_sha256": hashes,
        "original_bytes_preserved": True,
        "scope": "Original Emmy-generated FP16 ragged tensor IR; previously rejected by the scalar-only backend. Reference and tolerances unchanged.",
        "assembly": generated_assembly,
    }, indent=2) + "\n", encoding="utf-8")
    after = compiler_fingerprints(ROOT / "compile.py")
    assert before == after
    packages = sorted(p.parent for p in output.rglob("meta.json"))
    report = {
        "pass": True, "scope": "Fixture export and instruction assembly; no GPU execution",
        "compiler_sha256": before,
        "packages": [p.relative_to(ROOT).as_posix() for p in packages],
        "files_sha256": {p.relative_to(output).as_posix(): sha(p)
                         for folder in (output / n for n in ("tensor", "bulk", "combined", "generated_tensor"))
                         for p in folder.rglob("*") if p.is_file()},
    }
    (output / "feature-manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "packages": len(packages)}))


if __name__ == "__main__":
    main()
