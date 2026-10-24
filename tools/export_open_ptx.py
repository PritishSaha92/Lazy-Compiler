"""Export all public PTX modules and structural sidecars with source hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_ir import analyze_file
from tools.check_compiler import compile_ptx


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--example", type=Path, help="Also write the scalar-add example as ASCII bytes")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {name: digest(ROOT / name) for name in ("compile.py", "ptx_backend.py", "requirements.txt")}
    manifest = {"sources_sha256": sources, "kernels": [], "scope": "Emission and analysis only; numerical execution is recorded by separate runner gates."}
    for ir in sorted((ROOT / "problems").glob("*/*.ir.json")):
        name = ir.name.removesuffix(".ir.json")
        # Entry names can repeat across independent problem packages.
        folder = args.output / ir.parent.name
        folder.mkdir()
        path = folder / (name + ".ptx")
        # Capture the real CLI bytes, including this host's stdout newlines.
        path.write_bytes(compile_ptx(ROOT / "compile.py", ir)[0].encode("ascii"))
        analysis = folder / (name + ".analysis.json")
        analysis.write_text(json.dumps(analyze_file(ir), indent=2) + "\n", encoding="utf-8")
        manifest["kernels"].append({"problem": ir.parent.name, "ir": str(ir.relative_to(ROOT)),
                                    "ir_sha256": digest(ir), "metadata_sha256": digest(ir.with_name("meta.json")),
                                    "ptx": path.relative_to(args.output).as_posix(), "ptx_sha256": digest(path), "ptx_bytes": path.stat().st_size,
                                    "analysis": analysis.relative_to(args.output).as_posix(), "analysis_sha256": digest(analysis)})
        if args.example and ir.parent.name == "p00_scalar_add":
            args.example.write_bytes(path.read_bytes())
    if len(manifest["kernels"]) != 15:
        raise ValueError("Expected all15 public kernels")
    if sources != {name: digest(ROOT / name) for name in sources}:
        raise ValueError("Compiler changed during export")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported15 modules with analysis: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
