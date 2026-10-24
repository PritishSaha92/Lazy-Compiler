"""Hand-built typed IR for two operations blocked by Emmy's CUDA renderer.

These are NOT tools/generate.py outputs. They retain the mathematical
expressions, shapes and tolerances of oct24_p03_f64_rounding and
oct24_p09_remainder while constructing Kernel IR with the existing fixture API.
The compiler never imports this module. All default tests are CPU/assembler only.
"""
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from emmy.compiler.dtype import F32, F64
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.leaves import Assign, Let, Load, Write
from ir_cases import axis, case, op, write_case
from tools.check_compiler import assemble, compile_ptx, compiler_fingerprints


def rounding_reference(x):
    return [np.floor(x) + np.ceil(x) - np.trunc(x)]


def remainder_reference(x):
    return [np.remainder(x, np.float32(2.5))]


OPEN = [
    dict(id="typed_oct24_f64_rounding", original_id="oct24_p03_f64_rounding",
         shape=(13, 79), dtype="f64", rtol=0, atol=0, ref=rounding_reference),
    dict(id="typed_oct24_f32_remainder", original_id="oct24_p09_remainder",
         shape=(1049,), dtype="f32", rtol=1e-6, atol=1e-6, ref=remainder_reference),
]
SEEDS = ((0, "normal"), (17, "normal"), (101, "wide"))


def input_values(spec, seed=0, pattern="normal"):
    values = np.random.default_rng(seed).standard_normal(spec["shape"])
    return (values * (8 if pattern == "wide" else 1)).astype(
        np.float64 if spec["dtype"] == "f64" else np.float32
    )


def fixtures():
    for spec in OPEN:
        kind = F64 if spec["dtype"] == "f64" else F32
        shape = spec["shape"]
        axes = tuple(axis(f"i{d}", length) for d, length in enumerate(shape))
        index = tuple(Var(a.name) for a in axes)
        statements = [Load("value", "x", index, dtype=kind)]
        if kind == F64:
            statements.extend((
                Assign("down", op("floor"), ("value",), F64),
                Assign("up", op("ceil"), ("value",), F64),
                Assign("toward_zero", op("trunc"), ("value",), F64),
                Assign("combined", op("add"), ("down", "up"), F64),
                Assign("result", op("subtract"), ("combined", "toward_zero"), F64),
            ))
        else:
            statements.extend((
                Let("divisor", Literal(2.5, "float"), F32),
                Assign("result", op("remainder"), ("value", "divisor"), F32),
            ))
        statements.append(Write("out", index, "result", value_dtype=kind))
        graph, meta, _ = case(
            spec["id"], (Tile(axes, tuple(statements)),), {"x": shape}, shape,
            lambda x: x, block=128, grid=(int(np.prod(shape)) + 127) // 128,
        )
        for name, node in graph.nodes.items():
            graph.nodes[name] = replace(node, outputs=tuple(
                replace(tensor, dtype=kind) for tensor in node.outputs
            ))
        for arg in meta["kernels"][0]["args"]:
            arg["dtype"] = spec["dtype"]
        meta.update(rtol=spec["rtol"], atol=spec["atol"], fixture_provenance={
            "construction": "hand-built typed Kernel IR using tests/ir_cases.py",
            "generated_by_tools_generate": False,
            "corresponding_original_expression": spec["original_id"],
            "reason": "Original generator stopped in Emmy CUDA reference rendering; test backend independently",
        })
        yield graph, meta, {"out": spec["ref"](input_values(spec))[0]}


def cpu_references():
    import torch
    torch.set_num_threads(1)
    rows = []
    for spec in OPEN:
        for seed, pattern in SEEDS:
            array = input_values(spec, seed, pattern)
            tensor = torch.from_numpy(array.copy())
            actual = (torch.floor(tensor) + torch.ceil(tensor) - torch.trunc(tensor)
                      if spec["dtype"] == "f64" else torch.remainder(tensor, 2.5))
            actual = actual.numpy()
            expected = spec["ref"](array)[0]
            assert actual.shape == expected.shape == array.shape
            assert actual.dtype == expected.dtype == array.dtype
            matches = np.isclose(actual, expected, rtol=spec["rtol"], atol=spec["atol"])
            assert matches.all(), (spec["id"], seed, pattern)
            assert not np.isclose(np.zeros_like(expected), expected,
                                  rtol=spec["rtol"], atol=spec["atol"]).all()
            rows.append({"id":spec["id"], "seed":seed, "pattern":pattern,
                         "dtype":spec["dtype"], "shape":list(array.shape),
                         "pass":True, "all_zero_output_rejected":True,
                         "max_abs_error":float(np.max(np.abs(
                             actual.astype(np.float64)-expected.astype(np.float64))))})
    return rows


def export(destination):
    destination.mkdir(parents=True, exist_ok=False)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source_paths = [Path(__file__), ROOT/"tests/ir_cases.py", ROOT/"tests/expressions_oct24.py",
                    ROOT/"tools/check_compiler.py", ROOT/"compile.py", ROOT/"ptx_backend.py"]
    tracked = {str(p.resolve()):sha(p) for p in source_paths}
    rows = cpu_references()
    values = list(fixtures())
    emitted = []
    for value in values:
        graph, meta, expected = value
        name = meta["id"]
        path = write_case(destination/"packages"/name, value)
        spec = next(s for s in OPEN if s["id"] == name)
        arg_types = {a["name"]:a["dtype"] for a in meta["kernels"][0]["args"]}
        assert arg_types == {"x":spec["dtype"], "out":spec["dtype"]}
        expected_dtype = np.dtype("float64" if spec["dtype"] == "f64" else "float32")
        assert expected["out"].dtype == expected_dtype
        ptx, seconds = compile_ptx(ROOT/"compile.py", path)
        assembly = assemble(ptx, "sm_89")
        ptx_path = path.with_suffix(".ptx")
        ptx_path.write_bytes(ptx.encode("ascii"))
        emitted.append({"id":name, "pass":True, "compile_seconds":seconds,
                        "input_dtype":spec["dtype"],"output_dtype":spec["dtype"],
                        "ptx_sha256":sha(ptx_path), "assembly":assembly})
    assert all(sha(Path(p))==h for p,h in tracked.items())
    files = {p.relative_to(destination).as_posix():sha(p)
             for p in (destination/"packages").rglob("*") if p.is_file()}
    manifest = {"pass":True,"frozen_utc":datetime.now(timezone.utc).isoformat(),
                "provenance":"Hand-built typed IR, not generated by tools/generate.py",
                "packages":emitted,"planned_checks":6,"seeds_patterns":list(SEEDS),
                "cpu_references":rows,"cpu_reference_checks":len(rows),"local_gpu_used":False,
                "sources_sha256":tracked,"files_sha256":files,
                "compiler_sha256":compiler_fingerprints(ROOT/"compile.py"),"tracked_unchanged":True}
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"packages":len(emitted),"cpu_checks":len(rows),"assembly_pass":True,
                      "manifest_sha256":sha(destination/"manifest.json")}),flush=True)


class Oct24ElementwiseTests(unittest.TestCase):
    def test_independent_references(self):
        self.assertEqual(len(cpu_references()), 6)

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas required for instruction legality")
    def test_cli_assembles_typed_buffers_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in fixtures():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory)/value[1]["id"],value)
                    ptx, _ = compile_ptx(ROOT/"compile.py",path)
                    assemble(ptx,"sm_89")


if __name__ == "__main__":
    if "--export" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--export", type=Path, required=True)
        export(parser.parse_args().export)
    else:
        unittest.main()
