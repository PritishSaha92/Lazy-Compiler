"""Explicit FP64-state variants of three generated FP64-storage test kernels.

These variants are constructed by this test module, NOT emitted unchanged by
tools/generate.py. Original packages and their failed tight-FP64 comparisons
remain archived. We preserve input/output buffers, indices, launch geometry,
control flow and independent oracles while explicitly selecting FP64 local and
cooperative state types. This is test-data construction, never compiler policy.
"""
from collections import Counter
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/validation/2026-10-24-extra"
sys.path.insert(0, str(ROOT))
from emmy.compiler.dtype import DataType, F32, F64
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import KernelOp, Smem
from emmy.compiler.ir.stmt.leaves import Let, Select
from tools import check_compiler as checker


def norm_reference(x):
    centered = x - x.mean(-1, keepdims=True)
    return [centered / np.sqrt((centered * centered).sum(-1, keepdims=True) + 0.01)]


OPEN = [
    dict(id="typed_oct24_f64_state_sum", original_id="oct24_r03_f64_column_sum",
         rtol=2e-13, atol=2e-13, ref=lambda x: [x.sum(0, dtype=np.float64)]),
    dict(id="typed_oct24_f64_state_matmul", original_id="oct24_m01_f64_ragged",
         rtol=1e-12, atol=1e-12, ref=lambda a,b: [a @ b]),
    dict(id="typed_oct24_f64_state_centered_norm", original_id="oct24_f01_f64_centered_norm_inline",
         rtol=2e-12, atol=2e-12, ref=norm_reference),
]
SEEDS = ((0,"normal"), (17,"normal"), (101,"wide"))


def visit(value):
    yield value
    if isinstance(value, DataType):
        return
    if is_dataclass(value):
        for field in fields(value):
            yield from visit(getattr(value, field.name))
    elif isinstance(value, (tuple,list)):
        for child in value:
            yield from visit(child)


def promote(value, changes):
    if isinstance(value, DataType):
        if value == F32:
            changes["datatype_f32_to_f64"] += 1
            return F64
        return value
    if isinstance(value, Select):
        # The optional Select statement has a deliberately FP32 result in the
        # pinned IR renderer. Keep its predicates but select into explicit F64.
        assert len(value.branches) == 2
        first, fallback = value.branches
        changes["select_to_f64_ternary_let"] += 1
        return Let(value.name, TernaryExpr(first.select, Var(first.value), Var(fallback.value)), F64)
    if isinstance(value, Smem) and value.dtype == "float":
        changes["smem_float_to_double"] += 1
        return replace(value, dtype="double", align=max(8, value.align))
    if is_dataclass(value):
        kwargs = {}
        for field in fields(value):
            if not field.init:
                continue
            previous = getattr(value, field.name)
            if field.name in ("dtype", "value_dtype") and isinstance(previous, str) and previous in ("float", "f32"):
                changes["float_type_spelling_to_double"] += 1
                kwargs[field.name] = "double"
            else:
                kwargs[field.name] = promote(previous, changes)
        # Some immutable IR helpers (notably Dim) have custom constructors.
        # Identity-preserve unchanged shape/control objects instead of trying
        # to recreate them from their dataclass storage representation.
        if all(new is getattr(value, name) for name, new in kwargs.items()):
            return value
        return replace(value, **kwargs)
    if isinstance(value, tuple):
        updated = tuple(promote(item, changes) for item in value)
        return value if all(a is b for a,b in zip(value,updated)) else type(value)(updated)
    if isinstance(value, list):
        return [promote(item, changes) for item in value]
    return value


def layout_signature(body):
    """Record every scheduling, addressing and synchronization field unchanged."""
    kinds = {"Tile", "Loop", "StridedLoop", "Cond", "Sync", "WarpShuffle", "TreeHalve", "Load", "Write"}
    records = []
    for item in visit(body):
        if type(item).__name__ not in kinds:
            continue
        records.append((type(item).__name__, tuple(
            (f.name, repr(getattr(item, f.name))) for f in fields(item)
            if f.name not in ("body", "else_body", "combine_states", "dtype", "value_dtype")
        )))
    return records


def build(spec):
    parent = SOURCE/"packages"/spec["original_id"]
    meta = json.loads((parent/"meta.json").read_text())
    assert meta["rtol"] == spec["rtol"] and meta["atol"] == spec["atol"]
    assert len(meta["kernels"]) == 1
    kernel = meta["kernels"][0]
    assert all(arg["dtype"] == "f64" for arg in kernel["args"])
    path = parent/(kernel["entry"]+".ir.json")
    graph = Graph.from_dict(json.loads(path.read_text()))
    changes = Counter()
    shared_bytes = 0
    for name, node in list(graph.nodes.items()):
        if not isinstance(node.op, KernelOp):
            continue
        original_body = node.op.body
        promoted_body = promote(original_body, changes)
        assert layout_signature(original_body) == layout_signature(promoted_body)
        assert not any(isinstance(x, DataType) and x == F32 for x in visit(promoted_body))
        assert not any(isinstance(x, Select) for x in visit(promoted_body))
        for item in visit(promoted_body):
            if is_dataclass(item):
                for field in fields(item):
                    if field.name in ("dtype", "value_dtype"):
                        assert getattr(item, field.name) not in ("float", "f32")
            if isinstance(item, Smem):
                assert item.dtype == "double" and item.align >= 8
                shared_bytes = (shared_bytes + item.align - 1)//item.align*item.align
                shared_bytes += int(np.prod(item.extents))*8
        graph.nodes[name] = replace(node, op=replace(node.op, body=promoted_body))
    assert changes["datatype_f32_to_f64"] > 0
    assert shared_bytes <= 48*1024 and kernel["smem_bytes"] == 0
    # Deserialization is part of fixture validation, matching compile.py's input.
    encoded = json.dumps(graph.to_dict(), indent=2, default=repr)
    decoded = Graph.from_dict(json.loads(encoded))
    for node in decoded.nodes.values():
        if isinstance(node.op, KernelOp):
            assert not any(isinstance(x, DataType) and x == F32 for x in visit(node.op.body))
    edited = copy.deepcopy(meta)
    edited["id"] = spec["id"]
    edited["fixture_provenance"] = {
        "construction":"Explicit test-only FP64 type promotion of parsed Kernel IR",
        "generated_unchanged_by_tools_generate":False,"parent":str(parent.relative_to(ROOT)),
        "changes":dict(changes),"static_shared_bytes":shared_bytes,
        "preserved":"Launch geometry, addresses, control flow, sync, input/output ABI, constants and original tight reference tolerances",
        "original_failure_retained":True,
    }
    assert edited["kernels"] == meta["kernels"]
    return parent, encoded, edited, dict(changes)


def cpu_references(spec, meta):
    import torch
    torch.set_num_threads(1)
    by_name = {arg["name"]:arg for k in meta["kernels"] for arg in k["args"]}
    rows = []
    for seed, pattern in SEEDS:
        arrays = [checker.input_array(by_name[name],seed,pattern) for name in meta["graph_inputs"]]
        assert all(a.dtype == np.float64 for a in arrays)
        tensors = [torch.from_numpy(a.copy()) for a in arrays]
        if spec["id"].endswith("sum"):
            actual = tensors[0].sum(0)
        elif spec["id"].endswith("matmul"):
            actual = tensors[0] @ tensors[1]
        else:
            z = tensors[0] - tensors[0].mean(-1,keepdim=True)
            actual = z/torch.sqrt((z*z).sum(-1,keepdim=True)+0.01)
        actual = actual.numpy()
        expected = spec["ref"](*arrays)[0]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype == np.float64
        assert np.isclose(actual,expected,rtol=spec["rtol"],atol=spec["atol"]).all()
        if seed == 0:
            with np.load(SOURCE/"packages"/spec["original_id"]/"expected.npz",allow_pickle=False) as old:
                original = old[meta["graph_outputs"][0]]
                assert original.dtype == np.float64
                np.testing.assert_array_equal(original,expected)
        rows.append({"id":spec["id"],"seed":seed,"pattern":pattern,"pass":True,
                     "dtype":"f64","shape":list(expected.shape),
                     "max_abs_error":float(np.max(np.abs(actual-expected)))})
    return rows


def export(destination):
    destination.mkdir(parents=True,exist_ok=True)
    assert not (destination/"manifest.json").exists()
    assert not (destination/"packages").exists(), "Retain existing prepared packages"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sources = [Path(__file__),ROOT/"compile.py",ROOT/"ptx_backend.py",ROOT/"tools/check_compiler.py",
               ROOT/"tests/expressions_oct24.py",SOURCE/"expressions_inline.py"]
    for spec in OPEN:
        sources.extend(p for p in (SOURCE/"packages"/spec["original_id"]).iterdir() if p.is_file())
    tracked = {str(p.resolve()):sha(p) for p in sources}
    records, cpu = [], []
    for spec in OPEN:
        parent, encoded, meta, changes = build(spec)
        cpu.extend(cpu_references(spec,meta))
        folder = destination/"packages"/spec["id"]
        folder.mkdir(parents=True)
        path = folder/(meta["kernels"][0]["entry"]+".ir.json")
        path.write_text(encoded,encoding="utf-8")
        (folder/"meta.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
        shutil.copyfile(parent/"expected.npz",folder/"expected.npz")
        assert sha(parent/"expected.npz") == sha(folder/"expected.npz")
        ptx,seconds = checker.compile_ptx(ROOT/"compile.py",path)
        assert ".f32" not in ptx
        assembly = checker.assemble(ptx,"sm_89")
        ptx_path = path.with_suffix(".ptx");ptx_path.write_bytes(ptx.encode("ascii"))
        records.append({"id":spec["id"],"parent_id":spec["original_id"],"pass":True,
                        "changes":changes,"output_dtype":"f64","no_f32_instructions":True,
                        "layout_preserved":True,"expected_npz_byte_identical_to_parent":True,
                        "rtol":spec["rtol"],"atol":spec["atol"],"compile_seconds":seconds,
                        "ptx_sha256":sha(ptx_path),"assembly":assembly})
        print("PREPARED",spec["id"],flush=True)
    assert all(sha(Path(p))==h for p,h in tracked.items())
    files = {p.relative_to(destination).as_posix():sha(p) for p in (destination/"packages").rglob("*") if p.is_file()}
    manifest = {"pass":True,"frozen_utc":datetime.now(timezone.utc).isoformat(),
                "provenance":"Explicit test-only type-promoted IR variants, not unchanged generator outputs",
                "packages":records,"planned_checks":9,"seeds_patterns":list(SEEDS),
                "cpu_references":cpu,"cpu_reference_checks":len(cpu),"local_gpu_used":False,
                "sources_sha256":tracked,"files_sha256":files,
                "compiler_sha256":checker.compiler_fingerprints(ROOT/"compile.py"),"tracked_unchanged":True}
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"packages":len(records),"cpu_checks":len(cpu),"assembly_pass":True,
                      "manifest_sha256":sha(destination/"manifest.json")}),flush=True)


class Fp64StateVariantTests(unittest.TestCase):
    @unittest.skipUnless(all((SOURCE/"packages"/s["original_id"]/"meta.json").exists() for s in OPEN),
                         "Archived generated parent fixtures required")
    def test_fp64_promotion_preserves_layout_and_original_oracles(self):
        for spec in OPEN:
            with self.subTest(case=spec["id"]):
                _,_,meta,_ = build(spec)
                self.assertEqual(len(cpu_references(spec,meta)),3)


if __name__ == "__main__":
    if "--export" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--export",type=Path,required=True)
        export(parser.parse_args().export)
    else:
        unittest.main()
