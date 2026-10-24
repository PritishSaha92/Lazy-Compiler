"""Reviewable tensor-core examples with independent matrix references.

Default tests use CPU references and ptxas only, never a local GPU. Export the
same typed programs for the official runner with --export DIRECTORY. Shapes,
input bytes and tolerances are independent of compiler results.
"""
import argparse
from dataclasses import replace
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
from emmy.compiler.dtype import BF16, F16, F32, I32
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit, CpAsyncCopy, CpAsyncWait, FragmentRepack, LdmatrixLoad,
    MmaSyncPtx, RegFragment, RegStore, Smem, Sync, Tile,
)
from emmy.compiler.ir.pure.lam import Lambda
from emmy.compiler.ir.stmt.blocks import StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Write
from ir_cases import axis, case, expression as e, literal as lit, op, write_case
from ptx_backend import Compiler, CompileError
from tools.check_compiler import assemble, compile_ptx, input_array

SHAPE = (16, 8, 16)


def quantize(x, kind):
    x = np.asarray(x, np.float32)
    if kind == F16:
        return x.astype(np.float16).astype(np.float32)
    words = x.view(np.uint32)
    words = words + np.uint32(32767) + ((words >> 16) & np.uint32(1))
    return ((words >> 16) << 16).view(np.float32)


def matrix_case(name, *, kind=F16, m=16, n=8, k=16, shared=False,
                trans=False, swizzle="NONE", paired=False, family=False,
                epilogue=False, transpose_out=False, async_copy=False,
                atomic=False):
    lane, kk = Var("lane"), Var("kk")
    aname, bname = ("aslab", "bslab") if shared else ("a", "b")
    nfrags = 2 if paired else 1
    shape_b = (n, k) if trans else (k, n)
    shapes = {"a": (m, k), "b": shape_b}
    if epilogue:
        shapes["bias"] = (n,)
    body = []
    arow = e("*", Var("fi"), lit(16)) if family else lit(0)
    if shared:
        if k != 16 or m != 16 or n not in (8,16):
            raise ValueError("staged fixture dimensions must be a complete atom")
        stride_a = 64 if swizzle != "NONE" else 16
        stride_b = 64 if swizzle != "NONE" else shape_b[-1]
        spelling = "__half" if kind == F16 else "__nv_bfloat16"
        body += [Smem("aslab", (16,stride_a), spelling, 128),
                 Smem("bslab", (shape_b[0],stride_b), spelling, 128)]
        for src,dst,shape,stride in (("a","aslab",(m,k),stride_a),
                                     ("b","bslab",shape_b,stride_b)):
            if async_copy:
                chunk = int(np.prod(shape)) // 32
                offset = e("*", lane, lit(chunk))
                body.append(CpAsyncCopy(dst, (offset,), src, (offset,), nbytes=chunk*2))
            else:
                ix = Var("fill_"+src)
                row,col = e("/",ix,lit(shape[-1])),e("%",ix,lit(shape[-1]))
                body.append(StridedLoop(axis("fill_"+src,int(np.prod(shape))), lane,lit(32),(
                    Load("v_"+src,src,(ix,),dtype=F32),
                    Write(dst,(row,col),"v_"+src,value_dtype=kind,swizzle=swizzle),
                )))
        if async_copy:
            body.extend((CpAsyncCommit(),CpAsyncWait(0)))
        body.append(Sync())
    else:
        stride_a, stride_b = k, shape_b[-1]
    body.extend((RegFragment("af","a",SHAPE,kind,count=2 if family else 1),
                 RegFragment("bf","b",SHAPE,kind),
                 RegFragment("cf","c",SHAPE,F32,count=2 if family else 1)))
    if paired:
        body += [RegFragment("bf1","b",SHAPE,kind),RegFragment("cf1","c",SHAPE,F32)]
    af = "af[fi]" if family else "af"
    cf = "cf[fi]" if family else "cf"
    def aloader():
        return LdmatrixLoad(af,aname,(arow,kk),"a",ldm=stride_a,staged=shared,
                            swizzle=swizzle,gmem_guard=None if shared else (arow,lit(m)),
                            k_zero=None if shared else (kk,lit(k)))
    def bloader():
        return LdmatrixLoad("bf",bname,(lit(0),kk) if trans else (kk,lit(0)),"b",
                            ldm=stride_b,staged=shared,swizzle=swizzle,b_trans=trans,
                            gmem_guard=None if shared else (lit(0),lit(n)),
                            k_zero=None if shared else (kk,lit(k)),
                            pair_frag="bf1" if paired else None)
    mma = MmaSyncPtx(cf,af,"bf",SHAPE,ab_dtype=kind.name)
    if family:
        loop_body = (bloader(),StridedLoop(axis("fi",2),lit(0),lit(1),(aloader(),mma)))
    else:
        loop_body = (aloader(),bloader(),mma)
    if paired:
        loop_body += (MmaSyncPtx("cf1","af","bf1",SHAPE,ab_dtype=kind.name),)
    body.append(StridedLoop(axis("kk",k),lit(0),lit(16),loop_body))
    epi = None
    if epilogue:
        epi = Lambda(params=("acc","__N__"),body=(
            Load("bv","bias",(Var("__N__"),),dtype=F32),
            Assign("added",op("add"),("acc","bv"),F32),
            Assign("activated",op("relu"),("added",),F32),
        ),results=("activated",))
    def store(frag, row, col):
        index = (col,row) if transpose_out else (row,col)
        return RegStore("out",index,frag,SHAPE,epilogue=epi,
                        m_guard=(row,lit(m)),n_guard=(col,lit(n)),atomic=atomic,
                        row_dim=1 if transpose_out else 0,col_dim=0 if transpose_out else 1)
    if family:
        body.append(StridedLoop(axis("fi",2),lit(0),lit(1),(store(cf,arow,lit(0)),)))
    else:
        body.append(store("cf",lit(0),lit(0)))
    if paired:
        body.append(store("cf1",lit(0),lit(8)))
    def reference(a,b,**extra):
        aq,bq = quantize(a,kind),quantize(b,kind)
        if trans:
            bq = bq.T
        result = (aq.astype(np.float64) @ bq.astype(np.float64)).astype(np.float32)
        if epilogue:
            result = np.maximum(result + extra["bias"],np.float32(0))
        return [result.T if transpose_out else result]
    value = case(name,(Tile((axis("lane",32),),tuple(body)),),shapes,
                 (n,m) if transpose_out else (m,n),lambda **xs:reference(**xs)[0],block=32)
    graph,meta,_ = value
    meta["rtol"],meta["atol"] = 2e-4,2e-4
    if atomic:
        meta["kernels"][0]["zero_outputs"] = ["out"]
    if async_copy:
        # Real f16 buffers exercise the native 16/8-byte copy instructions.
        for key,node in graph.nodes.items():
            graph.nodes[key] = replace(node,outputs=tuple(
                replace(t,dtype=F16) if t.name in ("a","b") else t for t in node.outputs))
        for arg in meta["kernels"][0]["args"]:
            if arg["name"] in ("a","b"):
                arg["dtype"] = "f16"
        arrays = {a["name"]:input_array(a,0,"normal") for a in meta["kernels"][0]["args"] if not a.get("output")}
        value = graph,meta,{"out":reference(**arrays)[0]}
    return value,reference


def repack_case():
    name = "tensor_repack_f16"
    body = [RegFragment("a","a",SHAPE,F16),RegFragment("b0","b",SHAPE,F16),
            RegFragment("b1","b",SHAPE,F16),RegFragment("c0","c",SHAPE,F32),
            RegFragment("c1","c",SHAPE,F32),RegFragment("final","c",SHAPE,F32)]
    body += [LdmatrixLoad("a","x",(lit(0),lit(0)),"a",ldm=16,staged=False),
             LdmatrixLoad("b0","y",(lit(0),lit(0)),"b",ldm=16,staged=False),
             LdmatrixLoad("b1","y",(lit(0),lit(8)),"b",ldm=16,staged=False),
             MmaSyncPtx("c0","a","b0",SHAPE),MmaSyncPtx("c1","a","b1",SHAPE),
             FragmentRepack("a",("c0","c1")),
             LdmatrixLoad("b0","z",(lit(0),lit(0)),"b",ldm=8,staged=False),
             MmaSyncPtx("final","a","b0",SHAPE),
             RegStore("out",(lit(0),lit(0)),"final",SHAPE)]
    def reference(x,y,z):
        xy = quantize(x,F16).astype(np.float64) @ quantize(y,F16).astype(np.float64)
        return [(quantize(xy,F16).astype(np.float64) @ quantize(z,F16).astype(np.float64)).astype(np.float32)]
    value = case(name,(Tile((axis("lane",32),),tuple(body)),),
                 {"x":(16,16),"y":(16,16),"z":(16,8)},(16,8),
                 lambda **xs:reference(**xs)[0],block=32)
    value[1]["rtol"],value[1]["atol"] = 2e-3,2e-3
    return value,reference


def fixtures_with_refs():
    yield matrix_case("tensor_global_f16")
    yield matrix_case("tensor_global_bf16",kind=BF16)
    yield matrix_case("tensor_global_f16_ragged",m=11,n=7,k=19)
    yield matrix_case("tensor_global_bf16_trans_ragged",kind=BF16,m=9,n=7,k=17,trans=True)
    yield matrix_case("tensor_family_live_index",m=32,family=True)
    yield matrix_case("tensor_shared_f16",shared=True)
    yield matrix_case("tensor_shared_bf16_trans",kind=BF16,shared=True,trans=True)
    for swizzle in ("B32","B64","B128"):
        yield matrix_case("tensor_shared_"+swizzle.lower(),shared=True,swizzle=swizzle)
    yield matrix_case("tensor_shared_paired",shared=True,paired=True,n=16)
    yield matrix_case("tensor_shared_paired_trans",shared=True,paired=True,n=16,trans=True)
    yield matrix_case("tensor_cp_async_shared",shared=True,async_copy=True)
    yield matrix_case("tensor_fused_guarded_epilogue",m=11,n=7,k=19,epilogue=True)
    yield matrix_case("tensor_transposed_output",transpose_out=True)
    yield matrix_case("tensor_atomic_f32",atomic=True)
    yield repack_case()


# Compatible with tools/check_compiler.py --expressions this_file.py.
OPEN = [dict(id=value[1]["id"],ref=ref) for value,ref in fixtures_with_refs()]


def export(destination):
    destination.mkdir(parents=True,exist_ok=False)
    entries=[]
    for value,ref in fixtures_with_refs():
        graph,meta,expected=value
        path=write_case(destination/"packages"/meta["id"],value)
        ptx,seconds=compile_ptx(ROOT/"compile.py",path)
        assembly=assemble(ptx,"sm_89")
        path.with_suffix(".ptx").write_bytes(ptx.encode("ascii"))
        arrays={a["name"]:input_array(a,0,"normal") for a in meta["kernels"][0]["args"] if not a.get("output")}
        np.testing.assert_array_equal(ref(**arrays)[0],expected["out"])
        assert not np.allclose(np.zeros_like(expected["out"]),expected["out"],rtol=meta["rtol"],atol=meta["atol"])
        entries.append(dict(id=meta["id"],pass_=True,assembly=assembly,compile_seconds=seconds,
                            ptx_sha256=hashlib.sha256(ptx.encode("ascii")).hexdigest()))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={"pass":True,"provenance":"Hand-built typed IR, independent NumPy mathematical references",
              "local_gpu_used":False,"packages":entries,"sources_sha256":{
                  p.name:sha(p) for p in (ROOT/"compile.py",ROOT/"ptx_backend.py",ROOT/"ptx_tensor.py",Path(__file__))},
              "files_sha256":{p.relative_to(destination).as_posix():sha(p)
                               for p in (destination/"packages").rglob("*") if p.is_file()}}
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"pass":True,"packages":len(entries),"manifest_sha256":sha(destination/"manifest.json")}))


class TensorCoreTests(unittest.TestCase):
    def compiler_with_changed_body(self, transform, **fixture_options):
        graph,meta,_ = matrix_case("tensor_rejection",**fixture_options)[0]
        for name,node in graph.nodes.items():
            if hasattr(node.op,"body"):
                graph.nodes[name] = replace(node,op=replace(node.op,body=transform(node.op.body)))
        return Compiler(graph,meta["kernels"][0])

    def test_reference_distinguishes_wrong_shapes_and_zero_results(self):
        for value,ref in fixtures_with_refs():
            _,meta,expected=value
            with self.subTest(case=meta["id"]):
                arrays={a["name"]:input_array(a,0,"normal") for a in meta["kernels"][0]["args"] if not a.get("output")}
                np.testing.assert_array_equal(ref(**arrays)[0],expected["out"])
                self.assertFalse(np.allclose(0,expected["out"],rtol=meta["rtol"],atol=meta["atol"]))

    @unittest.skipUnless(shutil.which("ptxas"),"ptxas required")
    def test_actual_cli_assembles_all_examples(self):
        with tempfile.TemporaryDirectory() as folder:
            for value,_ in fixtures_with_refs():
                with self.subTest(case=value[1]["id"]):
                    path=write_case(Path(folder)/value[1]["id"],value)
                    ptx,_=compile_ptx(ROOT/"compile.py",path)
                    self.assertIn("mma.sync.aligned.m16n8k16",ptx)
                    assemble(ptx,"sm_89")

    def test_unsupported_architecture_rejected_before_output(self):
        value,_=matrix_case("tensor_old_arch")
        with self.assertRaisesRegex(CompileError,"sm_80"):
            Compiler(value[0],value[1]["kernels"][0],target="sm_75").compile()

    def test_fragment_contract_rejections(self):
        def replace_fragment(body, **changes):
            tile=body[0]
            changed=(replace(tile.body[0],**changes),)+tile.body[1:]
            return (replace(tile,body=changed),)
        for changes,diagnostic in ((dict(dtype=F32),"f16/bf16"),
                                   (dict(nregs=3),"4 registers"),
                                   (dict(count=65),"between 1 and 64"),
                                   (dict(shape=(8,8,4)),"tensor shape")):
            with self.subTest(changes=changes), self.assertRaisesRegex(CompileError,diagnostic):
                self.compiler_with_changed_body(lambda body:replace_fragment(body,**changes)).compile()

    def test_auxiliary_partition_rejected_if_it_splits_a_warp(self):
        with self.assertRaisesRegex(CompileError,"auxiliary Tile partition"):
            self.compiler_with_changed_body(lambda body:(replace(body[0],block_threads=40,aux_threads=32),)).compile()

    def test_epilogue_unbound_coordinate_rejected(self):
        def change(body):
            tile=body[0]
            epi=Lambda(params=("acc","missing_coordinate"),body=(),results=("acc",))
            statements=tile.body[:-1]+(replace(tile.body[-1],epilogue=epi),)
            return (replace(tile,body=statements),)
        with self.assertRaisesRegex(CompileError,"unbound tensor epilogue coordinate"):
            self.compiler_with_changed_body(change).compile()

    def test_reused_load_meets_alignment_facts_from_every_scope(self):
        def change(body):
            tile=body[0]
            statements=[]
            for statement in tile.body:
                if isinstance(statement,StridedLoop) and statement.axis.name=="kk":
                    # The same immutable load node is first used at an invalid
                    # row offset, then at an aligned one. Last occurrence wins
                    # would wrongly certify the first use.
                    load=statement.body[0]
                    statements.extend((
                        replace(statement,start=lit(1),body=(load,),end=lit(2)),
                        replace(statement,start=lit(0),body=(load,),end=lit(1)),
                    ))
                else:
                    statements.append(statement)
            return (replace(tile,body=tuple(statements)),)
        with self.assertRaisesRegex(CompileError,"alignment cannot be proved"):
            self.compiler_with_changed_body(change,shared=True).compile()

    def test_late_axis_update_invalidates_next_iteration_alignment(self):
        def change(body):
            tile=body[0]
            statements=[]
            for statement in tile.body:
                if isinstance(statement,StridedLoop) and statement.axis.name=="kk":
                    # The loop first seeds the carrier, then binds its induction
                    # register under the same name. The late update aliases that
                    # register, making a subsequent iteration's offset odd.
                    load=statement.body[0]
                    altered=(load,Accum("kk","one",op("add"),I32,("kk",)))
                    from emmy.compiler.ir.stmt.leaves import Let
                    statements.append(Let("one",lit(1),I32))
                    statements.append(replace(statement,body=altered,seed=True))
                else:
                    statements.append(statement)
            return (replace(tile,body=tuple(statements)),)
        with self.assertRaisesRegex(CompileError,"alignment cannot be proved"):
            self.compiler_with_changed_body(change,shared=True).compile()


if __name__ == "__main__":
    if "--export" in sys.argv:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--export",type=Path,required=True)
        export(parser.parse_args().export)
    else:
        unittest.main()
