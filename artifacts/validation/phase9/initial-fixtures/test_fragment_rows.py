"""Original fragment-algebra cases with independent host references.

Pinned Emmy 0.3.11 kernel/ir.py:993-1226 specifies FragmentApply and
FragmentRowReduce. Tests cover operand distributions, lazy guarded loads,
the exact per-lane reduction order, and composition with an actual MMA.
CPU/ptxas only by default. --export writes packages for the official runner.
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
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT / "tests"))
from emmy.compiler.dtype import F16, F32, F64, I32
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import (
    COORD, FRAG, FRAG_COL, FRAG_ROW, GMEM, M8N8K4, ROW, UNIFORM,
    FragmentApply, FragmentRepack, FragmentRowReduce, LdmatrixLoad, MmaSyncPtx,
    RegFragment, RegStore, Tile,
)
from emmy.compiler.ir.stmt.blocks import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Let
from ir_cases import axis, case, expression as e, literal as lit, op, write_case
from test_tensor_cores import matrix_case, quantize
from ptx_backend import Compiler, CompileError
from tools.check_compiler import assemble, compile_ptx, input_array

SHAPE = (16,8,16)


def replace_body(value, body):
    graph,meta,expected = value
    for key,node in graph.nodes.items():
        if hasattr(node.op,"body"):
            graph.nodes[key] = replace(node,op=replace(node.op,body=tuple(body)))
    return graph,meta,expected


def masked_gather_case():
    row,col = Var(FRAG_ROW),Var(FRAG_COL)
    valid = e("&&",e("<",row,lit(15)),e("<",col,lit(10)))
    body = (
        Let("row_origin",lit(2),I32),Let("col_origin",lit(3),I32),
        Let("scale",lit(0.5),F32),
        FragmentApply("gathered",op("where"),
            (valid,("x",(row,col)),lit(-5.0)),(COORD,GMEM,UNIFORM),
            post=(op("abs"),),row_base=Var("row_origin"),col_base=Var("col_origin")),
        FragmentApply("gathered",op("multiply"),("gathered","scale"),
                      (FRAG,UNIFORM),in_place=True),
        RegStore("out",(lit(0),lit(0)),"gathered",SHAPE),
    )
    def reference(x):
        result = np.full((16,8),np.float32(2.5))
        result[:13,:7] = np.abs(x[2:15,3:10]) * np.float32(0.5)
        return [result]
    value = case("fragment_masked_gmem_post",(Tile((axis("lane",32),),body),),
                 {"x":(15,10)},(16,8),lambda x:reference(x)[0],block=32)
    value[1]["rtol"] = value[1]["atol"] = 0.0
    return value,reference


def family_softmax_case():
    row,col = Var(FRAG_ROW),Var(FRAG_COL)
    fi = Var("fi")
    indexed = "cf[fi]"
    origins = dict(row_base=lit(0),col_base=e("*",fi,lit(8)))
    body = (
        RegFragment("cf","c",SHAPE,F32,count=2),
        StridedLoop(axis("fi",2),lit(0),lit(1),(
            FragmentApply(indexed,op("add"),(indexed,("x",(row,col))),
                          (FRAG,GMEM),in_place=True,**origins),)),
        FragmentRowReduce("top_max","bot_max",("cf[0]","cf[1]"),op("maximum")),
        StridedLoop(axis("fi",2),lit(0),lit(1),(
            FragmentApply(indexed,op("subtract"),(indexed,("top_max","bot_max")),
                          (FRAG,ROW),in_place=True,post=(op("exp"),)),)),
        FragmentRowReduce("top_sum","bot_sum",("cf[0]","cf[1]"),op("add")),
        StridedLoop(axis("fi",2),lit(0),lit(1),(
            FragmentApply(indexed,op("divide"),(indexed,("top_sum","bot_sum")),
                          (FRAG,ROW),in_place=True),
            RegStore("out",(lit(0),e("*",fi,lit(8))),indexed,SHAPE),)),
    )
    def reference(x):
        # Independent matrix formula; this does not use the fragment mapping.
        shifted = x.astype(np.float64) - np.max(x.astype(np.float64),axis=1,keepdims=True)
        weights = np.exp(shifted)
        return [(weights / np.sum(weights,axis=1,keepdims=True)).astype(np.float32)]
    value = case("fragment_family_row_softmax",(Tile((axis("lane",32),),body),),
                 {"x":(16,16)},(16,16),lambda x:reference(x)[0],block=32)
    value[1]["rtol"],value[1]["atol"] = 8e-6,1e-7
    return value,reference


def mma_normalization_case():
    value,_ = matrix_case("fragment_mma_row_normalization",kind=F16)
    graph,meta,_ = value
    tile = next(node.op.body[0] for node in graph.nodes.values() if hasattr(node.op,"body"))
    body = tile.body[:-1] + (
        FragmentRowReduce("row_max0","row_max1",("cf",),op("maximum")),
        FragmentApply("weights",op("subtract"),("cf",("row_max0","row_max1")),
                      (FRAG,ROW),post=(op("exp"),)),
        FragmentRowReduce("row_sum0","row_sum1",("weights",),op("add")),
        FragmentApply("normalized",op("divide"),("weights",("row_sum0","row_sum1")),
                      (FRAG,ROW)),
        replace(tile.body[-1],frag="normalized"),
    )
    def reference(a,b):
        scores = quantize(a,F16).astype(np.float64) @ quantize(b,F16).astype(np.float64)
        weights = np.exp(scores - np.max(scores,axis=1,keepdims=True))
        return [(weights / np.sum(weights,axis=1,keepdims=True)).astype(np.float32)]
    inputs = {a["name"]:input_array(a,0,"normal") for a in meta["kernels"][0]["args"] if not a.get("output")}
    value = graph,meta,{"out":reference(**inputs)[0]}
    value = replace_body(value,(replace(tile,body=body),))
    meta["rtol"],meta["atol"] = 5e-5,2e-6
    return value,reference


def lane_fold_reference(x, operation):
    """Direct lane/column ownership model, preserving the specified FP32 fold.

    Each lane first visits its two columns in each of two side-by-side tiles.
    XOR peers are read from the previous round, not an in-place host loop.
    Cancellation makes a scalar left fold over the row differ from this tree;
    the reference deliberately preserves the IR order instead of a row np.sum.
    """
    combine = {"add":np.add,"multiply":np.multiply,
               "maximum":np.maximum,"minimum":np.minimum}[operation]
    state = np.empty((32,2),np.float32)
    for lane in range(32):
        columns = [2*(lane%4)+offset for offset in (0,1,8,9)]
        for half,row in enumerate((lane//4,lane//4+8)):
            result = x[row,columns[0]]
            for column in columns[1:]:
                result = np.float32(combine(result,x[row,column]))
            state[lane,half] = result
    for mask in (2,1):
        previous = state.copy()
        for lane in range(32):
            state[lane] = combine(previous[lane],previous[lane^mask])
    result = np.empty((16,8),np.float32)
    for lane in range(32):
        for half,row in enumerate((lane//4,lane//4+8)):
            result[row,2*(lane%4):2*(lane%4)+2] = state[lane,half]
    return result


def attention_tile_case():
    """One complete 16x16 attention tile, not an online/multi-tile kernel.

    Q/K/V contain small dyadic values. Both dot stages are exact in FP32 for
    these inputs, while the host oracle explicitly rounds normalized weights
    to FP16 before the second product, as FragmentRepack requires.
    """
    body = [RegFragment("af","a",SHAPE,F16),
            RegFragment("bf","b",SHAPE,F16,count=2),
            RegFragment("scores","c",SHAPE,F32,count=2),
            RegFragment("result","c",SHAPE,F32),
            LdmatrixLoad("af","q",(lit(0),lit(0)),"a",ldm=16,staged=False)]
    for part in range(2):
        body.extend((
            LdmatrixLoad(f"bf[{part}]","kt",(lit(0),lit(part*8)),"b",ldm=16,staged=False),
            MmaSyncPtx(f"scores[{part}]","af",f"bf[{part}]",SHAPE),
        ))
    fragments = ("scores[0]","scores[1]")
    body.append(FragmentRowReduce("pivot0","pivot1",fragments,op("maximum")))
    for name in fragments:
        body.append(FragmentApply(name,op("subtract"),(name,("pivot0","pivot1")),
                                  (FRAG,ROW),in_place=True,post=(op("exp"),)))
    body.append(FragmentRowReduce("denom0","denom1",fragments,op("add")))
    for name in fragments:
        body.append(FragmentApply(name,op("divide"),(name,("denom0","denom1")),
                                  (FRAG,ROW),in_place=True))
    body.extend((FragmentRepack("af",fragments),
                 LdmatrixLoad("bf[0]","v",(lit(0),lit(0)),"b",ldm=8,staged=False),
                 MmaSyncPtx("result","af","bf[0]",SHAPE),
                 RegStore("out",(lit(0),lit(0)),"result",SHAPE)))

    def reference(q,kt,v):
        q,kt,v = (quantize(x,F16).astype(np.float64) for x in (q,kt,v))
        scores = (q @ kt).astype(np.float32)
        shifted = scores.astype(np.float64) - np.max(scores,axis=1,keepdims=True)
        weights = np.exp(shifted)
        weights = (weights / np.sum(weights,axis=1,keepdims=True)).astype(np.float32)
        rounded = quantize(weights,F16).astype(np.float64)
        return [(rounded @ v).astype(np.float32)]

    value = case("fragment_attention_tile_f16",(Tile((axis("lane",32),),tuple(body)),),
                 {"q":(16,16),"kt":(16,16),"v":(16,8)},(16,8),
                 lambda q,kt,v:reference(q,kt,v)[0],block=32)
    rng = np.random.default_rng(731)
    arrays = {name:rng.integers(-3,4,size=shape).astype(np.float32)/np.float32(8)
              for name,shape in (("q",(16,16)),("kt",(16,16)),("v",(16,8)))}
    for argument in value[1]["kernels"][0]["args"]:
        if argument["name"] in arrays:
            argument["data"] = arrays[argument["name"]].tolist()
    value[2]["out"] = reference(**arrays)[0]
    value[1]["rtol"],value[1]["atol"] = 2e-4,2e-5
    return value,reference


def ordered_fold_case(operation="add"):
    body = tuple(
        FragmentApply("part"+str(i),op("copy"),(("x",(Var(FRAG_ROW),Var(FRAG_COL))),),
                      (GMEM,),row_base=lit(0),col_base=lit(8*i)) for i in range(2)
    ) + (
        FragmentRowReduce("row0","row1",("part0","part1"),op(operation)),
        FragmentApply("broadcast",op("copy"),(("row0","row1"),),(ROW,)),
        RegStore("out",(lit(0),lit(0)),"broadcast",SHAPE),
    )
    def reference(x):
        return [lane_fold_reference(x,operation)]
    value = case("fragment_row_"+operation+"_ordered",(Tile((axis("lane",32),),body),),
                 {"x":(16,16)},(16,8),lambda x:reference(x)[0],block=32)
    if operation == "add":
        # This pattern distinguishes left-fold order and simultaneous shuffles.
        pattern = np.array([1e10,1,-1e10,2,3,-4,5,6,7,8,9,-10,11,12,-13,14],np.float32)
        x = np.stack([np.roll(pattern,row) for row in range(16)])
        x[-1] = np.arange(1,17,dtype=np.float32)
        value[1]["kernels"][0]["args"][0]["data"] = x.tolist()
        value[2]["out"] = reference(x)[0]
    value[1]["rtol"] = value[1]["atol"] = 0.0
    return value,reference


def fixtures_with_refs():
    yield masked_gather_case()
    yield family_softmax_case()
    yield mma_normalization_case()
    yield attention_tile_case()
    for operation in ("add","multiply","maximum","minimum"):
        yield ordered_fold_case(operation)


OPEN = [dict(id=value[1]["id"],ref=reference) for value,reference in fixtures_with_refs()]


def export(destination):
    destination.mkdir(parents=True,exist_ok=True)
    records = []
    for value,reference in fixtures_with_refs():
        graph,meta,expected = value
        path = write_case(destination/meta["id"],value)
        ptx,seconds = compile_ptx(ROOT/"compile.py",path)
        assembly = assemble(ptx,"sm_89")
        records.append(dict(id=meta["id"],assembly=assembly,compile_seconds=seconds,
                            ptx_sha256=hashlib.sha256(ptx.encode("ascii")).hexdigest()))
    report = dict(pass_=True,local_gpu_used=False,cases=records)
    print(json.dumps(report,indent=2))


class FragmentRowTests(unittest.TestCase):
    def test_references_are_independent_and_nontrivial(self):
        for value,reference in fixtures_with_refs():
            _,meta,expected = value
            with self.subTest(case=meta["id"]):
                arrays = {a["name"]:input_array(a,0,"normal") for a in meta["kernels"][0]["args"] if not a.get("output")}
                np.testing.assert_array_equal(reference(**arrays)[0],expected["out"])
                self.assertFalse(np.all(expected["out"] == 0))
        value,_ = ordered_fold_case()
        x = np.asarray(value[1]["kernels"][0]["args"][0]["data"],np.float32)
        scalar = np.zeros(16,np.float32)
        for column in range(16):
            scalar = scalar + x[:,column]
        self.assertTrue(np.any(value[2]["out"] != scalar[:,None]))

    @unittest.skipUnless(shutil.which("ptxas"),"ptxas required")
    def test_actual_cli_assembles_fragment_programs(self):
        with tempfile.TemporaryDirectory() as folder:
            for value,_ in fixtures_with_refs():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(folder)/value[1]["id"],value)
                    ptx,_ = compile_ptx(ROOT/"compile.py",path)
                    assemble(ptx,"sm_89")

    def compile_body(self, body, **tile_changes):
        value,_ = masked_gather_case()
        value = replace_body(value,(Tile((axis("lane",32),),tuple(body),**tile_changes),))
        return Compiler(value[0],value[1]["kernels"][0]).compile()

    def test_rejects_malformed_fragment_operands(self):
        initial = RegFragment("cf","c",SHAPE,F32)
        valid = FragmentApply("copy",op("copy"),("cf",),(FRAG,))
        changes = (
            (dict(layout=M8N8K4),"FP32 C layout"),
            (dict(kinds=("unknown",)),"unsupported fragment operand"),
            (dict(kinds=(ROW,)),"two scalar names"),
            (dict(kinds=(UNIFORM,)),"must be a scalar"),
            (dict(in_place=True),"must be a fragment operand"),
            (dict(out="copy[0]"),"unsubscripted identifier"),
            (dict(post=(op("add"),)),"expects 2 operands"),
        )
        for mutation,message in changes:
            with self.subTest(mutation=mutation),self.assertRaisesRegex(CompileError,message):
                self.compile_body((initial,replace(valid,**mutation)))
        # Reserved coordinates need origins, even if an enclosing scope happens
        # to have a scalar with the same reserved name.
        with self.assertRaisesRegex(CompileError,"unbound IR value '__frow'"):
            self.compile_body((Let(FRAG_ROW,lit(4),I32),FragmentApply("x",op("copy"),
                               (Var(FRAG_ROW),),(COORD,))))
        with self.assertRaisesRegex(CompileError,"unbound IR value '__fcol'"):
            self.compile_body((FragmentApply("x",op("copy"),(Var(FRAG_COL),),(COORD,)),))
        # Emmy rejects this malformed node while deriving the typed graph's
        # external reads, before the backend can visit it.
        with self.assertRaisesRegex(ValueError,"zip"):
            self.compile_body((initial,replace(valid,kinds=())))

    def test_rejects_double_expressions_that_would_round_early(self):
        double = Literal(1.0 + 2.0**-24,"double")
        initial = (RegFragment("cf","c",SHAPE,F32),Let("double",double,F64))
        for argument,kind in ((double,UNIFORM),(("double","double"),ROW),(double,COORD)):
            with self.subTest(kind=kind),self.assertRaisesRegex(CompileError,"f64 results"):
                self.compile_body(initial + (FragmentApply("result",op("add"),
                    ("cf",argument),(FRAG,kind)),))
        # Named uniform values are explicitly converted to f32 by pinned Emmy.
        self.compile_body(initial + (FragmentApply("result",op("add"),
            ("cf","double"),(FRAG,UNIFORM)),))

    def test_collective_admission_and_fold_contract(self):
        initial = RegFragment("cf","c",SHAPE,F32)
        fold = FragmentRowReduce("top","bot",("cf",),op("add"))
        for statement,message in (
            (replace(fold,frags=()),"nonempty FP32"),
            (replace(fold,dtype=F16),"nonempty FP32"),
            (replace(fold,op=op("subtract")),"supports add"),
            (replace(fold,top="bot"),"distinct names"),
            (Cond(e("<",Var("lane"),lit(16)),(fold,)),"unconditional, static"),
            (StridedLoop(axis("j",32),Var("lane"),lit(1),(fold,)),"unconditional, static"),
        ):
            with self.subTest(statement=statement),self.assertRaisesRegex(CompileError,message):
                self.compile_body((initial,statement))
        with self.assertRaisesRegex(CompileError,"auxiliary Tile partition"):
            self.compile_body((initial,fold),block_threads=40,aux_threads=32)
        value,_ = masked_gather_case()
        value = replace_body(value,(Tile((axis("lane",31),),(initial,fold)),))
        with self.assertRaisesRegex(CompileError,"complete warps"):
            Compiler(value[0],value[1]["kernels"][0]).compile()

    def test_masked_gmem_load_is_inside_the_taken_branch(self):
        value,_ = masked_gather_case()
        ptx = Compiler(value[0],value[1]["kernels"][0]).compile()
        # This structural test checks the safety condition that the numerical
        # oracle cannot catch if an out-of-range read happens to return a value.
        branch = ptx.index("bra fragment_else_")
        load = ptx.index("ld.global.f32",branch)
        end_true = ptx.index("bra fragment_end_",branch)
        self.assertLess(branch,load)
        self.assertLess(load,end_true)


if __name__ == "__main__":
    if "--export" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--export",type=Path,required=True)
        export(parser.parse_args().export)
    else:
        unittest.main()
