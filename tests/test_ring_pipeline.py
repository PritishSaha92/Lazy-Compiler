"""Two-slot bulk-copy/tensor pipelines with exact, independent references.

The producer starts the next slab before consumers wait on the current slab.
Five/seven slabs exercise ring wraparound, both barrier parities and an explicit
last drain. Tests use CPU reference calculations and ptxas, never a local GPU.
The BF16 case uses supported F32 wire buffers and explicit shared F32->BF16 IR
conversion; it makes no claim that the organizer transports BF16 arguments.
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
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from emmy.compiler.dtype import BF16, F16, F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import (
    LdmatrixLoad, MbarrierArriveExpectTx, MbarrierInit, MbarrierWait,
    MmaSyncPtx, RegFragment, RegStore, Smem, Sync, Tile,
    TmaDescriptor, TmaLoad,
)
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Write
from ir_cases import axis, case, expression as ex, literal as lit, op, write_case
from ptx_backend import CompileError, compile_file
from tools.check_compiler import assemble, compile_ptx, input_array

SHAPE = (16, 8, 16)
SPECS = (
    ('ring_tensor_k80_f16', 5, F16, 1),
    ('ring_tensor_k112_bf16', 7, BF16, 1),
    ('ring_tensor_k112_f16_two_cta', 7, F16, 2),
)


def bf16_values(values):
    """Independent nearest-even storage conversion for finite F32 inputs."""
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError('The exact fixture oracle expects finite inputs')
    words = values.view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def reference(a, b, *, kind):
    convert = (lambda x: np.asarray(x, np.float16).astype(np.float32)) if kind == F16 else bf16_values
    aq, bq = convert(a), convert(b)
    return [(aq.astype(np.float64) @ bq.astype(np.float64)).astype(np.float32)]


def explicit_inputs(slabs, ctas, kind):
    """Each slab contributes a distinct positive matrix; CTAs own distinct rows."""
    rows = np.arange(16 * ctas)[:, None]
    column = np.arange(16 * slabs)[None, :]
    slab, local_k = column // 16, column % 16
    a = (((rows * 3 + local_k) % 13 + 1) + 4 * slab) / 16
    k = np.arange(16 * slabs)[:, None]
    cols = np.arange(8)[None, :]
    b = ((((k % 16) * 5 + cols * 3) % 11 + 1) + 2 * (k // 16)) / 16
    storage = np.float16 if kind == F16 else np.float32
    return {'a': a.astype(storage), 'b': b.astype(storage)}


def make_case(name, slabs, kind, ctas):
    k, m = 16 * slabs, 16 * ctas
    phase = Var('slab')
    row_origin = ex('*', Var('batch'), lit(16))
    leader = ex('==', Var('_tid'), lit(0))
    wire_dtype = F16 if kind == F16 else F32
    wire_spelling = '__half' if kind == F16 else 'float'
    operand_spelling = '__half' if kind == F16 else '__nv_bfloat16'
    bytes_per_slab = 768 if kind == F16 else 1536

    def producer(slab):
        slot = ex('%', slab, lit(2))
        offset = ex('*', slab, lit(16))
        return Cond(leader, (
            MbarrierArriveExpectTx('bar', bytes_per_slab, slot),
            TmaLoad('aring', (ex('*', slot, lit(256)),), 'adesc',
                    (row_origin, offset), 'bar', slot),
            TmaLoad('bring', (ex('*', slot, lit(128)),), 'bdesc',
                    (offset, lit(0)), 'bar', slot),
        ))

    def consumer(slab):
        slot = ex('%', slab, lit(2))
        parity = ex('%', ex('//', slab, lit(2)), lit(2))
        result = [MbarrierWait('bar', parity, slot)]
        if kind == BF16:
            # The runner ABI carries F32. Conversion is an explicit scheduled
            # operation after raw-copy completion, before the BF16 matrix load.
            for label, size in (('a', 256), ('b', 128)):
                index = Var('convert_' + label)
                flat = ex('+', ex('*', slot, lit(size)), index)
                result.append(StridedLoop(axis('convert_' + label, size), Var('cell'), lit(32), (
                    Load('wire_' + label, label + 'ring', (flat,), dtype=F32),
                    Write(label + 'operands', (flat,), 'wire_' + label, value_dtype=BF16),
                )))
            result.append(Sync())
        operand_a = 'aring' if kind == F16 else 'aoperands'
        operand_b = 'bring' if kind == F16 else 'boperands'
        start_row = ex('*', slot, lit(16))
        result.extend((
            LdmatrixLoad('af', operand_a, (start_row, lit(0)), 'a', ldm=16),
            LdmatrixLoad('bf', operand_b, (start_row, lit(0)), 'b', ldm=8),
            MmaSyncPtx('cf', 'af', 'bf', SHAPE, ab_dtype=kind.name),
            # All consumers finish using this slot and its barrier generation
            # before the next iteration can start overwriting the released slot.
            Sync(),
        ))
        return tuple(result)

    body = [
        Smem('aring', (32, 16), wire_spelling, align=16),
        Smem('bring', (32, 8), wire_spelling, align=16),
        Smem('bar', (2,), 'unsigned long long', align=8),
    ]
    if kind == BF16:
        body.extend((Smem('aoperands', (32, 16), operand_spelling, align=16),
                     Smem('boperands', (32, 8), operand_spelling, align=16)))
    body.extend((
        TmaDescriptor('adesc', 'a', (m, k), (16, 16), dtype=wire_spelling),
        TmaDescriptor('bdesc', 'b', (k, 8), (16, 8), dtype=wire_spelling),
        RegFragment('af', 'a', SHAPE, kind),
        RegFragment('bf', 'b', SHAPE, kind),
        RegFragment('cf', 'c', SHAPE, F32),
        Cond(leader, (MbarrierInit('bar', 1, lit(0)), MbarrierInit('bar', 1, lit(1)))),
        Sync(),
        # Prime slot zero. Each loop turn begins the next transfer while the
        # current transfer may still be pending; the final slab drains outside.
        producer(lit(0)),
        Loop(axis('slab', slabs - 1), (producer(ex('+', phase, lit(1))),) + consumer(phase), unroll=False),
        *consumer(lit(slabs - 1)),
        RegStore('out', (row_origin, lit(0)), 'cf', SHAPE),
    ))
    graph, meta, _ = case(name, (Tile((axis('batch', ctas), axis('cell', 32)),
                                     tuple(body), block_threads=32),),
                         {'a': (m, k), 'b': (k, 8)}, (m, 8),
                         lambda **xs: reference(**xs, kind=kind)[0],
                         block=32, grid=ctas, reverse_args=ctas > 1)
    arrays = explicit_inputs(slabs, ctas, kind)
    for key, node in graph.nodes.items():
        graph.nodes[key] = replace(node, outputs=tuple(
            replace(t, dtype=wire_dtype) if t.name in arrays else t for t in node.outputs))
    for arg in meta['kernels'][0]['args']:
        if arg['name'] in arrays:
            arg.pop('seed', None)
            arg.update(dtype=wire_dtype.name, data=arrays[arg['name']].ravel().tolist())
    meta.update(rtol=0, atol=0, fixture_provenance={
        'construction': 'Hand-built typed IR and independent exact matrix references; no renderer used',
        'feature': 'Two-slot pipelined bulk-copy, shared ldmatrix and tensor MMA with explicit final drain',
        'slabs': slabs, 'ring_slots': 2, 'ctas': ctas,
        'operand_dtype': kind.name, 'wire_dtype': wire_dtype.name,
        'bf16_wire_transport': False,
        'bf16_shared_conversion': kind == BF16,
        'transaction_bytes_per_slab': bytes_per_slab,
        'local_gpu_used': False,
    })
    return graph, meta, {'out': reference(**arrays, kind=kind)[0]}


def cases():
    for spec in SPECS:
        yield make_case(*spec)


OPEN = [dict(id=name, ref=lambda a, b, kind=kind: reference(a, b, kind=kind))
        for name, _, kind, _ in SPECS]


def scalar_dot_case():
    i, k = Var('i'), Var('k')
    body = (Loop(axis('k', 4), (
        Load('xv', 'x', (i, k), dtype=F32),
        Load('yv', 'y', (i, k), dtype=F32),
        Assign('product', op('multiply'), ('xv', 'yv'), F32),
        Accum('dot', 'product', op('add'), F32),
    )), Write('out', (i,), 'dot', value_dtype=F32))
    value = case('ring_isolation_scalar_dot', (Tile((axis('i', 32),), body),),
                 {'x': (32, 4), 'y': (32, 4)}, (32,),
                 lambda x, y: np.sum(x.astype(np.float64) * y, axis=1), block=32)
    # This fixture is used only for compilation isolation, but its reference
    # remains a valid exact contract rather than relying on random FP32 sums.
    x = (np.arange(128).reshape(32, 4) % 17 / 16).astype(np.float32)
    y = np.broadcast_to(np.arange(1, 5, dtype=np.float32) / 8, (32, 4)).copy()
    for arg in value[1]['kernels'][0]['args']:
        if arg['name'] in ('x', 'y'):
            arg.pop('seed', None)
            arg['data'] = (x if arg['name'] == 'x' else y).ravel().tolist()
    value[1].update(rtol=0, atol=0)
    expected = np.sum(x.astype(np.float64) * y, axis=1).astype(np.float32)
    return value[0], value[1], {'out': expected}


def export(destination):
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for value in cases():
        folder = destination / value[1]['id']
        path = write_case(folder, value)
        records.append({'id': value[1]['id'], 'files_sha256': {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.is_file()
        }})
        print(path, flush=True)
    (destination / 'manifest.json').write_text(json.dumps({
        'scope': 'Exact typed IR/input/reference packages only; no GPU execution',
        'packages': records,
        'test_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + '\n', encoding='utf-8')


class RingPipelineTests(unittest.TestCase):
    def test_exact_references_detect_missing_stale_and_cross_slab_data(self):
        for spec, value in zip(SPECS, cases()):
            name, slabs, kind, ctas = spec
            graph, meta, expected = value
            with self.subTest(case=name):
                Graph.from_dict(graph.to_dict())
                arrays = {a['name']: input_array(a, 0) for a in meta['kernels'][0]['args'] if not a.get('output')}
                a, b = (arrays[n].astype(np.float64) for n in ('a', 'b'))
                if kind == BF16:
                    np.testing.assert_array_equal(bf16_values(arrays['a']), arrays['a'])
                    np.testing.assert_array_equal(bf16_values(arrays['b']), arrays['b'])
                pieces = [a[:, 16*s:16*(s+1)] @ b[16*s:16*(s+1), :] for s in range(slabs)]
                full = a @ b
                np.testing.assert_array_equal(sum(pieces), full)
                np.testing.assert_array_equal(full.astype(np.float32).astype(np.float64), full)
                np.testing.assert_array_equal(reference(**arrays, kind=kind)[0], expected['out'])
                for missing, contribution in enumerate(pieces):
                    self.assertFalse(np.array_equal(full - contribution, full), missing)
                    stale = pieces[(missing - 1) % slabs]
                    self.assertFalse(np.array_equal(full - contribution + stale, full), missing)
                for s in range(slabs - 1):
                    mismatched = a[:, 16*s:16*(s+1)] @ b[16*(s+1):16*(s+2), :]
                    self.assertFalse(np.array_equal(mismatched, pieces[s]), s)
                if ctas > 1:
                    self.assertFalse(np.array_equal(full[:16], full[16:]))
                self.assertEqual((meta['rtol'], meta['atol']), (0, 0))
                # Every exact product and every possible partial sum is a small
                # integer multiple of 1/256, within FP32's exact integer range.
                total_scaled = np.sum(np.abs(a[:, :, None] * b[None, :, :]), axis=1) * 256
                self.assertTrue(np.all(total_scaled < 2**24))
                self.assertTrue(np.all(total_scaled == np.floor(total_scaled)))

    def test_actual_cli_and_architecture_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            for spec, value in zip(SPECS, cases()):
                with self.subTest(case=spec[0]):
                    path = write_case(Path(directory) / spec[0], value)
                    ptx, _ = compile_ptx(ROOT / 'compile.py', path)
                    self.assertEqual(ptx.replace('\r\n', '\n'), compile_file(path))
                    self.assertIn('.target sm_90', ptx)
                    self.assertIn('cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes', ptx)
                    self.assertIn('mbarrier.try_wait.parity.acquire.cta.shared::cta.b64', ptx)
                    self.assertIn('ldmatrix.sync.aligned.m8n8.x4', ptx)
                    self.assertIn('ldmatrix.sync.aligned.m8n8.x2.trans', ptx)
                    self.assertIn(f'mma.sync.aligned.m16n8k16.row.col.f32.{spec[2].name}.{spec[2].name}.f32', ptx)
                    if spec[2] == BF16:
                        self.assertIn('cvt.rn.bf16.f32', ptx)
                    with self.assertRaisesRegex(CompileError, 'sm_90'):
                        compile_file(path, target='sm_89')

    @unittest.skipUnless(shutil.which('ptxas'), 'ptxas required; no GPU execution')
    def test_ring_pipelines_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in cases():
                with self.subTest(case=value[1]['id']):
                    path = write_case(Path(directory) / value[1]['id'], value)
                    assemble(compile_file(path), 'sm_90')

    def test_same_process_compile_state_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scalar = scalar_dot_case()
            strict_path = write_case(root / 'strict', scalar)
            tolerant_meta = json.loads(json.dumps(scalar[1]))
            tolerant_meta.update(rtol=1e-6, atol=1e-6)
            tolerant_path = write_case(root / 'tolerant', (scalar[0], tolerant_meta, scalar[2]))
            strict = compile_file(strict_path)
            self.assertIn('.target sm_89', strict)
            self.assertNotIn('fma.rn.f32', strict)
            snapshots = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            for value in cases():
                # Repeated names in independent kernels must not retain fragment
                # state, shared buffers, target choice, labels or virtual registers.
                path = write_case(root / value[1]['id'], value)
                before = compile_file(path)
                self.assertIn('.target sm_90', before)
                self.assertEqual(compile_file(strict_path), strict)
                compensated = compile_file(tolerant_path)
                self.assertIn('fma.rn.f32', compensated)
                self.assertNotEqual(compensated, strict)
                self.assertEqual(compile_file(strict_path), strict)
                widened = compile_file(tolerant_path, accurate_dots=True)
                self.assertIn('f64', widened)
                self.assertEqual(compile_file(strict_path), strict)
                self.assertEqual(compile_file(path), before)
            for path, data in snapshots.items():
                self.assertEqual(path.read_bytes(), data)


if __name__ == '__main__':
    if '--export' in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--export', required=True, type=Path)
        export(parser.parse_args().export)
    else:
        unittest.main()