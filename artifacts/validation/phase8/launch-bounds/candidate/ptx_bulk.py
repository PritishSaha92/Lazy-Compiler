"""Restricted, descriptor-free lowering of Emmy tensor box copies.

For an unswizzled, statically shaped in-bounds box, a tensor load is exactly a
sequence of contiguous row copies.  We issue those rows on the SM90 bulk-copy
engine and retain the IR's transaction barrier protocol.  No host CUtensorMap
argument is needed.  This is deliberately not general cp.async.bulk.tensor:
boundary zero-fill, swizzles and unproved address ranges are diagnosed.

PTX 8.0: sections 9.7.8.22.5 (cp.async.bulk), 9.7.12.4 (proxy fences), and
9.7.12.14 (mbarrier).  All helpers emit PTX directly; none call an IR renderer.
"""

from __future__ import annotations

import itertools
import math
import re

from emmy.compiler.ir.expr import BinaryExpr, Builtin, Literal, Var
from emmy.compiler.ir.kernel.ir import (
    MbarrierArrive,
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    Tile,
    TmaDescriptor,
    TmaLoad,
)
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop


BULK_NODES = (
    TmaDescriptor, TmaLoad, MbarrierInit, MbarrierArrive,
    MbarrierArriveExpectTx, MbarrierWait,
)


class BulkLowering:
    """Emit bulk copies using a parent Compiler's typed address machinery.

    Call prepare(kernel.body) once after the parent's buffer declarations have
    been collected, then handles(stmt) from its normal statement dispatch.
    Bounds are conservative proofs, not runtime predicates that could silently
    drop a copy and strand its barrier's expected transaction count.
    """

    def __init__(self, compiler):
        # Local import keeps the extension independent of backend import order.
        from ptx_backend import CompileError, _SIZE, builtin_operand, dtype, extent

        self.c = compiler
        self.error, self.size, self.dtype, self.extent = CompileError, _SIZE, dtype, extent
        self.builtin_operand = builtin_operand
        self.descriptors = {}
        self.ranges = {}

    def prepare(self, body):
        for name in ("block", "grid"):
            dimensions = self.c.meta.get(name, [1, 1, 1])
            if (len(dimensions) != 3 or any(type(n) is not int or n < 1 for n in dimensions)
                    or tuple(dimensions[1:]) != (1, 1)):
                raise self.error("bulk copies require positive one-dimensional block and grid launches")
        block = self.c.meta.get("block", [1, 1, 1])[0]
        grid = self.c.meta.get("grid", [1, 1, 1])[0]
        self.builtin_ranges = {"WARP_SZ": (32, 32)}
        for position, axis in enumerate("xyz"):
            block_extent = self.c.meta.get("block", [1, 1, 1])[position]
            grid_extent = self.c.meta.get("grid", [1, 1, 1])[position]
            self.builtin_ranges.update({
                f"%tid.{axis}": (0, block_extent - 1),
                f"%ctaid.{axis}": (0, grid_extent - 1),
                f"%ntid.{axis}": (block_extent, block_extent),
                f"%nctaid.{axis}": (grid_extent, grid_extent),
            })
        ranges = {
            "_tid": (0, block - 1), "_cta": (0, grid - 1),
            "lane": (0, min(block, 32) - 1), "warp": (0, (block - 1) // 32),
        }
        self.redefined = set()

        def definitions(statements):
            for statement in statements:
                self.redefined.update(statement.defines())
                if isinstance(statement, (Tile, Loop, StridedLoop)):
                    definitions(statement.body)
                elif isinstance(statement, Cond):
                    definitions(statement.body)
                    definitions(statement.else_body)

        definitions(body)
        self._collect(body, ranges)

    def _collect(self, body, ranges):
        # A user leaf can shadow an axis name or mutate an alias of its register.
        # Kill such names for the whole kernel, including prior loop iterations.
        # This deliberately sacrifices precision rather than guessing at scope.
        ranges = {k: v for k, v in ranges.items() if k not in self.redefined}
        for statement in body:
            previous = self.ranges.get(id(statement))
            self.ranges[id(statement)] = ranges.copy() if previous is None else {
                key: (min(previous[key][0], ranges[key][0]), max(previous[key][1], ranges[key][1]))
                for key in previous.keys() & ranges.keys()
            }
            if isinstance(statement, TmaDescriptor):
                old = self.descriptors.get(statement.name)
                if old is not None and old != statement:
                    raise self.error(f"inconsistent tensor descriptor {statement.name!r}")
                self.descriptors[statement.name] = statement
            if isinstance(statement, Tile):
                nested = ranges.copy()
                nested.update({a.name: (0, self.extent(a.extent) - 1) for a in statement.axes})
                self._collect(statement.body, nested)
            elif isinstance(statement, (Loop, StridedLoop)):
                nested = ranges.copy()
                # A StridedLoop may start outside its nominal axis extent; do
                # not import the ordinary counted-loop proof into that case.
                if isinstance(statement, StridedLoop):
                    nested.pop(statement.axis.name, None)
                else:
                    nested[statement.axis.name] = (0, self.extent(statement.axis.extent) - 1)
                self._collect(statement.body, nested)
            elif isinstance(statement, Cond):
                self._collect(statement.body, ranges.copy())
                self._collect(statement.else_body, ranges.copy())

    def _has_unsigned_builtin(self, expr):
        if isinstance(expr, Builtin):
            return self.builtin_operand(expr.name) != "WARP_SZ"
        if isinstance(expr, BinaryExpr):
            return self._has_unsigned_builtin(expr.left) or self._has_unsigned_builtin(expr.right)
        return False

    def _interval(self, expr, ranges):
        """Small signed-int32 proof domain: literals, axes, +,-,*,/,%.

        Each intermediate must fit signed int32, matching these address
        expressions' normal IR promotion.  Unknown values and overflow fail
        closed.  The domain intentionally does not guess from sample inputs.
        """
        result = None
        if isinstance(expr, int) and not isinstance(expr, bool):
            result = (expr, expr)
        elif isinstance(expr, Literal) and self.dtype(expr.dtype) == "s32":
            result = (int(expr.value), int(expr.value))
        elif isinstance(expr, Var):
            result = ranges.get(expr.name)
        elif isinstance(expr, Builtin):
            result = self.builtin_ranges.get(self.builtin_operand(expr.name))
        elif isinstance(expr, BinaryExpr):
            a, b = self._interval(expr.left, ranges), self._interval(expr.right, ranges)
            if expr.op == "+":
                result = a[0] + b[0], a[1] + b[1]
            elif expr.op == "-":
                result = a[0] - b[1], a[1] - b[0]
            elif expr.op == "*":
                products = [x * y for x in a for y in b]
                result = min(products), max(products)
            elif expr.op in ("/", "//") and a[0] >= 0 and b[0] == b[1] and b[0] > 0:
                result = a[0] // b[0], a[1] // b[0]
            elif expr.op == "%" and a[0] >= 0 and b[0] == b[1] and b[0] > 0:
                result = (0, min(a[1], b[0] - 1))
            elif expr.op == "&" and a[0] >= 0 and b[0] == b[1] and b[0] >= 0:
                result = (0, min(a[1], b[0]))
        if (result is None or not (-2**31 <= result[0] <= result[1] < 2**31)
                or (result[0] < 0 and self._has_unsigned_builtin(expr))):
            raise self.error("bulk copy requires provable, overflow-free int32 index bounds")
        return result

    def _flat_interval(self, name, indices, ranges):
        shape = self.c.buffers[name].shape
        if not indices:
            return 0, 0
        terms = [self._interval(x, ranges) for x in indices]
        strides = (
            [math.prod(shape[i + 1:]) for i in range(len(shape))]
            if len(indices) == len(shape) else [1] * len(indices)
        )
        lower = upper = 0
        for term, stride in zip(terms, strides):
            lo, hi = term[0] * stride, term[1] * stride
            lower, upper = lower + lo, upper + hi
            if not (-2**31 <= min(lo, lower) <= max(hi, upper) < 2**31):
                raise self.error("bulk shared index must not overflow int32 during flattening")
        return lower, upper

    def _barrier_address(self, name, slot, ranges):
        buf = self.c.buffers.get(name)
        if buf is None or buf.space != "shared" or buf.type not in ("u64", "s64", "b64"):
            raise self.error("mbarrier storage must be a declared shared 64-bit integer buffer")
        index = Literal(0, "int") if slot is None else slot
        lower, upper = self._interval(index, ranges)
        if lower < 0 or upper >= math.prod(buf.shape) or buf.align < 8:
            raise self.error("mbarrier slot must be in bounds and 8-byte aligned")
        return self.c.address(name, (index,))

    def _descriptor(self, descriptor):
        c = self.c
        src = c.buffers.get(descriptor.src_buf)
        shape, box = tuple(descriptor.src_shape), tuple(descriptor.box_extents)
        if descriptor.swizzle != "NONE":
            raise self.error("bulk tensor boxes support only NONE swizzle")
        if not 1 <= len(shape) <= 5 or len(box) != len(shape):
            raise self.error("bulk tensor boxes require matching rank 1 through 5")
        if any(type(x) is not int or x <= 0 for x in (*shape, *box)):
            raise self.error("bulk tensor boxes require positive static extents")
        if math.prod(shape) >= 2**31:
            raise self.error("bulk tensor source must contain fewer than 2**31 elements")
        if src is None or src.space != "global" or src.shape != shape:
            raise self.error("bulk tensor descriptor shape must match its global source buffer")
        if self.dtype(descriptor.dtype) != src.type:
            raise self.error("bulk tensor descriptor dtype must match its source buffer")
        if any(b > n for b, n in zip(box, shape)):
            raise self.error("bulk tensor box exceeds its source shape; boundary zero-fill is unsupported")
        row_bytes = box[-1] * self.size[src.type]
        if row_bytes % 16 or row_bytes == 0:
            raise self.error("bulk tensor box rows must contain a multiple of 16 bytes")
        if math.prod(box) * self.size[src.type] >= 2**20:
            raise self.error("bulk tensor box exceeds the supported transaction byte count")
        # Avoid a pathological compile-time expansion; the enclosing schedule
        # can divide a large box into more loads without changing the ABI.
        if math.prod(box[:-1]) > 1024:
            raise self.error("bulk tensor box exceeds 1024 contiguous row segments")
        return src, shape, box, row_bytes

    def _load(self, statement, ranges):
        c = self.c
        descriptor = self.descriptors.get(statement.desc)
        if descriptor is None:
            raise self.error(f"unknown tensor descriptor {statement.desc!r}")
        src, shape, box, row_bytes = self._descriptor(descriptor)
        dst = c.buffers.get(statement.smem)
        if dst is None or dst.space != "shared" or dst.type != src.type:
            raise self.error("bulk tensor destination must be shared memory of the source dtype")
        if len(statement.coords) != len(shape):
            raise self.error("bulk tensor coordinate rank differs from the descriptor")
        for coord, n, width in zip(statement.coords, shape, box):
            lower, upper = self._interval(coord, ranges)
            if lower < 0 or upper + width > n:
                raise self.error("bulk tensor box is not provably in bounds; boundary zero-fill is unsupported")
        lower, upper = self._flat_interval(statement.smem, statement.smem_index, ranges)
        if lower < 0 or upper + math.prod(box) > min(math.prod(dst.shape), 2**31 - 1):
            raise self.error("bulk tensor destination box is not provably in bounds")
        barrier = self._barrier_address(statement.mbar, statement.mbar_slot, ranges)
        src_base = c.flat_index(descriptor.src_buf, statement.coords)
        dst_base = c.flat_index(statement.smem, statement.smem_index)
        strides = [math.prod(shape[i + 1:]) for i in range(len(shape) - 1)]
        for row, offsets in enumerate(itertools.product(*(range(n) for n in box[:-1]))):
            src_delta = sum(x * stride for x, stride in zip(offsets, strides))
            source = c.address(descriptor.src_buf, flat=c.binary("+", src_base, c.literal(src_delta)))
            destination = c.address(statement.smem, flat=c.binary("+", dst_base, c.literal(row * box[-1])))
            if source.multiple % 16 or destination.multiple % 16:
                raise self.error("bulk copy source and destination must be provably 16-byte aligned")
            c.emit(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                f"[{destination}], [{source}], {row_bytes}, [{barrier}];"
            )

    def handles(self, statement):
        if not isinstance(statement, BULK_NODES):
            return False
        if int(re.search(r"[0-9]+", self.c.target).group()) < 90:
            raise self.error("bulk copies and transaction barriers require sm_90 or newer")
        c = self.c
        ranges = self.ranges.get(id(statement), {})
        if isinstance(statement, TmaDescriptor):
            self._descriptor(statement)
            return True  # The descriptor is resolved into pointer arithmetic.
        if isinstance(statement, TmaLoad):
            self._load(statement, ranges)
            return True
        address = self._barrier_address(statement.mbar, statement.slot, ranges)
        if isinstance(statement, MbarrierInit):
            if type(statement.count) is not int or not 1 <= statement.count < 2**20:
                raise self.error("mbarrier initialization count must be in [1, 2**20)")
            c.emit(f"mbarrier.init.shared::cta.b64 [{address}], {statement.count};")
            # Publish initialization to the asynchronous copy proxy.  The IR's
            # following CTA barrier still publishes it to other issuing threads.
            c.emit("fence.proxy.async.shared::cta;")
        elif isinstance(statement, MbarrierArriveExpectTx):
            if type(statement.bytes_) is not int or not 0 <= statement.bytes_ < 2**20:
                raise self.error("mbarrier transaction byte count must be in [0, 2**20)")
            c.emit(f"mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [{address}], {statement.bytes_};")
        elif isinstance(statement, MbarrierArrive):
            c.emit(f"mbarrier.arrive.release.cta.shared::cta.b64 _, [{address}];")
        elif isinstance(statement, MbarrierWait):
            lower, upper = self._interval(statement.phase, ranges)
            if lower < 0 or upper > 1:
                raise self.error("mbarrier wait phase parity must be provably zero or one")
            phase = c.convert(c.expr(statement.phase), "u32")
            done, retry = c.reg("pred"), c.label("bulk_wait")
            c.mark(retry)
            c.emit(f"mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 {done}, [{address}], {phase};")
            c.emit(f"@!{done} bra {retry};")
        return True
