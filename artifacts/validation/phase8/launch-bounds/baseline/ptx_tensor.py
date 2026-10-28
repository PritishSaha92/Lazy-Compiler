"""Direct PTX lowering for Emmy's explicit m16n8k16 register fragments.

The scheduled IR determines operand layout, storage, and work ownership. This
module does not recognize a matrix multiplication expression or reschedule it.
Each fragment holds the PTX-defined per-lane registers. Family indexing uses
register selects and predicated assignments, including when the index is live.
"""

from dataclasses import dataclass
import math
import re

from emmy.compiler.ir.kernel.ir import (
    ELEM_COL, ELEM_ROW, FragmentRepack, LdmatrixLoad, MmaSyncPtx,
    RegFragment, RegStore, Tile,
)
from emmy.compiler.ir.stmt.leaves import Assign, Load, Select
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var

NODE_TYPES = (RegFragment, LdmatrixLoad, MmaSyncPtx, RegStore, FragmentRepack)
SHAPE = (16, 8, 16)


@dataclass(frozen=True)
class Fragment:
    role: str
    dtype: str
    members: tuple


class TensorLowering:
    """An extension of the scalar compiler; imports its errors only at runtime."""

    def __init__(self, compiler):
        from ptx_backend import CompileError, dtype, extent
        self.c = compiler
        self.error, self.dtype, self.extent = CompileError, dtype, extent
        self.induction_facts = {}
        self.prepare(compiler.kernel.body)

    def prepare(self, body):
        c = self.c
        if int(re.search(r"\d+", c.target).group()) < 80:
            raise self.error("m16n8k16 tensor cores require sm_80 or newer")
        block = c.meta.get("block", (32, 1, 1))
        if block[0] % 32 or math.prod(block[1:]) != 1:
            raise self.error("tensor cores require complete one-dimensional warps")
        if math.prod(c.meta.get("grid",(1,1,1))[1:]) != 1:
            raise self.error("tensor cores currently require a one-dimensional grid")

        # An axis can be rebound after a load, invalidating its alignment on
        # the next iteration. Ignore induction facts for any name explicitly
        # defined by a leaf anywhere, rather than attempting loop dataflow.
        leaf_writes = set()
        def collect_writes(statements):
            for s in statements:
                if not isinstance(s,(Loop,StridedLoop,Tile,Cond)):
                    leaf_writes.update(s.defines())
                for attr in ("body","else_body"):
                    nested = getattr(s,attr,())
                    if nested:
                        collect_writes(nested)
        collect_writes(body)

        def visit(statements, tiles=(), facts=None):
            facts = {} if facts is None else dict(facts)
            for s in statements:
                active = tiles + (s,) if isinstance(s, Tile) else tiles
                inner_facts = dict(facts)
                if isinstance(s, (Loop, StridedLoop)):
                    multiple = 1
                    if isinstance(s, StridedLoop) and isinstance(s.start, Literal) and isinstance(s.step, Literal):
                        if isinstance(s.start.value,int) and isinstance(s.step.value,int):
                            multiple = math.gcd(s.start.value,s.step.value,2**32)
                    if s.axis.name in leaf_writes:
                        multiple = 1
                    inner_facts[s.axis.name] = multiple
                if isinstance(s, LdmatrixLoad):
                    # Frozen IR nodes may be reused in distinct loop scopes.
                    # Retain only facts true at every occurrence of that node.
                    previous = self.induction_facts.get(id(s))
                    if previous is None:
                        self.induction_facts[id(s)] = dict(facts)
                    else:
                        self.induction_facts[id(s)] = {
                            name: math.gcd(previous.get(name,1),facts.get(name,1))
                            for name in previous.keys() | facts.keys()
                        }
                if isinstance(s, NODE_TYPES):
                    for tile in tiles:
                        if math.prod(self.extent(a.extent) for a in tile.axes) % 32:
                            raise self.error("tensor Tile work count must contain complete warps")
                        if tile.aux_threads and (not tile.block_threads or
                                tile.block_threads % 32 or tile.aux_threads % 32):
                            raise self.error("tensor auxiliary Tile partition must contain complete warps")
                for attr in ("body", "else_body"):
                    nested = getattr(s, attr, ())
                    if nested:
                        visit(nested, active, inner_facts)
                # A body-local rebinding of an axis invalidates its induction
                # fact for later statements in this scope.
                for name in s.defines():
                    if name in facts:
                        facts[name] = 1
        visit(body)

    def expression_multiple(self, expr, facts):
        """Conservative power-of-two alignment proof for explicit loop indices."""
        if isinstance(expr, Literal) and isinstance(expr.value,int):
            return abs(expr.value)
        if isinstance(expr, Var):
            return facts.get(expr.name,1)
        if isinstance(expr, BinaryExpr):
            a,b = self.expression_multiple(expr.left,facts),self.expression_multiple(expr.right,facts)
            if expr.op in ("+","-"):
                return math.gcd(a,b,2**32)
            if expr.op == "*":
                return math.gcd(a*b,2**32)
        return 1

    def base_multiple(self, s):
        shape = self.c.buffers[s.src_buffer].shape
        facts = self.induction_facts.get(id(s),{})
        terms=[]
        for i,expr in enumerate(s.src_index):
            stride = math.prod(shape[i+1:]) if len(s.src_index)==len(shape) else 1
            terms.append(self.expression_multiple(expr,facts)*stride)
        return math.gcd(*terms) if terms else 0

    def handles(self, s):
        methods = {RegFragment: self.declare, LdmatrixLoad: self.load,
                   MmaSyncPtx: self.mma, RegStore: self.store,
                   FragmentRepack: self.repack}
        method = methods.get(type(s))
        if method is None:
            return False
        method(s)
        return True

    def check_shape(self, s):
        if tuple(s.shape) != SHAPE:
            raise self.error(f"tensor shape {s.shape!r} unsupported; expected {SHAPE}")

    def check_layout(self, s):
        if s.fragment_layout != "m16n8k16":
            raise self.error(f"tensor fragment layout {s.fragment_layout!r} unsupported")

    def declare(self, s):
        c = self.c
        self.check_shape(s)
        dt = self.dtype(s.dtype)
        expected = {"a": 4, "b": 2, "c": 4}.get(s.role)
        if expected is None or (s.role == "c" and dt != "f32") or (
                s.role != "c" and dt not in ("f16", "bf16")):
            raise self.error("tensor fragments require f16/bf16 A/B and f32 C")
        if s.nregs not in (None, expected):
            raise self.error(f"tensor fragment requires {expected} registers")
        if not isinstance(s.count, int) or not 1 <= s.count <= 64:
            raise self.error("tensor fragment family count must be between 1 and 64")
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", s.name, re.ASCII):
            raise self.error(f"invalid tensor fragment name {s.name!r}")
        members = tuple(tuple(c.mutable_reg("f32" if s.role == "c" else "b32")
                              for _ in range(expected)) for _ in range(s.count))
        if s.role == "c":
            for member in members:
                for value in member:
                    c.move(value, c.literal(0, "f32"))
        c.env[s.name] = Fragment(s.role, dt, members)

    def reference(self, text, role=None):
        """Resolve a fragment stem and optional integer/SSA family subscript."""
        c = self.c
        match = re.fullmatch(r"([A-Za-z_$][\w$]*)(?:\[([A-Za-z_$][\w$]*|[0-9]+)\])?",
                             text, re.ASCII)
        if match is None:
            raise self.error(f"unsupported tensor fragment reference {text!r}")
        stem, subscript = match.groups()
        fragment = c.env.get(stem)
        if not isinstance(fragment, Fragment):
            raise self.error(f"undeclared tensor fragment {stem!r}")
        if role is not None and fragment.role != role:
            raise self.error(f"tensor fragment {stem!r} must have role {role}")
        if subscript is None:
            if len(fragment.members) != 1:
                raise self.error(f"tensor family {stem!r} requires a subscript")
            index = c.literal(0)
        elif subscript.isdecimal():
            index = c.literal(int(subscript))
        else:
            index = c.lookup(subscript)
            if index.type not in ("s32", "u32", "s64", "u64"):
                raise self.error("tensor family subscript must be an integer")
        if index.constant is not None and not 0 <= index.constant < len(fragment.members):
            raise self.error(f"tensor family index {index.constant} outside {stem!r}")
        return fragment, index

    def read(self, text, role=None):
        c = self.c
        fragment, index = self.reference(text, role)
        if index.constant is not None:
            return fragment, fragment.members[int(index.constant)]
        values = tuple(c.reg(v.type) for v in fragment.members[0])
        for value in values:
            c.move(value,c.literal(0,value.type))
        for i, member in enumerate(fragment.members):
            pred = c.compare("eq", index, c.literal(i, index.type))
            for value, source in zip(values, member):
                c.emit(f"@{pred} mov.{value.type} {value}, {source};")
        return fragment, values

    def write(self, text, values, role=None):
        c = self.c
        fragment, index = self.reference(text, role)
        if len(values) != len(fragment.members[0]):
            raise self.error("tensor register count mismatch")
        if index.constant is not None:
            for target, source in zip(fragment.members[int(index.constant)], values):
                c.move(target, source)
        else:
            for i, member in enumerate(fragment.members):
                pred = c.compare("eq", index, c.literal(i, index.type))
                for target, source in zip(member, values):
                    if target.type != source.type:
                        raise self.error("tensor assignment register type mismatch")
                    c.emit(f"@{pred} mov.{target.type} {target}, {source};")

    def lane(self):
        return self.c.binary("&", self.c.lookup("_tid"), self.c.literal(31))

    def add(self, a, b):
        return self.c.binary("+", a, b if hasattr(b, "type") else self.c.literal(b))

    def mul(self, a, b):
        return self.c.binary("*", a, b if hasattr(b, "type") else self.c.literal(b))

    def pack(self, lo, hi):
        c = self.c
        dst = c.reg("b32")
        c.emit(f"mov.b32 {dst}, {{{lo}, {hi}}};")
        return dst

    def load(self, s):
        c = self.c
        self.check_layout(s)
        if s.role not in ("a", "b") or (s.role == "a" and s.b_trans):
            raise self.error("invalid tensor operand role/orientation")
        if s.byte_slab or s.scale_buffer is not None or s.scale_index or s.scale_ldm:
            raise self.error("byte/block-scaled tensor loads are unsupported")
        fragment, _ = self.reference(s.frag, s.role)
        buf = c.buffers[s.src_buffer]
        ldm = s.ldm or buf.shape[-1]
        if not isinstance(ldm, int) or ldm <= 0:
            raise self.error("tensor load row stride must be a positive static integer")
        if s.staged:
            if buf.space != "shared" or buf.type != fragment.dtype:
                raise self.error("ldmatrix requires matching 16-bit shared-memory storage")
            if s.gmem_guard is not None or s.k_zero is not None:
                raise self.error("staged tensor loads require padding in the shared producer")
            if ldm % 8 or buf.align < 16:
                raise self.error("ldmatrix requires 16-byte aligned rows and shared storage")
            self.shared_load(s, fragment, ldm)
        else:
            if buf.space != "global" or s.swizzle != "NONE" or s.pair_frag is not None:
                raise self.error("global tensor loads do not support swizzles or paired fragments")
            if buf.type not in ("f16", "bf16", "f32", "f64"):
                raise self.error("tensor gather source must have a floating dtype")
            self.global_load(s, fragment, ldm)

    def shared_load(self, s, fragment, ldm):
        c = self.c
        lane = self.lane()
        if s.pair_frag is not None:
            other, _ = self.reference(s.pair_frag, "b")
            if s.role != "b" or other.dtype != fragment.dtype or s.swizzle != "NONE":
                raise self.error("paired ldmatrix requires two matching unswizzled B fragments")
            if s.b_trans:
                row = self.add(c.binary("%", lane, c.literal(8)),
                               self.mul(c.binary("/", lane, c.literal(16)), 8))
                col = self.mul(c.binary("%", c.binary("/", lane, c.literal(8)), c.literal(2)), 8)
            else:
                row = c.binary("%", lane, c.literal(16))
                col = self.mul(c.binary("/", lane, c.literal(16)), 8)
            count = 4
        elif s.role == "a":
            row = c.binary("%", lane, c.literal(16))
            col = self.mul(c.binary("/", lane, c.literal(16)), 8)
            count = 4
        elif s.b_trans:
            row = c.binary("%", lane, c.literal(8))
            col = self.mul(c.binary("%", c.binary("/", lane, c.literal(8)), c.literal(2)), 8)
            count = 2
        else:
            row, col = c.binary("%", lane, c.literal(16)), c.literal(0)
            count = 2
        flat = self.add(c.flat_index(s.src_buffer, s.src_index), self.add(self.mul(row, ldm), col))
        address = c.address(s.src_buffer, flat=flat, swizzle=s.swizzle)
        # The scalar compiler intentionally does not attach invariants to a
        # mutable induction register. Prove the atom base separately from the
        # explicit loop start/step; XOR swizzles preserve its low three bits.
        if address.multiple % 16 and self.base_multiple(s) % 8:
            raise self.error("ldmatrix row-address alignment cannot be proved")
        values = tuple(c.reg("b32") for _ in range(count))
        trans = ".trans" if s.role == "b" and not s.b_trans else ""
        c.emit(f"ldmatrix.sync.aligned.m8n8.x{count}{trans}.shared.b16 "
               f"{{{', '.join(map(str, values))}}}, [{address}];")
        if s.pair_frag is None:
            self.write(s.frag, values, s.role)
        else:
            self.write(s.frag, values[:2], "b")
            self.write(s.pair_frag, values[2:], "b")

    def global_load(self, s, fragment, ldm):
        c = self.c
        lane = self.lane()
        group = c.binary("/", lane, c.literal(4))
        pair = self.mul(c.binary("%", lane, c.literal(4)), 2)
        base = c.flat_index(s.src_buffer, s.src_index)
        remaining_mn = (c.binary("-", c.expr(s.gmem_guard[1]), c.expr(s.gmem_guard[0]))
                        if s.gmem_guard is not None else None)
        remaining_k = (c.binary("-", c.expr(s.k_zero[1]), c.expr(s.k_zero[0]))
                       if s.k_zero is not None else None)
        values = []
        for i in range(4 if s.role == "a" else 2):
            mn = self.add(group, 8 if s.role == "a" and i & 1 else 0)
            k = self.add(pair, 8 if (i & 2 if s.role == "a" else i) else 0)
            valid_mn = None
            if remaining_mn is not None:
                # A completely overhanging fragment has no consumers. Zero it
                # without reading the invalid base; partial tiles clamp M/N.
                valid_mn = c.compare("gt", remaining_mn, c.literal(0))
                last = c.operation("maximum", [self.add(remaining_mn, -1), c.literal(0)], "s32")
                mn = c.operation("minimum", [mn, last], "s32")
            halves = []
            for half in (0, 1):
                kh = self.add(k, half)
                row, col = ((mn, kh) if s.role == "a" or s.b_trans else (kh, mn))
                address = c.address(s.src_buffer, flat=self.add(base, self.add(self.mul(row, ldm), col)))
                valid = valid_mn
                if remaining_k is not None:
                    valid_k = c.compare("lt", kh, remaining_k)
                    valid = valid_k if valid is None else c.binary("&&", valid, valid_k)
                source_type = c.buffers[s.src_buffer].type
                value = c.reg(source_type)
                memdt = "b16" if source_type in ("f16", "bf16") else source_type
                if valid is not None:
                    c.move(value,c.literal(0,source_type))
                prefix = f"@{valid} " if valid is not None else ""
                c.emit(f"{prefix}ld.global.{memdt} {value}, [{address}];")
                halves.append(c.convert(value, fragment.dtype))
            values.append(self.pack(*halves))
        self.write(s.frag, values, s.role)

    def mma(self, s):
        c = self.c
        self.check_shape(s)
        if s.ab_dtype not in ("f16", "bf16") or s.c_dtype != "f32":
            raise self.error("tensor MMA supports f16/bf16 operands with f32 accumulation")
        if s.b_row_major or s.sfa_frag is not None or s.sfb_frag is not None:
            raise self.error("row-major B and block-scaled MMA forms are unsupported")
        a, ar = self.read(s.a_frag, "a")
        b, br = self.read(s.b_frag, "b")
        acc, cr = self.read(s.c_frag, "c")
        if a.dtype != s.ab_dtype or b.dtype != s.ab_dtype or acc.dtype != "f32":
            raise self.error("MMA instruction dtype disagrees with its fragment declarations")
        out = tuple(c.reg("f32") for _ in range(4))
        vector = lambda regs: "{" + ", ".join(map(str, regs)) + "}"
        c.emit(f"mma.sync.aligned.m16n8k16.row.col.f32.{s.ab_dtype}.{s.ab_dtype}.f32 "
               f"{vector(out)}, {vector(ar)}, {vector(br)}, {vector(cr)};")
        self.write(s.c_frag, out, "c")

    def store(self, s):
        c = self.c
        self.check_shape(s)
        self.check_layout(s)
        if s.volta_interleaved:
            raise self.error("Volta interleaved tensor stores are unsupported")
        _, values = self.read(s.frag, "c")
        extra = [self.read(name, "c")[1] for name in s.extra_frags]
        if s.epilogue is None and extra:
            raise self.error("extra tensor fragments require an epilogue")
        buf = c.buffers[s.dst_buffer]
        if buf.type not in ("f16", "bf16", "f32", "f64"):
            raise self.error("tensor output must have a floating dtype")
        if s.atomic and buf.type not in ("f32", "f64"):
            raise self.error("tensor atomic output currently supports f32/f64 only")
        for dim in (s.row_dim, s.col_dim):
            if dim is not None and not 0 <= dim < len(buf.shape):
                raise self.error("tensor output dimension is outside its buffer rank")
        row_stride = s.ldm or (math.prod(buf.shape[s.row_dim+1:])
                              if s.row_dim is not None else buf.shape[-1])
        col_stride = math.prod(buf.shape[s.col_dim+1:]) if s.col_dim is not None else 1
        if row_stride <= 0 or col_stride <= 0:
            raise self.error("tensor output strides must be positive")
        base = c.flat_index(s.dst_buffer, s.dst_index)
        lane = self.lane()
        group = c.binary("/", lane, c.literal(4))
        pair = self.mul(c.binary("%", lane, c.literal(4)), 2)
        for i, value in enumerate(values):
            row, col = self.add(group, 8 if i >= 2 else 0), self.add(pair, i & 1)
            predicates = []
            for guard, offset in ((s.m_guard, row), (s.n_guard, col)):
                if guard is not None:
                    predicates.append(c.compare("lt", self.add(c.expr(guard[0]), offset), c.expr(guard[1])))
            end = None
            if predicates:
                valid = predicates[0]
                for pred in predicates[1:]:
                    valid = c.binary("&&", valid, pred)
                end = c.label("tensor_store_skip")
                c.emit(f"@!{valid} bra {end};")
            if s.epilogue is not None:
                value = self.epilogue(s.epilogue, [value] + [v[i] for v in extra], row, col)
            address = c.address(s.dst_buffer, flat=self.add(base, self.add(
                self.mul(row, row_stride), self.mul(col, col_stride))), swizzle=s.swizzle)
            c.store(s.dst_buffer, address, [value], s.atomic)
            if end is not None:
                c.mark(end)

    def epilogue(self, epi, inputs, row, col):
        c = self.c
        if len(epi.params) < len(inputs) or len(epi.results) != 1:
            raise self.error("tensor epilogue must bind its fragments and return one scalar")
        saved = c.env.copy()
        try:
            c.env.update(zip(epi.params, inputs))
            c.env[ELEM_ROW], c.env[ELEM_COL] = row, col
            # Lambda.closing appends captured coordinates after the fragment
            # parameters. They must have an explicit enclosing value, rather
            # than relying on a missing binding to fail deep in a load.
            for param in epi.params[len(inputs):]:
                if param not in c.env or isinstance(c.env[param],Fragment):
                    raise self.error(f"unbound tensor epilogue coordinate {param!r}")
            for statement in epi.body:
                if isinstance(statement, Load):
                    if len(statement.names) != 1:
                        raise self.error("tensor epilogue loads must be scalar")
                    value = c.load(statement.input, c.address(statement.input, statement.index))[0]
                    c.env[statement.names[0]] = c.convert(value, "f32")
                elif isinstance(statement, (Assign, Select)):
                    c.statement(statement)
                else:
                    raise self.error(f"unsupported tensor epilogue statement {type(statement).__name__}")
            return c.lookup(epi.results[0])
        finally:
            c.env = saved

    def repack(self, s):
        c = self.c
        self.check_layout(s)
        if s.part != 0 or len(s.srcs) != 2 or s.ab_dtype not in ("f16", "bf16"):
            raise self.error("tensor repack requires two C fragments and a 16-bit A dtype")
        fragment, _ = self.reference(s.frag, "a")
        if fragment.dtype != s.ab_dtype:
            raise self.error("tensor repack dtype disagrees with its A declaration")
        values = []
        for source in s.srcs:
            _, scalars = self.read(source, "c")
            for pair in (0, 2):
                values.append(self.pack(c.convert(scalars[pair], s.ab_dtype),
                                        c.convert(scalars[pair+1], s.ab_dtype)))
        self.write(s.frag, values, "a")
