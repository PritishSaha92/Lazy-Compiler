"""A direct, deterministic lowering of scheduled Emmy 0.3.11 Kernel IR.

Emmy supplies typed deserialization only. All instruction selection, addresses,
control flow, state lifetimes and cooperative operations are lowered here.
Virtual registers are deliberately plentiful: ptxas owns physical allocation.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    FlatIndex,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    FragmentRepack,
    KernelOp,
    LdmatrixLoad,
    MbarrierArrive,
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    MmaSyncPtx,
    RegFragment,
    RegStore,
    Smem,
    Sync,
    Tile,
    TmaDescriptor,
    TmaLoad,
    TreeHalve,
    WarpShuffle,
)
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Select, Write


class CompileError(ValueError):
    """An unsupported or inconsistent IR, diagnosed before printing any PTX."""


_BUILTIN_OPERANDS = {
    f"{spelling}.{axis}": f"%{register}.{axis}"
    for logical, cuda, register in (
        ("thread_idx", "threadIdx", "tid"),
        ("block_idx", "blockIdx", "ctaid"),
        ("block_dim", "blockDim", "ntid"),
        ("grid_dim", "gridDim", "nctaid"),
    )
    for spelling in (logical, cuda)
    for axis in "xyz"
}
_BUILTIN_OPERANDS.update(warp_size="WARP_SZ", warpSize="WARP_SZ")


def builtin_operand(name):
    """Resolve Emmy's logical GPU names and retained CUDA-spelling aliases.

    Emmy 0.3.11 emits names such as ``thread_idx.x``. The same resolver is
    used by instruction emission and the bulk-copy address bounds proof.
    WARP_SZ is PTX's predefined immediate, not a percent-prefixed register.
    """
    try:
        return _BUILTIN_OPERANDS[name]
    except KeyError as exc:
        raise CompileError(f"unknown builtin {name}") from exc


_TYPES = {
    "float": "f32",
    "double": "f64",
    "int": "s32",
    "long long": "s64",
    "unsigned int": "u32",
    "unsigned long long": "u64",
    "__half": "f16",
    "half": "f16",
    "__nv_bfloat16": "bf16",
    "bool": "pred",
    "char": "s8",
    "unsigned char": "u8",
    "short": "s16",
    "unsigned short": "u16",
    "i8": "s8",
    "i16": "s16",
    "i32": "s32",
    "i64": "s64",
}
_SIZE = {
    "pred": 1,
    "s8": 1,
    "u8": 1,
    "s16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "s32": 4,
    "u32": 4,
    "b32": 4,
    "f32": 4,
    "s64": 8,
    "u64": 8,
    "b64": 8,
    "f64": 8,
}
_REG = {
    "pred": ("p", "pred"),
    "f32": ("f", "f32"),
    "f64": ("d", "f64"),
    "f16": ("h", "b16"),
    "bf16": ("h", "b16"),
    "s8": ("r", "b32"),
    "u8": ("r", "b32"),
    "s16": ("r", "b32"),
    "u16": ("r", "b32"),
    "s32": ("r", "b32"),
    "u32": ("r", "b32"),
    "b32": ("r", "b32"),
    "s64": ("rd", "b64"),
    "u64": ("rd", "b64"),
    "b64": ("rd", "b64"),
}

# Admission is separate from instruction selection: recognizing a storage dtype
# does not make every instruction legal at that dtype. Assign/Accum/FuncCallExpr
# share this table; BinaryExpr additionally applies C expression promotions.
_INTEGERS = frozenset(("s8", "u8", "s16", "u16", "s32", "u32", "s64", "u64"))
_FLOATS = frozenset(("f16", "bf16", "f32", "f64"))
_NUMERIC = _INTEGERS | _FLOATS
_SCALARS = _NUMERIC | {"pred"}
_ELEMENTWISE_CAPABILITIES = {
    **{name: (1, _SCALARS) for name in ("copy", "pad", "logical_not")},
    "bitcast": (1, _NUMERIC | {"b32", "b64"}),
    "where": (3, _SCALARS),
    **{name: (2, _SCALARS) for name in (
        "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
        "logical_and", "logical_or",
    )},
    **{name: (2, _NUMERIC) for name in (
        "add", "subtract", "multiply", "divide", "floor_divide", "remainder",
        "mod", "maximum", "minimum", "fmax", "fmin",
    )},
    **{name: (1, _NUMERIC) for name in (
        "negative", "abs", "square", "reciprocal", "relu",
    )},
    # Historical f32 bitwise-and/or/not spellings represent logical masks.
    **{name: (2, _INTEGERS | {"pred", "f32"}) for name in ("bitwise_and", "bitwise_or")},
    "bitwise_not": (1, _INTEGERS | {"pred", "f32"}),
    **{name: (2, _INTEGERS) for name in ("bitwise_xor", "left_shift", "right_shift")},
    "bitwise_count": (1, _INTEGERS),
    "fma": (3, _FLOATS),
    **{name: (1, _FLOATS) for name in ("sqrt", "rsqrt", "floor", "ceil", "trunc", "rint")},
    **{name: (1, _FLOATS - {"f64"}) for name in (
        "exp", "exp2", "log", "log2", "log10", "sin", "cos", "tanh", "erf",
        "sigmoid", "silu",
    )},
    "power": (2, _FLOATS - {"f64"}),
}


def dtype(value, default="f32") -> str:
    name = getattr(value, "name", value) or default
    name = _TYPES.get(name, name)
    if name not in _SIZE:
        raise CompileError(f"unsupported dtype {name!r}")
    return name


def extent(dim) -> int:
    return int(dim.as_static()) if hasattr(dim, "as_static") else int(dim)


def walk(body):
    for stmt in body:
        yield stmt
        if isinstance(stmt, (Loop, StridedLoop, Tile)):
            yield from walk(stmt.body)
        elif isinstance(stmt, Cond):
            yield from walk(stmt.body)
            yield from walk(stmt.else_body)


@dataclass(frozen=True)
class Value:
    text: str
    type: str
    constant: int | float | None = None
    multiple: int = (
        1  # proven integer divisibility; zero is divisible by every alignment
    )

    def __str__(self):
        return self.text


@dataclass
class Buffer:
    shape: tuple[int, ...]
    type: str
    space: str
    pointer: Value | None = None
    symbol: str = ""
    offset: int = 0
    align: int = 1


class Compiler:
    def __init__(
        self,
        graph,
        meta,
        *,
        target="sm_89",
        synchronous_copy=False,
        unroll_budget=0,
        fuse_accumulations=False,
        accurate_dots=False,
        compensated_dots=False,
        launch_bounds=True,
    ):
        if not re.fullmatch(r"sm_[0-9]+[af]?", target):
            raise CompileError(f"invalid PTX target {target!r}")
        self.graph, self.meta, self.target = graph, meta, target
        self.launch_bounds = launch_bounds
        self.synchronous_copy = synchronous_copy
        self.unroll_budget = unroll_budget
        self.unroll_remaining = 4096
        self.fuse_accumulations = fuse_accumulations
        nodes = [n for n in graph.nodes.values() if isinstance(n.op, KernelOp)]
        if len(nodes) != 1:
            raise CompileError(f"expected one KernelOp, found {len(nodes)}")
        self.node, self.kernel = nodes[0], nodes[0].op
        # Each native cp.async operation requires SM80, including commit/wait
        # on empty groups. Keep the explicitly requested synchronous fallback.
        if not synchronous_copy and int(re.search(r"[0-9]+", target).group()) < 80:
            if any(isinstance(stmt, (CpAsyncCopy, CpAsyncCommit, CpAsyncWait))
                   for stmt in walk(self.kernel.body)):
                raise CompileError(
                    "native cp.async operations require sm_80 or newer; "
                    "use synchronous_copy=True for the explicit synchronous lowering"
                )
        self.wide_accumulators = (
            self.independent_dots(self.kernel.body) if accurate_dots else set()
        )
        self.compensated_accumulators = (
            self.independent_dots(self.kernel.body) if compensated_dots else set()
        )
        if accurate_dots and compensated_dots:
            raise CompileError("choose either widened or compensated dot accumulation")
        if self.kernel.serial:
            raise CompileError(
                "serial runtime axes are outside the competition contract"
            )
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", self.kernel.name, re.ASCII):
            raise CompileError(f"invalid entry identifier {self.kernel.name!r}")
        self.lines, self.declarations = [], []
        self.counts, self.regtypes = defaultdict(int), {}
        self.labels = 0
        self.env: dict[str, Value] = {}
        self.mutable: set[str] = set()
        self.corrections: dict[str, Value] = {}
        self.buffers: dict[str, Buffer] = {}
        # Optional instruction families remain isolated from scalar lowering.
        # Lazy imports keep the required vocabulary's emission byte-identical.
        self.tensor_lowering = None
        self.bulk_lowering = None
        for node in graph.nodes.values():
            for tensor in node.outputs:
                self.buffers[tensor.name] = Buffer(
                    tuple(extent(x) for x in tensor.shape),
                    dtype(tensor.dtype),
                    "global",
                    align=16,
                )
        self.smem = {}
        size = 0
        for stmt in walk(self.kernel.body):
            if isinstance(stmt, Smem):
                dt = dtype(stmt.dtype)
                align = max(stmt.align or _SIZE[dt], _SIZE[dt])
                shape = tuple(extent(x) for x in stmt.extents)
                if stmt.name in self.smem:
                    old = self.smem[stmt.name]
                    if old.shape != shape or old.type != dt or old.align != align:
                        raise CompileError(
                            f"inconsistent shared declaration {stmt.name!r}"
                        )
                    continue
                size = (size + align - 1) // align * align
                buf = Buffer(
                    shape,
                    dt,
                    "shared",
                    symbol=f"smem_{len(self.smem)}",
                    offset=size,
                    align=align,
                )
                self.smem[stmt.name] = buf
                self.buffers[stmt.name] = buf
                size += math.prod(shape) * _SIZE[dt]
        self.dynamic_smem = size > 48 * 1024
        if self.dynamic_smem:
            if int(meta.get("smem_bytes", 0)) < size:
                raise CompileError(
                    f"kernel needs {size} dynamic shared bytes; metadata provides {meta.get('smem_bytes', 0)}"
                )
            alignment = max([16] + [b.align for b in self.smem.values()])
            self.declarations.append(
                f".extern .shared .align {alignment} .b8 smem_pool[];"
            )
        else:
            for buf in self.smem.values():
                self.declarations.append(
                    f".shared .align {buf.align} .b8 {buf.symbol}[{math.prod(buf.shape) * _SIZE[buf.type]}];"
                )

    @staticmethod
    def independent_dots(body):
        """Find additive dot states without a consumer during accumulation.

        Reading a carrier anywhere in its loop, other than the implicit add
        base, disqualifies it. This conservatively excludes coupled online
        normalization and loop-carried recurrence transformations.
        """
        eligible, rejected = set(), set()
        for loop in walk(body):
            if not isinstance(loop, (Loop, StridedLoop)):
                continue
            states = {s.name for s in loop.body if isinstance(s, Accum)}
            for i, stmt in enumerate(loop.body):
                if not isinstance(stmt, Accum):
                    continue
                previous = loop.body[i - 1] if i else None
                if (
                    isinstance(previous, Assign)
                    and previous.op.name == "multiply"
                    and previous.name == stmt.value
                    and stmt.base is None
                    and stmt.op.name in ("add", "sum")
                    and dtype(stmt.dtype) == dtype(previous.dtype) == "f32"
                ):
                    eligible.add(stmt.name)
                else:
                    rejected.add(stmt.name)
            for stmt in walk(loop.body):
                # These typed-node queries expose operands only. They do not
                # render or lower code. Read all expressions, including branch
                # selectors, loop bounds and asynchronous lane indices.
                reads = set(stmt.deps())
                if isinstance(stmt, Accum):
                    reads = {stmt.value} | ({stmt.base} if stmt.base else set())
                for expr in stmt.exprs():
                    reads.update(expr.free_vars())
                rejected.update(states & reads)
        return eligible - rejected

    def emit(self, text):
        self.lines.append("    " + text)

    def label(self, name="L"):
        self.labels += 1
        return f"{name}_{self.labels}"

    def mark(self, label):
        self.lines.append(label + ":")

    def reg(self, dt):
        dt = dtype(dt)
        prefix, declaration = _REG[dt]
        self.counts[prefix] += 1
        self.regtypes[prefix] = declaration
        return Value(f"%{prefix}{self.counts[prefix]}", dt)

    def mutable_reg(self, dt):
        value = self.reg(dt)
        self.mutable.add(value.text)
        return value

    def snapshot(self, value):
        """An SSA copy must retain the old value of a later-mutated carrier."""
        if value.text in self.mutable:
            return self.instruction(
                f"mov.{self.move_type(value.type)}", value.type, value
            )
        return value

    def literal(self, x, dt="s32"):
        dt = dtype(dt)
        if dt == "f32":
            if isinstance(x, int):
                # An integer-spelled floating literal must round only once.
                # Python int -> binary64 -> binary32 can double-round near a
                # binary32 midpoint when the integer exceeds 53 significant bits.
                magnitude = abs(x)
                shift = max(0, magnitude.bit_length() - 24)
                significand, remainder = divmod(magnitude, 1 << shift)
                if shift:
                    halfway = 1 << (shift - 1)
                    if remainder > halfway or (
                        remainder == halfway and significand & 1
                    ):
                        significand += 1
                rounded = (
                    math.inf
                    if magnitude.bit_length() > 128
                    else math.ldexp(float(significand), shift)
                )
                x = -rounded if x < 0 else rounded
            try:
                bits = struct.pack(">f", float(x))
            except OverflowError:
                bits = struct.pack(">f", math.copysign(math.inf, x))
            return Value("0f" + bits.hex().upper(), dt, struct.unpack(">f", bits)[0])
        if dt == "f64":
            return Value("0d" + struct.pack(">d", float(x)).hex().upper(), dt, float(x))
        if dt in ("f16", "bf16"):
            return self.convert(self.literal(x, "f32"), dt)
        if dt == "pred":
            return self.compare("ne", self.literal(int(bool(x))), self.literal(0))
        # PTX integer operations and conversions use fixed-width bit patterns.
        # Canonicalize constants too, otherwise folding e.g. u32(-1) / 2
        # performs signed Python arithmetic instead of unsigned division.
        bits = _SIZE[dt] * 8
        number = int(x) % (1 << bits)
        if dt.startswith("s") and number >= 1 << (bits - 1):
            number -= 1 << bits
        return Value(str(number), dt, number, abs(number))

    def instruction(self, op, dt, *args):
        dst = self.reg(dt)
        operands = ", " + ", ".join(map(str, args)) if args else ""
        self.emit(f"{op} {dst}{operands};")
        return dst

    def materialize(self, value):
        if value.constant is None:
            return value
        return self.instruction(f"mov.{self.move_type(value.type)}", value.type, value)

    @staticmethod
    def move_type(dt):
        if dt in ("u8", "s8", "u16", "s16"):
            return "b32"
        return "b16" if dt in ("f16", "bf16") else dt

    def move(self, dest, source):
        source = self.convert(source, dest.type)
        if dest.text != source.text:
            self.emit(f"mov.{self.move_type(dest.type)} {dest}, {source};")

    def convert(self, value, dt):
        dt = dtype(dt)
        if value.type == dt:
            return value
        if value.type in ("b32", "b64") or dt in ("b32", "b64"):
            raise CompileError(
                f"numeric conversion {value.type} -> {dt} is unsupported; "
                "use an explicit equal-width bitcast for raw payloads"
            )
        if "bf16" in (value.type, dt):
            # PTX 8.0 on sm_80/89 has only the f32 <-> bf16 conversions.
            # Expanding a bf16 first is exact, as is expanding f16 or a small
            # integer before rounding to bf16. Do not silently double-round a
            # wide integer or f64 source through f32.
            target = int(re.search(r"[0-9]+", self.target).group())
            if target < 80:
                raise CompileError(f"conversion {value.type} -> {dt} requires sm_80 or newer")
            if value.type == "bf16" and dt != "f32":
                return self.convert(self.convert(value, "f32"), dt)
            if dt == "bf16" and value.type not in ("f32", "pred"):
                if value.type in ("f16", "s8", "u8", "s16", "u16"):
                    return self.convert(self.convert(value, "f32"), dt)
                if target < 90:
                    raise CompileError(
                        f"conversion {value.type} -> bf16 requires sm_90; "
                        "an implicit f32 intermediate would change rounding"
                    )
        if (
            value.constant is not None
            and dt not in ("f16", "bf16", "pred")
            and not (value.type.startswith("f") and dt[0] in "sub")
            and not (value.type == "f64" and dt == "f32")
            and not (
                value.type[0] in "sub" and dt == "f32" and abs(value.constant) > 2**53
            )
        ):
            return self.literal(value.constant, dt)
        if dt == "pred":
            return self.compare("ne", value, self.literal(0, value.type))
        if value.type == "pred":
            if dt == "bf16":
                return self.convert(self.convert(value, "f32"), dt)
            if dt in ("u8", "s8", "u16", "s16", "f16", "bf16"):
                return self.convert(self.convert(value, "u32"), dt)
            return self.instruction(
                f"selp.{dt}", dt, self.literal(1, dt), self.literal(0, dt), value
            )
        floats = ("f16", "bf16", "f32", "f64")
        is_float, was_float = dt in floats, value.type in floats
        if not is_float and not was_float and _SIZE[dt] == _SIZE[value.type]:
            return Value(value.text, dt, value.constant, value.multiple)
        rounding = (
            ".rn"
            if is_float
            and (not was_float or _SIZE[dt] < _SIZE[value.type] or dt == "bf16")
            else ""
        )
        if was_float and not is_float:
            rounding = ".rzi"
        result = self.instruction(f"cvt{rounding}.{dt}.{value.type}", dt, value)
        if not is_float and not was_float:
            multiple = (
                math.gcd(value.multiple, 1 << (_SIZE[dt] * 8))
                if _SIZE[dt] < _SIZE[value.type]
                else value.multiple
            )
            return Value(result.text, dt, multiple=multiple)
        return result

    def lookup(self, name):
        if name not in self.env:
            raise CompileError(f"unbound IR value {name!r}")
        value = self.env[name]
        if value.text in self.corrections:
            # A Dot2 high state evolves only by addition between resets. Once
            # nonfinite it stays nonfinite; every replacement resets low. Defer
            # discarding its irrelevant low component to materialization.
            finite = self.instruction("testp.finite.f32", "pred", value)
            correction = self.select(
                finite, self.corrections[value.text], self.literal(0, "f32")
            )
            return self.binary("+", value, correction)
        return value

    def seed(self, name, identity, dt):
        dt = "f64" if name in self.wide_accumulators else dt
        value = self.mutable_reg(dt)
        self.move(value, self.literal(identity, dt))
        self.env[name] = value
        if name in self.compensated_accumulators:
            correction = self.mutable_reg("f32")
            self.move(correction, self.literal(0, "f32"))
            self.corrections[value.text] = correction

    def write_state(self, name, value):
        """Write a carrier, consuming any prior compensation exactly once."""
        destination = self.env[name]
        self.move(destination, value)
        if destination.text in self.corrections:
            self.move(self.corrections[destination.text], self.literal(0, "f32"))

    def compensated_dot(self, stmt, product):
        """Dot2: TwoProductFMA + full TwoSum, followed by a correction sum.

        See Ogita, Rump and Oishi (2005), Accurate Sum and Dot Product,
        Algorithm 5.3. No magnitude ordering or reassociation is assumed.
        """
        high = self.env[stmt.name]
        low = self.corrections[high.text]
        p = self.lookup(product.name)
        a, b = (self.lookup(n) for n in product.args)
        residual = self.operation(
            "fma", [a, b, self.operation("negative", [p], "f32")], "f32"
        )
        total = self.binary("+", high, p)
        delta = self.binary("-", total, high)
        error = self.binary(
            "+",
            self.binary("-", high, self.binary("-", total, delta)),
            self.binary("-", p, delta),
        )
        correction = self.binary("+", low, self.binary("+", error, residual))
        self.move(low, correction)
        self.move(high, total)

    @staticmethod
    def assign_type(name, args):
        """Infer an unstamped Assign without turning mixed integer ops into f32.

        Keep the existing same-type and floating/mask defaults. Emmy's mixed
        integer rule uses rank and sign before the operation, so packed words
        must not pass through a floating value or the legacy logical-mask path.
        """
        fallback = (
            args[0].type if args and all(a.type == args[0].type for a in args)
            else "f32"
        )
        integer_ops = {
            "add", "sum", "subtract", "sub", "multiply", "mul", "prod",
            "floor_divide", "remainder", "mod", "left_shift", "right_shift",
            "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_count",
        }
        if name not in integer_ops or not args or not all(a.type in _INTEGERS for a in args):
            return fallback
        signed = max((_SIZE[a.type] for a in args if a.type.startswith("s")), default=0)
        unsigned = max((_SIZE[a.type] for a in args if a.type.startswith("u")), default=0)
        return f"{'s' if signed > unsigned else 'u'}{8 * max(signed, unsigned)}"

    def common_type(self, a, b):
        for t in ("f64", "f32", "f16", "bf16"):
            if t in (a.type, b.type):
                return t
        if a.type == b.type == "pred":
            return "pred"
        # C integer promotions: small signed and unsigned integers fit in int.
        # At equal rank unsigned wins; signed 64-bit can represent all u32.
        promoted = [t if _SIZE[t] >= 4 else "s32" for t in (a.type, b.type)]
        for t in ("u64", "s64", "u32", "s32"):
            if t in promoted:
                return t
        return a.type if a.type == b.type else "s32"

    def select(self, predicate, a, b, dt=None):
        dt = dt or self.common_type(a, b)
        if dt == "pred":
            return self.convert(
                self.select(predicate, self.convert(a, "u32"), self.convert(b, "u32")),
                "pred",
            )
        if dt in ("s8", "u8", "s16", "u16"):
            return self.convert(
                self.select(predicate, self.convert(a, "s32"), self.convert(b, "s32")),
                dt,
            )
        a, b = self.convert(a, dt), self.convert(b, dt)
        instruction_type = "b16" if dt in ("f16", "bf16") else dt
        return self.instruction(
            f"selp.{instruction_type}", dt, a, b, self.convert(predicate, "pred")
        )

    def compare(self, op, a, b):
        dt = self.common_type(a, b)
        if dt in ("b32", "b64"):
            raise CompileError(f"numeric comparison {op!r} is unsupported for raw dtype {dt}")
        if dt == "bf16":
            # bf16 setp requires sm_90; f32 expansion preserves every value,
            # including signed zero, subnormals and unordered comparisons.
            dt = "f32"
        a, b = self.convert(a, dt), self.convert(b, dt)
        if dt == "pred":
            a, b, dt = self.convert(a, "u32"), self.convert(b, "u32"), "u32"
        if op == "ne" and dt in ("f16", "bf16", "f32", "f64"):
            op = "neu"  # C/NumPy nonzero and != are true for NaN.
        return self.instruction(f"setp.{op}.{dt}", "pred", a, b)

    def binary(self, op, a, b, dt=None):
        if op in ("&&", "||"):
            return self.instruction(
                "and.pred" if op == "&&" else "or.pred",
                "pred",
                self.convert(a, "pred"),
                self.convert(b, "pred"),
            )
        if op in ("<", "<=", ">", ">=", "==", "!="):
            return self.compare(
                {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}[
                    op
                ],
                a,
                b,
            )
        if op in ("<<", ">>") and dt is None:
            # C shifts promote each operand independently. A wide or unsigned
            # count must not widen the value or change an arithmetic right
            # shift into a logical shift. Named operations pass explicit dt.
            if a.type not in _INTEGERS | {"pred"} or b.type not in _INTEGERS | {"pred"}:
                raise CompileError(f"shift operator {op!r} requires integer operands")
            dt = a.type if _SIZE[a.type] >= 4 else "s32"
        dt = dt or self.common_type(a, b)
        # Comparisons produce predicates, but C arithmetic and bitwise
        # operators promote two boolean operands to int. Logical operators
        # were handled above and deliberately retain predicate computation.
        if dt == "pred":
            dt = "s32"
        if dt not in _NUMERIC or (
            dt in _FLOATS and op not in ("+", "-", "*", "/", "//")
        ):
            raise CompileError(f"binary operator {op!r} is unsupported for {dt} operands")
        if dt in ("f16", "bf16"):
            # A pure expression has a rounding boundary at each binary node,
            # even when nested inside an F32 Let. Division has no f16/bf16 PTX
            # instruction. Use exact operand expansion and round the result.
            return self.convert(
                self.binary(
                    op,
                    self.convert(self.convert(a, dt), "f32"),
                    self.convert(self.convert(b, dt), "f32"),
                ), dt
            )
        if dt in ("s8", "u8", "s16", "u16"):
            return self.convert(
                self.binary(op, self.convert(a, "s32"), self.convert(b, "s32")), dt
            )
        a = self.convert(a, dt)
        # The shift count is an independent integer operand; PTX takes u32.
        # Keep its mathematical constant until folding, and convert at emit.
        b = b if op in ("<<", ">>") else self.convert(b, dt)
        integer = dt[0] in "sub"
        if integer and a.constant is not None and b.constant is not None:
            x, y = int(a.constant), int(b.constant)
            if op == "+":
                z = x + y
            elif op == "-":
                z = x - y
            elif op == "*":
                z = x * y
            elif op in ("/", "//", "%"):
                if not y:
                    raise CompileError("integer division by zero")
                # Match div/rem and the Kernel IR's C reference lowering.
                # Python // and % choose different answers for negative x.
                quotient = abs(x) // abs(y)
                if (x < 0) != (y < 0):
                    quotient = -quotient
                z = x - quotient * y if op == "%" else quotient
            elif op == "&":
                z = x & y
            elif op == "|":
                z = x | y
            elif op == "^":
                z = x ^ y
            elif op == "<<":
                z = x << y
            elif op == ">>":
                z = x >> y
            else:
                raise CompileError(f"unsupported binary operator {op}")
            return self.literal(z, dt)
        if integer:
            if op in ("+", "-", "|", "^") and b.constant == 0:
                return a
            if op in ("*", "/", "//") and b.constant == 1:
                return a
            if op == "+" and a.constant == 0:
                return b
            if op == "*" and (a.constant == 0 or b.constant == 0):
                return self.literal(0, dt)
            if op == "%" and b.constant == 1:
                return self.literal(0, dt)
        suffix = dt
        if op in ("&", "|", "^", "<<"):
            suffix = f"b{_SIZE[dt] * 8}"
        instruction = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "//": "div",
            "%": "rem",
            "&": "and",
            "|": "or",
            "^": "xor",
            "<<": "shl",
            ">>": "shr",
        }.get(op)
        if instruction is None:
            raise CompileError(f"unsupported binary operator {op!r}")
        modifier = (
            ".lo"
            if integer and op == "*"
            else ".rn" if not integer and op in ("+", "-", "*", "/", "//") else ""
        )
        if op in ("<<", ">>"):
            b = self.convert(b, "u32")
        result = self.instruction(f"{instruction}{modifier}.{suffix}", dt, a, b)
        multiple = 1
        if integer:
            if op in ("+", "-"):
                multiple = math.gcd(a.multiple, b.multiple)
            elif op == "*":
                multiple = a.multiple * b.multiple
            elif op in ("/", "//") and b.constant and a.multiple % int(b.constant) == 0:
                multiple = a.multiple // abs(int(b.constant))
            elif op == "%" and b.constant:
                multiple = math.gcd(a.multiple, abs(int(b.constant)))
            if op in ("+", "-", "*"):
                # Wrapping arithmetic preserves only divisors shared with
                # 2**width. Keeping odd factors can falsely prove alignment
                # after a later division: u32(1431655766 * 6) / 3 is 1,
                # although the unbounded product would be divisible by 6.
                multiple = math.gcd(multiple, 1 << (_SIZE[dt] * 8))
        return Value(result.text, dt, multiple=multiple)

    def expr(self, expr):
        if isinstance(expr, Value):
            return expr
        if isinstance(expr, bool):
            return self.literal(expr)
        if isinstance(expr, int):
            return self.literal(expr)
        if isinstance(expr, float):
            return self.literal(expr, "f32")
        if isinstance(expr, str):
            return self.lookup(expr)
        if isinstance(expr, Var):
            return self.lookup(expr.name)
        if isinstance(expr, Literal):
            return self.literal(expr.value, dtype(expr.dtype))
        if isinstance(expr, BinaryExpr):
            return self.binary(expr.op, self.expr(expr.left), self.expr(expr.right))
        if isinstance(expr, Builtin):
            operand = builtin_operand(expr.name)
            # CUDA geometry components are unsigned int; warpSize is int.
            # Keep this type through expression promotions, including a
            # comparison against -1 or arithmetic that crosses the high bit.
            return self.instruction("mov.u32", "s32" if operand == "WARP_SZ" else "u32", operand)
        if isinstance(expr, CastExpr):
            return self.convert(self.expr(expr.expr), dtype(expr.dtype))
        if isinstance(expr, FlatIndex):
            return self.flat_index(expr.buffer, expr.index)
        if isinstance(expr, FuncCallExpr):
            args = [self.expr(a) for a in expr.args]
            return self.operation(expr.name, args, args[0].type if args else "f32")
        if isinstance(expr, TernaryExpr):
            p = self.convert(self.expr(expr.cond), "pred")
            a, b = self.expr(expr.if_true), self.expr(expr.if_false)
            dt = self.common_type(a, b)
            return self.select(p, a, b, dt)
        raise CompileError(f"unsupported expression {type(expr).__name__}")

    def floating_remainder(self, x, y):
        """Floor remainder via exact binary long division, then divisor-sign adjust.

        No quotient is represented in floating point. All reduction subtractions
        are exact (Sterbenz); powers of two retain every bit, including subnormals.
        The finite-input loops have <=277 (f32) or <=2098 (f64) visits per phase.
        """
        dt = x.type
        if dt not in ("f32", "f64") or y.type != dt:
            raise ValueError("floating_remainder requires matching f32/f64 operands")
        zero = self.literal(0.0, dt)
        result, residual, scaled = self.reg(dt), self.reg(dt), self.reg(dt)
        count = self.reg("u32")
        up, down, signed, done = (self.label("rem_" + name)
                                 for name in ("up", "down", "signed", "done"))
        self.move(result, self.literal(float("nan"), dt))
        finite_x = self.instruction(f"testp.finite.{dt}", "pred", x)
        nan_y = self.instruction(f"testp.notanumber.{dt}", "pred", y)
        zero_y = self.compare("eq", y, zero)
        self.emit(f"@!{finite_x} bra {done};")
        self.emit(f"@{nan_y} bra {done};")
        self.emit(f"@{zero_y} bra {done};")
        self.emit(f"abs.{dt} {residual}, {x};")
        self.emit(f"abs.{dt} {scaled}, {y};")
        self.move(count, self.literal(0, "u32"))
        smaller = self.compare("lt", residual, scaled)
        self.emit(f"@{smaller} bra {signed};")
        self.mark(up)
        doubled = self.instruction(f"mul.rn.{dt}", dt, scaled, self.literal(2.0, dt))
        too_large = self.compare("gt", doubled, residual)
        self.emit(f"@{too_large} bra {down};")
        self.move(scaled, doubled)
        self.emit(f"add.u32 {count}, {count}, 1;")
        self.emit(f"bra {up};")
        self.mark(down)
        subtract = self.compare("ge", residual, scaled)
        self.emit(f"@{subtract} sub.rn.{dt} {residual}, {residual}, {scaled};")
        finished = self.compare("eq", count, self.literal(0, "u32"))
        self.emit(f"@{finished} bra {signed};")
        self.emit(f"mul.rn.{dt} {scaled}, {scaled}, {self.literal(0.5, dt)};")
        self.emit(f"sub.u32 {count}, {count}, 1;")
        self.emit(f"bra {down};")
        self.mark(signed)
        self.emit(f"copysign.{dt} {result}, {x}, {residual};")
        negative_x, negative_y = self.compare("lt", x, zero), self.compare("lt", y, zero)
        different = self.instruction("xor.pred", "pred", negative_x, negative_y)
        nonzero = self.compare("ne", result, zero)
        adjust = self.binary("&&", different, nonzero)
        self.emit(f"@{adjust} add.rn.{dt} {result}, {result}, {y};")
        is_zero = self.compare("eq", result, zero)
        self.emit(f"@{is_zero} copysign.{dt} {result}, {y}, {zero};")
        self.mark(done)
        return result

    def operation(self, name, args, dt):
        dt = dtype(dt)
        aliases = {
            "sum": "add",
            "prod": "multiply",
            "sub": "subtract",
            "mul": "multiply",
            "neg": "negative",
            "exp_fast": "exp",
            "true_divide": "divide",
            "amax": "maximum",
            "max": "maximum",
            "min": "minimum",
            "absolute": "abs",
            "pow": "power",
        }
        name = aliases.get(name, name)
        capability = _ELEMENTWISE_CAPABILITIES.get(name)
        if capability is None:
            raise CompileError(f"unsupported elementwise operation {name!r}")
        arity, types = capability
        if len(args) != arity:
            raise CompileError(f"elementwise {name!r} expects {arity} operands, got {len(args)}")
        if dt not in types:
            raise CompileError(f"elementwise {name!r} is unsupported for result dtype {dt}")
        if name in ("copy", "pad"):
            return self.convert(args[0], dt)
        if name == "bitcast":
            source = args[0]
            if source.type == "pred" or _SIZE[source.type] != _SIZE[dt]:
                raise CompileError(f"bitcast requires equal-width non-predicate types: {source.type} -> {dt}")
            if _SIZE[dt] < 4:
                # Small integer values occupy b32 registers; narrow floating
                # values occupy b16 registers. cvt truncates/extends integer
                # payloads, while mov preserves the floating representation.
                if source.type == dt:
                    return source
                if source.type in ("f16", "bf16") and dt in ("f16", "bf16"):
                    return Value(source.text, dt)
                payload = source
                if source.type in ("f16", "bf16"):
                    payload = self.reg("u32")
                    self.emit(f"mov.b32 {payload}, {{{source}, 0}};")
                if dt in ("f16", "bf16"):
                    result, high = self.reg(dt), self.reg("f16")
                    self.emit(f"mov.b32 {{{result}, {high}}}, {payload};")
                    return result
                return self.convert(Value(payload.text, "u32"), dt)
            return self.instruction(f"mov.b{_SIZE[dt] * 8}", dt, source)
        if name == "where":
            return self.select(args[0], args[1], args[2], dt)
        if name in (
            "equal",
            "not_equal",
            "less",
            "less_equal",
            "greater",
            "greater_equal",
        ):
            p = self.compare(
                {
                    "equal": "eq",
                    "not_equal": "ne",
                    "less": "lt",
                    "less_equal": "le",
                    "greater": "gt",
                    "greater_equal": "ge",
                }[name],
                *args,
            )
            return self.convert(p, dt)
        if name == "logical_not" or (name == "bitwise_not" and dt in ("f32", "pred")):
            return self.convert(
                self.compare("eq", args[0], self.literal(0, args[0].type)), dt
            )
        if name in ("logical_and", "logical_or") or (
            name in ("bitwise_and", "bitwise_or") and dt in ("f32", "pred")
        ):
            op = "&&" if name.endswith("and") else "||"
            return self.convert(self.binary(op, *args), dt)
        if dt in ("s8", "u8", "s16", "u16"):
            wide = "s32" if dt.startswith("s") else "u32"
            return self.convert(
                self.operation(name, [self.convert(a, wide) for a in args], wide), dt
            )
        # Narrow floating computation rounds at each IR statement, not at each
        # synthetic step of a composite operation.
        if dt in ("f16", "bf16"):
            return self.convert(
                self.operation(name, [self.convert(a, "f32") for a in args], "f32"), dt
            )
        args = [self.convert(a, dt) for a in args]
        if name == "floor_divide":
            q = self.binary("/", *args)
            if dt.startswith("f"):
                return self.operation("floor", [q], dt)
            if dt.startswith("u"):
                return q
            rem = self.binary("%", *args)
            different_signs = self.compare(
                "lt", self.binary("^", *args), self.literal(0, dt)
            )
            correction = self.binary(
                "&&", self.compare("ne", rem, self.literal(0, dt)), different_signs
            )
            return self.binary("-", q, self.convert(correction, dt))
        if name in ("mod", "remainder") and dt.startswith("f"):
            return self.floating_remainder(*args)
        if name in ("mod", "remainder") and dt.startswith("s"):
            rem = self.binary("%", *args)
            different_signs = self.compare(
                "lt", self.binary("^", args[0], args[1]), self.literal(0, dt)
            )
            correction = self.binary(
                "&&", self.compare("ne", rem, self.literal(0, dt)), different_signs
            )
            return self.select(correction, self.binary("+", rem, args[1]), rem)
        binary = {
            "add": "+",
            "subtract": "-",
            "multiply": "*",
            "divide": "/",
            "remainder": "%",
            "mod": "%",
            "bitwise_and": "&",
            "bitwise_or": "|",
            "bitwise_xor": "^",
            "left_shift": "<<",
            "right_shift": ">>",
        }
        if name in binary:
            return self.binary(binary[name], *args, dt=dt)
        x = args[0]
        if name == "bitwise_not":
            return self.instruction(f"not.b{_SIZE[dt]*8}", dt, x)
        if name == "bitwise_count":
            if dt.startswith("s"):
                x = self.operation("abs", [x], dt)
            return self.convert(self.instruction(f"popc.b{_SIZE[dt]*8}", "u32", x), dt)
        if name in ("maximum", "minimum", "fmax", "fmin"):
            return self.instruction(
                f"{'max' if name in ('maximum', 'fmax') else 'min'}.{dt}", dt, *args
            )
        if name in ("negative", "abs"):
            if dt.startswith("u"):
                return self.binary("-", self.literal(0, dt), x) if name == "negative" else x
            return self.instruction(
                f"{'neg' if name == 'negative' else 'abs'}.{dt}", dt, x
            )
        if name == "square":
            return self.binary("*", x, x)
        if name == "fma":
            return self.instruction(f"fma.rn.{dt}", dt, *args)
        if name == "reciprocal":
            return self.binary("/", self.literal(1, dt), x)
        if name == "sqrt":
            return self.instruction(f"sqrt.rn.{dt}", dt, x)
        if name == "rsqrt":
            return self.binary(
                "/", self.literal(1, dt), self.operation("sqrt", [x], dt)
            )
        if name in ("floor", "ceil", "trunc", "rint"):
            mode = {"floor": "rmi", "ceil": "rpi", "trunc": "rzi", "rint": "rni"}[name]
            return self.instruction(f"cvt.{mode}.{dt}.{dt}", dt, x)
        if name == "relu":
            return self.operation("maximum", [x, self.literal(0, dt)], dt)
        if dt != "f32":
            raise CompileError(f"elementwise {name!r} is not implemented at {dt}")
        if name in ("exp", "exp2"):
            y = (
                x
                if name == "exp2"
                else self.binary("*", x, self.literal(math.log2(math.e), dt))
            )
            return self.instruction("ex2.approx.f32", dt, y)
        if name in ("log", "log2", "log10"):
            y = self.instruction("lg2.approx.f32", dt, x)
            return (
                y
                if name == "log2"
                else self.binary(
                    "*",
                    y,
                    self.literal(math.log(2) if name == "log" else math.log10(2), dt),
                )
            )
        if name in ("sin", "cos"):
            result = self.instruction(f"{name}.approx.f32", dt, x)
            if name == "sin":
                # sin(x) rounds to x here. Some device implementations lose
                # tiny values during approximate argument reduction.
                small = self.compare(
                    "lt", self.operation("abs", [x], dt), self.literal(2**-12, dt)
                )
                result = self.select(small, x, result, dt)
            return result
        if name == "tanh":
            # tanh.approx only guarantees 2**-11 relative error. A short odd
            # polynomial avoids cancellation near zero; the stable exponential
            # formula handles the rest without positive exponential overflow.
            ax = self.operation("abs", [x], dt)
            z = self.binary("*", x, x)
            polynomial = self.literal(-1382 / 155925, dt)
            for coefficient in (62 / 2835, -17 / 315, 2 / 15, -1 / 3):
                polynomial = self.operation(
                    "fma", [polynomial, z, self.literal(coefficient, dt)], dt
                )
            small_result = self.operation(
                "fma", [self.binary("*", x, z), polynomial, x], dt
            )
            small_result = self.instruction("copysign.f32", dt, x, small_result)
            tail = self.operation(
                "exp", [self.binary("*", self.literal(-2, dt), ax)], dt
            )
            magnitude = self.binary(
                "/",
                self.binary("-", self.literal(1, dt), tail),
                self.binary("+", self.literal(1, dt), tail),
            )
            large_result = self.instruction("copysign.f32", dt, x, magnitude)
            return self.select(
                self.compare("le", ax, self.literal(0.25, dt)),
                small_result,
                large_result,
                dt,
            )
        if name == "power":
            return self.power(x, args[1])
        if name == "erf":
            # Abramowitz-Stegun 7.1.26, absolute error below 1.5e-7 in
            # real arithmetic; each generated operation is separately rounded.
            ax = self.operation("abs", [x], dt)
            t = self.binary(
                "/",
                self.literal(1, dt),
                self.binary(
                    "+",
                    self.literal(1, dt),
                    self.binary("*", self.literal(0.3275911, dt), ax),
                ),
            )
            poly = self.literal(1.061405429, dt)
            for coefficient in (-1.453152027, 1.421413741, -0.284496736, 0.254829592):
                poly = self.binary(
                    "+", self.binary("*", poly, t), self.literal(coefficient, dt)
                )
            tail = self.binary(
                "*",
                self.binary("*", poly, t),
                self.operation(
                    "exp",
                    [self.operation("negative", [self.binary("*", ax, ax)], dt)],
                    dt,
                ),
            )
            result = self.binary("-", self.literal(1, dt), tail)
            return self.instruction("copysign.f32", dt, x, result)
        if name in ("sigmoid", "silu"):
            den = self.binary(
                "+",
                self.literal(1, dt),
                self.operation("exp", [self.operation("negative", [x], dt)], dt),
            )
            return self.binary("/", x if name == "silu" else self.literal(1, dt), den)
        raise CompileError(f"unsupported elementwise operation {name!r}")

    def power(self, x, y):
        """Real f32 power with signed integral-base and IEEE special cases."""
        dt = "f32"
        ax = self.operation("abs", [x], dt)
        result = self.operation(
            "exp2", [self.binary("*", self.operation("log2", [ax], dt), y)], dt
        )
        integral = self.compare("eq", y, self.operation("trunc", [y], dt))
        half = self.binary("*", y, self.literal(0.5, dt))
        even = self.compare("eq", half, self.operation("trunc", [half], dt))
        odd = self.binary("&&", integral, self.instruction("not.pred", "pred", even))
        signed = self.instruction("copysign.f32", dt, x, result)
        result = self.instruction("selp.f32", dt, signed, result, odd)
        finite_base = self.compare("lt", ax, self.literal(float("inf"), dt))
        invalid = self.binary(
            "&&",
            self.binary("&&", self.compare("lt", x, self.literal(0, dt)), finite_base),
            self.instruction("not.pred", "pred", integral),
        )
        result = self.instruction(
            "selp.f32", dt, self.literal(float("nan"), dt), result, invalid
        )
        identity = self.binary(
            "||",
            self.compare("eq", y, self.literal(0, dt)),
            self.compare("eq", x, self.literal(1, dt)),
        )
        unit_at_infinity = self.binary(
            "&&",
            self.compare("eq", ax, self.literal(1, dt)),
            self.compare(
                "eq", self.operation("abs", [y], dt), self.literal(float("inf"), dt)
            ),
        )
        identity = self.binary("||", identity, unit_at_infinity)
        return self.instruction("selp.f32", dt, self.literal(1, dt), result, identity)

    def flat_index(self, name, indices):
        buf = self.buffers[name]
        dt = "s64" if math.prod(buf.shape) > 2**31 - 1 else "s32"
        if not indices:
            return self.literal(0, dt)
        if len(indices) == 1:
            return self.convert(self.expr(indices[0]), dt)
        if len(indices) != len(buf.shape):
            # Emmy's legacy partial-coordinate form contains already-flat
            # offsets: when rank differs, add them instead of applying shape
            # strides. Widen each term before summing for large buffers.
            total = self.literal(0, dt)
            for index in indices:
                total = self.binary("+", total, self.convert(self.expr(index), dt))
            return total
        total = self.literal(0, dt)
        for i, index in enumerate(indices):
            term = self.binary(
                "*",
                self.convert(self.expr(index), dt),
                self.literal(math.prod(buf.shape[i + 1 :]), dt),
            )
            total = self.binary("+", total, term)
        return total

    def address(self, name, indices=(), *, flat=None, swizzle="NONE"):
        buf = self.buffers[name]
        index = self.flat_index(name, indices) if flat is None else flat
        if swizzle != "NONE":
            mode, _, shift = swizzle.partition("@")
            masks = {"B32": 1, "B64": 3, "B128": 7}
            if mode not in masks or buf.space != "shared":
                raise CompileError(f"unsupported shared swizzle {swizzle}")
            bits = self.binary(
                "&",
                self.binary(">>", index, self.literal(int(shift) if shift else 6)),
                self.literal(masks[mode]),
            )
            index = self.binary("^", index, self.binary("<<", bits, self.literal(3)))
        if buf.space == "shared":
            off = self.binary(
                "*", self.convert(index, "u32"), self.literal(_SIZE[buf.type], "u32")
            )
            address = self.binary("+", buf.pointer, off, "u32")
            return Value(
                address.text,
                address.type,
                multiple=math.gcd(buf.align, index.multiple * _SIZE[buf.type]),
            )
        if index.type in ("s32", "u32"):
            off = self.instruction(
                f"mul.wide.{index.type}", "s64", index, _SIZE[buf.type]
            )
        else:
            off = self.binary(
                "*", self.convert(index, "s64"), self.literal(_SIZE[buf.type], "s64")
            )
        address = self.binary("+", buf.pointer, self.convert(off, "u64"), "u64")
        return Value(
            address.text,
            address.type,
            multiple=math.gcd(buf.align, index.multiple * _SIZE[buf.type]),
        )

    def load(self, name, address, count=1):
        buf = self.buffers[name]
        dt = "u8" if buf.type == "pred" else buf.type
        memdt = "b16" if dt in ("f16", "bf16") else dt
        values = [self.reg(dt) for _ in range(count)]
        if (
            count in (2, 4)
            and count * _SIZE[dt] <= 16
            and _SIZE[dt] >= 2
            and address.multiple % (count * _SIZE[dt]) == 0
        ):
            self.emit(
                f"ld.{buf.space}.v{count}.{memdt} {{{', '.join(map(str, values))}}}, [{address}];"
            )
        else:
            for i, val in enumerate(values):
                self.emit(f"ld.{buf.space}.{memdt} {val}, [{address}+{i * _SIZE[dt]}];")
        return (
            [self.convert(v, "pred") for v in values] if buf.type == "pred" else values
        )

    def store(self, name, address, values, atomic=False):
        buf = self.buffers[name]
        dt = "u8" if buf.type == "pred" else buf.type
        values = [self.convert(v, dt) for v in values]
        memdt = "b16" if dt in ("f16", "bf16") else dt
        count = len(values)
        if atomic:
            if dt not in ("f32", "f64", "u32", "s32", "u64"):
                raise CompileError(f"unsupported atomic-add type {dt}")
            for i, val in enumerate(values):
                if dt == "f64" or buf.space == "shared":
                    self.emit(
                        f"atom.{buf.space}.add.{dt} {self.reg(dt)}, [{address}+{i * _SIZE[dt]}], {val};"
                    )
                else:
                    self.emit(
                        f"red.{buf.space}.add.{dt} [{address}+{i * _SIZE[dt]}], {val};"
                    )
        elif (
            count in (2, 4)
            and count * _SIZE[dt] <= 16
            and _SIZE[dt] >= 2
            and address.multiple % (count * _SIZE[dt]) == 0
        ):
            self.emit(
                f"st.{buf.space}.v{count}.{memdt} [{address}], {{{', '.join(map(str, values))}}};"
            )
        else:
            for i, val in enumerate(values):
                self.emit(f"st.{buf.space}.{memdt} [{address}+{i * _SIZE[dt]}], {val};")

    def body(self, body):
        previous = None
        for stmt in body:
            try:
                if (
                    (
                        self.fuse_accumulations
                        or (
                            isinstance(stmt, Accum)
                            and stmt.base is None
                            and stmt.name
                            in (self.wide_accumulators | self.compensated_accumulators)
                        )
                    )
                    and isinstance(previous, Assign)
                    and previous.op.name == "multiply"
                    and isinstance(stmt, Accum)
                    and stmt.value == previous.name
                    and stmt.op.name in ("add", "sum")
                    and dtype(previous.dtype) == dtype(stmt.dtype) == "f32"
                ):
                    # Contract only an explicitly adjacent product/accumulator.
                    # The product binding remains available if another statement
                    # reads it. Exact pointwise arithmetic is never contracted.
                    if stmt.name in self.compensated_accumulators and stmt.base is None:
                        self.compensated_dot(stmt, previous)
                    else:
                        dst = self.env[stmt.name]
                        factors = [
                            self.convert(self.lookup(n), "f32") for n in previous.args
                        ]
                        value = self.operation(
                            "fma",
                            [*factors, self.lookup(stmt.base or stmt.name)],
                            dst.type,
                        )
                        self.write_state(stmt.name, value)
                else:
                    self.statement(stmt)
            except CompileError as exc:
                raise CompileError(f"{type(stmt).__name__}: {exc}") from exc
            previous = stmt

    def statement(self, s):
        if isinstance(s, (RegFragment, LdmatrixLoad, MmaSyncPtx, RegStore, FragmentRepack)):
            if self.tensor_lowering is None:
                from ptx_tensor import TensorLowering

                self.tensor_lowering = TensorLowering(self)
            if self.tensor_lowering.handles(s):
                return
        if isinstance(s, (TmaDescriptor, TmaLoad, MbarrierInit, MbarrierArrive,
                          MbarrierArriveExpectTx, MbarrierWait)):
            if self.bulk_lowering is None:
                from ptx_bulk import BulkLowering

                self.bulk_lowering = BulkLowering(self)
                self.bulk_lowering.prepare(self.kernel.body)
            if self.bulk_lowering.handles(s):
                return
        if isinstance(s, Tile):
            return self.tile(s)
        if isinstance(s, (Loop, StridedLoop)):
            return self.loop(s)
        if isinstance(s, Cond):
            otherwise, end = self.label("else"), self.label("endif")
            pred = self.convert(self.expr(s.cond), "pred")
            saved = self.env.copy()
            self.emit(f"@!{pred} bra {otherwise};")
            self.body(s.body)
            self.emit(f"bra {end};")
            self.mark(otherwise)
            self.env = saved.copy()
            self.body(s.else_body)
            self.mark(end)
            self.env = saved
        elif isinstance(s, Load):
            values = self.load(s.input, self.address(s.input, s.index), len(s.names))
            self.env.update(zip(s.names, values))
        elif isinstance(s, Write):
            self.store(
                s.output,
                self.address(s.output, s.index, swizzle=s.swizzle),
                [self.lookup(n) for n in s.values],
                s.atomic,
            )
        elif isinstance(s, Assign):
            args = [self.lookup(n) for n in s.args]
            dt = dtype(s.dtype, self.assign_type(s.op.name, args))
            self.env[s.name] = self.snapshot(self.operation(s.op.name, args, dt))
        elif isinstance(s, Let):
            value = self.expr(s.value)
            self.env[s.name] = self.snapshot(
                self.convert(value, dtype(s.dtype, value.type))
            )
        elif isinstance(s, Select):
            if not s.branches:
                raise CompileError("Select requires at least one branch")
            value = self.convert(self.lookup(s.branches[-1].value), "f32")
            for branch in reversed(s.branches[:-1]):
                value = self.select(
                    self.expr(branch.select),
                    self.convert(self.lookup(branch.value), "f32"),
                    value,
                )
            self.env[s.name] = self.snapshot(value)
        elif isinstance(s, Init):
            self.seed(s.name, s.identity, dtype(s.dtype))
        elif isinstance(s, Accum):
            dst = self.env[s.name]
            value = self.operation(
                s.op.name,
                [self.lookup(s.base or s.name), self.lookup(s.value)],
                dst.type,
            )
            self.write_state(s.name, value)
        elif isinstance(s, Smem):
            pass  # declarations and addresses are in the prologue
        elif isinstance(s, Sync):
            self.sync(s.barrier_id, s.count, s.warp)
        elif isinstance(s, WarpShuffle):
            self.shuffle(s)
        elif isinstance(s, TreeHalve):
            self.tree(s)
        elif isinstance(s, CpAsyncCopy):
            index = self.flat_index(s.smem, s.smem_index)
            if s.lane_index is not None:
                lane = self.binary(
                    "+",
                    self.binary(
                        "*",
                        self.expr(s.lane_index[0]),
                        self.literal(self.buffers[s.smem].shape[-1]),
                    ),
                    self.expr(s.lane_index[1]),
                )
                index = self.binary("+", index, lane)
            dst, src = self.address(
                s.smem, flat=index, swizzle=s.swizzle
            ), self.address(s.src, s.src_index)
            if s.nbytes not in (4, 8, 16):
                raise CompileError(f"invalid cp.async size {s.nbytes}")
            if self.synchronous_copy:
                for i in range(s.nbytes // 4):
                    val = self.reg("b32")
                    self.emit(f"ld.global.b32 {val}, [{src}+{4*i}];")
                    self.emit(f"st.shared.b32 [{dst}+{4*i}], {val};")
            else:
                cache = "cg" if s.nbytes == 16 else "ca"
                self.emit(
                    f"cp.async.{cache}.shared.global [{dst}], [{src}], {s.nbytes};"
                )
        elif isinstance(s, CpAsyncCommit):
            if not self.synchronous_copy:
                self.emit("cp.async.commit_group;")
        elif isinstance(s, CpAsyncWait):
            if not 0 <= s.group <= 7:
                raise CompileError(f"invalid cp.async wait group {s.group}")
            if not self.synchronous_copy:
                self.emit(f"cp.async.wait_group {s.group};")
        else:
            raise CompileError(f"unsupported statement {type(s).__name__}")

    def loop(self, s):
        # Seed exactly the immediate folds. Nested folds reset inside this loop;
        # a seed=False drain updates a surrounding Init across slab iterations.
        seen = set()
        for a in s.body:
            if isinstance(a, Accum) and a.name not in seen:
                seen.add(a.name)
                if s.seed:
                    if a.op.identity is None:
                        raise CompileError(f"no identity for {a.op.name}")
                    self.seed(a.name, a.op.identity, dtype(a.dtype))
                else:
                    self.lookup(a.name)
        saved = self.env.copy()
        # The schedule's unroll flag is a hint. Honor it only for compact,
        # statically counted loops; retain large outer attention loops.
        if isinstance(s, Loop) and s.unroll and self.unroll_budget:
            count = extent(s.axis.extent)
            cost = count * self.expansion_cost(s.body)
            if cost <= min(self.unroll_budget, self.unroll_remaining):
                self.unroll_remaining -= cost
                for i in range(count):
                    self.env = saved.copy()
                    self.env[s.axis.name] = self.literal(i)
                    self.body(s.body)
                self.env = saved
                return
        # Loop/StridedLoop declare an int induction variable, independently of
        # the start expression's type. Compound increments still compute with
        # the step's usual promotions before assigning back to this s32 slot.
        start = self.expr(s.start) if isinstance(s, StridedLoop) else self.literal(0)
        iv = self.mutable_reg("s32")
        self.move(iv, start)
        self.env[s.axis.name] = iv
        if isinstance(s, StridedLoop) and s.end is not None:
            # The second int for-init binding sees the initialized induction
            # variable and rounds/truncates independently of its source type.
            bound = self.convert(self.expr(s.end), "s32")
        else:
            # An implicit extent is a decimal C integer literal, not an int
            # binding. Preserve a large literal's width in the comparison.
            count = extent(s.axis.extent)
            bound = self.literal(count, "s64" if count > 2**31 - 1 else "s32")
        # The explicit end is a for-init binding, evaluated once on entry.
        # A bare carrier Var may otherwise alias a register mutated by the
        # body, changing both the stop condition and the <axis>_end binding.
        bound = self.snapshot(bound)
        if isinstance(s, StridedLoop) and s.end is not None:
            self.env[s.axis.name + "_end"] = bound
        head, end = self.label("loop"), self.label("endloop")
        self.mark(head)
        done = self.compare("ge", iv, bound)
        self.emit(f"@{done} bra {end};")
        self.body(s.body)
        step = self.expr(s.step) if isinstance(s, StridedLoop) else self.literal(1)
        self.move(iv, self.binary("+", iv, step))
        self.emit(f"bra {head};")
        self.mark(end)
        self.env = saved

    def expansion_cost(self, body):
        total = 0
        for stmt in body:
            if isinstance(stmt, Loop):
                factor = extent(stmt.axis.extent) if stmt.unroll else 1
                total += 1 + factor * self.expansion_cost(stmt.body)
            elif isinstance(stmt, (StridedLoop, Tile)):
                total += 1 + self.expansion_cost(stmt.body)
            elif isinstance(stmt, Cond):
                total += (
                    1
                    + self.expansion_cost(stmt.body)
                    + self.expansion_cost(stmt.else_body)
                )
            else:
                total += 1
        return total

    def tile(self, s):
        saved = self.env.copy()
        n = math.prod(extent(a.extent) for a in s.axes)
        dt = "s64" if n > 2**31 - 1 else "s32"
        tid, cta, ntid = (self.lookup(x) for x in ("_tid", "_cta", "_ntid"))
        cells = (
            self.literal(s.block_threads, dt)
            if s.aux_threads
            else self.convert(ntid, dt)
        )
        local = (
            self.binary("%", tid, self.literal(s.block_threads))
            if s.aux_threads
            else tid
        )
        gid = self.binary(
            "+", self.binary("*", self.convert(cta, dt), cells), self.convert(local, dt)
        )
        end = self.label("endtile")
        self.emit(f"@{self.compare('ge', gid, self.literal(n, dt))} bra {end};")
        sizes = [extent(a.extent) for a in s.axes]
        for i, a in enumerate(s.axes):
            val = self.binary("/", gid, self.literal(math.prod(sizes[i + 1 :]), dt))
            if i:
                val = self.binary("%", val, self.literal(sizes[i], dt))
            self.env[a.name] = self.convert(val, "s32")
        self.raster(s, gid, sizes)
        self.body(s.body)
        self.mark(end)
        self.env = saved

    def raster(self, s, gid, sizes):
        if not s.raster_group or not s.raster_axes:
            return
        names = [a.name for a in s.axes]
        if not all(a in names for a in s.raster_axes):
            return
        mi, ni = map(names.index, s.raster_axes)
        if ni != mi + 1:
            return
        gi, oi = (mi, ni) if s.raster_orient == "m" else (ni, mi)
        ea, eb, group = sizes[gi], sizes[oi], min(s.raster_group, sizes[gi])
        # With either raster extent equal to one, grouped traversal is the
        # same row-major order already decoded by Tile.
        if ea <= 1 or eb <= 1 or group <= 1:
            return
        sub = self.binary("/", gid, self.literal(math.prod(sizes[ni + 1 :]), gid.type))
        if mi:
            sub = self.binary("%", sub, self.literal(ea * eb, gid.type))
        grp = self.binary("/", sub, self.literal(group * eb, gid.type))
        base = self.binary("*", grp, self.literal(group, gid.type))
        # Tile's cell guard bounds sub to this raster plane. If the grouped
        # extent is divisible by the effective group, there is no short final
        # group, so avoid a runtime minimum and division by its result.
        gs = self.literal(group, gid.type)
        if ea % group:
            gs = self.operation(
                "minimum",
                [
                    gs,
                    self.binary("-", self.literal(ea, gid.type), base),
                ],
                gid.type,
            )
        rem = self.binary("%", sub, self.literal(group * eb, gid.type))
        self.env[names[gi]] = self.convert(
            self.binary("+", base, self.binary("%", rem, gs)), "s32"
        )
        self.env[names[oi]] = self.convert(self.binary("/", rem, gs), "s32")

    def sync(self, barrier_id=0, count=None, warp=False):
        if warp:
            if barrier_id:
                raise CompileError("warp synchronization cannot use a named barrier")
            # Sync(warp=True) means __syncwarp() with its full-warp mask.
            # An instantaneous active mask can omit lanes still finishing
            # divergent work and fail to order their shared-memory writes.
            self.emit("bar.warp.sync 0xffffffff;")
        elif barrier_id:
            if count is None:
                raise CompileError("named barrier requires participant count")
            self.emit(f"bar.sync {barrier_id}, {count};")
        else:
            self.emit("bar.sync 0;")

    def merge(self, s):
        dt = dtype(s.dtype)
        for a in s.combine_states:
            value = self.operation(a.op.name, [self.lookup(n) for n in a.args], dt)
            if a.name in s.state:
                self.write_state(a.name, value)
            else:
                self.env[a.name] = self.snapshot(value)

    def shuffle(self, s):
        if s.length < 1 or s.length > 32 or s.length & (s.length - 1):
            raise CompileError("shuffle length must be a power of two in 1..32")
        self.prepare_state(s.state)
        saved = self.env.copy()
        # Membership follows the IR's aligned power-of-two group, not the
        # instantaneous convergence of lanes after different loop trip counts.
        group_base = self.binary("&", self.lookup("lane"), self.literal(32 - s.length))
        mask = self.binary(
            "<<", self.literal((1 << s.length) - 1, "u32"), group_base, "u32"
        )
        delta = s.length // 2
        while delta:
            for a, b in zip(s.state, s.state_b, strict=True):
                src = self.materialize(self.convert(self.lookup(a), dtype(s.dtype)))
                if _SIZE[src.type] == 8:
                    lo, hi = self.reg("b32"), self.reg("b32")
                    self.emit(f"mov.b64 {{{lo}, {hi}}}, {src};")
                    parts = [
                        self.instruction(
                            "shfl.sync.bfly.b32", "b32", part, delta, 31, mask
                        )
                        for part in (lo, hi)
                    ]
                    dst = self.reg(src.type)
                    self.emit(f"mov.b64 {dst}, {{{parts[0]}, {parts[1]}}};")
                else:
                    if _SIZE[src.type] < 4:
                        src = self.convert(src, "f32")
                    dst = self.instruction(
                        "shfl.sync.bfly.b32", src.type, src, delta, 31, mask
                    )
                self.env[b] = dst
            self.merge(s)
            delta //= 2
        self.env = saved

    def tree(self, s):
        if s.length < 1 or s.length & (s.length - 1):
            raise CompileError("tree length must be a power of two")
        self.prepare_state(s.state)
        saved = self.env.copy()
        tid = self.lookup(s.tid_var)
        inner, scale = (
            (self.lookup(s.inner[0]), s.inner[1]) if s.inner else (self.literal(0), 1)
        )
        slot = self.binary("+", self.binary("*", tid, self.literal(scale)), inner)
        delta = s.length // 2
        while delta:
            end = self.label("tree_step")
            worker = self.compare("lt", tid, self.literal(delta))
            if s.tid_var == "warp" and s.inner is None:
                worker = self.binary(
                    "&&",
                    worker,
                    self.compare("eq", self.lookup("lane"), self.literal(0)),
                )
            self.emit(f"@!{worker} bra {end};")
            other = self.binary("+", slot, self.literal(delta * scale))
            for buf, a, b in zip(s.bufs, s.state, s.state_b, strict=True):
                loaded = self.convert(
                    self.load(buf, self.address(buf, flat=slot))[0], dtype(s.dtype)
                )
                self.env[a] = self.mutable_reg(dtype(s.dtype))
                self.move(self.env[a], loaded)
                self.env[b] = self.convert(
                    self.load(buf, self.address(buf, flat=other))[0], dtype(s.dtype)
                )
            self.merge(s)
            for buf, name in zip(s.bufs, s.state, strict=True):
                self.store(buf, self.address(buf, flat=slot), [self.lookup(name)])
            self.mark(end)
            self.sync(s.barrier_id, s.barrier_count)
            delta //= 2
        self.env = saved
        for buf, name in zip(s.bufs, s.state, strict=True):
            self.write_state(name, self.load(buf, self.address(buf, flat=inner))[0])

    def prepare_state(self, names):
        for name in names:
            value = self.lookup(name)
            if value.text not in self.mutable:
                state = self.mutable_reg(value.type)
                self.move(state, value)
                self.env[name] = state
                if name in self.compensated_accumulators:
                    correction = self.mutable_reg("f32")
                    self.move(correction, self.literal(0, "f32"))
                    self.corrections[state.text] = correction

    def compile(self):
        order = self.meta["arg_order"]
        expected = set(self.node.inputs) | {t.name for t in self.node.outputs}
        if len(order) != len(set(order)) or set(order) != expected:
            raise CompileError(
                f"arg_order {order!r} does not cover kernel buffers {sorted(expected)!r}"
            )
        for i, name in enumerate(order):
            pointer = self.instruction("ld.param.u64", "u64", f"[param_{i}]")
            self.buffers[name].pointer = self.instruction(
                "cvta.to.global.u64", "u64", pointer
            )
        for name, builtin in (
            ("_tid", "%tid.x"),
            ("_cta", "%ctaid.x"),
            ("_ntid", "%ntid.x"),
        ):
            self.env[name] = self.instruction("mov.u32", "s32", builtin)
        self.env["lane"] = self.binary("%", self.env["_tid"], self.literal(32))
        self.env["warp"] = self.binary("/", self.env["_tid"], self.literal(32))
        for buf in self.smem.values():
            ptr = self.instruction(
                "mov.u32", "u32", "smem_pool" if self.dynamic_smem else buf.symbol
            )
            buf.pointer = (
                self.binary("+", ptr, self.literal(buf.offset, "u32"))
                if self.dynamic_smem
                else ptr
            )
        self.body(self.kernel.body)
        self.emit("ret;")
        params = ",\n".join(f"    .param .u64 param_{i}" for i in range(len(order)))
        declarations = [
            f"    .reg .{self.regtypes[p]} %{p}<{n+1}>;" for p, n in self.counts.items()
        ]
        declarations += [
            "    " + line
            for line in self.declarations
            if not line.startswith(".extern")
        ]
        externs = "\n".join(
            line for line in self.declarations if line.startswith(".extern")
        )
        bounds = ""
        if self.launch_bounds:
            # The launch contract includes helper threads and every block
            # dimension. Give ptxas this truthful bound without changing the
            # schedule, register cap, or numerical instruction sequence.
            block = self.meta["block"]
            if (not isinstance(block, (list, tuple)) or len(block) != 3
                    or any(type(n) is not int or n < 1 for n in block)
                    or math.prod(block) > 1024):
                raise CompileError("block must contain three positive integers with at most 1024 threads")
            bounds = ".maxntid " + ", ".join(map(str, block)) + "\n"
        source = (
            f".version 8.0\n.target {self.target}\n.address_size 64\n{externs}\n\n.visible .entry {self.kernel.name}(\n{params}\n)\n{bounds}{{\n"
            + "\n".join(declarations + self.lines)
            + "\n}\n"
        )
        source.encode("ascii")
        return source


def compile_file(
    path,
    *,
    target=None,
    synchronous_copy=False,
    unroll_budget=1024,
    fuse_accumulations=False,
    accurate_dots=False,
    compensated_dots=None,
    launch_bounds=True,
):
    """Compile one packaged kernel using its ABI and numerical contract.

    By default, independent FP32 dots use compensation when the problem allows
    nonzero error. Exact or missing tolerances preserve ordinary FP32 arithmetic.
    Explicit boolean flags select development alternatives reproducibly.
    """
    path = Path(path)
    graph = Graph.from_dict(json.loads(path.read_text(encoding="utf-8")))
    kernels = [
        node.op for node in graph.nodes.values() if isinstance(node.op, KernelOp)
    ]
    if len(kernels) != 1:
        raise CompileError("input must contain exactly one KernelOp")
    if target is None:
        # Bulk transactions and parity try-waits require SM90. The ordinary
        # contest vocabulary keeps its original SM89 target; explicit caller
        # targets are never silently upgraded.
        needs_bulk = any(isinstance(s, (TmaDescriptor, TmaLoad, MbarrierInit,
                            MbarrierArrive, MbarrierArriveExpectTx, MbarrierWait))
                         for s in walk(kernels[0].body))
        target = "sm_90" if needs_bulk else "sm_89"
    meta_path = path.with_name("meta.json")
    if not meta_path.exists():
        raise CompileError(f"missing launch contract: {meta_path}")
    problem = json.loads(meta_path.read_text(encoding="utf-8"))
    matches = [m for m in problem["kernels"] if m["entry"] == kernels[0].name]
    if len(matches) != 1:
        raise CompileError(f"meta.json needs exactly one entry for {kernels[0].name}")
    if compensated_dots is None:
        compensated_dots = (
            not accurate_dots
            and not fuse_accumulations
            and any(float(problem.get(key, 0)) > 0 for key in ("rtol", "atol"))
        )
    return Compiler(
        graph,
        matches[0],
        target=target,
        synchronous_copy=synchronous_copy,
        unroll_budget=unroll_budget,
        fuse_accumulations=fuse_accumulations,
        accurate_dots=accurate_dots,
        compensated_dots=compensated_dots,
        launch_bounds=launch_bounds,
    ).compile()
