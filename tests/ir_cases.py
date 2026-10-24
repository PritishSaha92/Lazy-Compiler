"""Small typed-IR counterexamples with mathematical references, not PTX snapshots.

These fixtures do not invoke Emmy's renderer or any CUDA compiler. They construct
the same typed nodes accepted by the backend and serialize with Graph.to_dict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, I32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir import InputOp
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    KernelOp,
    Smem,
    Sync,
    Tile,
    TreeHalve,
    WarpShuffle,
)
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write


def axis(name, extent):
    return Axis(name, Dim(extent))


def op(name):
    return ElementwiseImpl(
        {"sub": "subtract", "mul": "multiply", "neg": "negative"}.get(name, name)
    )


def literal(value):
    return Literal(value, "int" if isinstance(value, int) else "float")


def expression(operation, left, right):
    return BinaryExpr(operation, left, right)


def case(
    name, body, shapes, output_shape, reference, block=64, grid=1, reverse_args=False
):
    graph = Graph()
    inputs = {}
    for seed, (buffer, shape) in enumerate(shapes.items()):
        graph.add_node(InputOp(), [], Tensor(buffer, tuple(map(Dim, shape)), F32))
        inputs[buffer] = (
            np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
        )
    graph.inputs = list(shapes)
    graph.outputs = ["out"]
    entry = "k_regression_" + name
    graph.add_node(
        KernelOp(body=tuple(body), name=entry),
        list(shapes),
        Tensor("out", tuple(map(Dim, output_shape)), F32),
    )
    names = list(shapes) + ["out"]
    if reverse_args:
        names.reverse()
    descriptors = {
        buffer: {"name": buffer, "dtype": "f32", "shape": list(shape), "seed": seed}
        for seed, (buffer, shape) in enumerate(shapes.items())
    }
    descriptors["out"] = {
        "name": "out",
        "dtype": "f32",
        "shape": list(output_shape),
        "output": True,
    }
    kernel = {
        "entry": entry,
        "grid": [grid, 1, 1],
        "block": [block, 1, 1],
        "smem_bytes": 0,
        "zero_outputs": [],
        "arg_order": names,
        "args": [descriptors[name] for name in names],
        "final": True,
    }
    metadata = {
        "id": name,
        "graph_inputs": list(shapes),
        "graph_outputs": ["out"],
        "rtol": 1e-5,
        "atol": 1e-6,
        "kernels": [kernel],
    }
    return graph, metadata, {"out": np.asarray(reference(**inputs), dtype=np.float32)}


def cases():
    i = Var("i")
    j = Var("j")
    # A non-warp-sized tile and output-first argument order. Every operation is
    # deliberately noncommutative, so swapping inputs is not accidentally valid.
    mirrored = TernaryExpr(
        expression("<", i, literal(18)), i, expression("-", literal(36), i)
    )
    body = (
        Let("j", mirrored, I32),
        Load("a", "x", (j,), dtype=F32),
        Load("b", "y", (i,), dtype=F32),
        Cond(
            expression("==", expression("%", i, literal(2)), literal(0)),
            (
                Assign("ab", op("sub"), ("a", "b"), F32),
                Write("out", (i,), "ab", value_dtype=F32),
            ),
            (
                Assign("ba", op("sub"), ("b", "a"), F32),
                Write("out", (i,), "ba", value_dtype=F32),
            ),
        ),
    )
    positions = np.arange(37)
    indices = np.where(positions < 18, positions, 36 - positions)
    yield case(
        "conditional_odd_reordered",
        (Tile((axis("i", 37),), body),),
        {"x": (37,), "y": (37,)},
        (37,),
        lambda x, y: np.where(positions % 2 == 0, x[indices] - y, y - x[indices]),
        reverse_args=True,
    )

    for width in (1, 2, 4):
        names = tuple(f"value_{lane}" for lane in range(width))
        results = tuple(f"result_{lane}" for lane in range(width))
        index = expression("*", i, literal(width))
        body = (
            (Load(input="x", index=(index,), names=names, dtype=F32),)
            + tuple(
                Assign(result, op("neg"), (value,), F32)
                for value, result in zip(names, results)
            )
            + (Write("out", (index,), values=results, value_dtype=F32),)
        )
        yield case(
            f"vector_{width}",
            (Tile((axis("i", 19),), body),),
            {"x": (19 * width,)},
            (19 * width,),
            lambda x: -x,
        )

    # Accum.base is a rescaled previous value; seed=False must preserve Init.
    k = Var("k")
    body = (
        Init("state", 1.0, dtype=F32),
        Let("half", literal(0.5), F32),
        Loop(
            axis("k", 5),
            (
                Load("value", "x", (i, k), dtype=F32),
                Assign("scaled", op("mul"), ("state", "half"), F32),
                Accum("state", "value", op("add"), F32, ("k",), "scaled"),
            ),
            seed=False,
        ),
        Write("out", (i,), "state", value_dtype=F32),
    )

    def recurrence(x):
        state = np.ones(x.shape[0], dtype=np.float32)
        for column in range(x.shape[1]):
            state = state * np.float32(0.5) + x[:, column]
        return state

    yield case(
        "rescaled_seed_false",
        (Tile((axis("i", 7),), body),),
        {"x": (7, 5)},
        (7,),
        recurrence,
        block=32,
    )

    # Inner sums reset at each outer iteration, while the outer sum survives.
    body = (
        Loop(
            axis("j", 3),
            (
                Loop(
                    axis("k", 5),
                    (
                        Load("value", "x", (i, j, k), dtype=F32),
                        Accum("inner", "value", op("add"), F32, ("k",)),
                    ),
                ),
                Accum("outer", "inner", op("add"), F32, ("j",)),
            ),
        ),
        Write("out", (i,), "outer", value_dtype=F32),
    )
    yield case(
        "nested_accumulator_scope",
        (Tile((axis("i", 7),), body),),
        {"x": (7, 3, 5)},
        (7,),
        lambda x: x.astype(np.float64).sum(axis=(1, 2)),
        block=32,
    )

    for length, threads in ((8, 8), (8, 40), (16, 16), (32, 32)):
        combine = (Assign("acc", op("add"), ("acc", "other"), F32),)
        body = (
            Load("acc", "x", (i,), dtype=F32),
            WarpShuffle(("acc",), ("other",), combine, length, F32),
            Write("out", (i,), "acc", value_dtype=F32),
        )
        yield case(
            f"shuffle_{length}_threads_{threads}",
            (Tile((axis("i", threads),), body),),
            {"x": (threads,)},
            (threads,),
            lambda x, n=length: np.repeat(
                x.reshape(-1, n).astype(np.float64).sum(axis=1), n
            ),
            block=threads,
        )

    segment, column = Var("segment"), Var("column")
    combine = (Assign("acc", op("add"), ("acc", "other"), F32),)
    body = (
        Smem("scratch", (4, 2), "float"),
        Load("acc", "x", (segment, column), dtype=F32),
        Write("scratch", (segment, column), "acc", value_dtype=F32),
        Sync(),
        TreeHalve(
            ("scratch",),
            ("acc",),
            ("other",),
            combine,
            4,
            "segment",
            F32,
            inner=("column", 2),
        ),
        Write("out", (segment, column), "acc", value_dtype=F32),
    )
    yield case(
        "segmented_tree_broadcast",
        (Tile((axis("segment", 4), axis("column", 2)), body),),
        {"x": (4, 2)},
        (4, 2),
        lambda x: np.broadcast_to(x.astype(np.float64).sum(axis=0), (4, 2)),
        block=8,
    )


def write_case(folder: Path, value):
    graph, meta, expected = value
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (meta["kernels"][0]["entry"] + ".ir.json")
    # Emmy's JSON dumps use repr for typed statement bodies, recovered by its
    # own Graph.from_dict deserializer; no body-string parsing is done here.
    path.write_text(
        json.dumps(graph.to_dict(), indent=2, default=repr), encoding="utf-8"
    )
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez(folder / "expected.npz", **expected)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for value in cases():
        path = write_case(args.out / value[1]["id"], value)
        print(path)
