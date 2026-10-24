"""Transport cases with host references frozen before PTX evaluation.

The native tests cover 4/8/16-byte copies and wait_group 1. Local devices below
sm_80 use an explicit synchronous alternative; that checks addresses and CTA
ordering only. Native asynchronous completion must be checked on the runner.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    Smem,
    Sync,
    Tile,
)
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.leaves import Load, Write
from ir_cases import axis, case, expression, literal, write_case
from ptx_backend import compile_file
from tools.check_compiler import assemble, local_run


def concrete(name, body, data, expected, *, grid=1, shared_bytes=0):
    """Exact integer-valued FP32 transport; no approximate reference arithmetic."""
    result = case(
        name,
        body,
        {"x": data.shape},
        expected.shape,
        lambda x: expected,
        block=32,
        grid=grid,
    )
    metadata = result[1]
    metadata["rtol"] = metadata["atol"] = 0
    kernel = metadata["kernels"][0]
    kernel["smem_bytes"] = shared_bytes
    descriptor = next(arg for arg in kernel["args"] if arg["name"] == "x")
    descriptor.pop("seed")
    descriptor["data"] = data.ravel().tolist()
    return result


def transport_cases():
    lane, batch, stripe = Var("lane"), Var("batch"), Var("stripe")
    reverse_lane = expression("-", literal(31), lane)
    for nbytes in (4, 8, 16):
        width, batches = nbytes // 4, 3
        data = (
            np.arange(batches * 2 * 32 * width, dtype=np.float32).reshape(
                batches, 2, 32, width
            )
            - 317
        )
        lane_start = expression("*", lane, literal(width))
        peer_start = expression("*", reverse_lane, literal(width))
        names = tuple(f"v{i}" for i in range(width))
        body = (
            Smem("first", (32 * width,), "float", align=16),
            Smem("second", (32 * width,), "float", align=16),
            CpAsyncCopy(
                "first",
                (lane_start,),
                "x",
                (batch, literal(0), lane, literal(0)),
                nbytes=nbytes,
            ),
            CpAsyncCommit(),
            CpAsyncCopy(
                "second",
                (lane_start,),
                "x",
                (batch, literal(1), lane, literal(0)),
                nbytes=nbytes,
            ),
            CpAsyncCommit(),
            # The first group must be complete even though the second may be
            # in flight. The CTA barrier orders peer reads after all waits.
            CpAsyncWait(1),
            Sync(),
            Load(input="first", index=(peer_start,), names=names, dtype=F32),
            Write(
                "out",
                (batch, literal(0), lane, literal(0)),
                values=names,
                value_dtype=F32,
            ),
            CpAsyncWait(0),
            Sync(),
            Load(input="second", index=(peer_start,), names=names, dtype=F32),
            Write(
                "out",
                (batch, literal(1), lane, literal(0)),
                values=names,
                value_dtype=F32,
            ),
        )
        yield concrete(
            f"async_two_groups_{nbytes}_bytes",
            (Tile((axis("batch", batches), axis("cell", 32)), body, block_threads=32),),
            data,
            data[:, :, ::-1, :].copy(),
            grid=batches,
        )

    # Physical layout inspection makes the write permutation observable. The
    # reference permutes whole eight-element chunks with a host chunk table;
    # the ordinary shared loads deliberately do not undo the permutation.
    for mode, columns, mask in (
        ("B32", 64, 1),
        ("B64", 64, 3),
        ("B128", 64, 7),
        ("B128@7", 128, 7),
    ):
        rows, width, batches = 8, 4, 2
        data = (
            np.arange(batches * rows * columns, dtype=np.float32).reshape(
                batches, rows, columns
            )
            - 1001
        )
        chunks = data.reshape(batches, rows, columns // 8, 8)
        expected_chunks = np.empty_like(chunks)
        for row in range(rows):
            destination_order = [chunk ^ (row & mask) for chunk in range(columns // 8)]
            expected_chunks[:, row, destination_order, :] = chunks[:, row, :, :]
        expected = expected_chunks.reshape(data.shape)
        rows_per_stripe = 32 * width // columns
        row = expression(
            "+",
            expression("*", stripe, literal(rows_per_stripe)),
            expression("/", lane, literal(columns // width)),
        )
        column = expression(
            "*", expression("%", lane, literal(columns // width)), literal(width)
        )
        names = tuple(f"value_{i}" for i in range(width))
        body = (
            Smem("layout", (rows, columns), "float", align=16),
            Loop(
                axis("stripe", rows // rows_per_stripe),
                (
                    Load(input="x", index=(batch, row, column), names=names, dtype=F32),
                    Write(
                        "layout",
                        (row, column),
                        values=names,
                        value_dtype=F32,
                        swizzle=mode,
                    ),
                ),
                unroll=True,
            ),
            Sync(),
            Loop(
                axis("stripe", rows // rows_per_stripe),
                (
                    Load(input="layout", index=(row, column), names=names, dtype=F32),
                    Write("out", (batch, row, column), values=names, value_dtype=F32),
                ),
                unroll=True,
            ),
        )
        yield concrete(
            "shared_write_swizzle_" + mode.replace("@", "_shift_"),
            (Tile((axis("batch", batches), axis("cell", 32)), body, block_threads=32),),
            data,
            expected,
            grid=batches,
        )

    # 8193 floats occupy 32772 bytes. A second 128-byte-aligned allocation
    # starts at 32896 and ends at 49284, above the static 48 KiB threshold.
    # Distinct writes on both sides of both buffers expose overlap and offset
    # mistakes. Only the initialized cells are read; padding is never observed.
    data = (np.arange(128, dtype=np.float32).reshape(4, 32) - 37) * 3
    instructions = [
        Smem("left", (8193,), "float", align=16),
        Smem("right", (4097,), "float", align=128),
    ]
    for row, buffer, start in (
        (0, "left", 0),
        (1, "left", 8161),
        (2, "right", 0),
        (3, "right", 4065),
    ):
        instructions.extend(
            (
                Load("input", "x", (literal(row), lane), dtype=F32),
                Write(
                    buffer,
                    (expression("+", literal(start), lane),),
                    "input",
                    value_dtype=F32,
                ),
            )
        )
    instructions.append(Sync())
    for row, buffer, start in (
        (0, "left", 0),
        (1, "left", 8161),
        (2, "right", 0),
        (3, "right", 4065),
    ):
        instructions.extend(
            (
                Load(
                    "result",
                    buffer,
                    (expression("+", literal(start), reverse_lane),),
                    dtype=F32,
                ),
                Write("out", (literal(row), lane), "result", value_dtype=F32),
            )
        )
    yield concrete(
        "dynamic_pool_distinct_alignment",
        (Tile((axis("cell", 32),), tuple(instructions), block_threads=32),),
        data,
        data[:, ::-1].copy(),
        shared_bytes=49284,
    )

    # A row stride of seven alternates alignment. Four consecutive values
    # remain a legal grouped Load even when one vector instruction is illegal.
    data = np.arange(19 * 7, dtype=np.float32).reshape(19, 7) - 61
    names = tuple(f"unaligned_{i}" for i in range(4))
    body = (
        Load(input="x", index=(Var("row"), literal(1)), names=names, dtype=F32),
        Write("out", (Var("row"), literal(0)), values=names, value_dtype=F32),
    )
    yield concrete(
        "unaligned_grouped_load_rows",
        (Tile((axis("row", 19),), body),),
        data,
        data[:, 1:5].copy(),
    )


def materialize(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in transport_cases():
        path = write_case(directory / value[1]["id"], value)
        rows.append({"id": value[1]["id"], "ir": str(path.relative_to(directory))})
    manifest = {
        "scope": "Independent exact host references materialized before compiler or GPU evaluation.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": rows,
        "files": {
            path.relative_to(directory).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Materialized {len(rows)} transport cases at {directory}")


class TransportEdgeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_transport_edges_assemble_native_and_synchronous(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in transport_cases():
                path = write_case(Path(directory) / value[1]["id"], value)
                for synchronous in (False, True):
                    with self.subTest(case=value[1]["id"], synchronous=synchronous):
                        assemble(
                            compile_file(path, synchronous_copy=synchronous), "sm_89"
                        )

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_transport_edges_match_exact_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in transport_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    kernel = value[1]["kernels"][0]
                    request = {
                        key: kernel[key]
                        for key in (
                            "entry",
                            "grid",
                            "block",
                            "smem_bytes",
                            "zero_outputs",
                            "args",
                        )
                    }
                    result = local_run(
                        {
                            **request,
                            "ptx": compile_file(path, synchronous_copy=True),
                            "adapt_target": True,
                        }
                    )
                    self.assertTrue(result.get("ok"), result)
                    output = result["outputs"]["out"]
                    np.testing.assert_array_equal(
                        np.asarray(output["values"]).reshape(output["shape"]),
                        value[2]["out"],
                    )


if __name__ == "__main__":
    if "--materialize" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--materialize", type=Path, required=True)
        materialize(parser.parse_args().materialize)
    else:
        unittest.main()
