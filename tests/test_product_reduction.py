"""Nonempty product folds expressed directly in the permitted typed IR.

Emmy 0.3.11 cannot trace the phase3 aten.prod expression, but multiplication
Accum and cooperative merge nodes are valid required vocabulary. Powers of two
make every individual product exact; host references sum integer exponents.
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
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt.blocks import StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Let, Load, Write
from ir_cases import axis, case, expression, literal, op, write_case
from ptx_backend import compile_file
from tools.check_compiler import assemble, local_run


def product_cases():
    i, k = Var("i"), Var("k")
    for name, threads, width, group, tree in (
        ("product_shuffle_varied_trip_counts", 40, 5, 8, False),
        ("product_tree_varied_trip_counts", 64, 3, 64, True),
    ):
        exponents = (
            np.arange(threads)[:, None] + 2 * np.arange(width)[None, :]
        ) % 3 - 1
        data = np.ldexp(np.ones_like(exponents, dtype=np.float32), exponents)
        trips = np.arange(threads) % (width + 1)
        per_thread_exponent = np.asarray(
            [
                sum(int(exponent) for exponent in row[:count])
                for row, count in zip(exponents, trips, strict=True)
            ]
        )
        grouped_exponent = np.repeat(
            per_thread_exponent.reshape(-1, group).sum(axis=1), group
        )
        expected = np.column_stack(
            (
                np.ldexp(np.ones(threads, np.float32), per_thread_exponent),
                np.ldexp(np.ones(threads, np.float32), grouped_exponent),
            )
        )
        merge = (Assign("product", op("multiply"), ("product", "other"), F32),)
        instructions = [
            StridedLoop(
                axis("k", width),
                literal(0),
                literal(1),
                (
                    Load("value", "x", (i, k), dtype=F32),
                    Accum("product", "value", op("multiply"), F32),
                ),
                end=expression("%", i, literal(width + 1)),
            ),
            Let("local_product", Var("product"), F32),
        ]
        if tree:
            instructions.extend(
                (
                    Smem("partials", (threads,), "float"),
                    Write("partials", (i,), "product", value_dtype=F32),
                    Sync(),
                    TreeHalve(
                        ("partials",),
                        ("product",),
                        ("other",),
                        merge,
                        threads,
                        "i",
                        F32,
                    ),
                )
            )
        else:
            instructions.append(
                WarpShuffle(("product",), ("other",), merge, group, F32)
            )
        instructions.append(
            Write(
                "out",
                (i, literal(0)),
                values=("local_product", "product"),
                value_dtype=F32,
            )
        )
        value = case(
            name,
            (Tile((axis("i", threads),), tuple(instructions), block_threads=threads),),
            {"x": data.shape},
            expected.shape,
            lambda x: expected,
            block=threads,
        )
        value[1]["rtol"] = value[1]["atol"] = 0
        descriptor = next(
            arg for arg in value[1]["kernels"][0]["args"] if arg["name"] == "x"
        )
        descriptor.pop("seed")
        descriptor["data"] = data.ravel().tolist()
        yield value


def materialize(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in product_cases():
        path = write_case(directory / value[1]["id"], value)
        rows.append({"id": value[1]["id"], "ir": str(path.relative_to(directory))})
    manifest = {
        "scope": "Exact exponent-sum host references materialized before PTX evaluation.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": rows,
        "files_sha256": {
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
    print(f"Materialized {len(rows)} exact product cases at {directory}")


class ProductReductionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_nonempty_products_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in product_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    assemble(compile_file(path), "sm_89")

    @unittest.skipUnless(
        os.environ.get("EMMY_GPU_TESTS") == "1",
        "set EMMY_GPU_TESTS=1 for CUDA execution",
    )
    def test_nonempty_products_match_exact_references(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in product_cases():
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
                        {**request, "ptx": compile_file(path), "adapt_target": True}
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
