"""Small independent interaction fixtures; this module never launches a GPU.

``interaction_cases()`` returns ordinary (Graph, metadata, expected) packages for
the remote runner. All inputs are explicit FP32 integers and every output uses
an exact oracle. CPU checks cover emitted guard dominance and assembly; they do
not establish native async execution safety or numerical GPU correctness.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
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
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write
from ir_cases import axis, case, expression, literal, op, write_case
from ptx_backend import compile_file
from tools.check_compiler import assemble


def explicit_case(name, body, inputs, expected, contract, *, block=32):
    """Freeze inputs, references, launch geometry and the fixture's contract."""
    value = case(
        "interaction_" + name,
        tuple(body),
        {key: array.shape for key, array in inputs.items()},
        np.shape(expected),
        lambda **_: expected,
        block=block,
    )
    for argument in value[1]["kernels"][0]["args"]:
        if argument["name"] in inputs:
            argument.pop("seed", None)
            argument["data"] = inputs[argument["name"]].ravel().tolist()
    value[1]["rtol"] = value[1]["atol"] = 0
    value[1]["interaction_contract"] = contract
    return value[0], value[1], {"out": np.asarray(expected, dtype=np.float32)}


def guard_cases():
    """Skipped statement loads have invalid addresses, with state live afterward.

    Cond permits accumulator targets to cross arm boundaries; ordinary SSA
    names remain scoped to each arm. The tests use Init/Accum for that reason.
    No invalid load executes in either fixture's source-defined control flow.
    """
    i = Var("i")
    x = np.arange(17, dtype=np.float32) * 3 - 11
    y = np.arange(17, dtype=np.float32) * 5 + 101
    body = (
        Init("state", 7.0, dtype=F32),
        Assign("before", op("copy"), ("state",), F32),
        Cond(
            expression(">", i, literal(0)),
            (
                Load("previous", "x", (expression("-", i, literal(1)),), dtype=F32),
                Accum("state", "previous", op("add"), F32),
            ),
            (
                Let("boundary", literal(-13.0), F32),
                Accum("state", "boundary", op("add"), F32),
            ),
        ),
        Load("current", "y", (i,), dtype=F32),
        Assign("combined", op("add"), ("current", "state"), F32),
        Assign("result", op("add"), ("combined", "before"), F32),
        Write("out", (i,), "result", value_dtype=F32),
    )
    # Mathematical neighbor selection, not interpretation of emitted PTX.
    neighbor = np.concatenate((np.asarray([-13], np.float32), x[:-1]))
    yield explicit_case(
        "guard_previous_live_state",
        (Tile((axis("i", 17),), body, block_threads=32),),
        {"x": x, "y": y},
        y + neighbor + 14,
        {
            "formula": "out[i] = y[i] + (x[i-1] if i>0 else -13) + 14",
            "invalid_if_speculated": "x[-1] at i=0; Tile excludes threads 17..31",
            "guarded_input": "x",
            "minimum_guard_edges": 2,
            "state": "snapshot 7 survives the branch; state becomes 7+neighbor",
        },
    )

    row, column = Var("row"), Var("column")
    x = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) * 3 - 41
    y = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) * 2 + 53
    body = (
        Init("state", 7.0, dtype=F32),
        Assign("before", op("copy"), ("state",), F32),
        Cond(
            expression("<", row, literal(4)),
            (
                Let("row_bias", literal(100.0), F32),
                Accum("state", "row_bias", op("add"), F32),
                Cond(
                    expression("<", column, literal(6)),
                    (
                        Load(
                            "diagonal",
                            "x",
                            (
                                expression("+", row, literal(1)),
                                expression("+", column, literal(1)),
                            ),
                            dtype=F32,
                        ),
                        Accum("state", "diagonal", op("add"), F32),
                    ),
                    (
                        Let("column_boundary", literal(-19.0), F32),
                        Accum("state", "column_boundary", op("add"), F32),
                    ),
                ),
            ),
            (
                Let("row_boundary", literal(-23.0), F32),
                Accum("state", "row_boundary", op("add"), F32),
            ),
        ),
        Load("current", "y", (row, column), dtype=F32),
        Assign("combined", op("add"), ("current", "state"), F32),
        Assign("result", op("add"), ("combined", "before"), F32),
        Write("out", (row, column), "result", value_dtype=F32),
    )
    contribution = np.full((5, 7), -23, np.float32)
    contribution[:-1, :-1] = 100 + x[1:, 1:]
    contribution[:-1, -1] = 81
    yield explicit_case(
        "guard_nested_row_tail_state",
        (Tile((axis("row", 5), axis("column", 7)), body, block_threads=64),),
        {"x": x, "y": y},
        y + contribution + 14,
        {
            "formula": (
                "out[r,c] = y[r,c] + 14 + (-23 if r==4 else "
                "100 + (-19 if c==6 else x[r+1,c+1]))"
            ),
            "invalid_if_speculated": (
                "last-row x[r+1,c+1] is outside x; inner tail at r=3,c=6 "
                "would load x[4,7] outside x; Tile excludes threads 35..63"
            ),
            "guarded_input": "x",
            "minimum_guard_edges": 3,
            "state": "two nested branch levels mutate state; snapshot remains 7",
        },
        block=64,
    )


def ragged_row_cases():
    """Two legal four-value chunks plus three scalar tail elements per row.

    The source row pitch is 13 floats and destination pitch is 11. Rows in each
    group of four therefore cycle through four alignment classes. With offset
    unrolling, the first row proves 16-byte alignment and the others cannot use
    a four-value memory instruction. In rolled emission the conservative path
    is also valid. The last two source columns are poison padding, never read.
    """
    group, offset, chunk, tail = map(Var, ("group", "offset", "chunk", "tail"))
    row = expression("+", expression("*", group, literal(4)), offset)
    column = expression("*", chunk, literal(4))
    names = tuple(f"loaded_{j}" for j in range(4))
    results = tuple(f"negated_{j}" for j in range(4))
    grouped = (
        Load(input="x", index=(row, column), names=names, dtype=F32),
        *(Assign(dst, op("negative"), (src,), F32) for src, dst in zip(names, results)),
        Write("out", (row, column), values=results, value_dtype=F32),
    )
    tail_column = expression("+", literal(8), tail)
    cleanup = (
        Load("remainder", "x", (row, tail_column), dtype=F32),
        Assign("negative_remainder", op("negative"), ("remainder",), F32),
        Write("out", (row, tail_column), "negative_remainder", value_dtype=F32),
    )
    body = (
        Loop(
            axis("offset", 4),
            (
                Loop(axis("chunk", 2), grouped, unroll=True),
                Loop(axis("tail", 3), cleanup, unroll=True),
            ),
            unroll=True,
        ),
    )
    x = np.arange(12 * 13, dtype=np.float32).reshape(12, 13) * 3 - 211
    x[:, 11:] = -8000 - np.arange(24, dtype=np.float32).reshape(12, 2)
    yield explicit_case(
        "ragged_inner_row_alignment_cleanup",
        (Tile((axis("group", 3),), body, block_threads=32),),
        {"x": x},
        -x[:, :11],
        {
            "formula": "out[r,c] = -x[r,c], 0<=r<12, 0<=c<11",
            "accesses": "grouped columns 0..3 and 4..7; scalar columns 8,9,10",
            "alignment": "input stride 52 bytes; output stride 44 bytes",
            "padding": "input columns 11,12 are explicit negative sentinels",
            "ownership": "three active threads each own four complete rows",
        },
    )


def rotating_async_cases():
    """Seven committed stages reuse two slots, with distinct producer/consumer warps.

    Warp 0 issues one 4-byte copy per lane; warp 1 reads reversed lanes. All 64
    threads commit, wait and reach both CTA barriers for every consumed stage.
    Consumer threads commit empty groups, which are legal. A producer wait plus
    a CTA barrier precedes cross-warp reads, and a second CTA barrier precedes
    slot overwrite. After the five steady-state iterations, two explicit drain
    stages consume the remaining groups. The final row observes persistent sum
    state, so silently dropping a final stage or resetting that state is visible.
    """
    lane, stage, warm = Var("lane"), Var("stage"), Var("warm")
    producer = expression("==", Var("warp"), literal(0))
    consumer = expression("==", Var("warp"), literal(1))
    peer = expression("-", literal(31), lane)

    def issue(index):
        slot = expression("%", index, literal(2))
        return (
            Cond(
                producer,
                (CpAsyncCopy("slots", (slot, lane), "x", (index, lane), nbytes=4),),
            ),
            CpAsyncCommit(),
        )

    def consume(index, pending):
        slot = expression("%", index, literal(2))
        return (
            CpAsyncWait(pending),
            Sync(),
            Cond(
                consumer,
                (
                    Load("peer_value", "slots", (slot, peer), dtype=F32),
                    Write("out", (index, lane), "peer_value", value_dtype=F32),
                    Accum("checksum", "peer_value", op("add"), F32),
                ),
            ),
            Sync(),
        )

    body = (
        Smem("slots", (2, 32), "float", align=16),
        Init("checksum", 0.0, dtype=F32),
        Loop(axis("warm", 2), issue(warm), unroll=True),
        Loop(
            axis("stage", 5),
            consume(stage, 1) + issue(expression("+", stage, literal(2))),
            unroll=True,
            seed=False,
        ),
        *consume(literal(5), 1),
        *consume(literal(6), 0),
        Cond(consumer, (Write("out", (literal(7), lane), "checksum", value_dtype=F32),)),
    )
    x = (
        np.arange(7, dtype=np.float32)[:, None] * 100
        + np.arange(32, dtype=np.float32)[None, :]
        - 200
    )
    expected = np.concatenate(
        (x[:, ::-1], x[:, ::-1].sum(axis=0, dtype=np.float64)[None, :]), axis=0
    ).astype(np.float32)
    yield explicit_case(
        "async_two_slots_seven_stages_cross_warp",
        (Tile((axis("cell", 64),), body, block_threads=64),),
        {"x": x},
        expected,
        {
            "formula": "out[t,l]=100*t-169-l for t<7; out[7,l]=917-7*l",
            "participants": "warp 0 copies, warp 1 consumes; all 64 commit/wait/barrier",
            "pipeline": "2 prefill + 5 consume/refill + 2 drain; 7 commits total",
            "waits": "six wait_group 1 stages, final wait_group 0 stage",
            "shared": "256 static bytes, 2 slots x 32 FP32 values; 4-byte aligned copies",
            "ordering": "wait then CTA barrier before reads; CTA barrier before reuse",
        },
        block=64,
    )


def interaction_cases():
    """Export all four independently checkable cases for the remote harness."""
    yield from guard_cases()
    yield from ragged_row_cases()
    yield from rotating_async_cases()


def _ptx_control_graph(ptx):
    """Read emitted instructions into a CFG; labels are not tied to emitter names."""
    instructions, labels = [], {}
    for line in ptx.splitlines():
        line = line.split("//", 1)[0].strip()
        if not line or line.startswith(".") or line in ("{", "}"):
            continue
        if line.endswith(":"):
            labels[line[:-1]] = len(instructions)
        elif line.endswith(";"):
            instructions.append(line)
    edges = {}
    for index, instruction in enumerate(instructions):
        branch = re.fullmatch(r"(@!?%\w+\s+)?bra\s+(\w+);", instruction)
        if branch:
            edges[index] = [labels[branch[2]]]
            if branch[1]:
                edges[index].append(index + 1)
        elif instruction == "ret;":
            edges[index] = []
        else:
            edges[index] = [index + 1] if index + 1 < len(instructions) else []
    return instructions, edges


def _reachable(edges, omitted=None):
    pending, seen = [0], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(
            target for target in edges.get(current, ()) if (current, target) != omitted
        )
    return seen


def _loads_from_parameter(instructions, parameter):
    """Trace pointer provenance through emitted address arithmetic only.

    Values read from memory do not inherit pointer provenance. Address registers
    in these straight-line guarded fixtures are immutable; this is intentionally
    a test-specific analysis, not an optimizer or general PTX interpreter.
    """
    pointers, loads = set(), []
    for index, instruction in enumerate(instructions):
        registers = re.findall(r"%\w+", instruction)
        if instruction.startswith("ld.param.") and f"[param_{parameter}]" in instruction:
            pointers.add(registers[0])
        elif instruction.startswith("ld.global."):
            address = re.search(r"\[([^]]+)\]", instruction)
            if address and set(re.findall(r"%\w+", address[1])) & pointers:
                loads.append(index)
        elif instruction.startswith(("cvta.", "add.", "mov.")) and registers:
            if set(registers[1:]) & pointers:
                pointers.add(registers[0])
    return loads


def materialize(directory):
    """Write reproducible input/expected packages without compiling or executing."""
    directory = Path(directory)
    rows = []
    for value in interaction_cases():
        path = write_case(directory / value[1]["id"], value)
        rows.append(
            {
                "id": value[1]["id"],
                "ir": path.relative_to(directory).as_posix(),
                "contract": value[1]["interaction_contract"],
            }
        )
    manifest = {
        "suite": "interaction fixtures",
        "gpu_executed": False,
        "cases": rows,
        "sha256": {
            path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Materialized {len(rows)} interaction cases at {directory}")


class InteractionTests(unittest.TestCase):
    def test_explicit_contracts_and_exact_reference_boundaries(self):
        values = list(interaction_cases())
        self.assertEqual(len(values), 4)
        self.assertEqual(len({value[1]["id"] for value in values}), 4)
        for _, metadata, expected in values:
            with self.subTest(case=metadata["id"]):
                self.assertEqual((metadata["rtol"], metadata["atol"]), (0, 0))
                kernel = metadata["kernels"][0]
                self.assertEqual(kernel["grid"], [1, 1, 1])
                self.assertEqual(kernel["smem_bytes"], 0)  # Shared storage is static.
                for argument in kernel["args"]:
                    if not argument.get("output"):
                        self.assertNotIn("seed", argument)
                        self.assertEqual(len(argument["data"]), np.prod(argument["shape"]))
                self.assertEqual(expected["out"].dtype, np.float32)
        np.testing.assert_array_equal(values[0][2]["out"][[0, 1, 16]], [102, 109, 229])
        np.testing.assert_array_equal(
            values[1][2]["out"][[0, 3, 4], [0, 6, 6]], [150, 202, 112]
        )
        np.testing.assert_array_equal(
            values[2][2]["out"][[0, 11], [10, 10]], [181, -248]
        )
        np.testing.assert_array_equal(
            values[3][2]["out"][:, 0], [-169, -69, 31, 131, 231, 331, 431, 917]
        )
        np.testing.assert_array_equal(
            values[3][2]["out"][:, 31], [-200, -100, 0, 100, 200, 300, 400, 700]
        )

    def test_false_arm_loads_require_all_guard_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in guard_cases():
                with self.subTest(case=value[1]["id"]):
                    path = write_case(Path(directory) / value[1]["id"], value)
                    instructions, edges = _ptx_control_graph(compile_file(path))
                    contract = value[1]["interaction_contract"]
                    order = value[1]["kernels"][0]["arg_order"]
                    loads = _loads_from_parameter(
                        instructions, order.index(contract["guarded_input"])
                    )
                    self.assertEqual(len(loads), 1, "each guarded input has one source Load")
                    load = loads[0]
                    self.assertIn(load, _reachable(edges))
                    controlling_edges = [
                        (source, destination)
                        for source, destinations in edges.items()
                        if len(destinations) == 2
                        for destination in destinations
                        if load not in _reachable(edges, (source, destination))
                    ]
                    # Includes the Tile guard. Hoisting past either nested Cond
                    # loses a required edge even if it remains inside the Tile.
                    self.assertGreaterEqual(
                        len(controlling_edges), contract["minimum_guard_edges"]
                    )

    def test_ragged_row_vector_and_scalar_paths_are_both_present(self):
        with tempfile.TemporaryDirectory() as directory:
            value = next(ragged_row_cases())
            path = write_case(Path(directory), value)
            ptx = compile_file(path, unroll_budget=1024)
            self.assertIn("ld.global.v4.f32", ptx)
            self.assertIn("ld.global.f32", ptx)
            self.assertIn("st.global.v4.f32", ptx)
            self.assertIn("st.global.f32", ptx)
            # Address alignment and touched-element bounds are distinct facts.
            for row in range(12):
                for start in (0, 4):
                    self.assertLess(start + 3, 11)
                    self.assertEqual((13 * row + start) % 4 == 0, row % 4 == 0)
                    self.assertEqual((11 * row + start) % 4 == 0, row % 4 == 0)

    def test_async_prefill_reuse_and_drain_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(Path(directory), next(rotating_async_cases()))
            ptx = compile_file(path, unroll_budget=1024)
            self.assertEqual(ptx.count("cp.async.ca.shared.global"), 7)
            self.assertEqual(ptx.count("cp.async.commit_group;"), 7)
            self.assertEqual(ptx.count("cp.async.wait_group 1;"), 6)
            self.assertEqual(ptx.count("cp.async.wait_group 0;"), 1)
            self.assertEqual(ptx.count("bar.sync 0;"), 14)

    @unittest.skipUnless(shutil.which("ptxas"), "ptxas not installed")
    def test_rolled_and_unrolled_interactions_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in interaction_cases():
                path = write_case(Path(directory) / value[1]["id"], value)
                alternatives = (False, True) if "async_two_slots" in value[1]["id"] else (False,)
                for budget in (0, 1024):
                    for synchronous in alternatives:
                        with self.subTest(
                            case=value[1]["id"], budget=budget, synchronous=synchronous
                        ):
                            assemble(
                                compile_file(
                                    path, unroll_budget=budget, synchronous_copy=synchronous
                                ),
                                "sm_89",
                            )


if __name__ == "__main__":
    if "--materialize" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--materialize", type=Path, required=True)
        materialize(parser.parse_args().materialize)
    else:
        unittest.main()
