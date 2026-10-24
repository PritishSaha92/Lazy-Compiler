"""Analysis must describe structure without leaking workload identity or data."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Smem, Tile
from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Load, Write, Select, SelectBranch
from ir_cases import axis, case, expression, literal
from tools.analyze_ir import analyze_graph, analyze_file


def fixture():
    body = (
        Smem("first", (3,), "float", align=4),
        Smem("second", (2,), "float", align=16),
        Tile((axis("i", 8),), (
            Loop(axis("j", 3), (
                Load("a", "x", (Var("i"),), dtype=F32),
                Write("out", (Var("i"),), "a", value_dtype=F32),
            ), unroll=True),
            StridedLoop(axis("k", 7), literal(1), literal(2), (
                Load("b", "x", (Var("i"),), dtype=F32),
            )),
        )),
    )
    return case("analysis_fixture", body, {"x": (8,)}, (8,), lambda x: x, block=32)


class AnalysisTests(unittest.TestCase):
    def test_counts_offsets_and_unknown_trip_counts(self):
        graph, meta, _ = fixture()
        report = analyze_graph(graph, meta["kernels"][0])
        features = report["features"]
        self.assertEqual(features["shared_bytes_declared"], 20)
        self.assertEqual(features["shared_bytes_aligned"], 24)
        self.assertEqual(report["buffers"]["second"]["aligned_offset"], 16)
        self.assertEqual(features["loop_count"], 2)
        self.assertEqual(features["max_loop_depth"], 1)
        self.assertEqual(features["hinted_loop_work_max"], 6)
        self.assertEqual(features["statement_count"], 8)
        self.assertEqual(report["static_memory_access_site_bytes"]["global_read"], 8)
        self.assertIsNone(report["loops"][1]["actual_trip_count"])

    def test_names_and_host_data_do_not_change_features(self):
        graph, meta, _ = fixture()
        baseline = analyze_graph(graph, meta["kernels"][0])["features"]
        changed = deepcopy(graph)
        for key, node in tuple(changed.nodes.items()):
            if hasattr(node.op, "body"):
                changed.nodes[key] = replace(node, op=replace(node.op, name="unrelated_identity"))
        launch = deepcopy(meta["kernels"][0])
        launch["entry"] = "unrelated_identity"
        launch["args"][0].update(seed=879, data=[-999] * 8)
        self.assertEqual(baseline, analyze_graph(changed, launch)["features"])
        self.assertTrue(all(isinstance(v, int) for v in baseline.values()))

    def test_select_branch_expressions_are_counted(self):
        graph, meta, _ = case("select_analysis", (
            Tile((axis("i", 8),), (
                Load("a", "x", (Var("i"),), dtype=F32),
                Select("b", (SelectBranch("a", expression("<", Var("i"), literal(4))),)),
                Write("out", (Var("i"),), "b", value_dtype=F32),
            )),
        ), {"x": (8,)}, (8,), lambda x: x, block=32)
        report = analyze_graph(graph, meta["kernels"][0])
        self.assertEqual(report["expression_counts"]["BinaryExpr"], 1)

    def test_every_open_problem_has_a_json_serializable_report(self):
        import json
        paths = sorted((ROOT / "problems").glob("*/*.ir.json"))
        self.assertEqual(len(paths), 15)
        for path in paths:
            with self.subTest(problem=path.parent.name):
                report = analyze_file(path)
                json.dumps(report, allow_nan=False)
                self.assertGreater(report["features"]["statement_count"], 0)
                self.assertGreater(report["features"]["block_threads"], 0)


if __name__ == "__main__":
    unittest.main()
