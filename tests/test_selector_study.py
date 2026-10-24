"""Protocol safety and independent evidence checks for the development study."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import selector_study as study


def measured(identifier, family, speeds=(1.1, 1.05, 1.0)):
    variants = {}
    for action, speed in zip(study.ACTIONS, speeds):
        variants[str(action)] = {"pass": True, "timings": [
            {"round": r, "median_us": (100 + r) / speed, "pass": True} for r in range(6)]}
    return {"id": identifier, "family": family, "pass": True, "variants": variants,
            "aliases": {str(a): str(a) for a in study.ACTIONS}}


def test_three_actions_balance_every_position_and_pair_order():
    orders = study.balanced_orders([1024, 0, 256])
    assert len({tuple(order) for order in orders}) == 6
    for action in study.ACTIONS:
        assert [sum(order[position] == action for order in orders) for position in range(3)] == [2, 2, 2]
    for first in study.ACTIONS:
        for second in study.ACTIONS:
            if first != second:
                assert sum(order.index(first) < order.index(second) for order in orders) == 3


def test_deduplicated_orders_stay_balanced():
    assert study.balanced_orders([1024]) == [[1024]] * 6
    assert study.balanced_orders([1024, 0]) == [[1024, 0], [0, 1024]] * 3


def test_split_groups_cannot_leak_even_if_shapes_differ():
    specs = list(study.load_specs(ROOT / "tests/expressions_selector.py").values())
    study.check_splits(specs)
    copied = copy.deepcopy(specs)
    copied[10]["family"] = copied[0]["family"]
    with pytest.raises(ValueError, match="Family crosses"):
        study.check_splits(copied)


def test_only_structural_features_enter_model_and_scaling_is_train_only():
    rows = [{"id": "secret", "family": "forbidden", "features": {"n": 1}},
            {"id": "other", "features": {"n": 9}}]
    x, transform = study.feature_matrix(rows)
    assert transform["keys"] == ["n"]
    assert np.allclose(x.mean(axis=0), 0)
    y, reused = study.feature_matrix([{"features": {"n": 99, "unseen": 1000}}], transform)
    assert reused == transform and y.shape == (1, 1) and y[0, 0] > x[-1, 0]


def test_offline_learners_receive_equal_budget_and_reproduce():
    workloads = [{"id": f"w{i}", "family": f"family{i // 2}", "features": {"cost": i, "loop": i % 3}}
                 for i in range(10)]
    results = [measured(w["id"], w["family"], (1.2, 0.9, 1.0) if i < 5 else (0.9, 1.2, 1.0))
               for i, w in enumerate(workloads)]
    first = study.train_learners(workloads, results)
    second = study.train_learners(workloads, results)
    assert first == second
    for learner in ("tree", "bandit"):
        feedback = first["feedback"][learner]
        assert feedback["logical_observations"] == 20
        assert feedback["distinct_physical_labels"] == 20
        for workload in workloads:
            seen = [o["action"] for o in feedback["observations"] if o["id"] == workload["id"]]
            assert seen[0] == 1024 and seen[1] in (0, 256) and len(seen) == 2


def test_tree_uses_baseline_for_insufficient_action_support():
    x = np.asarray([[0], [1], [2], [3]], dtype=float)
    observations = [{"index": i, "action": 1024, "reward": 0} for i in range(4)]
    observations.append({"index": 0, "action": 0, "reward": 20})
    tree = study.fit_tree(x, observations)
    assert all(study.tree_predict(tree, value) == 1024 for value in x)


def test_static_heuristic_thresholds_and_missing_feature_fallback():
    for value, expected in ((0, 1024), (256, 1024), (257, 256), (1024, 256), (1025, 0)):
        assert study.choose("heuristic", {"features": {"hinted_loop_work_max": value}}, {}) == expected
    assert study.choose("heuristic", {"features": {}}, {}) == 1024


def test_returned_output_gate_rejects_truncation_nonfinite_and_wrong_shape():
    snapshot = {"expected": {"out": {"shape": [2], "dtype": "float32", "values": [1, 2]}}}
    kernel = {"args": [{"name": "out", "dtype": "f32"}]}
    meta = {"rtol": 0, "atol": 0}
    response = {"ok": True, "median_us": 1, "outputs": {"out": {"shape": [2], "values": [1, 2]}}}
    assert study.validate_response(response, snapshot, kernel, meta)["pass"]
    for changed in (
        {"shape": [2], "values": [1, 2], "truncated": True},
        {"shape": [2], "values": [1, float("nan")]},
        {"shape": [1, 2], "values": [1, 2]},
    ):
        bad = {**response, "outputs": {"out": changed}}
        with pytest.raises(ValueError):
            study.validate_response(bad, snapshot, kernel, meta)
    for timing in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            study.validate_response({**response, "median_us": timing}, snapshot, kernel, meta)


def test_duplicate_or_failed_round_cannot_be_a_label():
    row = measured("x", "f")
    row["variants"]["0"]["timings"][1]["round"] = 0
    with pytest.raises(ValueError, match="Incomplete/duplicate"):
        study.timings(row, 0)
    row = measured("x", "f")
    row["variants"]["0"]["pass"] = False
    with pytest.raises(ValueError, match="Failed candidate"):
        study.reward(row, 0)


def test_identical_ptx_reuses_one_physical_measurement_without_fake_speedup():
    row = measured("x", "f")
    row["aliases"]["0"] = "1024"
    assert study.reward(row, 0) == 0


def test_family_bootstrap_and_promotion_gate():
    workloads = [{"id": f"w{i}", "family": f"f{i // 2}", "unique_ptx": 3} for i in range(10)]
    rows = [measured(w["id"], w["family"]) for w in workloads]
    summary = study.summarize(workloads, rows, {w["id"]: 0 for w in workloads}, draws=100)
    assert summary["families"] == 5 and summary["workloads"] == 10
    assert summary["geomean_speedup"] == pytest.approx(1.1)
    assert study.promotion_gate(summary)["pass"]
    for key, value in (("geomean_speedup", 1.02), ("family_bootstrap_95_ci", [1.0, 1.2]),
                       ("worst_workload_speedup", 0.95), ("distinct_workloads", 5)):
        assert not study.promotion_gate({**summary, key: value})["pass"]
    assert not study.promotion_gate(summary, all_correct=False)["pass"]


def test_seal_detects_mutated_artifact(tmp_path):
    artifact = tmp_path / "a.txt"
    artifact.write_text("frozen")
    study.write_json(tmp_path / "seal.json", {"artifacts": {"a.txt": study.sha(artifact)}, "source_evidence": {}})
    study.verify_seal(tmp_path)
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        study.verify_seal(tmp_path)


def test_final_embargo_prevents_refitting(tmp_path):
    (tmp_path / "results-final.json").write_text("{}")
    with patch.object(study, "verify_seal", return_value={}), patch.object(study, "train_learners") as train:
        with pytest.raises(ValueError, match="permanently closed"):
            study.fit(SimpleNamespace(out=tmp_path))
        train.assert_not_called()


def test_remote_requires_explicit_runner_turn(tmp_path):
    with patch.object(study, "remote_post") as remote:
        with pytest.raises(ValueError, match="runner-turn"):
            study.run_remote(SimpleNamespace(out=tmp_path, runner_turn=""))
        remote.assert_not_called()


def test_wrapper_attempt_audit_retains_retry_and_restores_requests():
    import requests
    first = SimpleNamespace(status_code=503, json=lambda: {"stage": "busy"})
    second = SimpleNamespace(status_code=200, json=lambda: {"ok": True})
    captured = []

    def wrapper(server, endpoint, payload):
        requests.post(server + endpoint, json=payload)
        return requests.post(server + endpoint, json=payload).json()

    with patch.object(requests, "post", side_effect=[first, second]) as fake:
        with patch.object(study, "remote_post", wrapper):
            assert study.audit_remote_post("http://no-network", "/run", {}, captured.append) == {"ok": True}
        assert requests.post is fake
    assert [entry["http_status"] for entry in captured] == [503, 200]


def test_invalid_response_numbers_are_retained_as_diagnostics(tmp_path):
    original = {"values": [1.0, float("nan"), float("inf")]}
    converted = study.diagnostic_json(original)
    study.write_json(tmp_path / "response.json", converted)
    assert study.read_json(tmp_path / "response.json")["values"][1] == {"invalid_nonfinite_float": "nan"}
    assert np.isnan(original["values"][1])  # Validation still receives the original invalid number.


def test_product_oracle_checks_all_shapes_and_inputs_and_rejects_zero():
    specs = study.load_specs(ROOT / "tests/expressions_selector.py").values()
    products = [s for s in specs if s["family"] == "sigmoid_product"]
    assert len(products) == 2
    for spec in products:
        for seed, pattern in study.INPUT_CASES:
            assert study.check_reference(spec, seed, pattern)["pass"]
        weak = {**spec, "atol": 1.0}
        with pytest.raises(ValueError, match="all-zero"):
            study.check_reference(weak, 0, "normal")


def test_package_provenance_rejects_tolerance_knob_and_source_mutation(tmp_path):
    spec = {"id": "test", "code": "x", "knobs": "WORK=t128", "rtol": 0.1, "atol": 0.01}
    study.write_json(tmp_path / "meta.json", {k: spec[k] for k in ("code", "rtol", "atol")})
    source = tmp_path / "source.py"
    source.write_text("own source")
    study.record_package_provenance(tmp_path, spec, source)
    assert study.verify_package(tmp_path, spec)["code"] == "x"
    for changed in ({**spec, "atol": 0.02}, {**spec, "knobs": "WORK=t64"}):
        with pytest.raises(ValueError):
            study.verify_package(tmp_path, changed)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source/provenance"):
        study.verify_package(tmp_path, spec)


def complete_mock_evidence(tmp_path, wrong_entry=False):
    """Exercise actual archival path; mocked transport never reaches a network."""
    kernel = {"entry": "k", "grid": [1, 1, 1], "block": [32, 1, 1], "smem_bytes": 0,
              "zero_outputs": False, "args": [{"name": "out", "dtype": "f32"}]}
    study.write_json(tmp_path / "meta.json", {"kernels": [kernel], "rtol": 0, "atol": 0})
    inputs = []
    for seed, pattern in study.INPUT_CASES:
        path = f"input-{seed}.json"
        study.write_json(tmp_path / path, {"seed": seed, "pattern": pattern, "args": [],
                         "expected": {"out": {"shape": [1], "dtype": "float32", "values": [7]}}})
        inputs.append(path)
    (tmp_path / "kernel.ptx").write_text("own PTX fixture", encoding="ascii")
    data = {"ptx": "kernel.ptx", "ptx_sha256": study.sha(tmp_path / "kernel.ptx")}
    workload = {"id": "w", "family": "f", "split": "train", "meta": "meta.json", "inputs": inputs,
                "actions": {str(action): data for action in study.ACTIONS}}
    study.write_json(tmp_path / "seal.json", {"rows": [workload]})

    def transport(server, endpoint, payload, callback):
        if endpoint == "/compile":
            response = {"ok": True, "cubin_id": "fake", "entry_points": ["other" if wrong_entry else "k"]}
        else:
            response = {"ok": True, "median_us": 4.0,
                        "outputs": {"out": {"shape": [1], "values": [7]}}}
        callback({"http_status": 200, "body": response, "seconds": 0.01})
        return response

    with patch.object(study, "verify_seal", return_value={"rows": [workload]}), \
            patch.object(study, "audit_remote_post", transport):
        study.run_remote(SimpleNamespace(out=tmp_path, split="train", runner_turn="mock-only", server="mock"))
    result = study.read_json(tmp_path / "results-train.json")
    return [workload], result


def test_raw_measurement_evidence_revalidated_and_missing_entry_rejected(tmp_path):
    workloads, result = complete_mock_evidence(tmp_path)
    study.verify_measurement_evidence(tmp_path, "train", workloads, result)
    changed = copy.deepcopy(result)
    changed["rows"][0]["aliases"]["0"] = "0"
    with pytest.raises(ValueError, match="aliases"):
        study.verify_measurement_evidence(tmp_path, "train", workloads, changed)
    changed = copy.deepcopy(result)
    changed["rows"][0]["variants"]["1024"]["timings"][0]["median_us"] = 3.0
    with pytest.raises(ValueError, match="returned output evidence"):
        study.verify_measurement_evidence(tmp_path, "train", workloads, changed)
    changed = copy.deepcopy(result)
    rounds = changed["rows"][0]["variants"]["1024"]["timings"]
    rounds[0]["event"], rounds[1]["event"] = rounds[1]["event"], rounds[0]["event"]
    with pytest.raises(ValueError, match="chronological"):
        study.verify_measurement_evidence(tmp_path, "train", workloads, changed)
    bad_dir = tmp_path / "bad-entry"
    bad_dir.mkdir()
    _, bad = complete_mock_evidence(bad_dir, wrong_entry=True)
    assert not bad["rows"][0]["pass"] and len(bad["events"]) == 1


@pytest.mark.parametrize("suffix", [".request.json", ".response.json", ".http-00.json"])
def test_raw_transport_hash_mutation_rejected(tmp_path, suffix):
    workloads, result = complete_mock_evidence(tmp_path)
    path = tmp_path / "events/train" / ("0001" + suffix)
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        study.verify_measurement_evidence(tmp_path, "train", workloads, result)
