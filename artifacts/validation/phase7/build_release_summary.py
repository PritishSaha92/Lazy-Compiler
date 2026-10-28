"""Admit the phase-7 release only from completed, source-bound evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/validation/phase7"
RUNTIME = ("compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def runtime(mapping):
    return {Path(k).name: v for k, v in mapping.items() if Path(k).name in RUNTIME}


def verify_mapping(mapping):
    for name, expected in mapping.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        assert sha(path) == expected, name


def main():
    destination = OUT / "release-summary.json"
    assert not destination.exists(), "refusing to overwrite retained release evidence"
    source = {name: sha(ROOT / name) for name in RUNTIME}
    paths = {
        "public": OUT / "public-submit/summary.json",
        "features": OUT / "feature-runner/summary.json",
        "new_features": OUT / "new-feature-runner/summary.json",
        "changed_scalar": OUT / "changed-scalar-runner/summary.json",
        "features_audit": OUT / "feature-runner/integrity-audit.json",
        "new_features_audit": OUT / "new-feature-runner/integrity-audit.json",
        "changed_scalar_audit": OUT / "changed-scalar-runner/integrity-audit.json",
        "features_manifest": OUT / "feature-manifest.json",
        "new_features_manifest": OUT / "new-feature-manifest.json",
        "changed_scalar_manifest": OUT / "changed-scalar-manifest.json",
        "cpu": OUT / "cpu-regressions-final.json",
        "clean": OUT / "clean-install/summary.json",
        "compatibility": OUT / "compatibility.json",
        "changed_scalar_copy": OUT / "changed-scalar-copy.json",
        "initial_cpu": OUT / "cpu-regressions.json",
        "pre_fix": OUT / "pre-fix-integer-runner/summary.json",
    }
    reports = {key: read(path) for key, path in paths.items()}
    pub = reports["public"]
    assert pub["pass"] and pub["passed_problems"] == len(pub["rows"]) == 15
    assert pub["tracked_unchanged"]
    assert runtime(pub["source_and_input_sha256"]) == source == runtime(pub["source_and_input_sha256_after"])
    verify_mapping(pub["source_and_input_sha256_after"])
    for row in pub["rows"]:
        assert row["pass"] and row["returncode"] == 0 and not row["timeout"]
        for stream in ("stdout", "stderr"):
            assert sha(paths["public"].parent / row[stream]) == row[stream + "_sha256"]

    audited_jobs = []
    for key, expected_count in (("features", 29), ("new_features", 11), ("changed_scalar", 1)):
        result, audit = reports[key], reports[key + "_audit"]
        assert result["pass"] and result["checks"] == result["passed_checks"] == result["planned_checks"] == expected_count
        assert result["tracked_unchanged"] and all(job["pass"] for job in result["jobs"])
        assert runtime(result["compiler_sha256"]) == source == runtime(result["compiler_sha256_after"])
        verify_mapping(result["tracked_sha256"])
        assert audit["integrity_pass"] and audit["numerical_gate_pass"] and audit["complete_replay"]
        assert audit["checks"] == audit["passed_checks"] == expected_count and audit["failed_checks"] == 0
        assert audit["compiler_sha256"] == source and audit["summary_sha256"] == sha(paths[key])
        assert audit["auditor_sha256"] == sha(OUT / "audit_feature_runner.py")
        manifest = reports[key + "_manifest"]
        assert audit["feature_manifest_sha256"] == sha(paths[key + "_manifest"])
        assert manifest["compiler_sha256"] == source and manifest["package_count"] == expected_count
        assert [str((ROOT / p).resolve()) for p in manifest["packages"]] == result["planned_packages"]
        for name, expected in manifest["files_sha256"].items():
            fixture = (ROOT / name).resolve()
            assert fixture.is_relative_to(ROOT / "examples") and sha(fixture) == expected
        for provenance in manifest["provenance"].values():
            if isinstance(provenance, dict) and "path" in provenance and "sha256" in provenance:
                assert sha(ROOT / provenance["path"]) == provenance["sha256"]
        audited_jobs.extend(result["jobs"])
    assert len({job["id"] for job in audited_jobs}) == 41, "duplicate packages are not additional coverage"

    cpu = reports["cpu"]
    assert cpu["pass"] and not cpu["gpu_execution_requested"] and cpu["sources_unchanged"] and cpu["executed_sources_unchanged"]
    assert not cpu["failures"] and not cpu["errors"] and cpu["tests_run"] == 129 and len(cpu["skipped"]) == 19
    assert runtime(cpu["sources_before_sha256"]) == source == runtime(cpu["sources_after_sha256"])
    verify_mapping(cpu["sources_after_sha256"])
    initial = reports["initial_cpu"]
    assert not initial["pass"] and len(initial["failures"]) == 2 and not initial["errors"]
    assert runtime(initial["sources_after_sha256"]) == source
    for failure, symbol in zip(initial["failures"], ("<<", ">>"), strict=True):
        assert "test_rejected_pairs_and_arity" in failure["test"] and f"symbol='{symbol}'" in failure["test"]
        assert "AssertionError" in failure["traceback"] and "binary operator.*f32" in failure["traceback"]
        assert f"shift operator '{symbol}' requires integer operands" in failure["traceback"]

    clean = reports["clean"]
    assert clean["ok"] and clean["source_unchanged_after_check"] and clean["source_sha256"] == source
    assert len(clean["cases"]) == 25 and len(clean["diagnostics"]) == 3
    assert clean["staged_package_files"] == sorted(RUNTIME)
    assert all(case["deterministic"] and case["audit"]["exit_code"] == 0
               and not case["audit"]["blocked_events"] and not case["audit"]["emmy_renderer_calls"] for case in clean["cases"])

    compat = reports["compatibility"]
    assert compat["pass"] and compat["sources_unchanged"] and compat["inputs_unchanged"] and compat["evidence_unchanged"]
    assert compat["baseline_inputs_identical"] and compat["expected_coverage_complete"]
    assert runtime(compat["compiler_and_audit_sources_before_sha256"]) == source == runtime(compat["compiler_and_audit_sources_after_sha256"])
    verify_mapping(compat["compiler_and_audit_sources_after_sha256"])
    verify_mapping(compat["inputs_after_sha256"])
    verify_mapping(compat["source_evidence_after_sha256"])
    assert compat["scalar"]["kernels"] == 120 and compat["scalar"]["identical"] == 119 and compat["scalar"]["changed"] == 1 and not compat["scalar"]["errors"]
    assert compat["features"]["kernels"] == compat["features"]["identical"] == 29
    changed = [row for row in compat["scalar"]["rows"] if not row["identical"]]
    assert len(changed) == 1 and "comparison_integer_promotion/" in changed[0]["ir"]
    copied = reports["changed_scalar_copy"]
    assert copied["byte_identical"]
    for name, expected in copied["sha256"].items():
        assert sha(ROOT / copied["source"] / name) == sha(ROOT / copied["destination"] / name) == expected
    changed_job = reports["changed_scalar"]["jobs"][0]
    assert changed_job["id"] == "comparison_integer_promotion"
    raw = read(paths["changed_scalar"].parent / "events/0000.request.json")
    canonical_ptx = raw["payload"]["ptx"].replace("\r\n", "\n").encode("ascii")
    assert hashlib.sha256(canonical_ptx).hexdigest() == changed[0]["candidate_ptx_sha256"]
    assert not reports["pre_fix"]["pass"] and len(reports["pre_fix"]["jobs"]) == 6

    report = {
        "schema": "emmy-phase7-release-v1", "pass": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source,
        "scope": "Fresh organizer-GPU and CPU/assembler checks for integer and logical-builtin corrections; prior release evidence remains historical.",
        "changes": ["Correct unstamped mixed-integer Assign promotion without a float fallback.",
                    "Use promoted left-operand type for expression shifts and independent unsigned PTX counts.",
                    "Support Emmy logical GPU builtin names and CUDA aliases with unsigned geometry semantics, sharing resolution with conservative bulk bounds proofs."],
        "organizer_gpu": {"public_checks": 15, "public_passes": 15, "prior_feature_checks": 29,
                          "new_counterexample_checks": 11, "changed_scalar_rechecks": 1,
                          "additional_checks": 41, "additional_passes": 41,
                          "total_checks": 56, "total_passes": 56, "local_gpu_used": False},
        "cpu": {"methods": 129, "passed": 110, "explicit_local_gpu_skips": 19,
                "initial_diagnostic_expectation_failures": 2, "runtime_unchanged_between_runs": True},
        "clean_install": {"packages": 25, "successful_compilations": 75, "expected_diagnostics": 3},
        "ptx_compatibility": {"scalar_modules": 120, "identical_scalar_modules": 119,
                              "changed_scalar_modules": 1, "changed_scalar_fresh_gpu_passes": 1,
                              "prior_feature_modules": 29, "identical_prior_features": 29,
                              "identity_audit_is_not_gpu_execution": True},
        "historical_fp64_reference_failures": 9,
        "limitations": ["Development counterexamples are not organizer hidden tests.",
                        "No new performance ranking or learned optimization benefit is asserted.",
                        "Auxiliary coverage is explicit tested IR; SetMaxNReg, WGMMA, host tensor-map TMA and general 3-D Tile schedules are outside the contract.",
                        "The nine earlier FP64-buffer/FP32-state reference failures remain failures.",
                        "The recorded video documents phase6 architecture and results; phase7 release notes document this subsequent correction."],
        "evidence": {key: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)} for key, path in paths.items()},
    }
    assert source == {name: sha(ROOT / name) for name in RUNTIME}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "gpu_checks": 56, "cpu_passed": 110, "clean_packages": 25, "summary_sha256": sha(destination)}, indent=2))


if __name__ == "__main__":
    main()
