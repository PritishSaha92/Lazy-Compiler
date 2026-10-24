"""Run the complete curated remote release gate, serially, on a frozen CLI.

Requires the retained generated packages in artifacts/validation. Local unit,
sanitizer and clean-install checks are separate commands. Reports never overwrite
an earlier run. This is development evaluation, not a competition submission.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from ir_cases import write_case
from test_alignment_overflow import alignment_overflow_cases
from test_index_layout import index_layout_cases
from test_load_packing import load_packing_cases
from test_loop_capture import loop_capture_cases
from test_predicate_promotion import predicate_promotion_cases
from tools.check_compiler import SERVER, compile_ptx, compiler_fingerprints, input_array


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packages(relative, count):
    paths = sorted((ROOT / relative).glob("*/meta.json"))
    if len(paths) != count:
        raise ValueError(
            f"Expected {count} packages under {relative}, found {len(paths)}"
        )
    return [p.parent for p in paths]


def unsigned_word_metadata(original):
    """Encode explicit unsigned input buffers as byte-identical signed words.

    The runner accepts i32 but not u32/u64, and its JSON number parsing loses
    low bits of large i64 values. Every i32 word is exact through float64.
    This changes only host transport; kernel IR, pointer ABI and bytes stay fixed.
    Unsigned outputs and by-value scalars intentionally remain unsupported.
    """
    updated = deepcopy(original)
    records = []
    for kernel in updated["kernels"]:
        for arg in kernel["args"]:
            if arg["dtype"] not in ("u32", "u64"):
                continue
            if arg.get("output") or arg.get("scalar") or "value" in arg:
                raise ValueError("Unsigned word transport supports input buffers only")
            if "data" not in arg:
                raise ValueError("Unsigned word transport requires explicit input data")
            if sys.byteorder != "little":
                raise ValueError(
                    "Unsigned word transport requires a little-endian host"
                )
            before = deepcopy(arg)
            initial = input_array(before, 0)
            words = initial.view(np.int32).reshape(-1)
            if not all(int(word) == int(float(word)) for word in words):
                raise ValueError("Input transport lost integer precision")
            arg.update(dtype="i32", shape=[int(words.size)], data=words.tolist())
            final = input_array(arg, 0)
            if initial.tobytes() != final.tobytes():
                raise ValueError("Unsigned word transport changed input bytes")
            records.append(
                {
                    "entry": kernel["entry"],
                    "name": arg["name"],
                    "original_dtype": before["dtype"],
                    "original_shape": list(initial.shape),
                    "transport_dtype": "i32",
                    "transport_shape": list(final.shape),
                    "bytes": initial.nbytes,
                    "original_bytes_sha256": hashlib.sha256(
                        initial.tobytes()
                    ).hexdigest(),
                    "transport_bytes_sha256": hashlib.sha256(
                        final.tobytes()
                    ).hexdigest(),
                    "float64_roundtrip_exact": True,
                }
            )
    return updated, records


def prepare_unsigned_transport(source, destination, compiler):
    """Retain original packages; prove copied packages emit identical PTX."""
    original = json.loads((source / "meta.json").read_text())
    updated, inputs = unsigned_word_metadata(original)
    if not inputs:
        return source, None
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file():
            (destination / path.name).write_bytes(path.read_bytes())
    (destination / "meta.json").write_text(
        json.dumps(updated, indent=2) + "\n", encoding="utf-8"
    )
    kernels = []
    for kernel in original["kernels"]:
        ir_name = kernel["entry"] + ".ir.json"
        if (source / ir_name).read_bytes() != (destination / ir_name).read_bytes():
            raise ValueError("Transport adaptation changed kernel IR")
        old_ptx = compile_ptx(compiler, source / ir_name)[0].encode("ascii")
        new_ptx = compile_ptx(compiler, destination / ir_name)[0].encode("ascii")
        if old_ptx != new_ptx:
            raise ValueError("Transport adaptation changed generated PTX")
        (destination / (kernel["entry"] + ".transport.ptx")).write_bytes(new_ptx)
        kernels.append(
            {
                "entry": kernel["entry"],
                "ir_sha256": digest(source / ir_name),
                "original_ptx_sha256": hashlib.sha256(old_ptx).hexdigest(),
                "transport_ptx_sha256": hashlib.sha256(new_ptx).hexdigest(),
                "ptx_byte_identical": True,
            }
        )
    if (source / "expected.npz").read_bytes() != (
        destination / "expected.npz"
    ).read_bytes():
        raise ValueError("Transport adaptation changed expected outputs")
    return destination, {
        "original": str(source),
        "transport": str(destination),
        "original_metadata_sha256": digest(source / "meta.json"),
        "transport_metadata_sha256": digest(destination / "meta.json"),
        "expected_npz_sha256": digest(source / "expected.npz"),
        "inputs": inputs,
        "kernels": kernels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--server", default=SERVER)
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    compiler_before = compiler_fingerprints(compiler)
    output = args.report_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    typed = output / "typed-inputs"
    typed_paths = []
    for factory in (
        loop_capture_cases,
        index_layout_cases,
        predicate_promotion_cases,
        alignment_overflow_cases,
        load_packing_cases,
    ):
        for value in factory():
            typed_paths.append(write_case(typed / value[1]["id"], value).parent)
    assert len(typed_paths) == 30
    original_typed_paths = typed_paths[:]
    adaptations = []
    for index, source in enumerate(typed_paths):
        typed_paths[index], record = prepare_unsigned_transport(
            source, output / "typed-wire-inputs" / source.name, compiler
        )
        if record is not None:
            adaptations.append(record)
    adaptation_path = output / "input-transport.json"
    adaptation_path.write_text(
        json.dumps(
            {
                "scope": "Input transport only; original IR, PTX, expected outputs and buffer bytes preserved.",
                "packages": adaptations,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    public = packages("problems", 15)
    fresh = packages("artifacts/validation/stress", 10)
    layout = packages("artifacts/validation/layouts", 8)
    old_holdout = packages("artifacts/validation/holdout-2026-10-23", 18)
    schedules = packages("artifacts/validation/phase3/schedules", 5)
    transport = packages("artifacts/validation/phase3/transport", 9)
    products = packages("artifacts/validation/phase3/products", 2)
    jobs = [
        ("public-normal", public, "0,17", "normal", None),
        ("public-stress", public, "101", "ones,alternating,wide", None),
        ("fresh", fresh, "0,17", "normal", "tests/expressions_stress.py"),
        ("layouts", layout, "0", "normal", None),
        (
            "prior-holdout-normal",
            old_holdout,
            "37,89",
            "normal",
            "tests/expressions_holdout.py",
        ),
        (
            "prior-holdout-wide",
            old_holdout,
            "137",
            "wide",
            "tests/expressions_holdout.py",
        ),
        (
            "schedules-normal",
            schedules,
            "37,89",
            "normal",
            "tests/expressions_phase3.py",
        ),
        ("schedules-wide", schedules, "137", "wide", "tests/expressions_phase3.py"),
        ("transport", transport, "0", "normal", None),
        ("products", products, "0", "normal", None),
        ("typed-semantics", typed_paths, "0", "normal", None),
    ]
    input_files = {adaptation_path}
    for _, folders, _, _, _ in jobs:
        for folder in folders:
            input_files.update(folder.glob("*.ir.json"))
            input_files.update((folder / "meta.json", folder / "expected.npz"))
    for folder in original_typed_paths:
        input_files.update(folder.glob("*.ir.json"))
        input_files.update((folder / "meta.json", folder / "expected.npz"))
    input_files.update(output.glob("typed-wire-inputs/*/*.transport.ptx"))
    source_files = [
        *sorted((ROOT / "tests").glob("*.py")),
        ROOT / "tools/expressions_open.py",
        ROOT / "tools/check_compiler.py",
        Path(__file__).resolve(),
    ]
    tracked = {str(p): digest(p) for p in [*input_files, *source_files]}
    summary = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "compiler_sources_before_sha256": compiler_before,
        "inputs_and_helpers_sha256": tracked,
        "planned_numerical_cases": 213,
        "distinct_workload_packages": len(
            {p for _, folders, _, _, _ in jobs for p in folders}
        ),
        "transported_input_packages": len(adaptations),
        "input_transport_manifest": adaptation_path.name,
        "jobs": [],
        "pass": False,
        "scope": "Curated development regressions, not organizer hidden tests. All calls serialized; timings are not paired performance evidence.",
    }
    summary_path = output / "summary.json"

    def save():
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    save()
    for label, folders, seeds, patterns, references in jobs:
        report = output / (label + ".json")
        command = [
            sys.executable,
            "-X",
            "utf8",
            str(ROOT / "tools/check_compiler.py"),
            *map(str, folders),
            "--compiler",
            str(compiler),
            "--mode",
            "remote",
            "--server",
            args.server,
            "--seeds",
            seeds,
            "--patterns",
            patterns,
            "--warmup",
            "1",
            "--iters",
            "3",
            "--report",
            str(report),
        ]
        if references:
            command.extend(("--references", str(ROOT / references)))
        row = {"name": label, "command": command, "pass": False}
        summary["jobs"].append(row)
        save()
        process = subprocess.run(command, cwd=ROOT, capture_output=True)
        transcript = process.stdout + process.stderr
        (output / (label + ".log")).write_bytes(transcript)
        print(transcript.decode("utf-8", errors="replace"), end="", flush=True)
        row["returncode"] = process.returncode
        if report.exists():
            results = json.loads(report.read_text())["results"]
            row.update(
                report=report.name,
                report_sha256=digest(report),
                numerical_cases=sum(len(r.get("cases", [])) for r in results),
                numerical_passes=sum(
                    c["pass"] for r in results for c in r.get("cases", [])
                ),
            )
            expected = len(folders) * len(seeds.split(",")) * len(patterns.split(","))
            row["pass"] = (
                process.returncode == 0
                and len(results) == len(folders)
                and all(r["pass"] and not r.get("skipped") for r in results)
                and row["numerical_passes"] == row["numerical_cases"] == expected
            )
        save()
    summary["compiler_sources_after_sha256"] = compiler_fingerprints(compiler)
    summary["compiler_unchanged"] = (
        summary["compiler_sources_before_sha256"]
        == summary["compiler_sources_after_sha256"]
    )
    summary["inputs_and_helpers_unchanged"] = all(
        Path(p).is_file() and digest(Path(p)) == expected
        for p, expected in tracked.items()
    )
    summary["numerical_cases"] = sum(
        row.get("numerical_cases", 0) for row in summary["jobs"]
    )
    summary["numerical_passes"] = sum(
        row.get("numerical_passes", 0) for row in summary["jobs"]
    )
    summary["pass"] = (
        summary["compiler_unchanged"]
        and summary["inputs_and_helpers_unchanged"]
        and all(row["pass"] for row in summary["jobs"])
        and summary["numerical_passes"]
        == summary["numerical_cases"]
        == summary["planned_numerical_cases"]
    )
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(
        f"{'PASS' if summary['pass'] else 'FAIL'}: {summary['numerical_passes']}/{summary['planned_numerical_cases']} numerical cases; {summary_path}"
    )
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
