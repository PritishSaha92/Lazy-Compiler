"""Supplement the historical oracle audit with runner-safe F64 word fixtures.

The earlier i64 fixture oracles were mathematically correct; that did not prove
lossless transport through the runner's intermediate float64 conversion. This
audit explicitly reproduces that conversion for each low/high i32 word.
"""
import hashlib
import importlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.dont_write_bytecode = True
from audit_final_oracles import ROOT, HERE, block_execution, digest, remainder_word, signed_word


def decode_pairs(words):
    return [struct.unpack("<d", struct.pack("<II", int(low) & 0xffffffff,
                                            int(high) & 0xffffffff))[0]
            for low, high in words]


def encode_pairs(words64):
    return np.asarray([[signed_word(int(word) & 0xffffffff, 32),
                        signed_word((int(word) >> 32) & 0xffffffff, 32)]
                       for word in words64], dtype=np.int32)


def wire_roundtrip(words):
    text = json.dumps(words.tolist(), allow_nan=False)
    decoded = json.loads(text)
    promoted = np.asarray(decoded, dtype=np.float64)
    restored = promoted.astype(np.int32)
    assert restored.dtype == words.dtype and restored.shape == words.shape
    assert restored.tobytes() == words.tobytes()
    return restored


def main():
    packages = sorted((ROOT / "examples/phase8/remainder_words").glob("*/meta.json"))
    assert len(packages) == 2
    sources = [ROOT / name for name in ("ptx_backend.py", "tests/test_floating_remainder.py",
                                      "tests/ir_cases.py", "tests/test_contract_edges.py")]
    sources += [path for meta in packages for path in meta.parent.iterdir() if path.is_file()]
    before = {path.relative_to(ROOT).as_posix(): digest(path) for path in sources}
    block_execution()
    sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]
    fixtures = importlib.import_module("test_floating_remainder")
    regenerated = {meta["id"]: (graph, meta, expected)
                   for graph, meta, expected in fixtures.floating_remainder_word_cases()}
    rows = []
    for meta_path in packages:
        meta = json.loads(meta_path.read_text())
        graph, regenerated_meta, regenerated_expected = regenerated[meta["id"]]
        assert meta == regenerated_meta
        irs = list(meta_path.parent.glob("*.ir.json"))
        assert len(irs) == 1
        assert json.loads(irs[0].read_text()) == json.loads(json.dumps(graph.to_dict(), default=repr))
        args = {arg["name"]: arg for arg in meta["kernels"][0]["args"]}
        assert all(arg["dtype"] == "i32" and arg["shape"] == [528, 2] for arg in args.values())
        inputs, input_words = {}, 0
        for name in ("x", "y"):
            data = args[name]["data"]
            assert len(data) == 1056 and all(type(value) is int and -2**31 <= value < 2**31 for value in data)
            words = np.asarray(data, dtype=np.int32).reshape(528, 2)
            restored = wire_roundtrip(words)
            inputs[name] = decode_pairs(restored)
            input_words += words.size
        # No NumPy remainder/fmod or fixture canonical_bits/word_pairs helper
        # participates in this second, exact rational answer calculation.
        answers = [remainder_word(x, y, 64) for x, y in zip(inputs["x"], inputs["y"], strict=True)]
        independent = encode_pairs(answers)
        with np.load(meta_path.with_name("expected.npz"), allow_pickle=False) as archive:
            assert archive.files == ["out"]
            expected = archive["out"].copy()
        assert expected.dtype == np.int32 and expected.shape == (528, 2)
        assert expected.tobytes() == independent.tobytes() == regenerated_expected["out"].tobytes()
        wire_roundtrip(expected)
        assert meta["rtol"] == meta["atol"] == 0
        rows.append({"id": meta["id"], "pass": True,
                     "remainder_values": len(answers), "input_i32_words": int(input_words),
                     "output_i32_words": int(expected.size),
                     "expected_array_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                     "ir_metadata_and_arrays_regenerated_exactly": True,
                     "float64_intermediate_transport_is_byte_exact": True})
    after = {path.relative_to(ROOT).as_posix(): digest(path) for path in sources}
    assert before == after
    old_probe = -(1 << 63) + 1
    assert int(float(old_probe)) != old_probe
    report = {"schema": "emmy-phase8-word-oracle-audit-v1", "pass": True,
              "scope": "Supplement for corrected two-i32-word F64 fixtures; historical i64 fixture references remain mathematical evidence only",
              "backend_sha256": before["ptx_backend.py"], "packages": len(rows),
              "remainder_values": sum(row["remainder_values"] for row in rows),
              "checked_input_words": sum(row["input_i32_words"] for row in rows),
              "checked_output_words": sum(row["output_i32_words"] for row in rows),
              "oracle": "Exact Fraction floor remainder + independent nearest-even IEEE encoding + explicit low/high signed-i32 extraction",
              "wire_model": "JSON signed integers -> float64 -> int32, checked byte-for-byte for inputs and outputs",
              "historical_i64_counterexample": {"source": old_probe, "after_float64": int(float(old_probe))},
              "compiler_process_network_execution_blocked": True, "gpu_used": False,
              "sources_and_fixtures_unchanged": True, "files_sha256": before,
              "audit_script_sha256": digest(Path(__file__)), "rows": rows}
    with (HERE / "final-word-oracle-audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("files_sha256", "rows")}, indent=2))


if __name__ == "__main__":
    main()
