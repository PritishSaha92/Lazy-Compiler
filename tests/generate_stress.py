#!/usr/bin/env python3
"""Generate independent practice packages, including on Windows with Emmy 0.3.11.

Usage: python tests/generate_stress.py --out artifacts/validation/stress
       python tests/generate_stress.py --expressions tests/expressions_holdout.py --out DIR

Emmy's CLI imports a Unix-only fcntl module even when compiling without a mutable
golden inventory. This worker supplies an import-only shim on Windows. Any actual
locking attempt fails loudly; it never pretends to acquire a lock. The isolated
test generation disables golden inventories and uses temporary tuning files.
"""

from pathlib import Path
import argparse
import os
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def trace_worker():
    if os.name == "nt":
        module = types.ModuleType("fcntl")
        module.LOCK_EX = 2

        def no_lock(*args, **kwargs):
            raise RuntimeError(
                "Windows practice generation does not support mutable golden-file locking"
            )

        module.flock = no_lock
        sys.modules["fcntl"] = module
    from emmy.emmy import main

    sys.argv = ["emmy"] + sys.argv[2:]
    main()


def main():
    from tools import generate

    def windows_trace(emmy, spec, dump, target):
        environment = {
            **os.environ,
            "EMMY_DUMP_DIR": str(dump),
            "EMMY_TUNE_DB": str(dump.parent / "autotune.db"),
            "EMMY_ONLINE_FILE": str(dump.parent / "online.json"),
            "EMMY_GOLDEN_FILE": "",
            "EMMY_KNOBS": spec.get("knobs", ""),
            "PYTHONUTF8": "1",
        }
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--trace-worker",
                "compile",
                "-c",
                spec["code"],
                "--ir",
                "cuda",
                "--target",
                target,
            ],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=900,
        )

    if os.name == "nt":
        generate.trace = windows_trace
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument(
        "--expressions", type=Path, default=ROOT / "tests/expressions_stress.py"
    )
    selected, remaining = options.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], str(selected.expressions)] + remaining
    return generate.main()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--trace-worker":
        trace_worker()
    else:
        raise SystemExit(main())
