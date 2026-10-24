#!/usr/bin/env python3
"""Check the runtime submission in an isolated, reproducibly recorded venv.

Install once with --install (requires package-index access), then rerun without
that flag. Compilation checks make no GPU calls. Each run retains its staged
sources and stripped input contracts so the tested snapshot remains inspectable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt")
DEFAULT_CASES = (
    "problems/p00_scalar_add",
    "problems/r01_array_sum",
    "problems/m01_matmul_512",
    "problems/f02_row_softmax",
    "problems/a01_attention",
    "artifacts/validation/stress/stress_matmul_tail",
    "artifacts/validation/stress/stress_attention_rect",
)

# Run in the clean interpreter, before importing the submission. The exact CLI
# is also executed separately: this guarded mode is a second, complementary test.
AUDITED_RUN = r"""
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import runpy
import sys

package, source_repo, input_path, audit_path = map(Path, sys.argv[1:])
sys.dont_write_bytecode = True
sys.path.insert(0, str(package))
source_root = str(source_repo.resolve()).casefold() + os.sep
emmy_root = str(metadata.distribution("emmy-ml").locate_file("emmy").resolve()).casefold() + os.sep
opened = set()
blocked = []
render_calls = []

def reject(kind, detail):
    blocked.append({"kind": kind, "detail": str(detail)})
    raise RuntimeError("clean-install guard blocked " + kind + ": " + str(detail))

def audit(event, args):
    if event.startswith("socket."):
        reject("network", event)
    if event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.fork", "os.forkpty") or event.startswith("os.spawn"):
        reject("child_process", event)
    if event == "ctypes.dlopen" and args and args[0] is not None:
        library = str(args[0]).casefold()
        if any(token in library for token in ("nvcuda", "cudart", "nvrtc", "nvvm", "libcuda")):
            reject("cuda_library", library)
    if event == "import" and args:
        name = str(args[0])
        if name.split(".")[0] in ("torch", "triton", "tvm", "llvmlite", "numba"):
            reject("compiler_backend_import", name)
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        absolute = os.path.normcase(os.path.abspath(path)).casefold()
        opened.add(absolute)
        if absolute.startswith(source_root):
            reject("original_repository_read", absolute)

def profile(frame, event, arg):
    if event != "call":
        return
    filename = os.path.normcase(os.path.abspath(frame.f_code.co_filename)).casefold()
    if filename.startswith(emmy_root) and frame.f_code.co_name.lstrip("_").startswith("render"):
        render_calls.append(filename + ":" + frame.f_code.co_name)
        reject("emmy_renderer_call", render_calls[-1])

sys.addaudithook(audit)
sys.setprofile(profile)
real_stdout, real_stderr = sys.stdout, sys.stderr
output, errors = io.StringIO(), io.StringIO()
sys.stdout, sys.stderr = output, errors
sys.argv = [str(package / "compile.py"), str(input_path)]
exit_code = 0
try:
    runpy.run_path(str(package / "compile.py"), run_name="__main__")
except SystemExit as exc:
    exit_code = int(exc.code or 0)
except BaseException as exc:
    exit_code = 1
    errors.write(type(exc).__name__ + ": " + str(exc) + "\n")
finally:
    sys.setprofile(None)
    sys.stdout, sys.stderr = real_stdout, real_stderr

origins = {name: str(getattr(module, "__file__", "")) for name, module in sys.modules.items()
           if name in ("ptx_backend", "ptx_tensor", "ptx_bulk", "emmy", "emmy.compiler.graph", "numpy")}
report = {"exit_code": exit_code, "blocked_events": blocked,
          "emmy_renderer_calls": render_calls, "opened_paths": sorted(opened),
          "module_origins": origins, "sys_path": sys.path,
          "emmy_version": metadata.version("emmy-ml"),
          "prefix": sys.prefix, "base_prefix": sys.base_prefix,
          "python_version": sys.version, "python_executable": sys.executable}
audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
real_stdout.write(output.getvalue())
real_stderr.write(errors.getvalue())
raise SystemExit(exit_code)
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, *, cwd, env, timeout=120):
    started = time.perf_counter()
    result = subprocess.run(
        argv, cwd=cwd, env=env, capture_output=True, timeout=timeout
    )
    return {
        "argv": list(map(str, argv)),
        "returncode": result.returncode,
        "stdout": result.stdout.decode("utf-8", errors="replace"),
        "stderr": result.stderr.decode("utf-8", errors="replace"),
        "elapsed_seconds": time.perf_counter() - started,
    }


def require_ok(result, label):
    if result["returncode"]:
        raise RuntimeError(label + " failed:\n" + result["stderr"] + result["stdout"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT.parent / ".build/clean-env")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install declared dependencies; may access package indexes",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts/validation/clean-install/summary.json",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Repository-relative input package; repeatable",
    )
    args = parser.parse_args()
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    build_root = ROOT.parent / ".build"
    build_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="clean-install-", dir=build_root)).resolve()
    package = stage / "package"
    package.mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / name, package / name)
    source_hashes = {name: sha256(package / name) for name in SOURCE_FILES}

    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CUDA_PATH", "CUDA_HOME"):
        env.pop(key, None)
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUTF8="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_NO_INPUT="1",
    )
    venv = args.venv.resolve()
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        if not args.install:
            parser.error("clean environment is absent; first run with --install")
        if importlib.util.find_spec("venv") is not None:
            create_argv = [sys.executable, "-m", "venv", str(venv)]
        elif importlib.util.find_spec("virtualenv") is not None:
            create_argv = [
                sys.executable,
                "-m",
                "virtualenv",
                "--no-periodic-update",
                "--app-data",
                str(build_root / "virtualenv"),
                str(venv),
            ]
        else:
            raise RuntimeError(
                "this Python lacks venv; install virtualenv or select a Python with venv"
            )
        created = command(create_argv, cwd=stage, env=env)
        require_ok(created, "venv creation")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "venv": str(venv),
        "stage": str(stage),
        "source_sha256": source_hashes,
        "source_files": list(SOURCE_FILES),
        "cases": [],
        "ok": False,
        "limitations": [
            "Same-host clean virtual environment, not a different OS/container or Python version.",
            "Dependency installation may use the network; compilation is tested separately with Python-level guards.",
            "Audit hooks are regression instrumentation, not an OS sandbox against arbitrary native code.",
            "Declared requirements pin Emmy but allow newer NumPy, requests, and transitive dependencies; installed versions are recorded, not hash-locked.",
            "This checks packaging, deterministic emission, and forbidden-path absence; it does not assemble or execute PTX.",
        ],
    }
    try:
        if args.install:
            installed = command(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(package / "requirements.txt"),
                ],
                cwd=stage,
                env=env,
                timeout=900,
            )
            (report_path.parent / "install.log").write_text(
                installed["stdout"] + installed["stderr"], encoding="utf-8"
            )
            report["install"] = {
                k: v for k, v in installed.items() if k not in ("stdout", "stderr")
            }
            require_ok(installed, "dependency installation")
        checked = command([str(python), "-m", "pip", "check"], cwd=stage, env=env)
        report["pip_check"] = checked
        require_ok(checked, "pip check")
        listed = command(
            [str(python), "-m", "pip", "list", "--format=json"], cwd=stage, env=env
        )
        require_ok(listed, "package inventory")
        report["installed_packages"] = json.loads(listed["stdout"])
        (report_path.parent / "installed-versions.txt").write_text(
            "\n".join(
                f"{p['name']}=={p['version']}" for p in report["installed_packages"]
            )
            + "\n",
            encoding="utf-8",
        )
        emmy = next(
            (
                p
                for p in report["installed_packages"]
                if p["name"].lower().replace("_", "-") == "emmy-ml"
            ),
            None,
        )
        if emmy is None or emmy["version"] != "0.3.11":
            raise RuntimeError("expected emmy-ml exactly 0.3.11")
        report["pyvenv_config"] = (venv / "pyvenv.cfg").read_text(encoding="utf-8")
        if "include-system-site-packages = false" not in report["pyvenv_config"]:
            raise RuntimeError("venv must exclude global site-packages")

        for number, relative in enumerate(args.cases or DEFAULT_CASES):
            folder = (ROOT / relative).resolve()
            paths = sorted(folder.glob("*.ir.json"))
            if len(paths) != 1:
                raise RuntimeError(
                    f"expected one input IR in {folder}, found {len(paths)}"
                )
            contract = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            case_dir = stage / f"case-{number:02d}"
            case_dir.mkdir()
            ir = case_dir / "kernel.ir.json"
            shutil.copy2(paths[0], ir)
            # No expressions, reference CUDA, expected arrays, or argument data.
            minimal = {
                key: contract[key] for key in ("rtol", "atol") if key in contract
            }
            minimal["kernels"] = [
                {
                    key: kernel[key]
                    for key in (
                        "entry",
                        "arg_order",
                        "grid",
                        "block",
                        "smem_bytes",
                        "zero_outputs",
                    )
                    if key in kernel
                }
                for kernel in contract["kernels"]
            ]
            (case_dir / "meta.json").write_text(
                json.dumps(minimal, indent=2), encoding="utf-8"
            )
            runs = [
                command([str(python), "compile.py", str(ir)], cwd=package, env=env)
                for _ in range(2)
            ]
            audited = command(
                [
                    str(python),
                    "-I",
                    "-c",
                    AUDITED_RUN,
                    str(package),
                    str(ROOT),
                    str(ir),
                    str(case_dir / "audit.json"),
                ],
                cwd=stage,
                env=env,
            )
            runs.append(audited)
            for run in runs:
                require_ok(run, "CLI compile for " + relative)
                if (
                    run["stderr"]
                    or not run["stdout"].isascii()
                    or not run["stdout"].startswith(".version")
                ):
                    raise RuntimeError(
                        "compiler must emit ASCII PTX only with clean stderr"
                    )
            if not all(run["stdout"] == runs[0]["stdout"] for run in runs):
                raise RuntimeError("repeated or guarded compilation changed PTX")
            expected_entries = {k["entry"] for k in minimal["kernels"]}
            emitted = re.findall(r"\.entry\s+([A-Za-z_$][\w$]*)", runs[0]["stdout"])
            if len(emitted) != 1 or emitted[0] not in expected_entries:
                raise RuntimeError("emitted entry does not match contract")
            audit = json.loads((case_dir / "audit.json").read_text(encoding="utf-8"))
            if audit["blocked_events"] or audit["emmy_renderer_calls"]:
                raise RuntimeError("guard observed a prohibited compilation dependency")
            if not Path(audit["module_origins"]["emmy.compiler.graph"]).is_relative_to(
                venv
            ):
                raise RuntimeError("Emmy was not imported from the clean environment")
            # Preserve the actual captured CLI bytes, including Windows CRLF.
            (case_dir / "out.ptx").write_bytes(runs[0]["stdout"].encode("ascii"))
            row = {
                "source_package": relative,
                "staged_ir": str(ir),
                "ir_sha256": sha256(ir),
                "metadata_sha256": sha256(case_dir / "meta.json"),
                "entry": emitted[0],
                "ptx_sha256": hashlib.sha256(
                    runs[0]["stdout"].encode("ascii")
                ).hexdigest(),
                "ptx_bytes": len(runs[0]["stdout"]),
                "deterministic": True,
                "plain_cli_seconds": [run["elapsed_seconds"] for run in runs[:2]],
                "guarded_seconds": audited["elapsed_seconds"],
                "audit": audit,
                "pass": True,
            }
            report["cases"].append(row)
            print("PASS " + relative, flush=True)
        malformed = stage / "malformed.ir.json"
        malformed.write_text("{invalid-json", encoding="ascii")
        report["diagnostics"] = []
        for label, arguments, code, prefix in (
            ("missing_argument", [], 2, "usage:"),
            ("extra_argument", ["one", "two"], 2, "usage:"),
            ("invalid_json", [str(malformed)], 1, "compile.py:"),
        ):
            result = command(
                [str(python), "compile.py", *arguments], cwd=package, env=env
            )
            if (
                result["returncode"] != code
                or result["stdout"]
                or not result["stderr"].startswith(prefix)
            ):
                raise RuntimeError("unexpected CLI diagnostic behavior: " + label)
            report["diagnostics"].append({"case": label, "pass": True, **result})
        report["source_unchanged_after_check"] = all(
            sha256(ROOT / name) == digest for name, digest in source_hashes.items()
        )
        if not report["source_unchanged_after_check"]:
            raise RuntimeError(
                "submission sources changed during the check; rerun against a stable snapshot"
            )
        report["staged_package_files"] = sorted(p.name for p in package.iterdir())
        if report["staged_package_files"] != sorted(SOURCE_FILES):
            raise RuntimeError("unexpected files appeared in the three-file package")
        report["ok"] = True
    except Exception as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        print(report["error"], file=sys.stderr)
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report: " + str(report_path))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
