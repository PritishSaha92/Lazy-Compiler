#!/usr/bin/env python3
"""Validate the submission interface, assemble PTX, or execute independent references.

Examples:
    python tools/check_compiler.py --mode assemble
    python tools/check_compiler.py --mode local --seeds 0,17
    python tools/check_compiler.py --mode remote --seeds 0,17 --report results.json

Remote requests are serialized. This is a development test client, not an entry
submission. CUDA driver execution lives in a subprocess so a bad kernel cannot
poison subsequent CUDA contexts. The local target adaptation is explicit and
never modifies the submitted PTX file or silently translates cp.async.
"""

from __future__ import annotations

import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "u32": np.uint32,
    "u64": np.uint64,
}
SERVER = "http://runner.riftstack.ai:48701"


class UnsupportedLocal(RuntimeError):
    """The local device cannot run this test; this is not a passing result."""


def compiler_fingerprints(compiler: Path) -> dict:
    sources = (compiler, *(compiler.parent / name for name in
                ("ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt")))
    return {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in sources
        if source.exists()
    }


def compile_ptx(compiler: Path, ir: Path, timeout: float = 90) -> tuple[str, float]:
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(compiler), str(ir)], capture_output=True, timeout=timeout
    )
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    try:
        ptx = result.stdout.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Compiler stdout contains non-ASCII bytes") from exc
    if not ptx.lstrip().startswith(".version"):
        raise ValueError(
            "Compiler stdout must begin with .version, and contain PTX only"
        )
    return ptx, elapsed


def assemble(ptx: str, arch: str) -> dict:
    assembler = shutil.which("ptxas")
    if not assembler:
        raise RuntimeError("ptxas is not installed or is not on PATH")
    with tempfile.TemporaryDirectory(prefix="emmy-ptx-") as folder:
        source = Path(folder) / "kernel.ptx"
        source.write_text(ptx, encoding="ascii")
        result = subprocess.run(
            [
                assembler,
                str(source),
                "-arch=" + arch,
                "-o",
                str(Path(folder) / "kernel.cubin"),
                "-v",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode:
            diagnostics = "\n".join(
                text.strip() for text in (result.stdout, result.stderr) if text.strip()
            )
            raise RuntimeError(
                diagnostics or f"ptxas exited with status {result.returncode}"
            )
        return {
            "assembler": assembler,
            "arch": arch,
            "resource_usage": result.stderr.strip(),
        }


def load_specs(path: Path | None = None) -> dict:
    spec = importlib.util.spec_from_file_location(
        "independent_problem_references", path or ROOT / "tools/expressions_open.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {item["id"]: item for item in module.OPEN}


def input_array(arg: dict, seed_offset: int, pattern: str = "normal") -> np.ndarray:
    dtype = DTYPES[arg["dtype"]]
    shape = tuple(arg.get("shape") or [1])
    if "data" in arg:
        return np.asarray(arg["data"], dtype=dtype).reshape(shape)
    seed = int(arg.get("seed", 0)) + seed_offset
    rng = np.random.default_rng(seed)
    if pattern == "zeros":
        result = np.zeros(shape)
    elif pattern == "ones":
        result = np.ones(shape)
    elif pattern == "alternating":
        result = (np.arange(math.prod(shape)).reshape(shape) % 2 * 2 - 1) * (
            1 + seed % 3
        )
    elif pattern == "wide":
        result = rng.standard_normal(shape) * 8
    else:
        result = rng.standard_normal(shape)
    return np.asarray(result, dtype=dtype)


def expected_outputs(
    folder: Path, meta: dict, seed_offset: int, pattern: str, specs: dict
) -> dict:
    if seed_offset == 0 and pattern == "normal":
        with np.load(folder / "expected.npz", allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    if meta["id"] not in specs:
        raise ValueError(
            "Alternate inputs require an independent reference in expressions_open.py"
        )
    arguments = {
        arg["name"]: arg for kernel in meta["kernels"] for arg in kernel["args"]
    }
    arrays = [
        input_array(arguments[name], seed_offset, pattern)
        for name in meta["graph_inputs"]
    ]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        outputs = specs[meta["id"]]["ref"](*arrays)
    return dict(zip(meta["graph_outputs"], map(np.asarray, outputs)))


def make_args(kernel: dict, produced: dict, seed_offset: int, pattern: str) -> list:
    by_name = {arg["name"]: arg for arg in kernel["args"]}
    if set(by_name) != set(kernel["arg_order"]):
        raise ValueError("meta.json args do not match arg_order")
    result = []
    for name in kernel["arg_order"]:
        arg = dict(by_name[name])
        if arg.pop("intermediate", False):
            arg["data"] = np.asarray(produced[name]).ravel().tolist()
        elif not arg.get("output") and not arg.get("scalar"):
            if pattern != "normal":
                arg["data"] = input_array(arg, seed_offset, pattern).ravel().tolist()
                arg.pop("seed", None)
            elif "seed" in arg:
                arg["seed"] += seed_offset
        result.append(arg)
    return result


def compare_outputs(produced: dict, expected: dict, rtol: float, atol: float) -> list:
    rows = []
    for name, reference in expected.items():
        if name not in produced:
            raise ValueError(f"Missing output {name}")
        actual = np.asarray(produced[name])
        if actual.shape != reference.shape:
            # Scalar graph outputs are represented as one-element device buffers.
            if actual.size == reference.size == 1 and (
                not actual.shape or not reference.shape
            ):
                actual = actual.reshape(reference.shape)
            else:
                raise ValueError(
                    f"Output {name} shape {actual.shape} != reference {reference.shape}"
                )
        if reference.dtype.kind in "iu" and rtol == atol == 0:
            if actual.dtype.kind not in "iu":
                raise ValueError(f"Output {name} requires exact integer values")
            # isclose (and mixed int64/uint64 equality) can promote to float64,
            # hiding adjacent integers above 2**53. Python integers preserve
            # both comparison and subtraction, including endpoint differences.
            exact_actual, exact_reference = actual.astype(object), reference.astype(object)
            good = exact_actual == exact_reference
            delta = np.asarray(np.abs(exact_actual - exact_reference), dtype=np.float64)
        else:
            good = np.isclose(actual, reference, rtol=rtol, atol=atol, equal_nan=False)
            delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
        finite_delta = delta[np.isfinite(delta)]
        row = {
            "name": name,
            "pass": bool(good.all()),
            "elements": actual.size,
            "mismatches": int(np.count_nonzero(~good)),
            "max_abs_error": float(finite_delta.max()) if finite_delta.size else None,
        }
        if not good.all():
            index = tuple(np.argwhere(~good)[0]) if actual.shape else ()
            row["first_failure"] = {
                "index": list(map(int, index)),
                "actual": actual[index].item(),
                "expected": reference[index].item(),
            }
        rows.append(row)
    return rows


class CudaDriver:
    """Minimal driver bindings. No CUDA source compilation or existing lowering."""

    def __init__(self):
        self.lib = (
            ct.WinDLL("nvcuda.dll") if os.name == "nt" else ct.CDLL("libcuda.so.1")
        )
        self.context = ct.c_void_p()
        self.allocations = []
        self.module = ct.c_void_p()
        self.call("cuInit", ct.c_uint(0))
        device = ct.c_int()
        self.call("cuDeviceGet", ct.byref(device), ct.c_int(0))
        self.device = device
        major, minor = ct.c_int(), ct.c_int()
        self.call("cuDeviceComputeCapability", ct.byref(major), ct.byref(minor), device)
        self.arch = f"sm_{major.value}{minor.value}"
        name = ct.create_string_buffer(256)
        self.call("cuDeviceGetName", name, ct.c_int(len(name)), device)
        self.name = name.value.decode("ascii", errors="replace")
        self.call("cuCtxCreate_v2", ct.byref(self.context), ct.c_uint(0), device)

    def call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result:
            message = ct.c_char_p()
            self.lib.cuGetErrorString(ct.c_int(result), ct.byref(message))
            detail = message.value.decode() if message.value else "unknown"
            raise RuntimeError(f"{name} failed: CUDA {result}: {detail}")

    def close(self):
        # A failed launch may invalidate the context. Best-effort cleanup only.
        for pointer in self.allocations:
            self.lib.cuMemFree_v2(pointer)
        if self.module.value:
            self.lib.cuModuleUnload(self.module)
        if self.context.value:
            self.lib.cuCtxDestroy_v2(self.context)

    def shared_memory_limits(self) -> tuple[int, int]:
        """Query hardware limits; architecture generation alone is insufficient."""
        default = ct.c_int()
        self.call("cuDeviceGetAttribute", ct.byref(default), ct.c_int(8), self.device)
        optin = ct.c_int()
        status = self.lib.cuDeviceGetAttribute(
            ct.byref(optin), ct.c_int(97), self.device
        )
        # Older drivers may not recognize the opt-in attribute. They retain
        # the ordinary shared-memory limit; other errors must remain visible.
        if status not in (0, 1):
            raise RuntimeError(
                f"cuDeviceGetAttribute(opt-in shared memory) failed: CUDA {status}"
            )
        return default.value, max(
            default.value, optin.value
        ) if status == 0 else default.value

    def run(self, request: dict) -> dict:
        ptx = request["ptx"]
        target = re.search(r"\.target\s+(sm_\d+)", ptx)
        adapted = False
        if target and int(target.group(1)[3:]) > int(self.arch[3:]):
            if not request.get("adapt_target"):
                raise UnsupportedLocal(
                    f"PTX targets {target.group(1)}, local GPU is {self.arch}; use --adapt-local-target explicitly"
                )
            if "cp.async" in ptx or "mma." in ptx or "mbarrier." in ptx:
                raise UnsupportedLocal(
                    "PTX requires instructions unavailable on the local GPU"
                )
            ptx = ptx[: target.start(1)] + self.arch + ptx[target.end(1) :]
            adapted = True
        shared_bytes = int(request.get("smem_bytes", 0))
        default_shared, optin_shared = self.shared_memory_limits()
        if shared_bytes > optin_shared:
            raise UnsupportedLocal(
                f"Requested {shared_bytes} dynamic shared bytes; "
                f"device opt-in limit is {optin_shared}"
            )
        source = ct.create_string_buffer(ptx.encode("ascii") + b"\0")
        self.call(
            "cuModuleLoadData", ct.byref(self.module), ct.cast(source, ct.c_void_p)
        )
        function = ct.c_void_p()
        self.call(
            "cuModuleGetFunction",
            ct.byref(function),
            self.module,
            request["entry"].encode("ascii"),
        )
        static_shared = ct.c_int()
        self.call("cuFuncGetAttribute", ct.byref(static_shared), ct.c_int(1), function)
        if static_shared.value + shared_bytes > optin_shared:
            raise UnsupportedLocal(
                f"Requested {shared_bytes} dynamic plus {static_shared.value} static "
                f"shared bytes; device opt-in limit is {optin_shared}"
            )
        if static_shared.value + shared_bytes > default_shared:
            self.call(
                "cuFuncSetAttribute", function, ct.c_int(8), ct.c_int(shared_bytes)
            )
        holders, arrays, pointers = [], {}, {}
        for arg in request["args"]:
            if arg.get("scalar"):
                holder = {
                    "i32": ct.c_int32,
                    "i64": ct.c_int64,
                    "f32": ct.c_float,
                    "f64": ct.c_double,
                }[arg["dtype"]](arg["value"])
            else:
                array = (
                    np.empty(arg["shape"] or [1], dtype=DTYPES[arg["dtype"]])
                    if arg.get("output")
                    else input_array(arg, 0)
                )
                if arg.get("output"):
                    array.fill(
                        np.nan if np.issubdtype(array.dtype, np.floating) else -1
                    )
                array = np.ascontiguousarray(array)
                pointer = ct.c_uint64()
                self.call(
                    "cuMemAlloc_v2",
                    ct.byref(pointer),
                    ct.c_size_t(max(1, array.nbytes)),
                )
                self.allocations.append(pointer)
                self.call(
                    "cuMemcpyHtoD_v2",
                    pointer,
                    ct.c_void_p(array.ctypes.data),
                    ct.c_size_t(array.nbytes),
                )
                arrays[arg["name"]], pointers[arg["name"]] = array, pointer
                holder = pointer
            holders.append(holder)
        parameters = (ct.c_void_p * len(holders))(
            *(ct.cast(ct.byref(v), ct.c_void_p) for v in holders)
        )
        elapsed = []
        warmup = request.get("warmup", 0)
        for iteration in range(warmup + request.get("iters", 1)):
            for name in request.get("zero_outputs", []):
                self.call(
                    "cuMemsetD8_v2",
                    pointers[name],
                    ct.c_ubyte(0),
                    ct.c_size_t(arrays[name].nbytes),
                )
            start = time.perf_counter()
            dimensions = [ct.c_uint(v) for v in request["grid"] + request["block"]]
            self.call(
                "cuLaunchKernel",
                function,
                *dimensions,
                ct.c_uint(request.get("smem_bytes", 0)),
                ct.c_void_p(),
                parameters,
                ct.c_void_p(),
            )
            self.call("cuCtxSynchronize")
            if iteration >= warmup:
                elapsed.append((time.perf_counter() - start) * 1e6)
        outputs = {}
        for arg in request["args"]:
            if arg.get("output"):
                array = arrays[arg["name"]]
                self.call(
                    "cuMemcpyDtoH_v2",
                    ct.c_void_p(array.ctypes.data),
                    pointers[arg["name"]],
                    ct.c_size_t(array.nbytes),
                )
                outputs[arg["name"]] = {
                    "shape": list(array.shape),
                    "dtype": arg["dtype"],
                    "values": array.ravel().tolist(),
                    "truncated": False,
                }
        return {
            "ok": True,
            "outputs": outputs,
            "median_us": float(np.median(elapsed)),
            "timing_method": "host launch and synchronize; not kernel-only",
            "device": self.name,
            "arch": self.arch,
            "target_adapted": adapted,
        }


def local_run(request: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="emmy-run-") as directory:
        source, result = (
            Path(directory) / "request.json",
            Path(directory) / "result.json",
        )
        source.write_text(json.dumps(request), encoding="utf-8")
        process = subprocess.run(
            [sys.executable, str(Path(__file__)), "--worker", str(source), str(result)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.exists():
            response = json.loads(result.read_text(encoding="utf-8"))
            if process.returncode and response.get("ok"):
                raise RuntimeError(
                    f"CUDA worker reported output but exited {process.returncode}: {process.stderr or process.stdout}"
                )
            return response
        raise RuntimeError(f"CUDA worker exited {process.returncode}: {process.stderr}")


def remote_post(server: str, endpoint: str, payload: dict) -> dict:
    import requests

    infrastructure_failures = 0
    for attempt in range(30):
        try:
            response = requests.post(
                server.rstrip("/") + endpoint, json=payload, timeout=(15, 90)
            )
        except (requests.ConnectionError, requests.Timeout):
            infrastructure_failures += 1
            if infrastructure_failures >= 3:
                raise
            time.sleep(2 * infrastructure_failures)
            continue
        if response.status_code in (429, 502, 503, 504):
            infrastructure_failures += 1
            if infrastructure_failures >= 3:
                response.raise_for_status()
            time.sleep(2 * infrastructure_failures)
            continue
        response.raise_for_status()
        result = response.json()
        if result.get("stage") != "busy":
            return result
        time.sleep(2)
    raise RuntimeError("Runner stayed busy after 30 serialized attempts")


def check_problem(folder: Path, args, specs: dict) -> dict:
    meta = json.loads((folder / "meta.json").read_text())
    report = {
        "problem": meta["id"],
        "mode": args.mode,
        "kernels": [],
        "cases": [],
        "pass": True,
    }
    compiled = []
    for kernel in meta["kernels"]:
        before = compiler_fingerprints(Path(args.compiler))
        ptx, seconds = compile_ptx(
            Path(args.compiler), folder / (kernel["entry"] + ".ir.json")
        )
        after = compiler_fingerprints(Path(args.compiler))
        if before != after:
            raise RuntimeError(
                "Compiler source changed while the subprocess was running; retry for an attributable result"
            )
        record = {
            "entry": kernel["entry"],
            "compile_seconds": seconds,
            "ptx_bytes": len(ptx),
        }
        record["ptx_sha256"] = hashlib.sha256(ptx.encode("ascii")).hexdigest()
        record["compiler_sources_sha256"] = after
        if args.mode == "assemble":
            record.update(assemble(ptx, args.arch))
        elif args.mode == "remote":
            assembly = remote_post(args.server, "/compile", {"ptx": ptx})
            if not assembly.get("ok"):
                raise RuntimeError(assembly.get("stderr", str(assembly)))
            if kernel["entry"] not in assembly.get("entry_points", []):
                raise ValueError("Runner compiled a different entry point")
            record["registers"] = assembly.get("registers", {}).get(kernel["entry"])
            record["cubin_id"] = assembly["cubin_id"]
        compiled.append((kernel, ptx, record))
        report["kernels"].append(record)
    if args.mode == "assemble":
        return report
    for seed_offset in map(int, args.seeds.split(",")):
        for pattern in args.patterns.split(","):
            expected = expected_outputs(folder, meta, seed_offset, pattern, specs)
            produced = {}
            case = {"seed_offset": seed_offset, "pattern": pattern, "runs": []}
            for kernel, ptx, record in compiled:
                run_args = make_args(kernel, produced, seed_offset, pattern)
                request = {
                    key: kernel[key]
                    for key in ("entry", "grid", "block", "smem_bytes", "zero_outputs")
                }
                request.update(
                    args=run_args,
                    warmup=args.warmup,
                    iters=args.iters,
                    max_return=max(
                        4096,
                        sum(math.prod(a["shape"]) for a in run_args if a.get("output")),
                    ),
                )
                if args.mode == "local":
                    result = local_run(
                        {**request, "ptx": ptx, "adapt_target": args.adapt_local_target}
                    )
                else:
                    result = remote_post(
                        args.server, "/run", {**request, "cubin_id": record["cubin_id"]}
                    )
                if not result.get("ok"):
                    if result.get("stage") == "unsupported":
                        raise UnsupportedLocal(result["stderr"])
                    raise RuntimeError(result.get("stderr", str(result)))
                for name, output in result["outputs"].items():
                    if output.get("truncated"):
                        raise ValueError(f"Runner truncated {name}")
                    produced[name] = np.asarray(
                        output["values"], dtype=DTYPES[output["dtype"]]
                    ).reshape(output["shape"])
                case["runs"].append(
                    {key: value for key, value in result.items() if key != "outputs"}
                )
            case["outputs"] = compare_outputs(
                produced, expected, meta["rtol"], meta["atol"]
            )
            case["pass"] = all(output["pass"] for output in case["outputs"])
            report["pass"] &= case["pass"]
            report["cases"].append(case)
    return report


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        driver = None
        try:
            driver = CudaDriver()
            result = driver.run(
                json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
            )
        except UnsupportedLocal as exc:
            result = {"ok": False, "stage": "unsupported", "stderr": str(exc)}
        except Exception as exc:
            result = {"ok": False, "stage": "run", "stderr": str(exc)}
        finally:
            if driver:
                driver.close()
        Path(sys.argv[3]).write_text(json.dumps(result), encoding="utf-8")
        return 0 if result["ok"] else 1
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "problems",
        nargs="*",
        help="problem ids or directories; default is all public problems",
    )
    parser.add_argument("--compiler", default=str(ROOT / "compile.py"))
    parser.add_argument(
        "--mode", choices=("assemble", "local", "remote"), default="assemble"
    )
    parser.add_argument(
        "--arch", default="sm_89", help="local ptxas target for assembly-only checks"
    )
    parser.add_argument("--server", default=SERVER)
    parser.add_argument(
        "--adapt-local-target",
        action="store_true",
        help="explicitly lower .target for older local devices when instructions permit",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="comma-separated offsets added to each original input seed",
    )
    parser.add_argument(
        "--patterns",
        default="normal",
        help="comma-separated normal,wide,ones,alternating,zeros; zeros can make normalization undefined",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument(
        "--report", type=Path, help="write machine-readable results, including failures"
    )
    parser.add_argument(
        "--references",
        action="append",
        default=[],
        help="additional expression-spec Python files for alternate-input references",
    )
    args = parser.parse_args()
    patterns = set(args.patterns.split(","))
    if patterns - {"normal", "wide", "ones", "alternating", "zeros"}:
        parser.error("Unknown input pattern: " + ", ".join(sorted(patterns)))
    if args.iters < 1 or args.iters > 200 or args.warmup < 0 or args.warmup > 20:
        parser.error("Require 1 <= iters <= 200 and 0 <= warmup <= 20")
    folders = (
        [
            (
                Path(name)
                if (Path(name) / "meta.json").exists()
                else ROOT / "problems" / name
            )
            for name in args.problems
        ]
        if args.problems
        else sorted((ROOT / "problems").glob("*"))
    )
    if args.problems and any(not (folder / "meta.json").exists() for folder in folders):
        parser.error(
            "Unknown problem directories: "
            + ", ".join(
                str(folder) for folder in folders if not (folder / "meta.json").exists()
            )
        )
    folders = [folder for folder in folders if (folder / "meta.json").exists()]
    if not folders:
        parser.error("No problem packages found")
    specs, rows = load_specs(), []
    for references in args.references:
        specs.update(load_specs(Path(references)))
    for folder in folders:
        try:
            row = check_problem(folder, args, specs)
        except UnsupportedLocal as exc:
            row = {
                "problem": folder.name,
                "mode": args.mode,
                "pass": False,
                "skipped": True,
                "error": str(exc),
            }
        except Exception as exc:
            row = {
                "problem": folder.name,
                "mode": args.mode,
                "pass": False,
                "error": str(exc),
            }
        rows.append(row)
        state = "PASS" if row["pass"] else "SKIP" if row.get("skipped") else "FAIL"
        print(
            f"{state:4} {folder.name}: {row.get('error', str(len(row.get('cases', []))) + ' execution cases')}",
            flush=True,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(
                    {
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "report_time_compiler_sources_sha256": compiler_fingerprints(
                            Path(args.compiler)
                        ),
                        "configuration": vars(args) | {"report": str(args.report)},
                        "results": rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    passed = sum(row["pass"] for row in rows)
    print(
        f"{passed}/{len(rows)} passed; {sum(row.get('skipped', False) for row in rows)} skipped"
    )
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
