#!/usr/bin/env python3
"""Emmy Kernel IR -> PTX. Stdout contains only the generated module."""
import sys

from ptx_backend import CompileError, compile_file


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python compile.py kernel.ir.json", file=sys.stderr)
        return 2
    try:
        ptx = compile_file(sys.argv[1])
    except (CompileError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f"compile.py: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
