#!/usr/bin/env python3
"""Emmy Kernel IR -> PTX. Stdout contains only the generated module."""
import sys
import json
from pathlib import Path

from ptx_backend import CompileError, compile_file


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python compile.py kernel.ir.json", file=sys.stderr)
        return 2
    try:
        ptx = compile_file(sys.argv[1])
        problem = json.loads(Path(sys.argv[1]).with_name('meta.json').read_text(encoding='utf-8'))
        matches = [k for k in problem['kernels'] if '.entry ' + k['entry'] + '(' in ptx]
        if len(matches) != 1:
            raise ValueError('Need exactly one matching launch contract')
        block = matches[0]['block']
        if len(block) != 3 or any(type(x) is not int or x < 1 for x in block):
            raise ValueError('Invalid block dimensions')
        marker = ')\n{\n'
        if ptx.count(marker) != 1:
            raise ValueError('Expected exactly one entry header')
        directive = '.maxntid ' + ', '.join(str(x) for x in block)
        ptx = ptx.replace(marker, ')\n' + directive + '\n{\n', 1)
    except (CompileError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f"compile.py: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
