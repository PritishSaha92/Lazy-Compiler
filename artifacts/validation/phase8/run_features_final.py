"""Freeze this revision's 51-package roster, then run the serialized GPU gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNTIME = ('compile.py','ptx_backend.py','ptx_tensor.py','ptx_bulk.py','requirements.txt')

def main():
    retained = json.loads((HERE/'retained-feature-roster.json').read_text())
    packages = [p.parent.relative_to(ROOT).as_posix()
                for p in sorted((ROOT/'examples/phase8').glob('*/*/meta.json'))]
    packages = [p for p in packages if p not in {
        'examples/phase8/remainder/floating_f64_mod',
        'examples/phase8/remainder/floating_f64_remainder'}]
    assert len(packages) == 10
    packages += retained['packages']
    assert len(packages) == len(set(packages)) == 51
    manifest = {'schema':'emmy-phase8-feature-inputs-v1',
                'compiler_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in RUNTIME},
                'package_count':len(packages),'packages':packages,
                'files_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in packages for p in (ROOT/folder).rglob('*') if p.is_file()}}
    with (HERE/'feature-manifest-final.json').open('x',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2);f.write('\n')
    return subprocess.call([sys.executable,str(ROOT/'tools/check_feature_examples.py'),
                            *[str(ROOT/p) for p in packages],
                            '--report-dir',str(HERE/'feature-runner-final')])

if __name__ == '__main__':
    raise SystemExit(main())
