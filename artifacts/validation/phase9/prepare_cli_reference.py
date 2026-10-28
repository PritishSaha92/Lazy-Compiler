"""Freeze actual CLI stdout for checkout reproduction, including all new features."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
RUNTIME=('compile.py','ptx_backend.py','ptx_tensor.py','ptx_bulk.py','requirements.txt')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination=HERE/'cli-reference.json'
    assert not destination.exists()
    roster=json.loads((HERE/'new-feature-manifest.json').read_text())['packages']
    roster += ['problems/p00_scalar_add','examples/bulk/bulk_rectangle',
               'examples/ring/ring_tensor_k112_bf16','examples/generated_tensor']
    paths=[p for name in roster for p in (ROOT/name).glob('*.ir.json')]
    assert len(paths)==len(set(paths))==20
    before={name:digest(ROOT/name) for name in RUNTIME}
    files={str(p.relative_to(ROOT)):digest(p) for ir in paths for p in (ir,ir.with_name('meta.json'))}
    report={'schema':'emmy-phase9-cli-reference-v1','pass':False,
            'scope':'16 new feature packages plus four retained path examples; real CLI stdout, no GPU execution',
            'runtime_source_sha256':before,'inputs_sha256':files,'cli_checks':[]}
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(EMMY_GPU_TESTS='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    for ir in paths:
        name=ir.relative_to(ROOT).as_posix()
        result=subprocess.run([sys.executable,'-B','compile.py',name],cwd=ROOT,env=env,capture_output=True,timeout=120)
        assert result.returncode==0 and not result.stderr,(name,result.stderr)
        report['cli_checks'].append({'ir':name,'pass':True,'ptx_sha256':hashlib.sha256(result.stdout).hexdigest(),
                                    'ptx_bytes':len(result.stdout),'stderr':''})
        print('CLI reference:',name,flush=True)
    assert before=={name:digest(ROOT/name) for name in RUNTIME}
    assert all(digest(ROOT/name)==value for name,value in files.items())
    report.update(sources_unchanged=True,finished_utc=datetime.now(timezone.utc).isoformat())
    report['pass']=True
    destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
