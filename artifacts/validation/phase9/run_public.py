"""Run every public package through the unchanged organizer submit.py client."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "public-submit"
RUNTIME = ("compile.py", "ptx_backend.py", "ptx_tensor.py", "ptx_bulk.py", "requirements.txt")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    packages=sorted(p.parent for p in (ROOT/'problems').glob('*/meta.json'))
    assert len(packages)==15
    files=[ROOT/p for p in RUNTIME]+[ROOT/'submit.py',Path(__file__)]
    files += [p for folder in packages for p in folder.iterdir()
              if p.name=='meta.json' or p.name=='expected.npz' or p.name.endswith('.ir.json')]
    before={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    report={'pass':False,'scope':'Unchanged submit.py, default compiler, all public packages',
            'started_utc':datetime.now(timezone.utc).isoformat(),
            'source_and_input_sha256':before,'planned_problems':[p.name for p in packages],'rows':[]}
    def save():
        (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    env=os.environ.copy();env['PYTHONUTF8']='1'
    save()
    for folder in packages:
        cmd=[sys.executable,'submit.py',folder.name,'--compiler','compile.py']
        result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,timeout=240)
        stdout=OUT/(folder.name+'.stdout.txt');stderr=OUT/(folder.name+'.stderr.txt')
        stdout.write_bytes(result.stdout);stderr.write_bytes(result.stderr)
        row={'problem':folder.name,'command':cmd,'returncode':result.returncode,
             'pass':result.returncode==0,'stdout_sha256':digest(stdout),'stderr_sha256':digest(stderr)}
        report['rows'].append(row);save()
        print(folder.name,'PASS' if row['pass'] else 'FAIL',flush=True)
    after={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    report.update(source_and_input_sha256_after=after,tracked_unchanged=before==after,
                  passed_problems=sum(row['pass'] for row in report['rows']),
                  finished_utc=datetime.now(timezone.utc).isoformat())
    report['pass']=report['passed_problems']==15 and before==after
    save()
    return 0 if report['pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
