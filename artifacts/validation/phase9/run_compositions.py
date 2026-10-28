"""Run the fixed phase9 generated compositions on the organizer GPU.

Three independent inputs per package. Requests are serialized and every raw
exchange is retained. This script does not use a local GPU or benchmark speed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from tools import check_compiler as checker
from tools import selector_study as transport


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--compiler',type=Path,default=ROOT/'compile.py')
    args=parser.parse_args()
    out=args.report_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'events').mkdir()
    protocol=json.loads((HERE/'composition-protocol.json').read_text())
    generation=json.loads((HERE/'composition-generation.json').read_text())
    sanity=json.loads((HERE/'composition-reference-sanity.json').read_text())
    assert generation['pass'] and sanity['pass']
    specs=checker.load_specs(ROOT/'tests/expressions_phase9.py')
    assert digest(ROOT/'tests/expressions_phase9.py')==protocol['expression_sha256']
    folders=[ROOT/'examples/phase9/generated'/spec['id'] for spec in protocol['planned']]
    compiler=args.compiler.resolve()
    source=checker.compiler_fingerprints(compiler)
    tracked={str(ROOT/name):value for name,value in generation['files_sha256'].items()}
    for p in (Path(__file__),ROOT/'tools/check_compiler.py',ROOT/'tools/selector_study.py',ROOT/'tests/expressions_phase9.py',
              HERE/'composition-protocol.json',HERE/'composition-generation.json',HERE/'composition-reference-sanity.json'):
        tracked[str(p)]=digest(p)
    assert all(digest(Path(p))==value for p,value in tracked.items())
    report={'schema':'emmy-phase9-composition-runner-v1','pass':False,
            'scope':'Selected compiler; eight original generated compositions, three input cases each; no performance claim',
            'compiler':str(compiler),
            'started_utc':datetime.now(timezone.utc).isoformat(),'compiler_sha256':source,
            'tracked_sha256':tracked,'planned_packages':[str(p) for p in folders],
            'planned_cases':protocol['input_cases'],'planned_checks':len(folders)*len(protocol['input_cases']),
            'local_gpu_used':False,'events':[],'jobs':[]}
    def save():
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def post(server,endpoint,payload):
        number=len(report['events'])
        base=out/'events'/f'{number:04d}'
        request=base.with_suffix('.request.json')
        request.write_text(json.dumps({'endpoint':endpoint,'payload':payload},indent=2)+'\n',encoding='utf-8')
        event={'index':number,'job':len(report['jobs'])-1,'endpoint':endpoint,
               'request_sha256':digest(request),'status':'pending','http':[]}
        report['events'].append(event);save()
        def attempt(value):
            path=base.with_name(base.name+f'.http-{len(event["http"]):02d}.json')
            path.write_text(json.dumps(transport.diagnostic_json(value),indent=2)+'\n',encoding='utf-8')
            event['http'].append({'path':path.relative_to(out).as_posix(),'sha256':digest(path)})
            save()
        try:
            response=transport.audit_remote_post(server,endpoint,payload,attempt)
            path=base.with_suffix('.response.json')
            path.write_text(json.dumps(transport.diagnostic_json(response),indent=2)+'\n',encoding='utf-8')
            event.update(status='received',response_sha256=digest(path));save()
            return response
        except Exception as exc:
            event.update(status='transport_error',error=f'{type(exc).__name__}: {exc}');save()
            raise
    original=checker.remote_post
    checker.remote_post=post
    save()
    try:
        for folder in folders:
            for seed,pattern in protocol['input_cases']:
                job={'package':str(folder),'seed':seed,'pattern':pattern,'pass':False}
                report['jobs'].append(job);save()
                config=SimpleNamespace(compiler=str(compiler),mode='remote',server=checker.SERVER,
                    seeds=str(seed),patterns=pattern,warmup=0,iters=1,adapt_local_target=False,arch='sm_89')
                try:
                    result=checker.check_problem(folder,config,specs)
                    job.update(result=result,pass_=result['pass'] and len(result['cases'])==1)
                    job['pass']=job.pop('pass_')
                except Exception as exc:
                    job['error']=f'{type(exc).__name__}: {exc}'
                save()
                print(folder.name,seed,pattern,'PASS' if job['pass'] else 'FAIL',job.get('error',''),flush=True)
    finally:
        checker.remote_post=original
    after=checker.compiler_fingerprints(compiler)
    report.update(compiler_sha256_after=after,
        tracked_unchanged=all(digest(Path(p))==value for p,value in tracked.items()),
        checks=len(report['jobs']),passed_checks=sum(j['pass'] for j in report['jobs']),
        finished_utc=datetime.now(timezone.utc).isoformat())
    report['pass']=(report['checks']==report['passed_checks']==report['planned_checks'] and report['tracked_unchanged'] and source==after)
    save();print({k:report[k] for k in ('pass','checks','passed_checks','tracked_unchanged')},flush=True)
    return 0 if report['pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
