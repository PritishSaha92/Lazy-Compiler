"""Verify and execute a fresh reviewer archive extraction; never launch a GPU."""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/validation/phase6'
RELEASE = ROOT / 'artifacts/release/phase6'
REPORT = OUT / 'review-bundle-verification.json'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

report = {'schema': 'emmy-review-extraction-verification-v1', 'pass': False,
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'local_gpu_used': False, 'network_used': False,
          'scope': 'Archive integrity, source identity, extracted CLI execution, focused CPU/assembler tests and primary reviewer links; no additional GPU results.'}
try:
    assert not REPORT.exists(), 'Refusing to overwrite retained verification'
    latest = read_json(RELEASE / 'review/latest.json')
    outer_path = RELEASE / 'review' / latest['manifest']
    assert sha(outer_path.read_bytes()) == latest['manifest_sha256']
    outer = read_json(outer_path)
    zip_path = RELEASE / 'review' / latest['zip']
    zip_bytes = zip_path.read_bytes()
    assert sha(zip_bytes) == latest['zip_sha256'] == outer['zip']['sha256']
    assert len(zip_bytes) == outer['zip']['bytes']
    frozen = read_json(OUT / 'release-summary.json')['source_sha256']
    assert frozen == outer['runtime_sha256']
    build_root = ROOT / '.build'
    build_root.mkdir(exist_ok=True)
    extracted = Path(tempfile.mkdtemp(prefix='phase6-review-', dir=build_root)).resolve()
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(names) == len(set(names)) == outer['zip']['members']
        member = outer['contents_manifest']['member']
        inner_bytes = z.read(member)
        assert sha(inner_bytes) == outer['contents_manifest']['sha256']
        inner = json.loads(inner_bytes)
        assert set(names) == set(inner['files']) | {member}
        assert frozen == inner['runtime_sha256']
        for info in z.infolist():
            name = info.filename
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and '..' not in rel.parts and ':' not in name and '\\' not in name
            assert not stat.S_ISLNK(info.external_attr >> 16)
            target = extracted.joinpath(*rel.parts).resolve()
            assert target.is_relative_to(extracted)
            data = z.read(name)
            if name != member:
                expected = inner['files'][name]
                assert len(data) == expected['bytes'] and sha(data) == expected['sha256'], name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    report['archive'] = {'path': str(zip_path.relative_to(ROOT)), 'sha256': sha(zip_bytes),
                         'bytes': len(zip_bytes), 'verified_members': len(names)}
    report['extraction'] = str(extracted.relative_to(ROOT))
    runtime_pointer = read_json(RELEASE / 'latest.json')
    runtime_name = runtime_pointer['zip']
    if isinstance(runtime_name, dict):
        runtime_name = runtime_name['path']
    runtime_path = RELEASE / runtime_name
    with zipfile.ZipFile(runtime_path) as rz:
        assert set(rz.namelist()) == set(frozen)
        for name, digest in frozen.items():
            assert sha((ROOT / name).read_bytes()) == digest
            assert sha((extracted / name).read_bytes()) == digest
            assert rz.read(name) == (extracted / name).read_bytes()
    report['runtime_source_sha256'] = frozen
    report['runtime_zip'] = {'path': str(runtime_path.relative_to(ROOT)), 'sha256': sha(runtime_path.read_bytes())}
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['EMMY_GPU_TESTS'] = '0'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONUTF8'] = '1'
    origin_code = 'import json,ptx_backend,ptx_tensor,ptx_bulk; print(json.dumps({m.__name__:m.__file__ for m in (ptx_backend,ptx_tensor,ptx_bulk)}))'
    origins_result = subprocess.run([sys.executable, '-B', '-c', origin_code], cwd=extracted, env=env, capture_output=True, timeout=120)
    assert origins_result.returncode == 0, origins_result.stderr.decode('utf8', 'replace')
    origins = json.loads(origins_result.stdout)
    assert all(Path(p).resolve().is_relative_to(extracted) for p in origins.values())
    report['extracted_module_origins'] = origins
    packages = ['problems/p00_scalar_add', 'examples/tensor/packages/tensor_global_f16',
                'examples/bulk/bulk_rectangle', 'examples/combined/bulk_tensor_k48_f16',
                'examples/generated_tensor', 'examples/ring/ring_tensor_k80_f16', 'examples/ring/ring_tensor_k112_bf16', 'examples/ring/ring_tensor_k112_f16_two_cta']
    report['cli_checks'] = []
    for package in packages:
        irs = sorted((extracted / package).glob('*.ir.json'))
        assert len(irs) == 1, package
        rel_ir = irs[0].relative_to(extracted).as_posix()
        results = [subprocess.run([sys.executable, '-B', 'compile.py', rel_ir], cwd=directory, env=env,
                                  capture_output=True, timeout=120) for directory in (extracted, ROOT)]
        assert all(r.returncode == 0 and not r.stderr for r in results), [(r.returncode, r.stderr.decode('utf8', 'replace')) for r in results]
        assert results[0].stdout == results[1].stdout and b'.visible .entry ' in results[0].stdout
        meta = read_json(extracted / package / 'meta.json')
        assert meta['kernels'][0]['entry'].encode('ascii') in results[0].stdout
        report['cli_checks'].append({'ir': rel_ir, 'pass': True, 'ptx_sha256': sha(results[0].stdout), 'ptx_bytes': len(results[0].stdout)})
        print('CLI verified: ' + package, flush=True)
    logs = OUT / 'review-bundle-logs'
    logs.mkdir(exist_ok=False)
    report['focused_tests'] = []
    for suite in ('test_tensor_cores.py', 'test_bulk_copy.py', 'test_feature_interactions.py', 'test_ring_pipeline.py', 'test_async_target.py'):
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', suite, '-v']
        completed = subprocess.run(command, cwd=extracted, env=env, capture_output=True, timeout=300)
        output = completed.stdout + completed.stderr
        log_path = logs / (suite + '.log')
        log_path.write_bytes(output)
        text = output.decode('utf8', 'replace')
        count = re.search(r'Ran (\d+) tests? in ', text)
        skipped = re.search(r'OK \(skipped=(\d+)\)', text)
        record = {'suite': suite, 'returncode': completed.returncode,
                  'methods': int(count.group(1)) if count else None,
                  'skipped': int(skipped.group(1)) if skipped else 0,
                  'log': str(log_path.relative_to(ROOT)), 'log_sha256': sha(output)}
        report['focused_tests'].append(record)
        assert completed.returncode == 0 and count and re.search(r'^OK(?: \(skipped=\d+\))?\s*$', text, re.M), text[-6000:]
        print('Tests verified: ' + json.dumps(record), flush=True)
    assert sum(t['methods'] for t in report['focused_tests']) == 30
    report['focused_test_totals'] = {'methods': 30, 'skipped': sum(t['skipped'] for t in report['focused_tests'])}
    report['focused_test_totals']['passed'] = 30 - report['focused_test_totals']['skipped']
    for name, expected in inner['files'].items():
        assert sha((extracted / name).read_bytes()) == expected['sha256'], 'Extracted source changed: ' + name
    for name, digest in frozen.items():
        assert sha((ROOT / name).read_bytes()) == digest
    primary = ['README.md', 'SUBMISSION.md', 'docs/FEATURES.md', 'examples/README.md', 'docs/VIDEO_SCRIPT.md', 'docs/PHASE6.md', 'docs/RESEARCH_REVIEW.md', 'docs/SUBMISSION_FORM.md']
    links = []
    external_release_links = []
    for name in primary:
        page = extracted / name
        text = page.read_text(encoding='utf-8-sig')
        assert '4?5' not in text and '3?5' not in text
        for raw in re.findall(r'\]\(([^)]+)\)', text):
            link = raw.strip().strip('<>').split('#', 1)[0]
            if not link or re.match(r'^[A-Za-z]+:', link):
                continue
            target = (page.parent / link).resolve()
            assert target.is_relative_to(extracted), (name, link)
            if target.exists():
                links.append({'page': name, 'link': link})
            elif target.is_relative_to(extracted / 'artifacts/release'):
                external_release_links.append({'page': name, 'link': link, 'reason': 'Distribution-only archive/pointer documented in SUBMISSION.md'})
            else:
                raise AssertionError('Broken primary reviewer link: ' + name + ' -> ' + link)
    report['primary_links'] = {'verified': len(links), 'external_release_artifacts': external_release_links}
    report['sources_unchanged'] = True
    report['pass'] = True
except Exception as exc:
    report['error'] = type(exc).__name__ + ': ' + str(exc)
finally:
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with REPORT.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report, indent=2), flush=True)
sys.exit(0 if report['pass'] else 1)