"""Select exact review evidence and invoke the deterministic archive builder."""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.package_review import ALLOWED_SUFFIXES, package_review

OUT = ROOT / 'artifacts/validation/phase6'
includes = set(json.loads((ROOT / 'artifacts/validation/phase5/review-includes.json').read_text(encoding='utf-8-sig')))
includes.update({
    'artifacts/validation/phase6/public-submit',
    'artifacts/validation/phase6/feature-runner',
    'artifacts/validation/phase6/clean-install',
    'artifacts/validation/phase6/cpu-regressions.json',
    'artifacts/validation/phase6/cpu-regressions.log',
    'artifacts/validation/phase6/ptx-compatibility.json',
    'artifacts/validation/phase6/release-summary.json',
    'artifacts/validation/phase6/audit_feature_runner.py',
    'artifacts/validation/phase6/async-target-before.json',
    'artifacts/validation/phase6/build_release_summary.py',
    'artifacts/validation/phase6/video-word-count.json',
    'artifacts/validation/phase6/verify_review_bundle.py',
    'artifacts/validation/phase6/build_review_bundle.py',
    'artifacts/validation/phase6/review-includes.json',
    'artifacts/release/phase6/latest.json',
    'artifacts/release/phase6/evidence',
})
runtime = json.loads((ROOT / 'artifacts/release/phase6/latest.json').read_text(encoding='utf-8-sig'))
includes.add('artifacts/release/phase6/' + runtime['manifest'])
pages = [ROOT / name for name in ('README.md', 'SOLUTION.md', 'SUBMISSION.md')]
pages += sorted((ROOT / 'docs').rglob('*.md'))
pages += sorted((ROOT / 'examples').rglob('*.md'))
for page in pages:
    for raw in re.findall(r'\]\(([^)]+)\)', page.read_text(encoding='utf-8-sig')):
        link = raw.strip().strip('<>').split('#', 1)[0]
        if not link or re.match(r'^[A-Za-z]+:', link):
            continue
        path = (page.parent / link).resolve()
        if path.is_relative_to(ROOT) and path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES:
            relative = path.relative_to(ROOT).as_posix()
            if not relative.startswith('artifacts/release/phase6/review/'):
                includes.add(relative)
selection = sorted(includes)
selection_path = OUT / 'review-includes.json'
assert not selection_path.exists(), 'Refusing to overwrite retained selection'
selection_path.write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
for name in selection:
    assert (ROOT / name).exists(), name
result = package_review(
    ROOT, ROOT / 'artifacts/release/phase6/review', includes=selection,
    evidence=['artifacts/validation/phase6/release-summary.json'], update_latest=True,
)
print(json.dumps(result, indent=2), flush=True)
