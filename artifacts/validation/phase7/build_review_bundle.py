"""Package phase-7 review evidence after the release summary is finalized.

Retain phase-6 and the initial phase-7 failed reports unchanged. Archive checks
run after packaging and are intentionally excluded to avoid a self-reference.
"""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.package_review import ALLOWED_SUFFIXES, package_review

OUT = ROOT / 'artifacts/validation/phase7'
includes = set(json.loads((ROOT / 'artifacts/validation/phase6/review-includes.json').read_text(encoding='utf-8-sig')))
includes.update({
    'artifacts/validation/phase7/public-submit',
    'artifacts/validation/phase7/feature-runner',
    'artifacts/validation/phase7/clean-install',
    'artifacts/validation/phase7/cpu-regressions.json',
    'artifacts/validation/phase7/cpu-regressions.log',
    'artifacts/validation/phase7/cpu-regressions-final.json',
    'artifacts/validation/phase7/cpu-regressions-final.log',
    'artifacts/validation/phase7/compatibility.json',
    'artifacts/validation/phase7/check_compatibility.py',
    'artifacts/validation/phase7/before-fix',
    'artifacts/validation/phase7/pre-fix-integer-runner',
    'artifacts/validation/phase7/new-feature-runner',
    'artifacts/validation/phase7/changed-scalar-runner',
    'artifacts/validation/phase7/changed-scalar-copy.json',
    'artifacts/validation/phase7/changed-scalar-manifest.json',
    'artifacts/validation/phase7/feature-manifest.json',
    'artifacts/validation/phase7/new-feature-manifest.json',
    'artifacts/validation/phase7/build_fixture_manifests.py',
    'artifacts/validation/phase7/clean-cases.json',
    'artifacts/validation/phase7/check_oracles.py',
    'artifacts/validation/phase7/oracle-audit.json',
    'artifacts/validation/phase7/runtime-changes.txt',
    'artifacts/validation/phase7/release-summary.json',
    'artifacts/validation/phase7/release-summary.initial.json',
    'artifacts/validation/phase7/audit_feature_runner.py',
    'artifacts/validation/phase7/audit_feature_runner-strict-integer-json.py',
    'artifacts/validation/phase7/build_release_summary.py',
    'artifacts/validation/phase7/verify_review_bundle.py',
    'artifacts/validation/phase7/build_review_bundle.py',
    'artifacts/validation/phase7/review-includes.json',
    'artifacts/release/phase7/latest.json',
    'artifacts/release/phase7/evidence',
})
runtime = json.loads((ROOT / 'artifacts/release/phase7/latest.json').read_text(encoding='utf-8-sig'))
includes.add('artifacts/release/phase7/' + runtime['manifest'])
summary = json.loads((OUT / 'release-summary.json').read_text(encoding='utf-8-sig'))
assert summary['pass'], 'Refusing to package an unsuccessful phase-7 release gate'
excluded = (
    'artifacts/release/phase7/review/',
    'artifacts/validation/phase7/review-bundle-verification.json',
    'artifacts/validation/phase7/review-bundle-logs/',
)
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
            if not relative.startswith(excluded):
                includes.add(relative)
assert not any(name.startswith(excluded) for name in includes), 'Post-package evidence must remain external'
selection = sorted(includes)
selection_path = OUT / 'review-includes.json'
assert not selection_path.exists(), 'Refusing to overwrite retained selection'
selection_path.write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
for name in selection:
    assert (ROOT / name).exists(), name
result = package_review(
    ROOT, ROOT / 'artifacts/release/phase7/review', includes=selection,
    evidence=['artifacts/validation/phase7/release-summary.json'], update_latest=True,
)
print(json.dumps(result, indent=2), flush=True)
