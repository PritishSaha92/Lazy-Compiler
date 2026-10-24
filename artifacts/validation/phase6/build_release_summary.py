from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/validation/phase6'
RUNTIME=('compile.py','ptx_backend.py','ptx_tensor.py','ptx_bulk.py','requirements.txt')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf8'))
def norm(mapping):return {Path(k).name:v for k,v in mapping.items() if Path(k).name in RUNTIME}
source={name:sha(ROOT/name) for name in RUNTIME}
paths={'public':OUT/'public-submit/summary.json','features':OUT/'feature-runner/summary.json','wire_audit':OUT/'feature-runner/integrity-audit.json','cpu':OUT/'cpu-regressions.json','clean_install':OUT/'clean-install/summary.json','baseline_ptx':OUT/'ptx-compatibility.json','source_review':ROOT/'docs/research/source-review-manifest.json'}
r={key:read(path) for key,path in paths.items()}
pub,features,wire,cpu,clean,compat=(r[k] for k in ('public','features','wire_audit','cpu','clean_install','baseline_ptx'))
assert pub['pass'] and pub['passed_problems']==15 and len(pub['rows'])==15 and pub['tracked_unchanged']
assert norm(pub['source_and_input_sha256'])==source==norm(pub['source_and_input_sha256_after'])
for name,digest in pub['source_and_input_sha256'].items():assert sha(ROOT/name)==digest,name
for row in pub['rows']:
    assert row['pass'] and row['returncode']==0 and not row['timeout']
    for stream in ('stdout','stderr'):assert sha(paths['public'].parent/row[stream])==row[stream+'_sha256']
assert features['pass'] and features['checks']==features['passed_checks']==features['planned_checks']==29
assert features['tracked_unchanged'] and norm(features['compiler_sha256'])==source==norm(features['compiler_sha256_after'])
for name,digest in features['tracked_sha256'].items():assert sha(Path(name))==digest,name
assert wire['integrity_pass'] and wire['numerical_gate_pass'] and wire['complete_replay']
assert wire['checks']==wire['passed_checks']==29 and wire['failed_checks']==0
assert wire['compiler_sha256']==source
assert wire['summary_sha256']==sha(paths['features'])
assert wire['feature_manifest_sha256']==sha(ROOT/'examples/research-feature-manifest.json')
assert wire['auditor_sha256']==sha(OUT/'audit_feature_runner.py')
assert cpu['pass'] and not cpu['gpu_execution_requested'] and cpu['sources_unchanged'] and cpu['executed_sources_unchanged']
assert not cpu['failures'] and not cpu['errors'] and cpu['tests_run']==117 and len(cpu['skipped'])==18
assert norm(cpu['sources_before_sha256'])==source==norm(cpu['sources_after_sha256'])
for name,digest in cpu['sources_after_sha256'].items():assert sha(ROOT/name)==digest,name
assert clean['ok'] and clean['source_unchanged_after_check'] and clean['source_sha256']==source
assert len(clean['cases'])==14 and len(clean['diagnostics'])==3 and clean['staged_package_files']==sorted(RUNTIME)
assert all(case['deterministic'] and case['audit']['exit_code']==0 and not case['audit']['blocked_events'] and not case['audit']['emmy_renderer_calls'] for case in clean['cases'])
assert compat['pass'] and compat['kernels']==compat['identical']==120 and compat['inputs_unchanged']
assert norm(compat['compiler_sources_before_sha256'])==source==norm(compat['compiler_sources_after_sha256'])
for name,digest in compat['input_sha256'].items():assert sha(Path(name))==digest,name
old=read(ROOT/'artifacts/validation/phase5/feature-runner/summary.json')
current={row['id']:row for row in features['jobs']};matches=[]
for row in old['jobs']:
    before=row['result']['kernels'][0]['ptx_sha256'];after=current[row['id']]['result']['kernels'][0]['ptx_sha256']
    assert before==after,row['id']
    matches.append({'id':row['id'],'ptx_sha256':after})
assert len(matches)==26
review=r['source_review'];assert review['complete_local_coverage'] and review['text_files_read']==336 and review['utf8_lines_read']==63630 and review['images_inspected']==6
assert review['article']['complete_unique_text_read'] and review['article']['technical_figures_inspected']==6
report={'schema':'emmy-phase6-release-v1','pass':True,'created_utc':datetime.now(timezone.utc).isoformat(),
 'source_sha256':source,'scope':'Current-source release gates after DeepInfer/MoK review. Counts describe this release run only; historical failures remain unchanged.',
 'changes':['Reject native CpAsyncCopy/Commit/Wait below SM80; preserve explicitly requested synchronous fallback.','Three independently referenced multiwrap ring pipelines and same-process compilation isolation tests.','Research review, exact reading coverage and refreshed submission documentation.'],
 'organizer_gpu':{'public_checks':15,'public_passes':15,'feature_checks':29,'feature_passes':29,'total_checks':44,'total_passes':44,'local_gpu_used':False},
 'feature_roster':{'typed_tensor_examples':17,'typed_bulk_examples':6,'typed_combined_examples':2,'unchanged_emmy_generated_ragged_matmul':1,'multiwrap_ring_pipelines':3},
 'cpu':{'methods':117,'passed':99,'explicit_local_gpu_skips':18},
 'clean_install':{'packages':14,'successful_compilations':42,'expected_diagnostics':3},
 'baseline_ptx_identity':{'retained_scalar_modules':120,'prior_feature_modules':26,'prior_features':matches,'additional_gpu_executions':0},
 'raw_feature_audit':{'events':wire['events_verified'],'http_attempts':wire['http_attempts_verified'],'files_verified':wire['files_verified'],'complete_replay':True},
 'research_review':{'deepinfer_commit':review['commit'],'text_files_read':336,'utf8_lines_read':63630,'images_inspected':6,'article_figures_inspected':6,'vendor_submodule_sources_present':False},
 'historical_fp64_reference_failures':9,
 'historical_note':'Earlier generated FP64-buffer/FP32-state mathematical-reference failures remain failures. This release does not relabel or rerun them as part of its 44 checks.',
 'performance_note':'One timed iteration per feature check is indicative only. No measured speed ranking or hidden-set success is asserted.',
 'evidence':{key:{'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha(path)} for key,path in paths.items()}}
for name,digest in source.items():assert sha(ROOT/name)==digest
path=OUT/'release-summary.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps({'pass':True,'gpu_checks':44,'cpu_passed':99,'clean_packages':14,'feature_ptx_identical':26,'summary_sha256':sha(path)},indent=2))