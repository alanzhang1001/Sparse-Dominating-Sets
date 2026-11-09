"""Offline integrity, cohort, source identity and displayed-value checks."""
from pathlib import Path
import hashlib, json, math, re, statistics
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=load('release_manifest.json')
    for name,expected in manifest['immutable_sha256'].items():
        p=ROOT/name
        assert p.is_file() and sha(p)==expected, 'Immutable file changed: '+name
    for name,expected in manifest['reference_table_sha256'].items():
        assert sha(ROOT/name)==expected, 'Displayed table differs: '+name
    for name,expected in load('supplementary/data/manifest.json')['snapshot_sha256'].items():
        assert sha(ROOT/'supplementary'/name)==expected,name
    reference=ROOT/'baselines/sparsedomsets/algorithms/c_dominating.cpp'
    assert sha(reference)=='0890116c13b4a006e76f17f63580eaff1a6c55ce966a46ce9f8bb8b45ccc6a0a'
    d=load('paper/experiments/data/v1/evidence.json')
    assert len(d['e1'])==160 and sum(x['opt'] is not None for x in d['e1'])==150
    assert all(not x['legacy_age'] for x in d['cc2fs'])
    retrieval=load('supplementary/data/retrieval.json')
    rows=retrieval['rows'] if isinstance(retrieval,dict) else retrieval
    assert len(rows)==144
    summaries=[]
    for batch in ['e5-subgraph-formal-20261108-v1','e5-subgraph-extension-20261109-v1']:
        summaries+=load('supplementary/data/'+batch+'/summary.json')
    paper=load('paper/experiments/data/subgraph_20261109/table_source.json')['rows']
    for row in paper:
        q=[x for x in summaries if x['graph']==row['graph'] and x['mode']=='file' and x['kind']=='partition' and x['group']=='primary']
        assert len(q)==9
        for method,v in row['mean_ms'].items():
            rr=[x['total_ms'] for x in q if x['method'].startswith('random-prune-')] if method=='random-prune-mean' else [x['total_ms'] for x in q if x['method']==method]
            assert len(rr)==(5 if method=='random-prune-mean' else 1)
            assert math.isclose(statistics.mean(rr),v,rel_tol=1e-12)
    py_count=0;json_count=0
    for name in manifest['immutable_sha256']:
        p=ROOT/name
        rel=p.relative_to(ROOT)
        if p.is_symlink():raise AssertionError('No external symlinks allowed: '+str(rel))
        assert p.stat().st_size<100*1024**2,'Large Git file: '+str(rel)
        if p.suffix=='.py':compile(p.read_text(),str(rel),'exec');py_count+=1
        if p.suffix=='.json':json.loads(p.read_text());json_count+=1
        if p.suffix in ['.py','.cpp','.hpp','.json','.tex','.md','.txt','.csv','.java']:
            source=p.read_text()
            for prefix in ['/'+'Users/','/'+'ai/data/','/'+'root/']:
                assert prefix not in source, 'Private absolute path: '+str(rel)
            assert not re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',source),'Private key: '+str(rel)
    out={'state':'passed','immutable_files':len(manifest['immutable_sha256']),'paper_tables_match_frozen_reference':len(manifest['reference_table_sha256']),'main_exact_instances':160,'certified_optima':150,'retrieval_configurations':144,'exact_query_graphs_cross_checked':len(paper),'python_sources_parse':py_count,'json_files_parse':json_count}
    (ROOT/'build').mkdir(exist_ok=True)
    (ROOT/'build/integrity_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
