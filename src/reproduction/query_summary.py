"""Paired aggregation and bootstrap for generated per-graph runs.

Uses paired stratified resampling with a fixed bootstrap seed.
"""
from pathlib import Path
from collections import defaultdict
import json, hashlib
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())

def aggregate(run_directory, output, bootstrap=True):
 R=Path(run_directory)
 manifest=load(R/'source_manifest.json')
 graphs=[manifest['graph_name']]
 g=graphs[0]
 runs=load(R/'runs.json')
 matrices=defaultdict(list);identities={};answers={};physical_batches=0;physical_bytes=0;records=0
 metrics=['total_ms','requested_bytes','read_calls','candidates']
 count=manifest['exact_queries']
 for r in runs:
  path=R/r['record']
  assert sha(path)==r['sha256']
  key=(g,r['mode'],r['layout'],r['kind']);identities.setdefault(key,set()).add(r['repeat'])
  rows=[json.loads(l) for l in path.open()];rows.sort(key=lambda x:x['query'])
  assert len(rows)==count and [x['query'] for x in rows]==list(range(count))
  for x in rows:
   assert x['correct'];k=(g,x['query']);answers.setdefault(k,x['answer_hash64']);assert answers[k]==x['answer_hash64']
   if r['mode']=='file':physical_batches+=x['physical_read_bytes_batch']>0;physical_bytes+=x['physical_read_bytes_batch']
  matrices[key].append(np.asarray([[x[m] for m in metrics] for x in rows]));records+=len(rows)
 repeats={r['repeat'] for r in runs}
 assert all(v==repeats for v in identities.values())
 assert len(matrices)==19*len({r['mode'] for r in runs})
 matrices={k:np.median(v,axis=0) for k,v in matrices.items()}
 summaries=[];comparisons=[];rng=np.random.default_rng(202611082232)
 for g in graphs:
  qs=load(R/'queries.json');groups=defaultdict(list)
  for q in qs:groups[(q['h'],q['query_size'],q['workload'])].append(q['id'])
  assert len(groups)==10
  if not manifest['quick']:assert sorted(map(len,groups.values()))==[100]*8+[1000]*2
  primary=[groups[(1,1,w)] for w in ['uniform_start','degree_stratified_start']]
  for (graph,mode,method,kind),a in matrices.items():
   if graph!=g:continue
   for label,ids in [('primary',primary[0]+primary[1])]+[(str(k),v) for k,v in groups.items()]:
    summaries.append(dict(graph=g,mode=mode,method=method,kind=kind,group=label,queries=len(ids),**{m:float(a[ids,i].mean()) for i,m in enumerate(metrics)},p50_ms=float(np.median(a[ids,0])),p95_ms=float(np.percentile(a[ids,0],95))))
  for mode in (sorted({r['mode'] for r in runs}) if bootstrap else []):
   ours=matrices[(g,mode,'ratio','partition')]
   for ref in ['degree','ratio-plus','cc2fs','random-mean','direct-bfs','same-layout-bfs']:
    random=None
    if ref=='random-mean':
     names=sorted(k[2] for k in matrices if k[0]==g and k[1]==mode and k[2].startswith('random-') and k[3]=='partition');assert len(names)==5
     random=np.asarray([matrices[(g,mode,n,'partition')] for n in names]);other=random.mean(axis=0)
    elif ref=='direct-bfs':other=matrices[(g,mode,'original','bfs')]
    elif ref=='same-layout-bfs':other=matrices[(g,mode,'ratio','bfs')]
    else:other=matrices[(g,mode,ref,'partition')]
    for label,strata in [('primary',primary),('uniform_start',[primary[0]]),('degree_stratified_start',[primary[1]])]:
     ids=np.concatenate(strata);den=other[ids].mean(axis=0);point=np.divide(ours[ids].mean(axis=0),den,out=np.full(4,np.nan),where=den>0);boot=[]
     for _ in range(2000):
      selected=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
      # Only selected rows are materialized for the resampled random seed mean.
      b=random[rng.integers(0,5,5)][:,selected].mean(axis=(0,1)) if random is not None else other[selected].mean(axis=0)
      boot.append(np.divide(ours[selected].mean(axis=0),b,out=np.full(4,np.nan),where=b>0))
     ci=np.percentile(boot,[2.5,97.5],axis=0)
     def finite(x):return float(x) if np.isfinite(x) else None
     comparisons.append(dict(graph=g,mode=mode,reference=ref,group=label,ratio_over_reference={m:dict(point=finite(point[i]),ci95=[finite(ci[0,i]),finite(ci[1,i])]) for i,m in enumerate(metrics)}))

 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
 if bootstrap:(output/'uncertainty.json').write_text(json.dumps(comparisons,indent=2)+'\n')
 return {'records':records,'groups':len(summaries),'comparisons':len(comparisons),'all_hashes_and_answers':True,'physical_read_bytes':physical_bytes}
