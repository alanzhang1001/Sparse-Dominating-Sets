"""Portable experiment runners over locally generated inputs."""
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
import json,hashlib,re,subprocess,sys,os,platform,time,struct,shutil,statistics
import numpy as np
from scipy import sparse
from congestionlab.graphs import _write_gxt,closed_radius_matrix,sample_bfs_graph
from congestionlab.solvers import solve_dom_mac
from congestionlab.query_benchmark import NativeQueries

ROOT=Path(__file__).resolve().parents[2]
STRATEGIES={'ratio':2,'degree':1,'ratio-plus':10,'degree-plus':17}
MAIN=['soc-Epinions1','web-Stanford','com-Amazon','web-Google','com-Youtube','uk-2002']
QUERY=['web-Stanford','web-Google','in-2004','eu-2005']
def load(p):return json.loads(Path(p).read_text())
def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def output(args):
 p=args.output.resolve()
 for x in [args.workspace.resolve(),ROOT/'paper',ROOT/'supplementary']:
  if p==x or x in p.parents:raise ValueError('Output overlaps immutable inputs')
 p.mkdir(parents=True,exist_ok=False)
 cpu=args.cpu if hasattr(args,'cpu') else None
 if hasattr(os,'sched_getaffinity'):
  allowed=os.sched_getaffinity(0);cpu=min(allowed) if cpu is None else cpu
  if cpu not in allowed:raise ValueError('CPU is outside the available affinity set')
  os.sched_setaffinity(0,{cpu})
 save(p/'run_protocol.json',{'command':args.command,'platform':platform.system(),'numpy':np.__version__,'cpu':cpu,'threads':1,'input_snapshot':'user-generated inputs','query_limit':getattr(args,'query_limit',None),'repeats':getattr(args,'repeats',None)})
 return p
def binaries(reference=False):
 d=ROOT/'build/native'
 if not (d/('reference_select' if reference else 'counter_heap')).exists():
  cmd=[sys.executable,str(ROOT/'scripts/build_native.py')]
  if reference:cmd+=['--with-reference' if platform.system()=='Linux' else '--serial-reference']
  subprocess.run(cmd,check=True)
 return d
@lru_cache(maxsize=4)
def npz_index(workspace):
 index=defaultdict(list)
 for p in Path(workspace).rglob('*.npz'):index[p.name].append(p)
 return index
def graph_path(workspace,name):
 names=[name+'.lcc.npz',name+'.npz']
 m=re.fullmatch(r'(.+)-bfs(\d+)-s(\d+)',name)
 if m:names.insert(0,f'{m[1]}.bfs{m[2]}.s{m[3]}.npz')
 idx=npz_index(str(workspace))
 for n in names:
  if idx[n]:return sorted(idx[n],key=lambda p:len(str(p)))[0]
 raise FileNotFoundError('Missing graph '+name+'; run prepare_data.py, prepare_webgraph.py or reproduce.py generate first')
def csr_directory(workspace,name):
 paths=sorted(Path(workspace).rglob(name+'/indptr.u64'))
 if not paths:raise FileNotFoundError('Run prepare_webgraph.py for '+name)
 return paths[0].parent

def sample_input(workspace,name,n,seed,out):
 try:parent=graph_path(workspace,name)
 except FileNotFoundError:
  from .csr_sampling import sample_csr
  return sample_csr(csr_directory(workspace,name),n,seed,out)
 meta=sample_bfs_graph(parent,n,seed,out)
 return out/meta['graph_file']

def adjacency(workspace,name,out):
 if name=='toy':
  d=load(ROOT/'data/toy/graph.json');rows=[];cols=[]
  for u,v in d['edges']:rows.extend([u,v]);cols.extend([v,u])
  a=sparse.csr_matrix((np.ones(len(rows),np.uint8),(rows,cols)),shape=(d['vertices'],d['vertices']))
  p=out/'inputs/toy.npz';p.parent.mkdir(exist_ok=True);sparse.save_npz(p,a)
 else:
  try:p=graph_path(workspace,name)
  except FileNotFoundError:
   m=re.fullmatch(r'(.+)-bfs(\d+)-s(\d+)',name)
   if not m:raise
   p=sample_input(workspace,m[1],int(m[2]),int(m[3]),out/'inputs/generated')
  a=sparse.load_npz(p).tocsr()
 a.sort_indices()
 return p,a
def gxt(a,out,name):
 p=out/'inputs'/(name+'.gxt');p.parent.mkdir(exist_ok=True)
 if not p.exists():_write_gxt(a,p)
 return p
def measured(cmd,folder,timeout=86400):
 folder.mkdir(parents=True,exist_ok=False);d=binaries();start=time.perf_counter()
 if platform.system()=='Linux' and (d/'measure').exists():
  run=subprocess.run([str(d/'measure'),str(timeout),str(folder/'stdout.txt'),str(folder/'stderr.txt'),*map(str,cmd)],capture_output=True,text=True,timeout=timeout+10)
  if run.returncode:raise RuntimeError(run.stderr)
  metrics=json.loads(run.stdout)
  if metrics['returncode']:raise RuntimeError((folder/'stderr.txt').read_text())
  text=(folder/'stdout.txt').read_text()
 else:
  run=subprocess.run(list(map(str,cmd)),capture_output=True,text=True,timeout=timeout)
  (folder/'stdout.txt').write_text(run.stdout);(folder/'stderr.txt').write_text(run.stderr)
  if run.returncode:raise RuntimeError(run.stderr)
  metrics={'process_wall_seconds':time.perf_counter()-start,'peak_rss_kib':None,'returncode':0,'timed_out':False};text=run.stdout
 metrics['native']=json.loads(text.strip().splitlines()[-1]);save(folder/'measurement.json',metrics);return metrics

def exact(args):
 out=output(args);rows=load(ROOT/'configs/instances.json')[args.suite]
 if args.limit:rows=rows[:args.limit]
 result=[];bins=binaries()
 for i,row in enumerate(rows):
  name=row['dataset'];path,a=adjacency(args.workspace,name,out);graph=gxt(a,out,name);r=row['radius'];tag=f'{i:03}-{name}-r{r}'
  target=out/(tag+'.vertices');metrics=measured([bins/'counter_heap',graph,r,2,target,args.max_entries],out/tag,args.timeout)
  coverage=closed_radius_matrix(a,r);costs=np.asarray(coverage.sum(axis=1)).ravel().astype(float)
  lp=solve_dom_mac(coverage,costs,'lp',args.lp_seconds)
  ip=solve_dom_mac(coverage,costs,'ilp',args.ip_seconds) if args.suite=='main' or (a.shape[0]<=256 and row['family']!='client_hub') else None
  cost=metrics['native']['total_cost']
  rr={'dataset':name,'radius':r,'graph_sha256':sha(path),'algorithm_cost':cost,'lp':lp,'ip':ip}
  result.append(rr);save(out/'results.json',result);print('completed',tag,flush=True)

def certificates(args):
 from .generators import client_hub_graph
 out=output(args);rows=[{'max_degree_parameter':d} for d in [3,8,15,24,48,80,120,255,511,1023]];result=[];bins=binaries()
 for row in rows:
  degree=row['max_degree_parameter'];a,certificate=client_hub_graph(degree);graph=gxt(a,out,'delta'+str(degree));solution=out/f'delta{degree}.vertices'
  m=measured([bins/'counter_heap',graph,1,2,solution,args.max_entries],out/f'delta{degree}',args.timeout)
  assert m['native']['total_cost']/a.shape[0]==certificate['dual_average_lower_bound']
  result.append(dict(degree=degree,certificate=certificate,primal_cost=m['native']['total_cost'],primal_dual_match=True))
 save(out/'certificates.json',result);print('Verified',len(result),'construction certificates')

def generate(args):
 from .generators import ensure_synthetic
 out=output(args);rows=load(ROOT/'configs/instances.json')[args.suite];result=[];seen=set()
 for row in rows:
  name=row['dataset']
  if name in seen:continue
  seen.add(name)
  if args.limit and len(result)>=args.limit:break
  if name.startswith('er-'):m=re.fullmatch(r'er-n(\d+)-s(\d+)',name);p=ensure_synthetic(out,'er',int(m[1]),int(m[2]))
  elif name.startswith('rr-'):m=re.fullmatch(r'rr-d(\d+)-n(\d+)-s(\d+)',name);p=ensure_synthetic(out,'rr',int(m[2]),int(m[3]),int(m[1]))
  elif name.startswith('pa-'):m=re.fullmatch(r'pa-m(\d+)-n(\d+)-s(\d+)',name);p=ensure_synthetic(out,'pa',int(m[2]),int(m[3]),int(m[1]))
  elif name.startswith('near-regular-'):m=re.fullmatch(r'near-regular-d(\d+)-n(\d+)-s(\d+)',name);p=ensure_synthetic(out,'nr',int(m[2]),int(m[3]),int(m[1]))
  elif name.startswith('client-hub-'):p=ensure_synthetic(out,'ch',0,0,int(name.split('Delta')[-1]))
  else:
   m=re.fullmatch(r'(.+)-bfs(\d+)-s(\d+)',name);p=sample_input(args.workspace,m[1],int(m[2]),int(m[3]),out/'graphs')
  a=sparse.load_npz(p)
  result.append({'dataset':name,'vertices':a.shape[0],'edges':a.nnz//2,'generated_npz_sha256':sha(p)})
 save(out/'generation.json',result);print('Generated',len(result),'graph inputs')

def greedy(args):
 out=output(args);reference=args.command in ['implementation','stability'];bins=binaries(reference);results=[]
 names=args.graphs.split(',');methods=args.methods.split(',')
 for name in names:
  csr=name=='uk-2002'
  if csr:
   if args.command=='stability' or args.radii!=[1]:raise ValueError('The full uk-2002 entry supports r=1 quality/implementation only')
   graph=csr_directory(args.workspace,name);path=graph/'indptr.u64';ptr=np.memmap(path,dtype='<u8',mode='r');n=len(ptr)-1;a=None
   if not (bins/'counter_csr').exists():raise RuntimeError('Rebuild native binaries to enable CSR input')
  else:path,a=adjacency(args.workspace,name,out);graph=gxt(a,out,name);n=a.shape[0]
  for radius in args.radii:
   for method in methods:
    if method not in STRATEGIES:raise ValueError(method)
    for rep in range(args.repeats):
     tag=f'{name}-r{radius}-{method}-{rep}';target=out/(tag+'.vertices')
     if args.command=='stability':cmd=[bins/'reference_select',graph,radius,STRATEGIES[method]|64,rep,target]
     else:cmd=[bins/('counter_csr' if csr else 'counter_heap'),graph,radius,STRATEGIES[method],target,args.max_entries]
     m=measured(cmd,out/tag,args.timeout);selected=np.loadtxt(target,dtype=int,ndmin=1)
     if radius==1 and not csr:
      covered=np.zeros(a.shape[0],bool)
      for v in selected:covered[v]=True;covered[a.indices[a.indptr[v]:a.indptr[v+1]]]=True
      assert covered.all()
     row={'graph':name,'radius':radius,'method':method,'repeat':rep,'landmarks':len(selected),'total_cost':m['native']['total_cost'],'average_congestion':m['native']['total_cost']/n,'new_measurement':m,'graph_sha256':sha(path),'graph_format':'CSR' if csr else 'NPZ'}
     if args.command=='implementation':
      old=out/(tag+'.reference.vertices');om=measured([bins/('reference_csr' if csr else 'reference'),graph,radius,STRATEGIES[method],old],out/(tag+'-reference'),args.timeout)
      assert np.array_equal(selected,np.loadtxt(old,dtype=int,ndmin=1)),tag
      row.update(reference_measurement=om,ordered_sequences_equal=True)
     results.append(row);save(out/'results.json',results)
  if args.command=='quality':
   for budget in args.budgets:
    for rep in range(args.cc_repeats):
     tag=f'{name}-cc2fs-{budget}-{rep}';target=out/(tag+'.vertices');m=measured([bins/('cc2fs_csr' if csr else 'cc2fs'),graph,budget,target],out/tag,args.timeout)
     results.append({'graph':name,'radius':1,'method':'cc2fs','budget_seconds':budget,'repeat':rep,'measurement':m});save(out/'results.json',results)
 if args.command=='stability':
  summary=[]
  for name in names:
   for method in methods:
    vv=[x['average_congestion'] for x in results if x['graph']==name and x['method']==method]
    summary.append({'graph':name,'method':method,'seeds':len(vv),'min':min(vv),'max':max(vv),'mean':statistics.mean(vv),'range_percent':100*(max(vv)-min(vv))/statistics.mean(vv)})
  save(out/'summary.json',summary)
 print('Completed',len(results),'native runs')

def radius_cohort(args):
 from argparse import Namespace
 out=output(args);key='paper_radius_quality' if args.cohort=='paper' else 'report_radius_implementation'
 rows=load(ROOT/'data/rerun_cohorts.json')[key]
 if args.limit:rows=rows[:args.limit]
 save(out/'conditions.json',rows)
 for i,row in enumerate(rows):
  one=Namespace(**vars(args));one.command='radii' if args.cohort=='paper' else 'implementation';one.graphs=row['graph'];one.radii=[row['radius']];one.output=out/f'condition-{i:02}'
  one.methods=args.methods or (','.join(STRATEGIES) if args.cohort=='paper' else 'ratio')
  greedy(one)

def register(sub):
 def parser(name,fn):
  p=sub.add_parser(name);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu',type=int);p.add_argument('--timeout',type=float,default=86400);p.add_argument('--max-entries',type=int,default=2_000_000_000);p.set_defaults(function=fn);return p
 for name,fn in [('exact',exact),('generate',generate)]:
  p=parser(name,fn);p.add_argument('--suite',choices=['main','theory'],default='main');p.add_argument('--limit',type=int,default=0)
  if name=='exact':p.add_argument('--lp-seconds',type=float,default=300);p.add_argument('--ip-seconds',type=float,default=600)
 parser('certificates',certificates)
 p=parser('radius-cohort',radius_cohort);p.add_argument('--cohort',choices=['paper','report'],default='paper');p.add_argument('--repeats',type=int,default=3);p.add_argument('--methods');p.add_argument('--limit',type=int,default=0)
 for name in ['quality','implementation','stability','radii']:
  p=parser(name,greedy);p.add_argument('--graphs',default=','.join(MAIN[:5]));p.add_argument('--methods',default=','.join(STRATEGIES));p.add_argument('--radii',type=int,nargs='+',default=[1]);p.add_argument('--repeats',type=int,default=10 if name=='stability' else 5)
  if name=='quality':p.add_argument('--budgets',type=float,nargs='+',default=[1,10,60,300]);p.add_argument('--cc-repeats',type=int,default=3)
