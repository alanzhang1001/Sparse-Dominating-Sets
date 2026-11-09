"""Portable input generation extracted from the original experiment sources."""
from pathlib import Path
import hashlib, json, struct, time
import numpy as np
from scipy import sparse

def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2)+"\n")

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def raw(p,a,dtype):np.asarray(a,dtype=dtype).tofile(p)

def layout(a,idx,d):
 t=time.perf_counter();d.mkdir(parents=True,exist_ok=True);n=a.shape[0];degrees=np.diff(a.indptr)
 order=idx['members'] if idx is not None else np.arange(n,dtype=np.int32)
 assert np.array_equal(np.sort(order),np.arange(n))
 offsets=np.r_[0,np.cumsum(degrees[order],dtype=np.int64)];pos=np.empty(n,np.int64);pos[order]=offsets[:-1]
 # CSR row permutation preserves each original adjacency list and original IDs.
 reordered=a[order].tocsr();raw(d/'adj.bin',reordered.indices,'<i4');raw(d/'pos.bin',pos,'<i8')
 for v in np.random.default_rng(19).choice(n,min(32,n),replace=False):
  assert np.array_equal(reordered.indices[pos[v]:pos[v]+degrees[v]],a.indices[a.indptr[v]:a.indptr[v+1]])
 if idx is not None:
  for k in ['owner','members','offsets','boundary_indptr','boundary_indices']:raw(d/(k+'.bin'),idx[k],'<i8' if k in ['offsets','boundary_indptr'] else '<i4')
 return dict(seconds=time.perf_counter()-t,additional_index_bytes=n*8,adj_bytes=(d/'adj.bin').stat().st_size,hashes={x.name:sha(x) for x in d.glob('*.bin')})

def expected(a,q):
 seen=np.zeros(a.shape[0],bool);front=np.unique(q['query_nodes']);seen[front]=True
 for _ in range(q['h']):
  adj=[a.indices[a.indptr[v]:a.indptr[v+1]] for v in front]
  nxt=np.unique(np.concatenate(adj)) if adj else np.empty(0,np.int32)
  front=nxt[~seen[nxt]];seen[front]=True
 vertices=np.flatnonzero(seen).astype('<i4');edges=sparse.triu(a[vertices][:,vertices],k=1).tocoo()
 packed=(vertices[edges.row].astype(np.uint64)<<np.uint64(32))|vertices[edges.col].astype(np.uint64)
 return vertices,np.sort(packed).astype('<u8')

def queries_file(a,queries,d):
 records=[]
 with (d/'queries.bin').open('wb') as f:
  f.write(struct.pack('<i',len(queries)))
  for i,q in enumerate(queries):
   v,e=expected(a,q);seeds=np.asarray(q['query_nodes'],dtype='<i4');f.write(struct.pack('<iii',i,q['h'],len(seeds)));f.write(seeds.tobytes());f.write(struct.pack('<qq',len(v),len(e)));f.write(v.tobytes());f.write(e.tobytes())
   records.append(dict(q,id=i,target_vertices=len(v),target_edges=len(e),vertices_sha256=hashlib.sha256(v.tobytes()).hexdigest(),edges_sha256=hashlib.sha256(e.tobytes()).hexdigest()))
 save(d/'queries.json',records);return records

def prefix(a,start,size,rng):
 visited={int(start)};order=[int(start)];head=0
 while len(order)<size and head<len(order):
  v=order[head];head+=1;neighbors=a.indices[a.indptr[v]:a.indptr[v+1]].copy();rng.shuffle(neighbors)
  for u in neighbors:
   u=int(u)
   if u not in visited:
    visited.add(u);order.append(u)
    if len(order)==size:break
 assert len(order)==size
 return order

def sample(degree,available,count,workload,rng):
 if workload=='uniform_start':return rng.choice(available,count,replace=False)
 ordered=available[np.argsort(degree[available],kind='stable')]
 result=np.concatenate([rng.choice(s,count//5,replace=False) for s in np.array_split(ordered,5)])
 rng.shuffle(result);return result

def _degree_stratified_sample(degrees: np.ndarray, available: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    ordered = available[np.argsort(degrees[available], kind="stable")]
    strata = np.array_split(ordered, 5)
    picks = []
    base, extra = divmod(count, len(strata))
    for i, stratum in enumerate(strata):
        take = min(len(stratum), base + (1 if i < extra else 0))
        if take:
            picks.extend(map(int, rng.choice(stratum, size=take, replace=False)))
    if len(picks) < count:
        remaining = np.setdiff1d(available, np.asarray(picks, dtype=np.int64), assume_unique=False)
        picks.extend(map(int, rng.choice(remaining, size=count - len(picks), replace=False)))
    result = np.asarray(picks, dtype=np.int64)
    rng.shuffle(result)
    return result
