"""Offline native correctness demo with independent set-based answers."""
from pathlib import Path
from fractions import Fraction
from collections import deque
import argparse, json, platform, random, struct, subprocess
from build_native import build, ROOT

def balls(adj,r):
    out=[]
    for s in range(len(adj)):
        seen={s};front={s}
        for _ in range(r):front=set().union(*(adj[v] for v in front))-seen if front else set();seen|=front
        out.append(seen)
    return out
def oracle(nb,strategy):
    remaining=set(range(len(nb)));chosen=[]
    while remaining:
        def key(v):
            gain=len(nb[v]&remaining);m={0:0,1:gain,2:Fraction(gain,len(nb[v]))}
            return m[strategy&7],m[(strategy>>3)&7],-v
        v=max(range(len(nb)),key=key);chosen.append(v);remaining-=nb[v]
    return chosen
def write_gxt(adj,p):
    p.write_text(str(len(adj))+'\n'+''.join(f'{u} {v}\n' for u in range(len(adj)) for v in sorted(adj[u])))
def execute(args):
    p=subprocess.run(list(map(str,args)),capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr)
    return json.loads(p.stdout)
def write_values(path,fmt,values):path.write_bytes(struct.pack('<'+str(len(values))+fmt,*values))
def layout(adj,selected,path):
    path.mkdir(parents=True,exist_ok=True);n=len(adj)
    owner=[-1]*n;q=deque()
    for i,v in enumerate(sorted(selected)):owner[v]=i;q.append(v)
    while q:
        v=q.popleft()
        for u in sorted(adj[v]):
            if owner[u]<0:owner[u]=owner[v];q.append(u)
    assert min(owner)>=0
    members=sorted(range(n),key=lambda v:(owner[v],v));offsets=[0]
    for p in range(len(selected)):offsets.append(offsets[-1]+owner.count(p))
    pos=[0]*n;neighbors=[]
    for v in members:pos[v]=len(neighbors);neighbors+=sorted(adj[v])
    bp=[0];bi=[]
    for v in range(n):bi+=sorted({owner[u] for u in adj[v]}-{owner[v]});bp.append(len(bi))
    for name,fmt,values in [('pos','q',pos),('adj','i',neighbors),('owner','i',owner),('members','i',members),('offsets','q',offsets),('boundary_indptr','q',bp),('boundary_indices','i',bi)]:write_values(path/(name+'.bin'),fmt,values)
def main():
    p=argparse.ArgumentParser();p.add_argument('--with-reference',action='store_true');a=p.parse_args()
    binaries=build(a.with_reference);out=ROOT/'build/smoke';out.mkdir(parents=True,exist_ok=True)
    toy=json.loads((ROOT/'data/toy/graph.json').read_text())
    cases=[('toy',toy['vertices'],toy['edges']),('singleton',1,[]),('isolates',6,[]),('path',8,[(v,v+1) for v in range(7)]),('cycle',8,[(v,(v+1)%8) for v in range(8)]),('star',9,[(0,v) for v in range(1,9)]),('clique',7,[(u,v) for u in range(7) for v in range(u+1,7)])]
    rng=random.Random(41)
    cases.append(('seeded',12,[(u,v) for u in range(12) for v in range(u+1,12) if rng.random()<.25]))
    count=0;cc_count=0;query_count=0
    for name,n,edges in cases:
        adj=[set() for _ in range(n)]
        for u,v in edges:adj[u].add(v);adj[v].add(u)
        d=out/name;d.mkdir(exist_ok=True);g=d/'graph.gxt';write_gxt(adj,g)
        csr=d/'csr';csr.mkdir(exist_ok=True);ptr=[0]
        for row in adj:ptr.append(ptr[-1]+len(row))
        write_values(csr/'indptr.u64','Q',ptr);write_values(csr/'indices.u32','I',[u for row in adj for u in sorted(row)])
        chosen_by_strategy={}
        for r in [1,2,3]:
            nb=balls(adj,r)
            for strategy in [1,2,10,17]:
                expected=oracle(nb,strategy);target=d/f'r{r}-s{strategy}.vertices'
                receipt=execute([binaries/'counter_heap',g,r,strategy,target,100000])
                got=list(map(int,target.read_text().split()));assert got==expected,(name,r,strategy,got,expected)
                assert receipt['total_cost']==sum(len(nb[v]) for v in got)
                csr_target=d/'csr.vertices';execute([binaries/'counter_csr',csr,r,strategy,csr_target,100000]);assert list(map(int,csr_target.read_text().split()))==expected
                assert set().union(*(nb[v] for v in got))==set(range(n))
                if a.with_reference:
                    ref=d/'reference.vertices';execute([binaries/'reference',g,r,strategy,ref])
                    assert list(map(int,ref.read_text().split()))==expected
                    execute([binaries/'reference_csr',csr,r,strategy,ref]);assert list(map(int,ref.read_text().split()))==expected
                if r==1:chosen_by_strategy[strategy]=got
                count+=1
        cc=d/'cc.vertices';execute([binaries/'cc2fs',g,1,cc,'--max-iterations',30])
        vertices=list(map(int,cc.read_text().split()));nb=balls(adj,1)
        execute([binaries/'cc2fs_csr',csr,1,d/'cc-csr.vertices','--max-iterations',30]);assert (d/'cc-csr.vertices').read_text()==cc.read_text()
        assert len(vertices)==len(set(vertices));assert set().union(*(nb[v] for v in vertices))==set(range(n));cc_count+=1
        ptr=[0]
        for row in adj:ptr.append(ptr[-1]+len(row))
        write_values(d/'ptr.bin','q',ptr)
        queries=[]
        for h in [1,2]:
            for seeds in [[0],[n-1],sorted({0,n-1})]:
                reached=set(seeds);front=set(seeds)
                for _ in range(h):front=set().union(*(adj[v] for v in front))-reached if front else set();reached|=front
                vs=sorted(reached);es=sorted((v<<32)|u for v in vs for u in adj[v] if u>v and u in reached)
                queries.append((h,seeds,vs,es))
        payload=struct.pack('<i',len(queries))
        for i,(h,seeds,vs,es) in enumerate(queries):
            payload+=struct.pack('<iii',i,h,len(seeds))+struct.pack('<'+str(len(seeds))+'i',*seeds)+struct.pack('<qq',len(vs),len(es))+struct.pack('<'+str(len(vs))+'i',*vs)+struct.pack('<'+str(len(es))+'Q',*es)
        (d/'queries.bin').write_bytes(payload)
        layout(adj,list(range(n)),d/'original')
        for strategy,selected in chosen_by_strategy.items():
            lay=d/f'layout-{strategy}';layout(adj,selected,lay)
            for mode in ['memory','file']:
                for approach,lp in [('partition',lay),('bfs',lay),('bfs',d/'original')]:
                    dest=d/f'q-{strategy}-{mode}-{approach}-{lp.name}.jsonl'
                    p=subprocess.run(list(map(str,[binaries/'subgraph_engine',d,lp,mode,approach,0,dest,42])),capture_output=True,text=True)
                    if p.returncode:raise RuntimeError(p.stderr)
                    rows=[json.loads(x) for x in dest.read_text().splitlines()]
                    assert len(rows)==len(queries) and all(x['correct'] for x in rows)
                    query_count+=len(rows)
    summary={'state':'passed','graphs':len(cases),'greedy_sequence_checks':count,'reference_sequence_checks':count if a.with_reference else 0,'csr_sequence_checks':count,'csr_reference_sequence_checks':count if a.with_reference else 0,'cc2fs_coverage_checks':cc_count,'exact_query_checks':query_count,'query_modes':['memory','file'],'controls':['partition','same-layout BFS','original-layout BFS'],'platform':platform.system(),'workload':'small-graph correctness examples'}
    (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
