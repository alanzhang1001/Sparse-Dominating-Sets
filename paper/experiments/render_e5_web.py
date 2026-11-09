"""Four-graph retrieval displays from the audited 2026-10-31 evidence."""
from pathlib import Path
import json,statistics as st
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'experiments/data/e5_web_20261031'
GRAPHS=['web-Stanford','web-Google','in-2004','eu-2005'];NAMES=dict(zip(GRAPHS,['Stanford','Google','in-2004','eu-2005']))
METHODS=['ratio','ratio-plus','degree','cc2fs','random'];LABELS=['Ratio','Ratio+', 'Degree',r'CC$^2$FS','Random-P']
def groups():
 rows=json.loads((DATA/'analysis.json').read_text())['rows'];result={}
 for g in GRAPHS:
  for control,levels in [('matched-budget',[1,2,3]),('matched-count',[0])]:
   for k in levels:
    for m in METHODS:
     rr=[r for r in rows if r['dataset_id']==g and r['control']==control and r['budget_index']==k and (r['selection_method'].startswith('random-prune-') if m=='random' else r['selection_method']==m)]
     assert len(rr)==(5 if m=='random' else 1)
     result[g,control,k,m]=dict(mean=st.mean(r['mean_candidates'] for r in rr),sd=st.stdev(r['mean_candidates'] for r in rr) if len(rr)>1 else 0,count=st.mean(r['landmark_count'] for r in rr),bytes=st.mean(r['index_payload_bytes'] for r in rr),congestion=st.mean(r['average_congestion'] for r in rr),p99=st.mean(r['partition_metrics']['p99'] for r in rr),maximum=st.mean(r['partition_metrics']['max'] for r in rr))
 return rows,result

def table(path,spec,head,body):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text('\n'.join([r'\begin{tabular}{@{}'+spec+'@{}}',r'\toprule',head+r' \\',r'\midrule']+[' & '.join(row)+r' \\' for row in body]+[r'\bottomrule',r'\end{tabular}','']))
def render_controls():
 rows,G=groups();body=[]
 for g in GRAPHS:
  a=G[g,'matched-count',0,'ratio'];r=G[g,'matched-count',0,'random']
  body.append([NAMES[g],f"{a['count']:,.0f}",f"{a['mean']/1000:.2f}",f"${r['mean']/1000:.2f}\\pm{r['sd']/1000:.2f}$"])
 table(ROOT/'experiments/data/icde_compact/controls.tex','lrrr',r'Graph & $|D|$ & Ratio & Random-P',body)

def render():
 from plot_main_figures import render_candidates, render_display_tables
 rows,G=groups();render_controls();tables=DATA/'tables';tables.mkdir(exist_ok=True)
 for g in GRAPHS:
  body=[]
  for m,label in zip(METHODS,LABELS):
   rr=[G[g,'matched-budget',k,m] for k in [1,2,3]]
   triple=lambda key,scale,digits:'/'.join(f"{r[key]/scale:.{digits}f}" for r in rr)
   body.append([label,triple('count',1000,1),triple('bytes',2**20,2),triple('congestion',1,2),triple('p99',1,0),triple('maximum',1000,1),triple('mean',1000,2)])
   if m=='random':body.append(['Seed SD','','','','','',triple('sd',1000,2)])
  table(tables/f'budget_{g}.tex','lrrrrrr',r'Selector & $|D|$ (k) & MiB & Cong. & P99 & Max (k) & Cand. (k)',body)
 body=[]
 for g in GRAPHS:
  for m,label in zip(METHODS,LABELS):
   r=G[g,'matched-count',0,m];value=f"{r['mean']/1000:.2f}"+(f" $\\pm$ {r['sd']/1000:.2f}" if m=='random' else '')
   body.append([NAMES[g] if m=='ratio' else '',label,f"{r['bytes']/2**20:.2f}",f"{r['congestion']:.3f}",f"{r['p99']:.0f}",f"{r['maximum']/1000:.1f}",value])
 table(tables/'matched_count.tex','llrrrrr',r'Graph & Selector & MiB & Cong. & P99 & Max (k) & Cand. (k)',body)
 render_candidates(G, ROOT/'figures')
 render_display_tables(grouped=G)
 save={'rows':len(rows),'graphs':GRAPHS,'ratio_reductions':{g:{m:[100*(1-G[g,'matched-budget',k,'ratio']['mean']/G[g,'matched-budget',k,m]['mean']) for k in [1,2,3]] for m in ['degree','cc2fs','random']} for g in GRAPHS},'same_count_random_reduction':{g:100*(1-G[g,'matched-count',0,'ratio']['mean']/G[g,'matched-count',0,'random']['mean']) for g in GRAPHS}}
 (DATA/'display_audit.json').write_text(json.dumps(save,indent=2)+'\n');print(json.dumps(save,indent=2))
if __name__=='__main__':render()
