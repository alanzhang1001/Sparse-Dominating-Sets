"""Regenerate every supplemental display from the frozen, hash-checked evidence."""
from pathlib import Path
import csv
import hashlib
import json
import os
import statistics as st

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parent/'build/supplement/mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import numpy as np

DATA, TABLES, FIGURES = ROOT/'data', ROOT/'tables', ROOT/'figures'
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
def load(name): return json.loads((DATA/name).read_text())
manifest = load('manifest.json')
for name, expected in manifest['snapshot_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name

GRAPHS = ['web-Stanford','web-Google','in-2004','eu-2005']
NAMES = {'web-Stanford':'Stanford','web-Google':'Google','in-2004':'in-2004',
         'eu-2005':'eu-2005','soc-Epinions1':'Epinions','com-Amazon':'Amazon',
         'com-Youtube':'YouTube','uk-2002':'uk-2002'}
METHODS = ['ratio','ratio-plus','degree','cc2fs','random']
MN = {'ratio':'Ratio','ratio-plus':'Ratio+','degree':'Degree','cc2fs':r'CC$^2$FS','random':'Random-P'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.spines.left':False})

def tex_table(name, columns, header, rows):
    text = ['\\begin{tabular}{@{}'+columns+'@{}}',r'\toprule',header+r' \\',r'\midrule']
    for row in rows:
        text.append(row if isinstance(row,str) else ' & '.join(map(str,row))+r' \\')
    text += [r'\bottomrule',r'\end{tabular}']
    (TABLES/(name+'.tex')).write_text('\n'.join(text)+'\n')

def record_csv(name, rows):
    if not rows:return
    with (TABLES/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

phase = load('implementation.json')
fig, axs = plt.subplots(2,3,figsize=(6.5,2.8),layout='constrained')
colors=['#c3cbd2','#7aa7b8','#236784','#d9ba79']
for ax, row in zip(axs.flat,phase):
    norm=row['old']['whole'];left=np.zeros(2)
    for key,color in zip(['read','prepare','select','write'],colors):
        val=np.array([row[i][key]/norm for i in ['old','new']])
        ax.barh([1,0],val,left=left,color=color,height=.5,label=key)
        left+=val
    ax.scatter([1,row['new']['whole']/norm],[1,0],marker='|',color='#111111',s=95,zorder=3)
    ax.set(yticks=[1,0],yticklabels=['Old','New'],xlim=(0,1.08),xticks=[0,.5,1])
    ax.set_title(NAMES[row['graph']]+f"  ({norm/row['new']['whole']:.2f}x)",loc='left')
    ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
handles, labels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,['Input','Prepare','Select','Output'],loc='outside lower center',ncol=4,frameon=False)
fig.savefig(FIGURES/'implementation_phases.pdf');plt.close(fig)
phase_rows=[]
for x in phase:
    phase_rows.append({'graph':x['graph'],**{i+'_'+k:v for i in ['old','new'] for k,v in x[i].items()}})
record_csv('implementation_phases',phase_rows)

radius=load('radius.json')
assert len(radius)==12 and load('radius_audit.json')['state']=='passed'
def rkey(x):
    g=x['condition']['graph']; n=int(g.split('bfs')[1].split('-')[0]) if 'bfs' in g else 255265
    seed=int(g.rsplit('s',1)[1]) if 'bfs' in g else 0
    return n,seed,x['condition']['radius']
rr=[];rcsv=[]
for x in sorted(radius,key=rkey):
    n,seed,r=rkey(x)
    label=f'{n//1000}k / {str(seed)[-2:]}' if seed else 'Full graph'
    speed=x['old_seconds']/x['new_seconds'];mem=x['new_peak_rss_gib']/x['old_peak_rss_gib']
    rr.append([label,r,f"{x['old_seconds']:.1f}",f"{x['new_seconds']:.1f}",f'{speed:.2f}',
               f"{x['old_peak_rss_gib']:.2f}",f"{x['new_peak_rss_gib']:.2f}",f'{100*(1-mem):.1f}'])
    rcsv.append({'graph':x['condition']['graph'],'radius':r,**{k:v for k,v in x.items() if k!='condition'},'speedup':speed,'memory_reduction_pct':100*(1-mem)})
tex_table('radius','lrrrrrrr',r'View / seed & $r$ & Old (s) & New (s) & Speedup & Old GiB & New GiB & Saved (\%)',rr)
record_csv('radius',rcsv)

retr=load('retrieval.json')
def retr_group(g,method,control,budget=0):
    return [x for x in retr if x['dataset_id']==g and x['control']==control and x['budget_index']==budget
            and (x['selection_method'].startswith('random-prune') if method=='random' else x['selection_method']==method)]
audit=[];structure=[];scsv=[]
for g in GRAPHS:
    br=[x for x in retr if x['dataset_id']==g and x['control']=='matched-budget']
    assert len(br)==27 and all(.95<=x['index_payload_bytes']/x['budget_bytes']<=1 for x in br)
    caps=[next(x['budget_bytes'] for x in br if x['budget_index']==i)/2**20 for i in [1,2,3]]
    util=[x['index_payload_bytes']/x['budget_bytes'] for x in br]
    a=retr_group(g,'ratio','matched-count')[0]
    audit.append([NAMES[g],f"{a['n']:,}",'/'.join(f'{x:.2f}' for x in caps),f'{100*min(util):.2f}--{100*max(util):.2f}',a['query_count']])
    structure.append(r'\multicolumn{6}{@{}l}{\textit{'+NAMES[g]+f": {a['landmark_count']:,} pieces; mean piece size {a['partition_metrics']['mean']:.2f}"+r'}} \\')
    for m in METHODS:
        rows=retr_group(g,m,'matched-count');assert len(rows)==(5 if m=='random' else 1)
        assert all(x['landmark_count']==a['landmark_count'] for x in rows)
        vals={'graph':g,'method':m,'landmarks':a['landmark_count'],
              'index_mib':st.mean(x['index_payload_bytes']/2**20 for x in rows),
              'piece_p99':st.mean(x['partition_metrics']['p99'] for x in rows),
              'piece_max':st.mean(x['partition_metrics']['max'] for x in rows),
              'candidates_k':st.mean(x['mean_candidates']/1000 for x in rows),
              'candidate_seed_sd_k':st.stdev(x['mean_candidates']/1000 for x in rows) if m=='random' else 0}
        structure.append([MN[m],f"{vals['index_mib']:.2f}",f"{vals['piece_p99']:.1f}",f"{vals['piece_max']:,.0f}",f"{vals['candidates_k']:.2f}",f"{vals['candidate_seed_sd_k']:.2f}" if m=='random' else '--'])
        scsv.append(vals)
    if g!=GRAPHS[-1]:structure.append(r'\midrule')
tex_table('budget_audit','lrcrr',r'Graph & Vertices & Caps (MiB), low/medium/high & Use (\%) & Queries',audit)
tex_table('fixed_count','lrrrrr',r'Method & Index MiB & Piece P99 & Largest piece & Cand. ($10^3$) & Seed SD',structure)
record_csv('fixed_count',scsv)

groups=[(h,q,w) for h,q in [(1,1),(1,5),(1,20),(2,1),(2,5)] for w in ['uniform_start','degree_stratified_start']]
glabels=[f'{h}/{q}/'+('U' if w=='uniform_start' else 'S') for h,q,w in groups]
cmap=LinearSegmentedColormap.from_list('gains',['#b55757','#fffdfa','#1f7f82'])
def heatmap(filename, array, labels, columns, title, unit='%', bound=None):
    a=np.array(array);v=max(abs(a.min()),abs(a.max()),1) if bound is None else bound
    fig,ax=plt.subplots(figsize=(6.5,2.65 if filename=='query_context.pdf' else 3.3),layout='constrained')
    im=ax.imshow(a,cmap=cmap,norm=TwoSlopeNorm(vmin=-v,vcenter=0,vmax=v),aspect='auto')
    ax.set(xticks=range(len(columns)),xticklabels=columns,yticks=range(len(labels)),yticklabels=labels)
    ax.set_title(title,loc='left',pad=10)
    ax.tick_params(length=0,labelsize=8)
    for (i,j),val in np.ndenumerate(a):
        label='0.0' if abs(val)<.05 else f'{val:.1f}' if abs(val)<1 else f'{val:.0f}'
        ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if abs(val)>.63*v else '#172c36')
    cb=fig.colorbar(im,ax=ax,pad=.025,fraction=.035);cb.set_label('Reduction ('+unit+')')
    fig.savefig(FIGURES/filename);plt.close(fig)
candidate_matrix=[];labels=[];ccsv=[]
for g in GRAPHS:
    ours=retr_group(g,'ratio','matched-budget',1)[0]
    for m in ['degree','random']:
        refs=retr_group(g,m,'matched-budget',1);vals=[]
        for group in groups:
            def metric(row):return next(s['mean_candidates'] for s in row['strata'] if (s['h'],s['query_size'],s['workload'])==group)
            val=100*(1-metric(ours)/st.mean(metric(x) for x in refs));vals.append(val)
            ccsv.append({'graph':g,'reference':m,'h':group[0],'query_size':group[1],'workload':group[2],'reduction_pct':val})
        candidate_matrix.append(vals);labels.append(NAMES[g]+' / '+('Degree' if m=='degree' else 'Random'))
heatmap('candidate_strata.pdf',candidate_matrix,labels,glabels,'Candidate reduction at the lowest matched budget')
record_csv('candidate_strata',ccsv)

formal=[];uncert=[];cohorts=['e5-subgraph-formal-20261108-v1','e5-subgraph-extension-20261109-v1']
for cohort in cohorts:
    audit=load(cohort+'/completion_audit.json');assert audit['state']=='passed' and audit['records']==638400
    assert load(cohort+'/decision.json')['physical_read_bytes']==0
    formal+=load(cohort+'/summary.json');uncert+=load(cohort+'/uncertainty.json')
def qrows(g,m,mode='file',group='primary',kind='partition'):
    return [x for x in formal if x['graph']==g and x['mode']==mode and x['group']==group and x['kind']==kind
            and (x['method'].startswith('random-prune') if m=='random' else x['method']==m)]
def qmean(g,m,metric='total_ms',mode='file',group='primary',kind='partition'):
    a=qrows(g,m,mode,group,kind);assert len(a)==(5 if m=='random' else 1),(g,m,mode,group,kind)
    return st.mean(x[metric] for x in a)
def uncertainty(g,ref,mode='file',group='primary'):
    return next(x['ratio_over_reference'] for x in uncert if (x['graph'],x['reference'],x['mode'],x['group'])==(g,ref,mode,group))
qtable=[];qcsv=[]
for g in GRAPHS:
    for j,m in enumerate(METHODS):
        vals={'graph':g,'method':m,'memory_mean_us':1000*qmean(g,m,mode='memory'),
              'file_mean_us':1000*qmean(g,m),'file_p95_us':1000*qmean(g,m,'p95_ms'),
              'requested_kib':qmean(g,m,'requested_bytes')/1024}
        qtable.append([NAMES[g] if j==0 else '',MN[m]]+[f'{vals[k]:.2f}' for k in ['memory_mean_us','file_mean_us','file_p95_us','requested_kib']]);qcsv.append(vals)
    if g!=GRAPHS[-1]:qtable.append(r'\midrule')
tex_table('query_methods','llrrrr',r'Graph & Method & Memory mean & Warm-file mean & Warm-file P95 & Req. KiB',qtable)
record_csv('query_methods',qcsv)

cit=[];cicsv=[]
for g in GRAPHS:
    for ref,label in [('degree','Degree'),('random-mean','Random-P')]:
        u=uncertainty(g,ref);t=u['total_ms'];b=u['requested_bytes'];assert t['ci95'][1]<1
        cit.append([NAMES[g],label,f"{t['point']:.3f}",f"[{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]",f"{b['point']:.3f}",f"[{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]"])
        cicsv.append({'graph':g,'reference':ref,'time_ratio':t['point'],'time_ci_lo':t['ci95'][0],'time_ci_hi':t['ci95'][1],'bytes_ratio':b['point'],'bytes_ci_lo':b['ci95'][0],'bytes_ci_hi':b['ci95'][1]})
tex_table('query_ci','llrcr c'.replace(' ',''),r'Graph & Reference & Time ratio & 95\% CI & Byte ratio & 95\% CI',cit)
record_csv('query_ci',cicsv)
bfs=[];bcsv=[]
for g in GRAPHS:
    for mode in ['memory','file']:
        a=1000*qmean(g,'ratio',mode=mode);b=1000*qmean(g,'original',mode=mode,kind='bfs');c=1000*qmean(g,'ratio',mode=mode,kind='bfs')
        bfs.append([NAMES[g], 'Memory' if mode=='memory' else 'Warm file',f'{a:.2f}',f'{b:.2f}',f'{c:.2f}',f'{a/c:.2f}'])
        bcsv.append({'graph':g,'mode':mode,'ratio_partition_us':a,'original_bfs_us':b,'same_layout_bfs_us':c,'partition_over_same_layout':a/c})
tex_table('bfs','llrrrr',r'Graph & Mode & Ratio partition & Original BFS & Same-layout BFS & Ratio/same',bfs)
record_csv('bfs',bcsv)

contextgroups=groups[2:];context=[];ctxcsv=[]
for g in GRAPHS:
    for m in ['degree','random']:
        vals=[]
        for group in contextgroups:
            label=str(group);val=100*(1-qmean(g,'ratio',group=label)/qmean(g,m,group=label));vals.append(val)
            ctxcsv.append({'graph':g,'reference':m,'h':group[0],'query_size':group[1],'workload':group[2],'warm_file_time_reduction_pct':val})
        context.append(vals)
heatmap('query_context.pdf',context,labels,glabels[2:],'Secondary workloads: warm-file time reduction')
record_csv('query_context',ctxcsv)

stability=load('stability.json');stable=[];sv=[]
for g in ['soc-Epinions1','web-Stanford','com-Amazon','web-Google','com-Youtube']:
    vals=[]
    for m in ['ratio','ratio-plus','degree','degree-plus']:
        x=next(x for x in stability if x['dataset_id']==g and x['method']==m+'-random-tie')
        assert x['seeds']==10 and x['full_coverage']
        value=100*x['relative_range'];vals.append(f'{value:.3f}')
        sv.append({'graph':g,'method':m,'range_pct':value,'seeds':10})
    stable.append([NAMES[g]]+vals)
tex_table('stability','lrrrr',r'Graph & Ratio & Ratio+ & Degree & Degree+',stable)
record_csv('stability',sv)

stats={'sources_verified':len(manifest['snapshot_sha256']),
       'query_records_audited_in_source_batches':1276800,'query_primary_unique':8000,
       'query_context_unique':3200,'radius_conditions':12,
       'radius_speedup_range':[min(x['speedup'] for x in rcsv),max(x['speedup'] for x in rcsv)],
       'radius_memory_saved_range':[min(x['memory_reduction_pct'] for x in rcsv),max(x['memory_reduction_pct'] for x in rcsv)],
       'primary_time_reduction_pct':{x['graph']+'/'+x['reference']:100*(1-x['time_ratio']) for x in cicsv},
       'candidate_strata_reduction_range':[min(x['reduction_pct'] for x in ccsv),max(x['reduction_pct'] for x in ccsv)],
       'context_time_reduction_range':[min(x['warm_file_time_reduction_pct'] for x in ctxcsv),max(x['warm_file_time_reduction_pct'] for x in ctxcsv)],
       'context_positive_cells':sum(x['warm_file_time_reduction_pct']>0 for x in ctxcsv),
       'context_total_cells':len(ctxcsv)}
(ROOT/'verification.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats,indent=2))
