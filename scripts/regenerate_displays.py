"""Rebuild paper displays and report displays from the packaged evidence."""
from pathlib import Path
import csv, json, os, runpy, statistics as st, sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/matplotlib'))
sys.path.insert(0,str(ROOT/'paper/experiments'))
import plot_main_figures as plot

def table(path,spec,header,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join([r'\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule']+[' & '.join(row)+r' \\' for row in rows]+[r'\bottomrule',r'\end{tabular}','']))
def main():
    plot.main();base=ROOT/'paper/experiments/data';d=json.loads((base/'v1/evidence.json').read_text())
    uk=list(csv.DictReader((base/'uk2002_20261014/greedy/summary.csv').open()))
    cc=list(csv.DictReader((base/'uk2002_20261014/cc2fs/summary.csv').open()))
    names=['Epinions','Stanford','Amazon','Google','YouTube'];types=['Trust','Hyperlinks','Products','Hyperlinks','Social']
    graphs=['soc-Epinions1','web-Stanford','com-Amazon','web-Google','com-Youtube'];methods=['ratio','ratio-plus','degree','degree-plus']
    datasets=[];quality=[]
    for g,name,kind in zip(graphs,names,types):
        perf=[x for x in d['performance'] if x['graph_id']==g and x['radius']==1 and x['implementation']=='new']
        datasets.append([name,kind,f"{perf[0]['n']:,}",f"{perf[0]['m']:,}"])
        vv=[st.median(x['average_congestion'] for x in perf if x['method']==m) for m in methods]
        vv.append(st.median(x['average_congestion'] for x in d['cc2fs'] if x['dataset_id']==g and x['budget_seconds']==300 and not x['legacy_age']))
        quality.append([name]+[r'\textbf{'+f'{v:.4f}'+'}' if abs(v-min(vv))<1e-9 else f'{v:.4f}' for v in vv])
    datasets.append(['uk-2002','Web crawl',f"{int(uk[0]['n']):,}",f"{int(uk[0]['m']):,}"])
    vv=[float(next(x for x in uk if x['method']==m)['average_congestion']) for m in methods]+[float(next(x for x in cc if int(x['budget_seconds'])==300)['average_congestion_median'])]
    quality.append(['uk-2002']+[r'\textbf{'+f'{v:.4f}'+'}' if abs(v-min(vv))<1e-9 else f'{v:.4f}' for v in vv])
    table(base/'v1/tables/datasets.tex','@{}llrr@{}',r'Graph & Type & Vertices & Edges',datasets)
    table(base/'v1/tables/quality.tex',r'@{}l >{\columncolor{expblue!5}}r >{\columncolor{expteal!6}}r rrr@{}',r'Graph & \textcolor{expblue}{\textbf{Ratio}} & \textcolor{expteal}{\textbf{Ratio+}} & Degree & Degree+ & CC$^2$FS',quality)
    extraction=json.loads((base/'subgraph_20261109/table_source.json').read_text())['rows'];rows=[]
    for x in extraction:
        v=[x['mean_ms'][m] for m in ['ratio','ratio-plus','degree','cc2fs','random-prune-mean']]
        rows.append([{'web-Stanford':'Stanford','web-Google':'Google'}.get(x['graph'],x['graph'])]+[r'\textbf{'+f'{t:.3f}'+'}' if t==min(v) else f'{t:.3f}' for t in v])
    table(base/'subgraph_20261109/extraction_time.tex','lrrrrr',r'Graph & Ratio & Ratio+ & Degree & CC$^2$FS & Random-P',rows)
    runpy.run_path(str(ROOT/'supplementary/render.py'),run_name='__main__')
    print('Rebuilt 4 main-paper figures, 7 main-paper tables and all supplemental displays.')
if __name__=='__main__':main()
