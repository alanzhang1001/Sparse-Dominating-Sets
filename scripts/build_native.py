"""Build released sources; formal timing requires Linux/GCC and --with-reference."""
from pathlib import Path
import argparse, hashlib, json, os, platform, shlex, shutil, subprocess
ROOT=Path(__file__).resolve().parents[1]
def build(with_reference=False, serial_reference=False):
    out=ROOT/'build/native';out.mkdir(parents=True,exist_ok=True)
    compiler=shlex.split(os.environ.get('CXX','c++'))
    if not shutil.which(compiler[0]):raise RuntimeError('Install a C++17 compiler or set CXX')
    flags=['-std=c++17','-O3','-DNDEBUG']+(['-fopenmp'] if with_reference and not serial_reference else [])
    native=ROOT/'src/native'
    targets={'counter_heap':[native/'dom_ratio_count_heap.cpp'],'cc2fs':[native/'cc2fs.cpp'],
             'subgraph_engine':[native/'subgraph_engine.cpp']}
    ref=ROOT/'baselines/sparsedomsets'
    targets['counter_csr']=[native/'csr/counter.cpp']
    targets['cc2fs_csr']=[native/'csr/cc2fs.cpp']
    if with_reference or serial_reference:
        common=[ref/p for p in ['graph/CIntGraph.cpp','graph/CIntDiGraph.cpp','graph/c_graph_ops.cpp','readwrite/c_edgelist.cpp']]
        targets['reference']=[native/'reference_profile.cpp']+common
        targets['reference_csr']=[native/'csr/reference.cpp']+common
        targets['reference_select']=[native/'reference_select.cpp']+common
        targets['partition']=[native/'partition.cpp',ref/'algorithms/c_partition.cpp',ref/'algorithms/c_flow.cpp']+common
    if platform.system()=='Linux':targets['measure']=[native/'measure_native.cpp']
    records={}
    for name,sources in targets.items():
        is_reference=name in ['reference','reference_csr','reference_select','partition']
        extra=['-I'+str(ROOT/'src/portability'),'-Wno-unknown-pragmas'] if serial_reference and is_reference else []
        cmd=compiler+flags+extra+(['-I'+str(ref)] if is_reference else [])+list(map(str,sources))+['-o',str(out/name)]
        p=subprocess.run(cmd,capture_output=True,text=True)
        (out/(name+'.build.txt')).write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(p.stderr)
        records[name]={'sources':{str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},'flags':flags,'binary_sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()}
    for name, source in [('libqueries', 'query_engine.cpp'), ('libcontrols', 'retrieval_controls.cpp')]:
        library = out / (name + '.so')
        cmd = compiler + flags + ['-shared', '-fPIC', str(native / source), '-o', str(library)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        records[name] = {'sources': {'src/native/' + source: hashlib.sha256((native / source).read_bytes()).hexdigest()},
                         'flags': flags, 'binary_sha256': hashlib.sha256(library.read_bytes()).hexdigest()}
    (out/'build.json').write_text(json.dumps({'platform':platform.system(),'compiler':subprocess.check_output(compiler+['--version'],text=True).splitlines()[0],'serial_reference_shim':serial_reference,'targets':records},indent=2)+'\n')
    print('Built: '+', '.join(targets))
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--with-reference',action='store_true');p.add_argument('--serial-reference',action='store_true');a=p.parse_args();build(a.with_reference,a.serial_reference)
