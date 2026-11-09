"""Generate selectors, byte-controlled indexes, held-out workloads and exact layouts."""
from pathlib import Path
from collections import defaultdict
import argparse
import ctypes as ct
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
from scipy import sparse
from congestionlab.graphs import _write_gxt
from congestionlab.query_benchmark import NativeQueries
from .query_inputs import layout, queries_file, expected, prefix, _degree_stratified_sample

ROOT = Path(__file__).resolve().parents[2]
METHODS = {'ratio': 2, 'ratio-plus': 10, 'degree': 1}


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def new_output(path):
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def build():
    directory = ROOT / 'build/native'
    if not all((directory / name).exists() for name in ['partition', 'libcontrols.so', 'subgraph_engine']):
        mode = '--with-reference' if platform.system() == 'Linux' else '--serial-reference'
        subprocess.run([sys.executable, str(ROOT / 'scripts/build_native.py'), mode], check=True)
    return directory


class Controls:
    def __init__(self, library):
        self.native = NativeQueries(library)
        self.lib = self.native.lib
        p64, p32 = ct.POINTER(ct.c_int64), ct.POINTER(ct.c_int32)
        self.lib.ex_random_prune.argtypes = [p64, p32, ct.c_int64, p32, p32]
        self.lib.ex_random_prune.restype = ct.c_int64
        self.lib.ex_validate_partition.argtypes = [p64, p32, ct.c_int64, p32, ct.c_int64, p32]
        self.lib.ex_validate_partition.restype = ct.c_int64

    def pointers(self, graph):
        ptr = np.ascontiguousarray(graph.indptr, dtype=np.int64)
        ids = np.ascontiguousarray(graph.indices, dtype=np.int32)
        return ptr, ids

    @staticmethod
    def p(values, dtype):
        return values.ctypes.data_as(ct.POINTER(dtype))

    def random_prune(self, graph, seed):
        ptr, ids = self.pointers(graph)
        order = np.random.default_rng(seed).permutation(graph.shape[0]).astype(np.int32)
        result = np.empty(graph.shape[0], np.int32)
        count = self.lib.ex_random_prune(self.p(ptr, ct.c_int64), self.p(ids, ct.c_int32),
                                        graph.shape[0], self.p(order, ct.c_int32), self.p(result, ct.c_int32))
        if not 0 < count <= graph.shape[0]:
            raise RuntimeError('Random-Prune failed')
        return np.sort(result[:count])

    def validate(self, graph, labels, owner):
        ptr, ids = self.pointers(graph)
        code = self.lib.ex_validate_partition(self.p(ptr, ct.c_int64), self.p(ids, ct.c_int32),
                                              graph.shape[0], self.p(labels, ct.c_int32), len(labels),
                                              self.p(owner, ct.c_int32))
        if code != 0:
            raise RuntimeError('Partition is not connected and nearest-landmark: ' + str(code))


def check_graph(a):
    a = a.tocsr(copy=True)
    a.eliminate_zeros()
    a.sum_duplicates()
    a.sort_indices()
    if a.shape[0] != a.shape[1] or a.shape[0] == 0 or a.diagonal().any() or (a != a.T).nnz:
        raise ValueError('Input must be a nonempty simple undirected CSR graph')
    a.data[:] = 1
    return a


def create_index(a, graph_file, selected, folder, engine, bins):
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    selected = np.asarray(selected, np.int32)
    labels = np.sort(selected)
    covered = np.zeros(a.shape[0], bool)
    for v in labels:
        covered[v] = True
        covered[a.indices[a.indptr[v]:a.indptr[v+1]]] = True
    if len(np.unique(labels)) != len(labels) or not covered.all():
        raise ValueError('Selector did not produce an r=1 dominating set')
    np.savetxt(folder / 'landmarks.txt', selected, fmt='%d')
    subprocess.run([str(bins / 'partition'), str(graph_file), str(folder / 'landmarks.txt'),
                    '1', '0', str(folder / 'pieces.txt')], check=True, capture_output=True, text=True)
    owner = np.full(a.shape[0], -1, np.int32)
    groups = {}
    label_set = set(map(int, labels))
    for line in (folder / 'pieces.txt').read_text().splitlines():
        row = np.fromstring(line, sep=' ', dtype=np.int32)
        if not len(row):
            continue
        label, members = int(row[0]), np.sort(row[1:])
        if label not in label_set or label in groups or not len(members):
            raise ValueError('Invalid partition label')
        if members.min() < 0 or members.max() >= len(owner) or np.any(owner[members] >= 0):
            raise ValueError('Invalid or repeated partition member')
        owner[members] = np.searchsorted(labels, label)
        groups[label] = members
    if np.any(owner < 0) or len(groups) != len(labels):
        raise ValueError('Partition does not cover graph')
    members = np.concatenate([groups[int(v)] for v in labels]).astype(np.int32)
    offsets = np.r_[0, np.cumsum([len(groups[int(v)]) for v in labels], dtype=np.int64)]
    if len(np.unique(members)) != len(owner) or not np.array_equal(owner[labels], np.arange(len(labels))):
        raise ValueError('Partition membership is invalid')
    engine.validate(a, labels, owner)
    bp, bi, _ = engine.native.build_boundary(a.indptr, a.indices, owner, len(labels))
    index = dict(owner=owner, landmarks=labels, offsets=offsets, members=members,
                 boundary_indptr=bp, boundary_indices=bi)
    np.savez(folder / 'index.npz', **index)
    return {'path': str(folder.name + '/index.npz'), 'landmarks': len(labels),
            'payload_bytes': sum(x.nbytes for x in index.values()),
            'sha256': sha(folder / 'index.npz'), 'build_seconds': time.perf_counter() - started}


def augment(base, ranking, count):
    keep = np.zeros(len(ranking), bool)
    keep[base] = True
    return np.sort(np.r_[base, ranking[~keep[ranking]][:count-len(base)]]).astype(np.int32)


def workloads(a, plan, quick=False):
    """Original sampling order, fixed seeds, dev/test split and confirmation exclusions."""
    rows = []
    used = set(plan.get('excluded_pilot_starts', [])) if not quick else set()
    degree = np.diff(a.indptr)
    for h in [1, 2]:
        if plan['profile'] == 'snap':
            used = set()  # The original SNAP h=1/h=2 generators were independent.
        rng = np.random.default_rng(plan['candidate_seed'] + h)
        sizes = [1, 5, 20] if h == 1 else [1, 5]
        dev = (200 if h == 1 else 50) if plan['profile'] == 'snap' else (100 if h == 1 else 20)
        test = 1000 if h == 1 else 200
        for distribution in ['uniform_start', 'degree_stratified_start']:
            for split, count in [('dev', 5 if quick else dev), ('test', 5 if quick else test)]:
                available = np.setdiff1d(np.arange(a.shape[0]), list(used))
                if count > len(available):
                    raise ValueError('Graph too small for disjoint starts; use --quick for a demonstration')
                starts = (rng.choice(available, count, replace=False) if distribution == 'uniform_start'
                          else _degree_stratified_sample(degree, available, count, rng))
                used.update(map(int, starts))
                for start in starts:
                    nodes = prefix(a, int(start), max(sizes), rng)
                    for size in sizes:
                        rows.append(dict(h=h, split=split, workload=distribution, query_size=size,
                                         start_vertex=int(start), query_nodes=nodes[:size]))
    excluded = {q['start_vertex'] for q in rows} | set(plan.get('excluded_pilot_starts', []) if not quick else [])
    used = set(excluded)
    rng = np.random.default_rng(plan['exact_seed'])
    exact = []
    for h, sizes, count, cohort in [(1, [1], 5 if quick else 1000, 'primary'),
                                    (1, [5, 20], 5 if quick else 100, 'context'),
                                    (2, [1, 5], 5 if quick else 100, 'context')]:
        for distribution in ['uniform_start', 'degree_stratified_start']:
            available = np.setdiff1d(np.arange(a.shape[0]), list(used))
            if count > len(available):
                raise ValueError('Insufficient confirmation starts; use --quick')
            starts = (rng.choice(available, count, replace=False) if distribution == 'uniform_start'
                      else _degree_stratified_sample(degree, available, count, rng))
            used.update(map(int, starts))
            for start in starts:
                nodes = prefix(a, int(start), max(sizes), rng)
                for size in sizes:
                    exact.append(dict(h=h, query_size=size, query_nodes=nodes[:size],
                                      start_vertex=int(start), workload=distribution, cohort=cohort))
    assert not excluded & {q['start_vertex'] for q in exact}
    return rows, exact


def prepare(args):
    out = new_output(args.output)
    bins = build()
    engine = Controls(bins / 'libcontrols.so')
    if args.graph:
        a = check_graph(sparse.load_npz(args.graph))
    else:
        # A connected, reproducible small graph with heterogeneous degrees.
        n = 256
        rng = np.random.default_rng(41)
        edges = {(i, (i+1) % n) for i in range(n)}
        edges |= {(int(u), int(v)) for u, v in rng.integers(n, size=(320, 2)) if u != v}
        rr, cc = zip(*edges)
        a = sparse.csr_matrix((np.ones(2*len(rr), np.uint8), (rr+cc, cc+rr)), shape=(n, n))
        a = check_graph(a)
        args.quick = True
    sparse.save_npz(out / 'graph.npz', a)
    _write_gxt(a, out / 'graph.gxt')
    (out / 'indexes').mkdir()
    (out / 'selectors').mkdir()
    bases = {}
    for method, strategy in METHODS.items():
        dest = out / 'selectors' / (method + '.txt')
        p = subprocess.run([str(bins / 'counter_heap'), str(out / 'graph.gxt'), '1', str(strategy),
                            str(dest), str(args.max_entries)], check=True, capture_output=True, text=True)
        bases[method] = np.loadtxt(dest, dtype=np.int32, ndmin=1)
        save(dest.with_suffix('.json'), json.loads(p.stdout.strip().splitlines()[-1]))
    dest = out / 'selectors/cc2fs.txt'
    p = subprocess.run([str(bins / 'cc2fs'), str(out / 'graph.gxt'), str(args.cc_seconds), str(dest)],
                       check=True, capture_output=True, text=True)
    bases['cc2fs'] = np.loadtxt(dest, dtype=np.int32, ndmin=1)
    save(dest.with_suffix('.json'), json.loads(p.stdout.strip().splitlines()[-1]))
    for seed in range(20261013, 20261018):
        bases['random-prune-' + str(seed)] = engine.random_prune(a, seed)
    ranking = np.random.default_rng(20261111).permutation(a.shape[0]).astype(np.int32)
    cache = {}
    published = load(ROOT / 'configs/retrieval.json')['published_controls']
    published = [r for r in published if r['dataset_id'] == args.graph_name]

    def index(method, count):
        key = method, count
        if key not in cache:
            base = bases[method]
            # Native Ratio partitions for SNAP preserve the original greedy order.
            selected = base if count == len(base) and method == 'ratio' and args.graph_name in ['web-Stanford', 'web-Google'] else augment(base, ranking, count)
            folder = out / 'indexes' / f'{method}-{count}'
            cache[key] = create_index(a, out / 'graph.gxt', selected, folder, engine, bins)
        return cache[key]

    native = {method: index(method, len(base)) for method, base in bases.items()}
    caps = args.budgets
    if not caps and published:
        caps = sorted({r['budget_bytes'] for r in published if r['control'] == 'matched-budget'})
    if not caps:
        maximum = max(row['payload_bytes'] for row in native.values())
        caps = [int(maximum * f) for f in [1.0, 1.15, 1.3]]
    configurations = []
    for control in args.controls.split(','):
        for method, base in bases.items():
            if control == 'native':
                configurations.append(dict(native[method], method=method, control=control, budget_index=None))
            elif control == 'fixed-count':
                k = max(map(len, bases.values()))
                configurations.append(dict(index(method, k), method=method, control=control, budget_index=None))
            elif control == 'budget':
                for budget_index, cap in enumerate(caps, 1):
                    best = native[method]
                    attempted = {len(base)}
                    hint = next((r['landmark_count'] for r in published if r['selection_method'] == method and r['control'] == 'matched-budget' and r['budget_index'] == budget_index), None)
                    low, high = len(base), a.shape[0]
                    for step in range(args.refine_steps):
                        if .95 * cap <= best['payload_bytes'] <= cap or native[method]['payload_bytes'] > cap:
                            break
                        k = hint if step == 0 and hint is not None else (low + high) // 2
                        k = max(len(base), min(a.shape[0], k))
                        if k in attempted:
                            break
                        attempted.add(k)
                        row = index(method, k)
                        if row['payload_bytes'] <= cap:
                            if row['payload_bytes'] > best['payload_bytes']:
                                best = row
                            low = k + 1
                        else:
                            high = k - 1
                        if high < low:
                            break
                    configurations.append(dict(best, method=method, control='matched-budget', budget_index=budget_index,
                                               budget_bytes=cap, matched=.95 <= best['payload_bytes']/cap <= 1,
                                               calibration_counts=sorted(attempted)))
            else:
                raise ValueError('Unknown control: ' + control)
    # Lock graph-only selection before reading or generating query content.
    save(out / 'index_lock.json', configurations)
    plan = load(ROOT / 'configs/query_workloads.json')['graphs'].get(args.graph_name,
           dict(profile='snap', candidate_seed=20261004, exact_seed=202611082231, excluded_pilot_starts=[]))
    candidate, exact = workloads(a, plan, args.quick)
    save(out / 'candidate_queries.json', candidate)
    graph = out / 'exact'
    graph.mkdir()
    np.asarray(a.indptr, dtype='<i8').tofile(graph / 'ptr.bin')
    queries_file(a, exact, graph)
    layout(a, None, graph / 'layouts/original')
    selected = [r for r in configurations if r['control'] == args.exact_control and (args.exact_control != 'matched-budget' or r['budget_index'] == args.exact_budget)]
    if len(selected) != 9:
        raise ValueError('The exact control must select nine indexes; choose a control included in --controls')
    for row in selected:
        idx = dict(np.load(out / 'indexes' / row['path']))
        layout(a, idx, graph / 'layouts' / row['method'])
    save(out / 'manifest.json', dict(graph_name=args.graph_name or 'example', vertices=a.shape[0], edges=a.nnz//2,
        graph_sha256=sha(out / 'graph.npz'), quick=args.quick, query_plan=plan,
        candidate_test_queries=sum(q['split'] == 'test' for q in candidate), exact_queries=len(exact),
        indexes=len(configurations), index_lock_sha256=sha(out / 'index_lock.json'),
        exact_configurations=selected, exact_eligible=all(r.get('matched', True) for r in selected),
        mapping_bytes_per_layout=8*a.shape[0], cc_seconds=args.cc_seconds,
        workload_profile="example" if args.quick else "full"))
    print(f'Prepared {len(configurations)} indexes and {len(exact)} exact queries in {out}')


def candidates(args):
    out = new_output(args.output)
    source = args.inputs.resolve()
    a = sparse.load_npz(source / 'graph.npz')
    engine = Controls(build() / 'libcontrols.so')
    qs = [q for q in load(source / 'candidate_queries.json') if q['split'] == 'test']
    target = engine.native.context(a.indptr, a.indices, a.shape[0])
    result = []
    for i, row in enumerate(load(source / 'index_lock.json')):
        idx = dict(np.load(source / 'indexes' / row['path']))
        ctx = engine.native.boundary_context(a.indptr, a.indices, idx['owner'], idx['offsets'], idx['members'], idx['boundary_indptr'], idx['boundary_indices'])
        strata = defaultdict(list)
        with (out / f'config-{i:03}.jsonl').open('w') as record:
            for qi, q in enumerate(qs):
                truth = target.run(q['query_nodes'], q['h'])
                found = ctx.run(q['query_nodes'], q['h'])
                # Independent BFS oracle plus exact quotient-piece union.
                vertices, _ = expected(a, q)
                assert np.array_equal(np.sort(truth), vertices)
                expected_count = np.diff(idx['offsets'])[np.unique(idx['owner'][truth])].sum()
                assert ctx.missing(truth) == 0 and len(found) == expected_count
                strata[(q['h'], q['query_size'], q['workload'])].append(len(found))
                record.write(json.dumps(dict(query=qi, candidate_nodes=len(found), target_nodes=len(truth), recall=1.0))+'\n')
        ctx.close()
        result.append(dict(row, queries=len(qs), mean_candidates=statistics.mean(v for s in strata.values() for v in s),
                           strata=[dict(h=k[0], query_size=k[1], workload=k[2], mean_candidates=statistics.mean(v)) for k,v in strata.items()],
                           recall=1.0, eligible_for_budget_comparison=row.get('matched', True)))
        save(out / 'summary.json', result)
    target.close()
    print(f'Checked {len(result)} configurations × {len(qs)} queries')


def subgraphs(args):
    out = new_output(args.output)
    source = args.inputs.resolve()
    manifest = load(source / 'manifest.json')
    if not manifest['exact_eligible'] and not manifest['quick']:
        raise ValueError('Unmatched byte controls: revise graph-only calibration before formal exact timing')
    graph = source / 'exact'
    jobs = []
    rng = np.random.default_rng(manifest['query_plan']['exact_seed'])
    for rep in range(args.repeats):
        group = []
        for mode in args.modes.split(','):
            group.append(dict(layout='original', mode=mode, kind='bfs', repeat=rep))
            for row in manifest['exact_configurations']:
                for kind in ['partition', 'bfs']:
                    group.append(dict(layout=row['method'], mode=mode, kind=kind, repeat=rep))
        rng.shuffle(group)
        jobs.extend(group)
    save(out / 'schedule.json', jobs)
    save(out / 'source_manifest.json', manifest)
    runs = []
    binary = build() / 'subgraph_engine'
    for i, job in enumerate(jobs):
        dest = out / f'run-{i:04}.jsonl'
        subprocess.run([str(binary), str(graph), str(graph / 'layouts' / job['layout']), job['mode'], job['kind'],
                        str(job['repeat']), str(dest), str(manifest['query_plan']['exact_seed'] + i)], check=True, capture_output=True, text=True)
        rows = [json.loads(line) for line in dest.read_text().splitlines()]
        assert len(rows) == manifest['exact_queries'] and all(x['correct'] for x in rows)
        runs.append(dict(job, record=dest.name, queries=len(rows), sha256=sha(dest)))
        save(out / 'runs.json', runs)
    save(out / 'queries.json', load(graph / 'queries.json'))
    summarize(out)
    print(f'Validated {sum(r["queries"] for r in runs)} complete vertex/edge outputs in {len(runs)} jobs')


def summarize(out):
    qs = load(out / 'queries.json')
    grouped = defaultdict(list)
    for run in load(out / 'runs.json'):
        for line in (out / run['record']).read_text().splitlines():
            row = json.loads(line)
            grouped[(run['mode'], run['layout'], run['kind'], row['query'])].append(row)
    strata = defaultdict(list)
    for (mode, method, kind, qi), values in grouped.items():
        q = qs[qi]
        median = {key: statistics.median(v[key] for v in values) for key in ['total_ms', 'requested_bytes', 'read_calls']}
        strata[(mode, method, kind, q['h'], q['query_size'], q['workload'])].append(median)
    result = []
    for key, values in strata.items():
        result.append(dict(zip(['mode', 'method', 'kind', 'h', 'query_size', 'workload'], key),
                           queries=len(values), **{k:statistics.mean(v[k] for v in values) for k in values[0]}))
    save(out / 'strata.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('prepare')
    s.add_argument('--graph', type=Path, help='Prepared simple undirected NPZ; omit for the 256-vertex example')
    s.add_argument('--graph-name', default='')
    s.add_argument('--quick', action='store_true')
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--controls', default='budget,fixed-count')
    s.add_argument('--budgets', type=int, nargs='+')
    s.add_argument('--exact-control', choices=['native', 'fixed-count', 'matched-budget'], default='matched-budget')
    s.add_argument('--exact-budget', type=int, default=1)
    s.add_argument('--refine-steps', type=int, default=12)
    s.add_argument('--cc-seconds', type=float, default=60)
    s.add_argument('--max-entries', type=int, default=2_000_000_000)
    s.set_defaults(function=prepare)
    for name, function in [('candidates', candidates), ('subgraphs', subgraphs)]:
        s = sub.add_parser(name)
        s.add_argument('--inputs', type=Path, required=True)
        s.add_argument('--output', type=Path, required=True)
        s.set_defaults(function=function)
        if name == 'subgraphs':
            s.add_argument('--repeats', type=int, default=3)
            s.add_argument('--modes', default='memory,file', choices=['memory', 'file', 'memory,file'])
    args = p.parse_args()
    if hasattr(args, 'cc_seconds') and args.cc_seconds <= 0:
        p.error('--cc-seconds must be positive')
    if hasattr(args, 'repeats') and args.repeats < 1:
        p.error('--repeats must be positive')
    args.function(args)
