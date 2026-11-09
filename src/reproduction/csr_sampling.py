"""BFS-induced samples from large raw CSR arrays without allocating full adjacency."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from congestionlab.graphs import _write_gxt


def sample_csr(directory, target_nodes, seed, output_dir):
    directory, output_dir = Path(directory), Path(output_dir)
    ptr = np.memmap(directory / 'indptr.u64', mode='r', dtype='<u8')
    ids = np.memmap(directory / 'indices.u32', mode='r', dtype='<u4')
    n = len(ptr) - 1
    if not 1 <= target_nodes <= n:
        raise ValueError('Invalid sample size')
    start = int(np.random.default_rng(seed).integers(n))
    selected, seen, head = [start], {start}, 0
    while head < len(selected) and len(selected) < target_nodes:
        v = selected[head]
        head += 1
        for value in ids[int(ptr[v]):int(ptr[v+1])]:
            u = int(value)
            if u not in seen:
                seen.add(u)
                selected.append(u)
                if len(selected) == target_nodes:
                    break
    if len(selected) != target_nodes:
        raise ValueError('Insufficient reachable vertices')
    mapping = {v:i for i,v in enumerate(selected)}
    rows, cols = [], []
    for i,v in enumerate(selected):
        for value in ids[int(ptr[v]):int(ptr[v+1])]:
            if int(value) in mapping:
                rows.append(i)
                cols.append(mapping[int(value)])
    a = sparse.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(target_nodes, target_nodes))
    a.sort_indices()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{directory.name}.bfs{target_nodes}.s{seed}'
    path = output_dir / (stem + '.npz')
    sparse.save_npz(path, a)
    _write_gxt(a, path.with_suffix('.gxt'))
    original = np.memmap(directory / 'original_ids.u32', mode='r', dtype='<u4')
    np.save(output_dir / (stem + '.node_ids.npy'), np.asarray(original[selected], np.int64))
    (output_dir / (stem + '.metadata.json')).write_text(json.dumps(dict(graph_file=path.name,
        sampler='BFS traversal then induced subgraph', seed=seed, n=target_nodes, m=a.nnz//2), indent=2)+'\n')
    return path
