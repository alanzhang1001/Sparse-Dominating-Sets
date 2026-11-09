from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


DATASETS = {
    "ego-Facebook": {
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "expected_nodes": 4039,
        "expected_edges": 88234,
        "kind": "undirected social graph",
        "contiguous_ids": True,
    },
    "soc-Epinions1": {
        "url": "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
        "expected_nodes": 75879,
        "expected_edges": 508837,
        "kind": "directed trust graph; symmetrized for this paper's undirected method",
        "contiguous_ids": False,
    },
    "web-Stanford": {
        "url": "https://snap.stanford.edu/data/web-Stanford.txt.gz",
        "expected_nodes": 281903,
        "expected_edges": 2312497,
        "kind": "directed web graph; symmetrized for this paper's undirected method",
        "contiguous_ids": True,
    },
    "com-Amazon": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        "expected_nodes": 334863,
        "expected_edges": 925872,
        "kind": "undirected product graph",
        "contiguous_ids": False,
    },
    "web-Google": {
        "url": "https://snap.stanford.edu/data/web-Google.txt.gz",
        "expected_nodes": 875713,
        "expected_edges": 5105039,
        "kind": "directed web graph; symmetrized for this paper's undirected method",
        "contiguous_ids": True,
    },
    "com-Youtube": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
        "expected_nodes": 1134890,
        "expected_edges": 2987624,
        "kind": "undirected online social graph",
        "contiguous_ids": False,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(dataset_id: str, raw_dir: Path) -> Path:
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset_id!r}; choose one of {sorted(DATASETS)}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{dataset_id}.txt.gz"
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    request = urllib.request.Request(DATASETS[dataset_id]["url"], headers={"User-Agent": "congestionlab/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)
    return destination


def _load_edges(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        edges = np.loadtxt(f, dtype=np.int64, comments="#", usecols=(0, 1), ndmin=2)
    if edges.size == 0:
        raise ValueError(f"No edge rows found in {path}")
    return edges


def _write_gxt(matrix: sparse.csr_matrix, path: Path) -> None:
    """Write the two-direction edge list format consumed by the reference C++ code."""
    coo = matrix.tocoo(copy=False)
    with path.open("w", encoding="ascii") as f:
        f.write(f"{matrix.shape[0]}\n")
        for u, v in zip(coo.row, coo.col):
            f.write(f"{int(u)} {int(v)}\n")


def prepare_dataset(dataset_id: str, raw_path: Path, output_dir: Path, view: str = "lcc") -> dict:
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset_id!r}; choose one of {sorted(DATASETS)}")
    if view not in {"lcc", "full"}:
        raise ValueError("view must be lcc or full")
    spec = DATASETS[dataset_id]
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_edges = _load_edges(raw_path)
    raw_node_ids = np.unique(raw_edges)
    if (
        spec["contiguous_ids"]
        and len(raw_node_ids) != spec["expected_nodes"]
        and int(raw_node_ids.min()) == 0
        and int(raw_node_ids.max()) == spec["expected_nodes"] - 1
    ):
        # If IDs span 0..N-1 but some are absent from the edge list, preserve them as isolates.
        raw_node_ids = np.arange(spec["expected_nodes"], dtype=np.int64)
    src = np.searchsorted(raw_node_ids, raw_edges[:, 0])
    dst = np.searchsorted(raw_node_ids, raw_edges[:, 1])
    n_raw = len(raw_node_ids)
    directed_rows = int(len(raw_edges))
    keep = src != dst
    src, dst = src[keep], dst[keep]
    rows = np.concatenate((src, dst))
    cols = np.concatenate((dst, src))
    matrix = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.uint32), (rows, cols)), shape=(n_raw, n_raw), dtype=np.uint32
    ).tocsr()
    matrix.data[:] = 1
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    n_unique_endpoint = len(np.unique(raw_edges))
    components_before = int(connected_components(matrix, directed=False, return_labels=False))
    if view == "lcc":
        count, labels = connected_components(matrix, directed=False, return_labels=True)
        sizes = np.bincount(labels, minlength=count)
        keep_component = int(np.argmax(sizes))
        keep_nodes = np.flatnonzero(labels == keep_component)
        matrix = matrix[keep_nodes][:, keep_nodes].tocsr()
        node_ids = raw_node_ids[keep_nodes]
    else:
        node_ids = raw_node_ids
    matrix.sort_indices()
    n_processed = int(matrix.shape[0])
    m_processed = int(matrix.nnz // 2)
    stem = f"{dataset_id}.{view}"
    matrix_path = output_dir / f"{stem}.npz"
    ids_path = output_dir / f"{stem}.node_ids.npy"
    gxt_path = output_dir / f"{stem}.gxt"
    metadata_path = output_dir / f"{stem}.metadata.json"
    sparse.save_npz(matrix_path, matrix, compressed=True)
    np.save(ids_path, node_ids, allow_pickle=False)
    _write_gxt(matrix, gxt_path)
    metadata = {
        "dataset_id": dataset_id,
        "source_url": spec["url"],
        "raw_file": str(raw_path.resolve()),
        "raw_sha256": sha256(raw_path),
        "view": view,
        "projection": "simple undirected projection; duplicate edges merged; self-loops removed",
        "raw_edge_rows": directed_rows,
        "raw_endpoint_node_count": n_unique_endpoint,
        "raw_node_count_used": n_raw,
        "expected_raw_nodes_from_snap": spec["expected_nodes"],
        "expected_raw_edges_from_snap": spec["expected_edges"],
        "components_before_view": components_before,
        "processed_nodes": n_processed,
        "processed_undirected_edges": m_processed,
        "isolated_nodes_in_processed_view": int(np.count_nonzero(np.diff(matrix.indptr) == 0)),
        "matrix_file": matrix_path.name,
        "matrix_sha256": sha256(matrix_path),
        "gxt_file": gxt_path.name,
        "gxt_sha256": sha256(gxt_path),
        "node_ids_file": ids_path.name,
        "node_ids_sha256": sha256(ids_path),
        "note": "SNAP metadata counts describe raw files. Processed counts are measured after projection and view selection.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_graph(matrix_path: Path) -> sparse.csr_matrix:
    graph = sparse.load_npz(matrix_path).tocsr()
    graph.sort_indices()
    return graph


def sample_bfs_graph(matrix_path: Path, target_nodes: int, seed: int, output_dir: Path) -> dict:
    """Create a reproducible BFS-induced subgraph for small exact-solver instances."""
    adjacency = load_graph(matrix_path)
    if target_nodes < 1 or target_nodes > adjacency.shape[0]:
        raise ValueError(f"target_nodes must be in 1..{adjacency.shape[0]}")
    ids_path = matrix_path.with_name(matrix_path.name.replace(".npz", ".node_ids.npy"))
    source_ids = np.load(ids_path, mmap_mode="r") if ids_path.exists() else np.arange(adjacency.shape[0])
    rng = np.random.default_rng(seed)
    start = int(rng.integers(adjacency.shape[0]))
    visited = np.zeros(adjacency.shape[0], dtype=np.bool_)
    selected = np.empty(target_nodes, dtype=np.int64)
    visited[start] = True
    selected[0] = start
    head = 0
    count = 1
    while head < count and count < target_nodes:
        vertex = int(selected[head])
        begin, end = adjacency.indptr[vertex : vertex + 2]
        for neighbor in adjacency.indices[begin:end]:
            neighbor = int(neighbor)
            if not visited[neighbor]:
                visited[neighbor] = True
                selected[count] = neighbor
                count += 1
                if count == target_nodes:
                    break
        head += 1
    if count != target_nodes:
        raise ValueError(f"BFS reached only {count} vertices; input graph may be disconnected")
    induced = adjacency[selected][:, selected].tocsr()
    induced.sort_indices()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = matrix_path.name.split(".")[0]
    stem = f"{dataset_id}.bfs{target_nodes}.s{seed}"
    graph_out = output_dir / f"{stem}.npz"
    ids_out = output_dir / f"{stem}.node_ids.npy"
    gxt_out = output_dir / f"{stem}.gxt"
    metadata_out = output_dir / f"{stem}.metadata.json"
    np.save(ids_out, np.asarray(source_ids[selected]), allow_pickle=False)
    sparse.save_npz(graph_out, induced, compressed=True)
    _write_gxt(induced, gxt_out)
    metadata = {
        "dataset_id": dataset_id,
        "source_graph": str(matrix_path.resolve()),
        "source_graph_sha256": sha256(matrix_path),
        "sampler": "BFS traversal then induced subgraph",
        "seed": int(seed),
        "start_vertex_original_id": int(source_ids[start]),
        "n": int(induced.shape[0]),
        "m": int(induced.nnz // 2),
        "graph_file": graph_out.name,
        "graph_sha256": sha256(graph_out),
        "gxt_file": gxt_out.name,
        "gxt_sha256": sha256(gxt_out),
        "node_ids_file": ids_out.name,
        "node_ids_sha256": sha256(ids_out),
    }
    metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def closed_radius_matrix(adjacency: sparse.csr_matrix, radius: int, max_walks: int = 100_000_000) -> sparse.csr_matrix:
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if radius == 1:
        result = adjacency.copy().astype(np.bool_)
        result.setdiag(True)
        result.eliminate_zeros()
        result.data[:] = True
        result.sort_indices()
        return result
    degrees = np.diff(adjacency.indptr).astype(np.int64)
    max_degree = int(degrees.max(initial=0))
    estimate = int(np.sum(degrees * degrees, dtype=np.int64))
    for _ in range(3, radius + 1):
        estimate *= max(1, max_degree)
        if estimate > max_walks:
            raise MemoryError(
                f"radius-{radius} walk upper bound {estimate:,} exceeds --max-walks {max_walks:,}; "
                "increase the budget only after checking available memory"
            )
    if estimate > max_walks:
        raise MemoryError(
            f"radius-{radius} walk upper bound {estimate:,} exceeds --max-walks {max_walks:,}; "
            "increase the budget only after checking available memory"
        )
    base = adjacency.astype(np.bool_)
    power = base.copy()
    result = base.copy()
    result.setdiag(True)
    for _distance in range(2, radius + 1):
        power = (power @ base).tocsr()
        power.eliminate_zeros()
        power.data[:] = True
        result = (result + power).tocsr()
        result.eliminate_zeros()
        result.data[:] = True
    result.sort_indices()
    return result
