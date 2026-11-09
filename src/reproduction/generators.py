"""Pure generators extracted unchanged from the formal controller."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib,json,math
import numpy as np
from scipy import sparse

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)

def simple_adjacency(n: int, edge_pairs: list[tuple[int, int]]) -> sparse.csr_matrix:
    if edge_pairs:
        left = np.fromiter((edge[0] for edge in edge_pairs), dtype=np.int32, count=len(edge_pairs))
        right = np.fromiter((edge[1] for edge in edge_pairs), dtype=np.int32, count=len(edge_pairs))
        rows = np.concatenate((left, right))
        cols = np.concatenate((right, left))
        matrix = sparse.coo_matrix((np.ones(rows.size, dtype=np.uint8), (rows, cols)), shape=(n, n)).tocsr()
        matrix.data[:] = 1
        matrix.setdiag(0)
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return matrix
    return sparse.csr_matrix((n, n), dtype=np.uint8)

def erdos_renyi_edges(n: int, p: float, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    edges: list[tuple[int, int]] = []
    for u in range(n - 1):
        selected = np.flatnonzero(rng.random(n - u - 1) < p) + u + 1
        edges.extend((u, int(v)) for v in selected)
    return edges

def circulant_edges(n: int, degree: int) -> list[tuple[int, int]]:
    if degree < 0 or degree >= n:
        raise ValueError(f"degree {degree} must be in [0, n-1] for n={n}")
    if degree % 2 and n % 2:
        raise ValueError("odd regular degree requires an even number of vertices")
    edges: set[tuple[int, int]] = set()
    for distance in range(1, degree // 2 + 1):
        for u in range(n):
            v = (u + distance) % n
            edges.add((min(u, v), max(u, v)))
    if degree % 2:
        for u in range(n // 2):
            v = u + n // 2
            edges.add((u, v))
    return sorted(edges)

def randomized_regular_edges(n: int, degree: int, seed: int) -> list[tuple[int, int]]:
    """Create a valid regular graph, then randomize it by degree-preserving switches."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    edge_list = [(int(permutation[u]), int(permutation[v])) for u, v in circulant_edges(n, degree)]
    edge_list = [(min(u, v), max(u, v)) for u, v in edge_list]
    edge_set = set(edge_list)
    target_successes = max(1000, 8 * len(edge_list))
    successes = 0
    attempts = 0
    max_attempts = target_successes * 20
    while successes < target_successes and attempts < max_attempts:
        attempts += 1
        first = int(rng.integers(len(edge_list)))
        second = int(rng.integers(len(edge_list) - 1))
        if second >= first:
            second += 1
        a, b = edge_list[first]
        c, d = edge_list[second]
        if rng.integers(2):
            c, d = d, c
        if len({a, b, c, d}) != 4:
            continue
        new1 = (min(a, d), max(a, d))
        new2 = (min(c, b), max(c, b))
        if new1 == new2 or new1 in edge_set or new2 in edge_set:
            continue
        edge_set.remove(edge_list[first])
        edge_set.remove(edge_list[second])
        edge_set.add(new1)
        edge_set.add(new2)
        edge_list[first] = new1
        edge_list[second] = new2
        successes += 1
    if successes < target_successes:
        raise RuntimeError(f"regular-graph switch mixing stopped at {successes}/{target_successes} successful swaps")
    return edge_list

def near_regular_edges(n: int, degree: int, seed: int) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Perturb disjoint vertex pairs in a randomized regular graph by one degree."""
    if n < 4 or degree < 2 or degree >= n - 1:
        raise ValueError("near-regular generation requires n>=4 and 2<=degree<n-1")
    rng = np.random.default_rng(seed)
    edges = {tuple(sorted(edge)) for edge in randomized_regular_edges(n, degree, seed)}
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    permutation = [int(x) for x in rng.permutation(n)]
    pair_limit = n // 10
    transfer_vertices = permutation[: 2 * pair_limit]
    anchor_vertices = set(permutation[2 * pair_limit :])
    completed = 0
    for i in range(0, len(transfer_vertices) - 1, 2):
        lower, higher = transfer_vertices[i], transfer_vertices[i + 1]
        moved = False
        for anchor in list(neighbors[lower]):
            if anchor not in anchor_vertices or anchor in neighbors[higher]:
                continue
            old_edge = tuple(sorted((lower, anchor)))
            new_edge = tuple(sorted((higher, anchor)))
            edges.remove(old_edge)
            edges.add(new_edge)
            neighbors[lower].remove(anchor)
            neighbors[anchor].remove(lower)
            neighbors[higher].add(anchor)
            neighbors[anchor].add(higher)
            completed += 1
            moved = True
            break
        if not moved:
            continue
    if completed == 0:
        raise RuntimeError("near-regular graph generator could not perform a degree move")
    degrees = np.asarray([len(row) for row in neighbors], dtype=np.int64)
    if degrees.min() < degree - 1 or degrees.max() > degree + 1:
        raise RuntimeError("near-regular graph degrees escaped d±1")
    metadata = {
        "family": "randomized d-regular graph with disjoint endpoint degree transfers",
        "target_degree": int(degree),
        "seed": int(seed),
        "degree_transfers": completed,
        "degree_min": int(degrees.min()),
        "degree_max": int(degrees.max()),
        "degree_histogram": {str(int(d)): int(np.count_nonzero(degrees == d)) for d in np.unique(degrees)},
    }
    return sorted(edges), metadata

def preferential_attachment_edges(n: int, m: int, seed: int) -> list[tuple[int, int]]:
    if m < 1 or n <= m:
        raise ValueError("preferential attachment requires n > m >= 1")
    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = set()
    degree = np.zeros(n, dtype=np.int64)
    for u in range(m + 1):
        for v in range(u + 1, m + 1):
            edges.add((u, v))
            degree[u] += 1
            degree[v] += 1
    for new_vertex in range(m + 1, n):
        selected: set[int] = set()
        while len(selected) < m:
            weights = degree[:new_vertex].astype(float)
            weights /= weights.sum()
            selected.add(int(rng.choice(new_vertex, p=weights)))
        for old in selected:
            edges.add((old, new_vertex))
            degree[old] += 1
            degree[new_vertex] += 1
    return sorted(edges)

def client_hub_graph(max_degree: int) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    t = math.floor(math.sqrt(max_degree + 1)) - 1
    L = max_degree + 1
    if L % 2:
        L += 1
    hub_internal_degree = max_degree - t
    edges: list[tuple[int, int]] = []
    for client in range(L):
        for offset in range(t):
            edges.append((client, L + ((client + offset) % L)))
    edges.extend((L + u, L + v) for u, v in circulant_edges(L, hub_internal_degree))
    adjacency = simple_adjacency(2 * L, edges)
    degrees = np.diff(adjacency.indptr)
    if not np.all(degrees[:L] == t) or not np.all(degrees[L:] == max_degree):
        raise RuntimeError("client-hub construction did not produce the prescribed degrees")
    costs = degrees + 1
    dual = np.zeros(2 * L, dtype=np.int64)
    dual[:L] = t + 1
    coverage = adjacency + sparse.eye(2 * L, format="csr", dtype=np.uint8)
    if np.any(np.asarray(coverage @ dual).ravel() > costs):
        raise RuntimeError("client-hub dual certificate is infeasible")
    metadata = {
        "construction": "client-hub theorem family",
        "max_degree_parameter": max_degree,
        "client_count": L,
        "hub_count": L,
        "t": t,
        "hub_internal_degree": hub_internal_degree,
        "degree_check_passed": True,
        "dual_feasibility_passed": True,
        "dual_average_lower_bound": float(dual.sum() / (2 * L)),
        "dual_expected_lower_bound": float((t + 1) / 2),
    }
    return adjacency, metadata

def save_graph(matrix: sparse.csr_matrix, graph_path: Path, graph_id: str, metadata: dict[str, Any]) -> None:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(graph_path, matrix, compressed=True)
    ids_path = graph_path.with_name(graph_path.name.replace(".npz", ".node_ids.npy"))
    np.save(ids_path, np.arange(matrix.shape[0], dtype=np.int64), allow_pickle=False)
    gxt_path = graph_path.with_suffix(".gxt")
    coo = matrix.tocoo(copy=False)
    with gxt_path.open("w", encoding="ascii") as stream:
        stream.write(f"{matrix.shape[0]}\n")
        for u, v in zip(coo.row, coo.col):
            stream.write(f"{int(u)} {int(v)}\n")
    degree = np.diff(matrix.indptr)
    graph_metadata = {
        "dataset_id": graph_id,
        "view": "synthetic_full",
        "generator": metadata,
        "n": int(matrix.shape[0]),
        "m": int(matrix.nnz // 2),
        "degree_min": int(degree.min()) if degree.size else 0,
        "degree_mean": float(degree.mean()) if degree.size else 0.0,
        "degree_max": int(degree.max(initial=0)),
        "matrix_sha256": sha256(graph_path),
        "node_ids_sha256": sha256(ids_path),
        "gxt_sha256": sha256(gxt_path),
    }
    graph_path.with_suffix(".metadata.json").write_text(
        json.dumps(graph_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

def ensure_synthetic(run_root: Path, family: str, n: int, seed: int, degree: int | None = None) -> Path:
    if family == "er":
        graph_id = f"er-n{n}-s{seed}"
    elif family == "rr":
        graph_id = f"rr-d{degree}-n{n}-s{seed}"
    elif family == "pa":
        graph_id = f"pa-m{degree}-n{n}-s{seed}"
    elif family == "ch":
        graph_id = f"client-hub-Delta{degree}"
    elif family == "nr":
        graph_id = f"near-regular-d{degree}-n{n}-s{seed}"
    else:
        raise ValueError(f"unknown synthetic family {family}")
    graph_path = run_root / "graphs" / f"{graph_id}.npz"
    if graph_path.exists():
        metadata_path = graph_path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            raise RuntimeError(f"synthetic graph exists without metadata: {graph_path}")
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata.get("matrix_sha256") != sha256(graph_path):
            raise RuntimeError(f"synthetic graph hash mismatch: {graph_path}")
        return graph_path
    if family == "er":
        edges = erdos_renyi_edges(n, 0.5, seed)
        matrix = simple_adjacency(n, edges)
        generation = {"family": "G(n,1/2)", "p": 0.5, "seed": seed}
    elif family == "rr":
        edges = randomized_regular_edges(n, int(degree), seed)
        matrix = simple_adjacency(n, edges)
        actual = np.diff(matrix.indptr)
        if not np.all(actual == int(degree)):
            raise RuntimeError("degree-preserving randomization changed the regular degree")
        generation = {
            "family": "simple d-regular graph randomized by degree-preserving edge switches",
            "degree": int(degree),
            "seed": seed,
            "switch_mix_target": max(1000, 8 * len(edges)),
            "uniform_over_all_regular_graphs": False,
        }
    elif family == "pa":
        edges = preferential_attachment_edges(n, int(degree), seed)
        matrix = simple_adjacency(n, edges)
        generation = {"family": "preferential attachment", "m": int(degree), "seed": seed}
    elif family == "nr":
        edges, generation = near_regular_edges(n, int(degree), seed)
        matrix = simple_adjacency(n, edges)
    else:
        matrix, generation = client_hub_graph(int(degree))
    save_graph(matrix, graph_path, graph_id, generation)
    return graph_path
