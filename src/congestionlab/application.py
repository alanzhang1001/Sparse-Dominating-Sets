from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

from .graphs import load_graph, sha256


def _read_vertices(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64, ndmin=1)
    return np.unique(values.astype(np.int64))


def _multi_source_bfs(adjacency, roots: np.ndarray, owner: np.ndarray | None = None):
    n = adjacency.shape[0]
    distance = np.full(n, -1, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)
    roots = np.unique(np.asarray(roots, dtype=np.int64))
    distance[roots] = 0
    queue[: roots.size] = roots
    tail = int(roots.size)
    head = 0
    while head < tail:
        vertex = int(queue[head])
        head += 1
        begin, end = adjacency.indptr[vertex : vertex + 2]
        next_distance = distance[vertex] + 1
        part = owner[vertex] if owner is not None else -1
        for neighbor in adjacency.indices[begin:end]:
            neighbor = int(neighbor)
            if distance[neighbor] >= 0 or (owner is not None and owner[neighbor] != part):
                continue
            distance[neighbor] = next_distance
            queue[tail] = neighbor
            tail += 1
    return distance, tail


def _partition_metrics(owner: np.ndarray, labels: np.ndarray, offsets: np.ndarray) -> dict:
    sizes = np.diff(offsets).astype(np.int64)
    mean = float(np.mean(sizes)) if sizes.size else 0.0
    std = float(np.std(sizes)) if sizes.size else 0.0
    return {
        "piece_count": int(sizes.size),
        "mean_size": mean,
        "std_size": std,
        "cv": std / mean if mean else None,
        "p95_size": float(np.percentile(sizes, 95)) if sizes.size else 0.0,
        "p99_size": float(np.percentile(sizes, 99)) if sizes.size else 0.0,
        "max_size": int(sizes.max(initial=0)),
        "max_over_mean": float(sizes.max(initial=0) / mean) if mean else None,
        "size_min": int(sizes.min()) if sizes.size else 0,
        "vertices_per_piece": int(owner.size),
        "landmark_count": int(labels.size),
    }


def _build_quotient_graph(adjacency, owner: np.ndarray):
    """Build partition adjacency: two pieces are adjacent if an input edge crosses them."""
    edge_list = adjacency.tocoo(copy=False)
    source_parts = owner[edge_list.row]
    target_parts = owner[edge_list.col]
    cross = source_parts != target_parts
    count = int(owner.max()) + 1 if owner.size else 0
    quotient = sparse.coo_matrix(
        (np.ones(int(np.count_nonzero(cross)), dtype=np.uint8),
         (source_parts[cross], target_parts[cross])),
        shape=(count, count), dtype=np.uint8,
    ).tocsr()
    quotient.data[:] = 1
    quotient.eliminate_zeros()
    quotient.sort_indices()
    return quotient


def command_partition(args) -> None:
    if args.r != 1:
        raise ValueError("the compiled Prt-Weight baseline supports radius-1 dominating sets only")
    adjacency = load_graph(args.graph)
    landmarks = _read_vertices(args.landmarks)
    n = adjacency.shape[0]
    if landmarks.size == 0 or landmarks.min() < 0 or landmarks.max() >= n:
        raise ValueError("landmark IDs must be nonempty and in the processed graph's internal ID range")

    parts_path = args.output.with_suffix(".parts.txt")
    index_path = args.output.with_suffix(".index.npz")
    metadata_path = args.output.with_suffix(".metadata.json")
    parts_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.binary.resolve()), str(args.graph.with_suffix(".gxt")), str(args.landmarks.resolve()), str(args.strategy), str(args.seed), str(parts_path)]
    started = time.perf_counter()
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"nbrprt exit={process.returncode}; stdout={process.stdout}; stderr={process.stderr}")

    parsed: dict[int, list[int]] = {}
    with parts_path.open("r", encoding="ascii") as source:
        for line_no, line in enumerate(source, 1):
            fields = line.split()
            if not fields:
                continue
            label = int(fields[0])
            if label in parsed:
                raise ValueError(f"duplicate partition label {label} on line {line_no}")
            parsed[label] = [int(x) for x in fields[1:]]
    if set(parsed) != set(map(int, landmarks)):
        raise ValueError("partition labels do not match the input landmark IDs")

    owner_landmark = np.full(n, -1, dtype=np.int64)
    for label, members in parsed.items():
        member_array = np.asarray(members, dtype=np.int64)
        if member_array.size == 0 or member_array.min() < 0 or member_array.max() >= n:
            raise ValueError(f"partition {label} is empty or contains an out-of-range vertex")
        if np.any(owner_landmark[member_array] != -1):
            raise ValueError("partition output assigns some vertices more than once")
        owner_landmark[member_array] = label
    if np.any(owner_landmark < 0):
        missing = int(np.count_nonzero(owner_landmark < 0))
        raise ValueError(f"partition output omits {missing} vertices")
    if any(owner_landmark[int(label)] != int(label) for label in landmarks):
        raise ValueError("a partition does not contain its landmark")

    labels = np.asarray(sorted(parsed), dtype=np.int32)
    owner = np.searchsorted(labels, owner_landmark).astype(np.int32)
    grouped = [np.sort(np.asarray(parsed[int(label)], dtype=np.int32)) for label in labels]
    offsets = np.zeros(len(grouped) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(x) for x in grouped], dtype=np.int64)
    members = np.concatenate(grouped) if grouped else np.empty(0, dtype=np.int32)
    if members.size != n:
        raise ValueError("partition membership count does not equal n")

    # Check connectedness and the nearest-landmark property using graph traversals.
    assigned_distance, assigned_reached = _multi_source_bfs(adjacency, labels, owner)
    if assigned_reached != n:
        raise ValueError(f"partition pieces are not connected to their landmarks ({assigned_reached}/{n} reached)")
    nearest_distance, nearest_reached = _multi_source_bfs(adjacency, labels)
    if nearest_reached != n:
        raise ValueError("processed graph has vertices outside the landmark-reachable components")
    nearest_ok = bool(np.array_equal(assigned_distance, nearest_distance))
    if not nearest_ok:
        raise ValueError("Prt-Weight assigned at least one vertex to a non-nearest landmark")

    quotient = _build_quotient_graph(adjacency, owner)
    np.savez(
        index_path, owner=owner, landmarks=labels, offsets=offsets, members=members,
        quotient_indptr=quotient.indptr, quotient_indices=quotient.indices,
    )
    metadata = {
        "dataset_id": args.dataset_id,
        "graph": str(args.graph.resolve()),
        "graph_sha256": sha256(args.graph),
        "landmarks_file": str(args.landmarks.resolve()),
        "landmarks_sha256": sha256(args.landmarks),
        "landmark_count": int(landmarks.size),
        "landmark_method": args.landmark_method,
        "partition_algorithm": "Prt-Weight (sparsedomsets nbrprt, strategy=1)",
        "landmark_radius": int(args.r),
        "partition_seed": int(args.seed),
        "command": command,
        "partition_output": str(parts_path.resolve()),
        "partition_output_sha256": sha256(parts_path),
        "index_file": str(index_path.resolve()),
        "index_sha256": sha256(index_path),
        "index_payload_bytes": int(owner.nbytes + labels.nbytes + offsets.nbytes + members.nbytes + quotient.indptr.nbytes + quotient.indices.nbytes),
        "quotient_graph_vertices": int(quotient.shape[0]),
        "quotient_graph_edges": int(quotient.nnz // 2),
        "quotient_graph_payload_bytes": int(quotient.indptr.nbytes + quotient.indices.nbytes),
        "partition_program_seconds": elapsed,
        "index_build_seconds": time.perf_counter() - started,
        "validation": {
            "all_vertices_assigned_once": True,
            "each_piece_contains_its_landmark": True,
            "all_pieces_connected": True,
            "every_vertex_assigned_to_nearest_landmark": nearest_ok,
        },
        "partition_metrics": _partition_metrics(owner, labels, offsets),
        "environment_stdout": process.stdout,
        "environment_stderr": process.stderr,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "index_file": str(index_path), "index_build_seconds": elapsed, **metadata["partition_metrics"]}, ensure_ascii=False))


def _sample_bfs_prefix(adjacency, start: int, size: int, rng: np.random.Generator) -> list[int]:
    visited = {int(start)}
    order = [int(start)]
    head = 0
    while len(order) < size and head < len(order):
        vertex = order[head]
        head += 1
        begin, end = adjacency.indptr[vertex : vertex + 2]
        neighbors = adjacency.indices[begin:end].copy()
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            node = int(neighbor)
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            if len(order) == size:
                break
    if len(order) < size:
        raise ValueError(f"BFS from vertex {start} reached only {len(order)} vertices, expected {size}")
    return order


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


def command_make_queries(args) -> None:
    adjacency = load_graph(args.graph)
    ids_path = args.graph.with_name(args.graph.name.replace(".npz", ".node_ids.npy"))
    original_ids = np.load(ids_path, mmap_mode="r") if ids_path.exists() else np.arange(adjacency.shape[0], dtype=np.int64)
    n = adjacency.shape[0]
    degrees = np.diff(adjacency.indptr).astype(np.int64)
    sizes = sorted(set(args.query_sizes))
    if not sizes or sizes[0] < 1 or sizes[-1] > n:
        raise ValueError("query sizes must be positive and no greater than n")
    rng = np.random.default_rng(args.seed)
    used = np.zeros(n, dtype=np.bool_)
    query_rows = []
    query_number = 0
    for workload in ("uniform_start", "degree_stratified_start"):
        for split, count in (("dev", args.dev_starts), ("test", args.test_starts)):
            available = np.flatnonzero(~used)
            if count > available.size:
                raise ValueError("not enough distinct start vertices for disjoint dev/test workloads")
            if workload == "uniform_start":
                starts = rng.choice(available, size=count, replace=False).astype(np.int64)
            else:
                starts = _degree_stratified_sample(degrees, available, count, rng)
            used[starts] = True
            for start in starts:
                bfs_order = _sample_bfs_prefix(adjacency, int(start), sizes[-1], rng)
                for query_size in sizes:
                    query_number += 1
                    query_rows.append({
                        "query_id": f"q{query_number:07d}",
                        "split": split,
                        "workload": workload,
                        "query_size": int(query_size),
                        "query_nodes": bfs_order[:query_size],
                        "start_vertex": int(start),
                        "start_vertex_original_id": int(original_ids[int(start)]),
                        "query_nodes_original_ids": [int(original_ids[int(node)]) for node in bfs_order[:query_size]],
                        "query_seed": int(args.seed),
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in query_rows:
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    metadata = {
        "dataset_id": args.dataset_id,
        "graph": str(args.graph.resolve()),
        "graph_sha256": sha256(args.graph),
        "node_ids_sha256": sha256(ids_path) if ids_path.exists() else None,
        "query_file": str(args.output.resolve()),
        "query_sha256": sha256(args.output),
        "query_seed": int(args.seed),
        "dev_starts_per_workload": int(args.dev_starts),
        "test_starts_per_workload": int(args.test_starts),
        "query_sizes": sizes,
        "query_count": len(query_rows),
        "start_ids_disjoint_between_dev_test": True,
        "start_ids_disjoint_between_workloads": True,
        "query_definition": "Q is a deterministic-size BFS prefix around the sampled start; one set is generated once and shared across methods",
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


class _ExactNeighborhood:
    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.visited = np.zeros(adjacency.shape[0], dtype=np.uint32)
        self.generation = 0

    def query(self, seeds: np.ndarray, radius: int):
        self.generation += 1
        if self.generation >= np.iinfo(np.uint32).max:
            self.visited.fill(0)
            self.generation = 1
        generation = self.generation
        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        self._distance[seeds] = 0
        queue = np.empty(self.adjacency.shape[0], dtype=np.int32)
        self.visited[seeds] = generation
        queue[: len(seeds)] = seeds
        tail = len(seeds)
        head = 0
        expanded = 0
        edges_scanned = 0
        while head < tail:
            vertex = int(queue[head])
            head += 1
            if int(self._distance[vertex]) >= radius:
                continue
            expanded += 1
            begin, end = self.adjacency.indptr[vertex : vertex + 2]
            edges_scanned += end - begin
            for neighbor in self.adjacency.indices[begin:end]:
                node = int(neighbor)
                if self.visited[node] == generation:
                    continue
                self.visited[node] = generation
                self._distance[node] = self._distance[vertex] + 1
                queue[tail] = node
                tail += 1
        target = queue[:tail].copy()
        return target, expanded, edges_scanned

    @property
    def _distance(self):
        # Distances only need to persist for vertices visited in the current query.
        if not hasattr(self, "_dist"):
            self._dist = np.zeros(self.adjacency.shape[0], dtype=np.uint8)
        return self._dist


def _summary_key(row):
    method = row["method"]
    if method == "Prt-Weight-quotient-neighborhood":
        method = f"{row.get('selection_method', 'unspecified')}/quotient-neighborhood"
    return row["split"], row["workload"], row["query_size"], method


def _quotient_bfs(indptr: np.ndarray, indices: np.ndarray, starts: np.ndarray, radius: int):
    """Expand seed partitions in the quotient graph and return touched partition IDs."""
    n = len(indptr) - 1
    distance = np.full(n, -1, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)
    starts = np.unique(np.asarray(starts, dtype=np.int64))
    distance[starts] = 0
    queue[:starts.size] = starts
    head = 0
    tail = int(starts.size)
    scanned_edges = 0
    while head < tail:
        part = int(queue[head])
        head += 1
        if distance[part] >= radius:
            continue
        begin, end = int(indptr[part]), int(indptr[part + 1])
        scanned_edges += end - begin
        for neighbor in indices[begin:end]:
            neighbor = int(neighbor)
            if distance[neighbor] != -1:
                continue
            distance[neighbor] = distance[part] + 1
            queue[tail] = neighbor
            tail += 1
    return queue[:tail].copy(), head, scanned_edges


def command_evaluate_queries(args) -> None:
    if args.r != 1:
        raise ValueError("the current Prt-Weight partition/query pipeline supports landmark radius r=1 only")
    if args.timing_repeats < 0:
        raise ValueError("--timing-repeats must be nonnegative")
    adjacency = load_graph(args.graph)
    with np.load(args.index) as index:
        owner = index["owner"]
        landmarks = index["landmarks"]
        offsets = index["offsets"]
        members = index["members"]
        if "quotient_indptr" not in index or "quotient_indices" not in index:
            raise ValueError("partition index lacks the quotient graph; rebuild it with the current runner")
        quotient_indptr = index["quotient_indptr"]
        quotient_indices = index["quotient_indices"]
    if owner.size != adjacency.shape[0]:
        raise ValueError("partition index size does not match graph")
    part_metadata_path = args.index.with_name(args.index.name.replace(".index.npz", ".metadata.json"))
    part_metadata = json.loads(part_metadata_path.read_text(encoding="utf-8")) if part_metadata_path.exists() else {}
    graph_hash = sha256(args.graph)
    if part_metadata.get("graph_sha256") and part_metadata["graph_sha256"] != graph_hash:
        raise ValueError("partition index was built from a different processed graph")
    if part_metadata.get("landmark_radius") is not None and int(part_metadata["landmark_radius"]) != args.r:
        raise ValueError("--r does not match the radius used to generate the landmark solution")
    if part_metadata.get("index_sha256") and part_metadata["index_sha256"] != sha256(args.index):
        raise ValueError("partition index file hash does not match its metadata")
    query_metadata_path = args.queries.with_suffix(args.queries.suffix + ".metadata.json")
    query_metadata = json.loads(query_metadata_path.read_text(encoding="utf-8")) if query_metadata_path.exists() else {}
    if query_metadata.get("graph_sha256") and query_metadata["graph_sha256"] != graph_hash:
        raise ValueError("query workload was generated from a different processed graph")
    query_hash = sha256(args.queries)
    if query_metadata.get("query_sha256") and query_metadata["query_sha256"] != query_hash:
        raise ValueError("query workload hash does not match its metadata")
    targets_path = args.targets_output or args.queries.with_name(f"{args.queries.stem}.h{args.h}.targets.bin")
    targets_metadata_path = targets_path.with_suffix(targets_path.suffix + ".metadata.json")
    targets_metadata = json.loads(targets_metadata_path.read_text(encoding="utf-8")) if targets_metadata_path.exists() else {}
    cache_valid = bool(
        targets_path.exists()
        and targets_metadata
        and targets_metadata.get("graph_sha256") == graph_hash
        and targets_metadata.get("queries_sha256") == query_hash
        and int(targets_metadata.get("h", -1)) == args.h
        and targets_metadata.get("target_file_sha256") == sha256(targets_path)
    )
    if targets_path.exists() and targets_metadata and not cache_valid:
        raise ValueError("target cache exists but its graph/query/radius hashes do not match this run")
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_valid:
        target_writer = targets_path.open("wb")
        target_index: dict[str, dict] = {}
    else:
        target_writer = None
        target_index = targets_metadata["queries"]
        target_reader = targets_path.open("rb")
    sizes = np.diff(offsets)
    groups = [members[offsets[i] : offsets[i + 1]] for i in range(len(landmarks))]
    exact = _ExactNeighborhood(adjacency)
    summaries: dict[tuple, list[dict]] = defaultdict(list)
    query_records = []
    processed_query_count = 0
    with args.queries.open("r", encoding="utf-8") as source:
        for query_index, line in enumerate(source):
            processed_query_count += 1
            query = json.loads(line)
            q = np.asarray(query["query_nodes"], dtype=np.int64)
            exact_timings = []
            for repeat in range(args.timing_repeats + 1):
                exact._distance[q] = 0
                started = time.perf_counter()
                target, expanded, edges_scanned = exact.query(q, args.h)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if repeat:
                    exact_timings.append(elapsed_ms)
            exact_ms = float(np.median(exact_timings)) if exact_timings else elapsed_ms
            target_sorted = np.sort(target).astype("<i4", copy=False)
            computed_target_hash = hashlib.sha256(target_sorted.astype("<i4", copy=False).tobytes()).hexdigest()
            if cache_valid:
                cached = target_index.get(query["query_id"])
                if cached is None or int(cached["count"]) != target_sorted.size:
                    raise ValueError(f"target cache is missing or has a different size for {query['query_id']}")
                target_reader.seek(int(cached["offset_bytes"]))
                cached_target = np.frombuffer(target_reader.read(int(cached["count"]) * 4), dtype="<i4").copy()
                if cached_target.size != target_sorted.size or not np.array_equal(cached_target, target_sorted):
                    raise ValueError(f"cached exact target does not match BFS for {query['query_id']}")
                target_hash = str(cached["sha256"])
                target_offset = int(cached["offset_bytes"])
            else:
                target_bytes = target_sorted.tobytes()
                target_offset = target_writer.tell()
                target_writer.write(target_bytes)
                target_hash = computed_target_hash
                target_index[query["query_id"]] = {
                    "offset_bytes": target_offset,
                    "count": int(target_sorted.size),
                    "sha256": target_hash,
                }
            retrieval_timings = []
            for repeat in range(args.timing_repeats + 1):
                started = time.perf_counter()
                touched, quotient_parts_scanned, quotient_edges_scanned = _quotient_bfs(
                    quotient_indptr, quotient_indices, owner[q], args.h
                )
                retrieved = np.concatenate([groups[int(part)] for part in touched]) if touched.size else np.empty(0, dtype=np.int32)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if repeat:
                    retrieval_timings.append(elapsed_ms)
            partition_ms = float(np.median(retrieval_timings)) if retrieval_timings else elapsed_ms
            intersection_size = int(np.intersect1d(retrieved, target_sorted, assume_unique=True).size)
            if intersection_size != target_sorted.size:
                raise RuntimeError("quotient-graph retrieval failed to include the exact h-hop neighborhood")
            union_size = int(retrieved.size + target_sorted.size - intersection_size)
            partition_result = {
                "is_template": False,
                "run_id": args.run_id,
                "dataset_id": args.dataset_id,
                "processed_graph_sha256": sha256(args.graph),
                "method": "Prt-Weight-quotient-neighborhood",
                "selection_method": part_metadata.get("landmark_method", "unspecified"),
                "partition_index": str(args.index.resolve()),
                "partition_index_sha256": sha256(args.index),
                "r": args.r,
                "h": args.h,
                "query_id": query["query_id"],
                "split": query["split"],
                "workload": query["workload"],
                "query_seed": query["query_seed"],
                "partition_seed": part_metadata.get("partition_seed"),
                "query_size": query["query_size"],
                "query_nodes": q.tolist(),
                "query_nodes_original_ids": query.get("query_nodes_original_ids"),
                "start_vertex_original_id": query.get("start_vertex_original_id"),
                "target_definition": "exact_closed_h_neighborhood_of_Q",
                "target_nodes_file": str(targets_path.resolve()),
                "target_offset_bytes": target_offset,
                "target_count": int(target_sorted.size),
                "target_sha256": target_hash,
                "status": "ok",
                "counts": {
                    "query_nodes": int(q.size),
                    "target_nodes": int(target_sorted.size),
                    "retrieved_nodes": int(retrieved.size),
                    "intersection_nodes": intersection_size,
                    "union_nodes": union_size,
                    "irrelevant_nodes": int(retrieved.size - intersection_size),
                    "nodes_scanned": int(retrieved.size),
                    "pieces_touched": int(touched.size),
                    "quotient_parts_scanned": int(quotient_parts_scanned),
                    "quotient_edges_scanned": int(quotient_edges_scanned),
                },
                "quality": {
                    "recall_to_target": intersection_size / max(1, target_sorted.size),
                    "precision_to_target": intersection_size / max(1, retrieved.size),
                    "jaccard_to_target": intersection_size / max(1, union_size),
                },
                "runtime": {
                    "latency_ms": partition_ms,
                    "timing_repeat": args.timing_repeats,
                    "latency_repeat_p95_ms": float(np.percentile(retrieval_timings, 95)) if retrieval_timings else partition_ms,
                    "cache_policy": "graph, partition index, and quotient graph resident in memory; first-touch order",
                    "page_list_materialization_included": True,
                    "ground_truth_preparation_ms": exact_ms,
                    "exact_bfs_expanded_nodes": int(expanded),
                    "exact_bfs_edges_scanned": int(edges_scanned),
                    "index_build_seconds": part_metadata.get("index_build_seconds"),
                    "index_bytes": part_metadata.get("index_payload_bytes"),
                },
            }
            exact_record = {
                "is_template": False,
                "run_id": args.run_id,
                "dataset_id": args.dataset_id,
                "processed_graph_sha256": sha256(args.graph),
                "method": "exact-BFS",
                "selection_method": "exact-reference",
                "r": args.r,
                "h": args.h,
                "query_id": query["query_id"],
                "split": query["split"],
                "workload": query["workload"],
                "query_seed": query["query_seed"],
                "query_size": query["query_size"],
                "query_nodes": q.tolist(),
                "query_nodes_original_ids": query.get("query_nodes_original_ids"),
                "start_vertex_original_id": query.get("start_vertex_original_id"),
                "target_definition": "exact_closed_h_neighborhood_of_Q",
                "target_nodes_file": str(targets_path.resolve()),
                "target_offset_bytes": target_offset,
                "target_count": int(target_sorted.size),
                "target_sha256": target_hash,
                "status": "ok",
                "counts": {
                    "query_nodes": int(q.size),
                    "target_nodes": int(target_sorted.size),
                    "retrieved_nodes": int(target_sorted.size),
                    "intersection_nodes": int(target_sorted.size),
                    "union_nodes": int(target_sorted.size),
                    "irrelevant_nodes": 0,
                    "nodes_scanned": expanded,
                    "pieces_touched": None,
                },
                "quality": {"recall_to_target": 1.0, "precision_to_target": 1.0, "jaccard_to_target": 1.0},
                "runtime": {
                    "latency_ms": exact_ms,
                    "timing_repeat": args.timing_repeats,
                    "latency_repeat_p95_ms": float(np.percentile(exact_timings, 95)) if exact_timings else exact_ms,
                    "cache_policy": "graph resident in memory; generation-stamped visited array",
                    "ground_truth_preparation_ms": 0.0,
                    "edges_scanned": int(edges_scanned),
                },
            }
            query_records.extend((partition_result, exact_record))
            summaries[_summary_key(partition_result)].append(partition_result)
            summaries[_summary_key(exact_record)].append(exact_record)

    if cache_valid:
        target_reader.close()
        if processed_query_count != len(target_index):
            raise ValueError("target cache query count does not match the workload")
    else:
        target_writer.close()
        targets_metadata = {
            "graph_sha256": graph_hash,
            "queries_sha256": query_hash,
            "h": args.h,
            "target_dtype": "little-endian int32, sorted vertex IDs",
            "target_file": str(targets_path.resolve()),
            "target_file_sha256": sha256(targets_path),
            "target_file_bytes": targets_path.stat().st_size,
            "queries": target_index,
        }
        targets_metadata_path.write_text(json.dumps(targets_metadata, separators=(",", ":")) + "\n", encoding="utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in query_records:
            out.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    summary_rows = []
    for key, records in sorted(summaries.items()):
        recall = np.asarray([x["quality"]["recall_to_target"] for x in records])
        precision = np.asarray([x["quality"]["precision_to_target"] for x in records])
        latency = np.asarray([x["runtime"]["latency_ms"] for x in records])
        returned = np.asarray([x["counts"]["retrieved_nodes"] for x in records])
        summary_rows.append({
            "split": key[0], "workload": key[1], "query_size": key[2], "method": key[3], "n_queries": len(records),
            "recall_mean": float(recall.mean()), "precision_mean": float(precision.mean()),
            "latency_p50_ms": float(np.percentile(latency, 50)), "latency_p95_ms": float(np.percentile(latency, 95)),
            "latency_p99_ms": float(np.percentile(latency, 99)), "returned_nodes_mean": float(returned.mean()),
        })
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps({"run_id": args.run_id, "graph_sha256": graph_hash, "queries_sha256": query_hash, "partition_index_sha256": sha256(args.index), "target_file": str(targets_path.resolve()), "target_file_sha256": sha256(targets_path), "h": args.h, "summary": summary_rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "records": len(query_records), "targets_file": str(targets_path), "summary_file": str(summary_path), "summary": summary_rows}, ensure_ascii=False))
