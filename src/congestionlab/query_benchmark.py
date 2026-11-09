"""Native, paired timing of the frozen exact and quotient query definitions."""
from __future__ import annotations

import ctypes as ct
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np


def compile_engine(directory: Path) -> tuple[Path, dict]:
    source = Path(__file__).resolve().parents[1] / "native/query_engine.cpp"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "libcongestion_queries.so"
    command = ["g++", "-O3", "-DNDEBUG", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return output, {"compile_command": command, "source_sha256": digest(source),
                    "binary_sha256": digest(output), "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]}


class NativeQueries:
    def __init__(self, library: Path):
        self.lib = ct.CDLL(str(library))
        p32, p64 = ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int64)
        self.lib.cq_create.argtypes = [p64, p32, ct.c_int64, ct.c_int64, p32, p64, p32]
        self.lib.cq_create.restype = ct.c_void_p
        self.lib.cq_destroy.argtypes = [ct.c_void_p]
        self.lib.cq_run.argtypes = [ct.c_void_p, p32, ct.c_int64, ct.c_int]
        self.lib.cq_output.argtypes = [ct.c_void_p]
        self.lib.cq_output.restype = p32
        self.lib.cq_counts.argtypes = [ct.c_void_p, p64]
        self.lib.cq_target_missing.argtypes = [ct.c_void_p, p32, ct.c_int64]
        self.lib.cq_target_missing.restype = ct.c_int64
        self.lib.cq_measure_pair.argtypes = [ct.c_void_p, ct.c_void_p, p32, ct.c_int64, ct.c_int,
                                           ct.c_int, ct.c_int, ct.c_uint64, ct.POINTER(ct.c_double)]
        self.lib.cq_create_boundary.argtypes = [p64, p32, ct.c_int64, p32, p64, p32, p64, p32, ct.c_int64]
        self.lib.cq_create_boundary.restype = ct.c_void_p
        self.lib.cq_build_boundary.argtypes = [p64, p32, ct.c_int64, p32, ct.c_int64, p64, p32]
        self.lib.cq_build_boundary.restype = ct.c_int64
        self.lib.cq_boundary_work.argtypes = [ct.c_void_p, p64]
        self.lib.cq_measure_three.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, p32, ct.c_int64,
                                            ct.c_int, ct.c_int, ct.c_int, ct.c_uint64, ct.POINTER(ct.c_double)]
        for name in ("cq_destroy", "cq_run", "cq_counts", "cq_measure_pair", "cq_boundary_work", "cq_measure_three"):
            getattr(self.lib, name).restype = None

    def context(self, indptr, indices, output_capacity, owner=None, offsets=None, members=None):
        return QueryContext(self, indptr, indices, output_capacity, owner, offsets, members)

    def build_boundary(self, indptr, indices, owner, pieces):
        started = time.perf_counter()
        indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        owner = np.ascontiguousarray(owner, dtype=np.int32)
        ptr = np.empty(len(indptr), dtype=np.int64)
        capacity = np.empty(len(indices), dtype=np.int32)
        as_ptr = lambda a, t: a.ctypes.data_as(ct.POINTER(t))
        size = self.lib.cq_build_boundary(as_ptr(indptr, ct.c_int64), as_ptr(indices, ct.c_int32),
            len(owner), as_ptr(owner, ct.c_int32), pieces, as_ptr(ptr, ct.c_int64), as_ptr(capacity, ct.c_int32))
        values = capacity[:size].copy()
        return ptr, values, {"build_seconds": time.perf_counter()-started, "entries": int(size),
            "payload_bytes": int(ptr.nbytes+values.nbytes), "original_directed_edges": int(len(indices)),
            "entry_fraction_of_original_edges": size/max(1, len(indices))}

    def boundary_context(self, indptr, indices, owner, offsets, members, boundary_indptr, boundary_indices):
        return QueryContext(self, indptr, indices, len(owner), owner, offsets, members,
                            boundary_indptr, boundary_indices)

    def measure_pair(self, exact, part, seeds, radius, repeats, inner, seed):
        q = np.ascontiguousarray(seeds, dtype=np.int32)
        timings = np.empty((repeats, 2), dtype=np.float64)
        self.lib.cq_measure_pair(exact.handle, part.handle, q.ctypes.data_as(ct.POINTER(ct.c_int32)),
            q.size, radius, repeats, inner, seed, timings.ctypes.data_as(ct.POINTER(ct.c_double)))
        return timings / 1e6  # milliseconds per query, excluding the Python/FFI boundary

    def measure_three(self, exact, coarse, fine, seeds, radius, repeats, inner, seed):
        q = np.ascontiguousarray(seeds, dtype=np.int32)
        timings = np.empty((repeats, 3), dtype=np.float64)
        self.lib.cq_measure_three(exact.handle, coarse.handle, fine.handle,
            q.ctypes.data_as(ct.POINTER(ct.c_int32)), q.size, radius, repeats, inner, seed,
            timings.ctypes.data_as(ct.POINTER(ct.c_double)))
        return timings / 1e6


class QueryContext:
    def __init__(self, engine, indptr, indices, output_capacity, owner, offsets, members,
                 boundary_indptr=None, boundary_indices=None):
        self.engine = engine
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.owner = None if owner is None else np.ascontiguousarray(owner, dtype=np.int32)
        self.offsets = None if offsets is None else np.ascontiguousarray(offsets, dtype=np.int64)
        self.members = None if members is None else np.ascontiguousarray(members, dtype=np.int32)
        ptr = lambda x, t: None if x is None else x.ctypes.data_as(ct.POINTER(t))
        self.boundary_indptr = None if boundary_indptr is None else np.ascontiguousarray(boundary_indptr, dtype=np.int64)
        self.boundary_indices = None if boundary_indices is None else np.ascontiguousarray(boundary_indices, dtype=np.int32)
        if self.boundary_indptr is None:
            self.handle = engine.lib.cq_create(ptr(self.indptr, ct.c_int64), ptr(self.indices, ct.c_int32),
                len(self.indptr) - 1, output_capacity, ptr(self.owner, ct.c_int32),
                ptr(self.offsets, ct.c_int64), ptr(self.members, ct.c_int32))
            self.workspace_bytes = 8 * (len(self.indptr) - 1) + 4 * output_capacity
        else:
            self.handle = engine.lib.cq_create_boundary(ptr(self.indptr, ct.c_int64), ptr(self.indices, ct.c_int32),
                len(self.owner), ptr(self.owner, ct.c_int32), ptr(self.offsets, ct.c_int64),
                ptr(self.members, ct.c_int32), ptr(self.boundary_indptr, ct.c_int64),
                ptr(self.boundary_indices, ct.c_int32), len(self.offsets)-1)
            self.workspace_bytes = 12 * len(self.owner) + 8 * (len(self.offsets)-1)
        if not self.handle:
            raise MemoryError("native query workspace allocation failed")

    def run(self, seeds, radius):
        q = np.ascontiguousarray(seeds, dtype=np.int32)
        self.engine.lib.cq_run(self.handle, q.ctypes.data_as(ct.POINTER(ct.c_int32)), q.size, radius)
        return self.output()

    def counts(self):
        result = np.empty(4, dtype=np.int64)
        self.engine.lib.cq_counts(self.handle, result.ctypes.data_as(ct.POINTER(ct.c_int64)))
        return tuple(map(int, result))

    def output(self):
        size = self.counts()[0]
        return np.ctypeslib.as_array(self.engine.lib.cq_output(self.handle), shape=(size,)).copy()

    def missing(self, target):
        target = np.ascontiguousarray(target, dtype=np.int32)
        return self.engine.lib.cq_target_missing(self.handle, target.ctypes.data_as(ct.POINTER(ct.c_int32)), target.size)

    def boundary_work(self):
        result = np.empty(4, dtype=np.int64)
        self.engine.lib.cq_boundary_work(self.handle, result.ctypes.data_as(ct.POINTER(ct.c_int64)))
        return tuple(map(int, result))

    def close(self):
        if getattr(self, "handle", None):
            self.engine.lib.cq_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
