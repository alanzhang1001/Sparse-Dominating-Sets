// The same resident-CSR traversal and output materialization serve both engines.
// This changes execution only; the frozen quotient-neighborhood query is intact.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

struct QueryContext {
    const int64_t* indptr;
    const int32_t* indices;
    const int32_t* owner;
    const int64_t* offsets;
    const int32_t* members;
    std::vector<uint32_t> seen;
    std::vector<int32_t> queue, output;
    const int64_t* boundary_indptr = nullptr;
    const int32_t* boundary_indices = nullptr;
    std::vector<uint32_t> piece_seen;
    std::vector<int32_t> piece_queue;
    uint32_t generation = 0;
    int64_t reached = 0, output_size = 0, expanded = 0, scanned_edges = 0;
    int64_t original_edges_scanned = 0, frontier_nodes = 0;
    bool single_hop = false;
    int32_t single_root = -1;

    QueryContext(const int64_t* p, const int32_t* i, int64_t n,
                 int64_t output_capacity, const int32_t* o,
                 const int64_t* off, const int32_t* mem)
        : indptr(p), indices(i), owner(o), offsets(off), members(mem),
          seen(n, 0), queue(n), output(output_capacity) {}

    void run(const int32_t* seeds, int64_t count, int radius) {
        if (boundary_indptr) {
            run_boundary(seeds, count, radius);
            return;
        }
        original_edges_scanned = frontier_nodes = 0;
        reached = expanded = scanned_edges = 0;
        single_hop = count == 1 && radius == 1;
        if (single_hop) {
            // Both CSR inputs are validated simple graphs. A single seed's
            // one-hop neighbors need neither duplicate removal nor visited writes.
            single_root = owner ? owner[seeds[0]] : seeds[0];
            queue[0] = single_root;
            const int64_t begin = indptr[single_root], end = indptr[single_root + 1];
            std::memcpy(queue.data() + 1, indices + begin, (end - begin) * sizeof(int32_t));
            reached = end - begin + 1;
            expanded = 1;
            scanned_edges = end - begin;
        } else {
            if (++generation == 0) {
                std::fill(seen.begin(), seen.end(), 0);
                generation = 1;
            }
            for (int64_t k = 0; k < count; ++k) {
                const int32_t v = owner ? owner[seeds[k]] : seeds[k];
                if (seen[v] == generation) continue;
                seen[v] = generation;
                queue[reached++] = v;
            }
            int64_t layer_begin = 0, layer_end = reached;
            for (int depth = 0; depth < radius && layer_begin < layer_end; ++depth) {
                for (int64_t pos = layer_begin; pos < layer_end; ++pos) {
                    const int32_t v = queue[pos];
                    const int64_t begin = indptr[v], end = indptr[v + 1];
                    ++expanded;
                    scanned_edges += end - begin;
                    for (int64_t j = begin; j < end; ++j) {
                        const int32_t u = indices[j];
                        if (seen[u] == generation) continue;
                        seen[u] = generation;
                        queue[reached++] = u;
                    }
                }
                layer_begin = layer_end;
                layer_end = reached;
            }
        }
        output_size = 0;
        if (!owner) {
            std::memcpy(output.data(), queue.data(), reached * sizeof(int32_t));
            output_size = reached;
        } else {
            for (int64_t pos = 0; pos < reached; ++pos) {
                const int32_t piece = queue[pos];
                const int64_t begin = offsets[piece], length = offsets[piece + 1] - begin;
                std::memcpy(output.data() + output_size, members + begin, length * sizeof(int32_t));
                output_size += length;
            }
        }
    }

    void run_boundary(const int32_t* seeds, int64_t count, int radius) {
        reached = expanded = scanned_edges = original_edges_scanned = frontier_nodes = 0;
        single_hop = count == 1 && radius == 1;
        if (single_hop) {
            // Rows contain distinct neighboring pieces, excluding the node's own
            // piece. The symmetric one-seed shortcut still materializes the list.
            single_root = seeds[0];
            piece_queue[0] = owner[single_root];
            const int64_t begin = boundary_indptr[single_root], end = boundary_indptr[single_root + 1];
            std::memcpy(piece_queue.data() + 1, boundary_indices + begin, (end - begin) * sizeof(int32_t));
            reached = end - begin + 1;
            expanded = frontier_nodes = 1;
            scanned_edges = end - begin;
        } else {
            if (++generation == 0) {
                std::fill(seen.begin(), seen.end(), 0);
                std::fill(piece_seen.begin(), piece_seen.end(), 0);
                generation = 1;
            }
            int64_t tail = 0;
            for (int64_t k = 0; k < count; ++k) {
                const int32_t v = seeds[k];
                if (seen[v] == generation) continue;
                seen[v] = generation;
                queue[tail++] = v;
            }
            if (radius == 2) {
                const int64_t seed_count = tail;
                for (int64_t pos = 0; pos < seed_count; ++pos) {
                    const int32_t v = queue[pos];
                    const int64_t begin = indptr[v], end = indptr[v + 1];
                    original_edges_scanned += end - begin;
                    for (int64_t j = begin; j < end; ++j) {
                        const int32_t u = indices[j];
                        if (seen[u] == generation) continue;
                        seen[u] = generation;
                        queue[tail++] = u;
                    }
                }
            }
            frontier_nodes = expanded = tail;
            auto add_piece = [&](int32_t p) {
                if (piece_seen[p] == generation) return;
                piece_seen[p] = generation;
                piece_queue[reached++] = p;
            };
            for (int64_t pos = 0; pos < tail; ++pos) {
                const int32_t v = queue[pos];
                add_piece(owner[v]);
                const int64_t begin = boundary_indptr[v], end = boundary_indptr[v + 1];
                scanned_edges += end - begin;
                for (int64_t j = begin; j < end; ++j) add_piece(boundary_indices[j]);
            }
        }
        output_size = 0;
        for (int64_t pos = 0; pos < reached; ++pos) {
            const int32_t p = piece_queue[pos];
            const int64_t begin = offsets[p], length = offsets[p + 1] - begin;
            std::memcpy(output.data() + output_size, members + begin, length * sizeof(int32_t));
            output_size += length;
        }
    }

    bool contains(int32_t original_node) const {
        const int32_t v = owner ? owner[original_node] : original_node;
        if (boundary_indptr) {
            if (!single_hop) return piece_seen[v] == generation;
            return v == owner[single_root] || std::binary_search(
                boundary_indices + boundary_indptr[single_root],
                boundary_indices + boundary_indptr[single_root + 1], v);
        }
        if (!single_hop) return seen[v] == generation;
        return v == single_root || std::binary_search(
            indices + indptr[single_root], indices + indptr[single_root + 1], v);
    }
};

extern "C" {
void* cq_create(const int64_t* ptr, const int32_t* indices, int64_t n,
                int64_t output_capacity, const int32_t* owner,
                const int64_t* offsets, const int32_t* members) {
    try { return new QueryContext(ptr, indices, n, output_capacity, owner, offsets, members); }
    catch (...) { return nullptr; }
}
void cq_destroy(void* raw) { delete static_cast<QueryContext*>(raw); }
void* cq_create_boundary(const int64_t* ptr, const int32_t* indices, int64_t n,
                         const int32_t* owner, const int64_t* offsets,
                         const int32_t* members, const int64_t* boundary_ptr,
                         const int32_t* boundary_indices, int64_t pieces) {
    try {
        auto* c = new QueryContext(ptr, indices, n, n, owner, offsets, members);
        c->boundary_indptr = boundary_ptr;
        c->boundary_indices = boundary_indices;
        c->piece_seen.assign(pieces, 0);
        c->piece_queue.resize(pieces);
        return c;
    } catch (...) { return nullptr; }
}
int64_t cq_build_boundary(const int64_t* ptr, const int32_t* indices, int64_t n,
                          const int32_t* owner, int64_t pieces,
                          int64_t* output_ptr, int32_t* output_indices) {
    std::vector<int32_t> marks(pieces, -1);
    int64_t tail = 0;
    output_ptr[0] = 0;
    for (int64_t v = 0; v < n; ++v) {
        const int64_t begin = tail;
        for (int64_t j = ptr[v]; j < ptr[v + 1]; ++j) {
            const int32_t p = owner[indices[j]];
            if (p == owner[v] || marks[p] == v) continue;
            marks[p] = static_cast<int32_t>(v);
            output_indices[tail++] = p;
        }
        std::sort(output_indices + begin, output_indices + tail);
        output_ptr[v + 1] = tail;
    }
    return tail;
}
void cq_run(void* raw, const int32_t* q, int64_t size, int radius) {
    static_cast<QueryContext*>(raw)->run(q, size, radius);
}
const int32_t* cq_output(void* raw) { return static_cast<QueryContext*>(raw)->output.data(); }
void cq_counts(void* raw, int64_t* counts) {
    const auto& c = *static_cast<QueryContext*>(raw);
    counts[0] = c.output_size; counts[1] = c.reached;
    counts[2] = c.expanded; counts[3] = c.scanned_edges;
}
int64_t cq_target_missing(void* raw, const int32_t* target, int64_t size) {
    const auto& c = *static_cast<QueryContext*>(raw);
    int64_t missing = 0;
    for (int64_t i = 0; i < size; ++i) missing += !c.contains(target[i]);
    return missing;
}
void cq_boundary_work(void* raw, int64_t* counts) {
    const auto& c = *static_cast<QueryContext*>(raw);
    counts[0] = c.frontier_nodes;
    counts[1] = c.original_edges_scanned;
    counts[2] = c.expanded;
    counts[3] = c.scanned_edges;
}
void cq_measure_pair(void* exact_raw, void* part_raw, const int32_t* q,
                     int64_t size, int radius, int repeats, int inner,
                     uint64_t seed, double* timings) {
    auto* exact = static_cast<QueryContext*>(exact_raw);
    auto* part = static_cast<QueryContext*>(part_raw);
    auto time_one = [&](QueryContext* c) {
        const auto start = std::chrono::steady_clock::now();
        for (int j = 0; j < inner; ++j) c->run(q, size, radius);
        return std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now() - start).count() / inner;
    };
    // Randomized AB/BA order in each paired repetition; one explicit warm-up each.
    exact->run(q, size, radius);
    part->run(q, size, radius);
    for (int i = 0; i < repeats; ++i) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        if (seed & 1) {
            timings[2 * i] = time_one(exact);
            timings[2 * i + 1] = time_one(part);
        } else {
            timings[2 * i + 1] = time_one(part);
            timings[2 * i] = time_one(exact);
        }
    }
}
void cq_measure_three(void* exact_raw, void* coarse_raw, void* fine_raw,
                      const int32_t* q, int64_t size, int radius,
                      int repeats, int inner, uint64_t seed, double* timings) {
    QueryContext* contexts[3] = {static_cast<QueryContext*>(exact_raw),
        static_cast<QueryContext*>(coarse_raw), static_cast<QueryContext*>(fine_raw)};
    auto next = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    for (auto* c : contexts) c->run(q, size, radius);
    for (int rep = 0; rep < repeats; ++rep) {
        int order[3] = {0, 1, 2};
        std::swap(order[2], order[next() % 3]);
        std::swap(order[1], order[next() % 2]);
        for (int k : order) {
            auto* c = contexts[k];
            const auto start = std::chrono::steady_clock::now();
            for (int j = 0; j < inner; ++j) {
                c->run(q, size, radius);
                // Prevent the compiler from eliding repeated result writes.
                asm volatile("" : : "g"(c->output.data()), "g"(c->output_size) : "memory");
            }
            timings[3 * rep + k] = std::chrono::duration<double, std::nano>(
                std::chrono::steady_clock::now() - start).count() / inner;
        }
    }
}
}
