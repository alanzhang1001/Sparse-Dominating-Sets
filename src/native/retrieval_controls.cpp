// E5 controls and exact refinement. The paper's selectors are not modified.
#include "query_engine.cpp"
#include <numeric>

struct Refinement {
    QueryContext* candidate;
    std::vector<uint32_t> seen;
    std::vector<int32_t> queue, output;
    uint32_t generation = 0;
    int64_t size = 0;
    explicit Refinement(QueryContext* c) : candidate(c), seen(c->seen.size(), 0),
        queue(c->seen.size()), output(c->seen.size()) {}
    void run(const int32_t* seeds, int64_t count, int radius) {
        if (++generation == 0) { std::fill(seen.begin(), seen.end(), 0); generation = 1; }
        size = 0;
        for (int64_t i = 0; i < count; ++i) {
            int32_t v = seeds[i];
            if (seen[v] == generation || !candidate->contains(v)) continue;
            seen[v] = generation; queue[size++] = v;
        }
        int64_t begin = 0, end = size;
        for (int depth = 0; depth < radius && begin < end; ++depth) {
            for (int64_t pos = begin; pos < end; ++pos) {
                int32_t v = queue[pos];
                for (int64_t j = candidate->indptr[v]; j < candidate->indptr[v+1]; ++j) {
                    int32_t u = candidate->indices[j];
                    if (seen[u] == generation || !candidate->contains(u)) continue;
                    seen[u] = generation; queue[size++] = u;
                }
            }
            begin = end; end = size;
        }
        std::memcpy(output.data(), queue.data(), size * sizeof(int32_t));
    }
};

extern "C" {
int64_t ex_random_prune(const int64_t* ptr, const int32_t* indices, int64_t n,
                        const int32_t* permutation, int32_t* result) {
    std::vector<int64_t> coverage(n);
    std::vector<uint8_t> keep(n, 1);
    for (int64_t v = 0; v < n; ++v) coverage[v] = ptr[v+1]-ptr[v]+1;
    for (int64_t i = 0; i < n; ++i) {
        int32_t v = permutation[i];
        bool redundant = coverage[v] > 1;
        for (int64_t j = ptr[v]; redundant && j < ptr[v+1]; ++j)
            redundant = coverage[indices[j]] > 1;
        if (!redundant) continue;
        keep[v] = 0; --coverage[v];
        for (int64_t j = ptr[v]; j < ptr[v+1]; ++j) --coverage[indices[j]];
    }
    int64_t size = 0;
    for (int64_t v = 0; v < n; ++v) if (keep[v]) result[size++] = v;
    return size;
}
int64_t ex_random_cover(const int64_t* ptr, const int32_t* indices, int64_t n,
                        const int32_t* permutation, int32_t* result) {
    std::vector<uint8_t> covered(n, 0);
    int64_t size = 0;
    for (int64_t i = 0; i < n; ++i) {
        int32_t v = permutation[i];
        if (covered[v]) continue;
        result[size++] = v; covered[v] = 1;
        for (int64_t j = ptr[v]; j < ptr[v+1]; ++j) covered[indices[j]] = 1;
    }
    return size;
}
int64_t ex_covered(const int64_t* ptr, const int32_t* indices, int64_t n,
                   const int32_t* landmarks, int64_t k) {
    std::vector<uint8_t> covered(n, 0);
    for (int64_t i = 0; i < k; ++i) {
        int32_t v = landmarks[i]; covered[v] = 1;
        for (int64_t j = ptr[v]; j < ptr[v+1]; ++j) covered[indices[j]] = 1;
    }
    return std::accumulate(covered.begin(), covered.end(), int64_t(0));
}
// Check connected pieces and exact nearest-landmark distances independently.
int64_t ex_validate_partition(const int64_t* ptr, const int32_t* indices, int64_t n,
    const int32_t* labels, int64_t k, const int32_t* owner) {
    std::vector<int32_t> global(n, -1), assigned(n, -1), queue(n);
    auto bfs = [&](std::vector<int32_t>& distance, bool constrained) {
        int64_t head = 0, tail = k;
        for (int64_t i = 0; i < k; ++i) { queue[i] = labels[i]; distance[labels[i]] = 0; }
        while (head < tail) {
            int32_t v = queue[head++];
            for (int64_t j = ptr[v]; j < ptr[v+1]; ++j) {
                int32_t u = indices[j];
                if (distance[u] >= 0 || (constrained && owner[u] != owner[v])) continue;
                distance[u] = distance[v] + 1; queue[tail++] = u;
            }
        }
        return tail;
    };
    if (bfs(global, false) != n || bfs(assigned, true) != n) return -1;
    int64_t mismatch = 0;
    for (int64_t v = 0; v < n; ++v) mismatch += global[v] != assigned[v];
    return mismatch;
}
void* ex_refine_create(void* raw) {
    try { return new Refinement(static_cast<QueryContext*>(raw)); } catch (...) { return nullptr; }
}
void ex_refine_destroy(void* raw) { delete static_cast<Refinement*>(raw); }
int64_t ex_refine_run(void* raw, const int32_t* q, int64_t size, int radius) {
    auto* r = static_cast<Refinement*>(raw); r->run(q, size, radius); return r->size;
}
const int32_t* ex_refine_output(void* raw) { return static_cast<Refinement*>(raw)->output.data(); }
// Independent batch timings: direct BFS, candidates, restricted refinement,
// restricted pipeline, and the pipeline using optimized original-graph BFS.
// Separate clock pairs avoid inflating sub-microsecond stages by per-stage timers.
void ex_measure(void* exact_raw, void* candidate_raw, void* refine_raw,
                const int32_t* q, int64_t size, int radius, int repeats,
                int inner, uint64_t seed, double* timings) {
    auto* exact = static_cast<QueryContext*>(exact_raw);
    auto* candidate = static_cast<QueryContext*>(candidate_raw);
    auto* refine = static_cast<Refinement*>(refine_raw);
    auto run = [&](int mode) {
        if (mode == 0) exact->run(q, size, radius);
        else if (mode == 1) candidate->run(q, size, radius);
        else if (mode == 2) refine->run(q, size, radius);
        else if (mode == 3) { candidate->run(q, size, radius); refine->run(q, size, radius); }
        else { candidate->run(q, size, radius); exact->run(q, size, radius); }
        asm volatile("" : : "g"(exact->output.data()), "g"(candidate->output.data()),
            "g"(refine->output.data()), "g"(refine->size) : "memory");
    };
    candidate->run(q, size, radius); exact->run(q, size, radius); refine->run(q, size, radius);
    auto next = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    for (int rep = 0; rep < repeats; ++rep) {
        int order[5] = {0,1,2,3,4};
        for (int i = 4; i > 0; --i) std::swap(order[i], order[next() % (i+1)]);
        for (int mode : order) {
            const auto start = std::chrono::steady_clock::now();
            for (int j = 0; j < inner; ++j) run(mode);
            timings[5*rep+mode] = std::chrono::duration<double, std::nano>(
                std::chrono::steady_clock::now()-start).count()/inner;
        }
    }
}
}
