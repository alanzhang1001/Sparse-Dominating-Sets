// Counter-based implementation of the manuscript's Dom-Ratio update rule.
// An indexed heap stores at most n entries. Each newly covered x decrements
// the gain of u in N_r[x] exactly once (the input graph is undirected).
#include "csr_input.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

class IndexedHeap {
    std::vector<int> heap, position;
    const std::vector<int>& volume;
    std::vector<int>& gain;
    int strategy;
    bool better(int a, int b) const {
        auto ratio = [&]() {
            const __int128 left = static_cast<__int128>(gain[a]) * volume[b];
            const __int128 right = static_cast<__int128>(gain[b]) * volume[a];
            return left == right ? 0 : (left > right ? 1 : -1);
        };
        for (int metric : {strategy & 7, (strategy >> 3) & 7}) {
            int c = 0;
            if (metric == 1) c = gain[a] == gain[b] ? 0 : (gain[a] > gain[b] ? 1 : -1);
            if (metric == 2) c = ratio();
            if (c) return c > 0;
        }
        return a < b; // Same final tie rule as the published reference.
    }
    void swap_at(int a, int b) {
        std::swap(heap[a], heap[b]);
        position[heap[a]] = a; position[heap[b]] = b;
    }
    void down(int i) {
        for (;;) {
            int b = i, l = 2*i+1, r = l+1;
            if (l < static_cast<int>(heap.size()) && better(heap[l], heap[b])) b = l;
            if (r < static_cast<int>(heap.size()) && better(heap[r], heap[b])) b = r;
            if (b == i) break;
            swap_at(i, b); i = b;
        }
    }
public:
    IndexedHeap(const std::vector<int>& v, std::vector<int>& g, int s)
        : heap(v.size()), position(v.size()), volume(v), gain(g), strategy(s) {
        std::iota(heap.begin(), heap.end(), 0);
        std::iota(position.begin(), position.end(), 0);
        for (int i = static_cast<int>(heap.size())/2-1; i >= 0; --i) down(i);
    }
    bool empty() const { return heap.empty(); }
    int pop() {
        int v = heap.front();
        swap_at(0, static_cast<int>(heap.size())-1);
        heap.pop_back(); position[v] = -1;
        if (!heap.empty()) down(0);
        return v;
    }
    void decrease(int v) {
        if (position[v] < 0) return;
        --gain[v]; down(position[v]);
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: counter_heap GRAPH R STRATEGY OUTPUT MAX_NEIGHBORHOOD_ENTRIES");
        const int radius = std::stoi(argv[2]), strategy = std::stoi(argv[3]);
        const long long limit = std::stoll(argv[5]);
        if (radius < 1 || limit < 1 || (strategy != 1 && strategy != 2 && strategy != 10 && strategy != 17))
            throw std::runtime_error("invalid radius, strategy or resource limit");
        auto all_started = Clock::now(), phase = all_started;
        int n;
        std::vector<std::vector<int>> adj;
        if (std::filesystem::is_directory(argv[1])) {
            adj = read_csr_adj(argv[1]); n = adj.size();
        } else {
        std::ifstream input(argv[1]); int u, v;
        if (!(input >> n) || n <= 0) throw std::runtime_error("invalid graph header");
        adj.resize(n);
        while (input >> u >> v) {
            if (u < 0 || u >= n || v < 0 || v >= n || u == v) throw std::runtime_error("invalid edge");
            adj[u].push_back(v);
        }
        for (auto& row : adj) {
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }
        for (u=0; u<n; ++u) for (int w : adj[u])
            if (!std::binary_search(adj[w].begin(), adj[w].end(), u)) throw std::runtime_error("graph must contain both edge directions");
        }
        int u;
        const double read_seconds = elapsed(phase); phase = Clock::now();
        std::vector<std::vector<int>> neighborhoods(n);
        std::vector<int> seen(n, -1), distance(n), queue(n), volume(n);
        long long entries = 0;
        int maximum_volume = 0;
        for (int source = 0; source < n; ++source) {
            int head = 0, tail = 1;
            queue[0] = source; seen[source] = source; distance[source] = 0;
            while (head < tail) {
                int x = queue[head++];
                if (distance[x] == radius) continue;
                for (int y : adj[x]) if (seen[y] != source) {
                    seen[y] = source; distance[y] = distance[x]+1; queue[tail++] = y;
                }
            }
            entries += tail;
            if (entries > limit) throw std::runtime_error("neighborhood entry limit exceeded");
            neighborhoods[source].assign(queue.begin(), queue.begin()+tail);
            volume[source] = tail; maximum_volume = std::max(maximum_volume, tail);
        }
        std::vector<int> gain = volume;
        const double preprocessing_seconds = elapsed(phase); phase = Clock::now();
        IndexedHeap heap(volume, gain, strategy);
        std::vector<unsigned char> covered(n, 0);
        std::vector<int> selected;
        int remaining = n; long long updates = 0, cost = 0;
        while (remaining) {
            if (heap.empty()) throw std::runtime_error("no covering candidate");
            int pick = heap.pop();
            if (gain[pick] <= 0) throw std::runtime_error("zero-gain selection");
            selected.push_back(pick); cost += volume[pick];
            for (int x : neighborhoods[pick]) if (!covered[x]) {
                covered[x] = 1; --remaining;
                for (int candidate : neighborhoods[x]) { heap.decrease(candidate); ++updates; }
            }
        }
        const double greedy_seconds = elapsed(phase); phase = Clock::now();
        std::ofstream output(argv[4]);
        if (!output) throw std::runtime_error("cannot open solution file");
        for (int x : selected) output << x << '\n';
        output.close();
        const double write_seconds = elapsed(phase);
        std::cout << std::setprecision(17)
          << "{\"implementation\":\"counter-indexed-heap-v1\",\"n\":" << n
          << ",\"radius\":" << radius << ",\"strategy\":" << strategy
          << ",\"total_cost\":" << cost << ",\"selected\":" << selected.size()
          << ",\"neighborhood_entries\":" << entries << ",\"maximum_neighborhood_size\":" << maximum_volume
          << ",\"gain_update_visits\":" << updates << ",\"heap_capacity\":" << n
          << ",\"read_seconds\":" << read_seconds << ",\"preprocessing_seconds\":" << preprocessing_seconds
          << ",\"greedy_seconds\":" << greedy_seconds << ",\"write_seconds\":" << write_seconds
          << ",\"internal_total_seconds\":" << elapsed(all_started) << "}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
    return 0;
}
