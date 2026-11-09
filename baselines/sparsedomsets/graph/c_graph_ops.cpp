/**
 * Graph Operations.
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <queue>

#include "c_graph_ops.hpp"

#ifdef USE_SLOW_IMPLEMENTATION
#include "CIntGraphSlow.hpp"
#endif

#include "CIntDiGraph.hpp"
#include "CIntGraph.hpp"

namespace sparsedomsets {
namespace graph {

template <typename T>
static inline bool contains(std::unordered_set<T> const& s, T const& x) {
  return s.find(x) != s.end();
}

template <typename Graph>
std::unordered_set<int> get_closed_neighborhood(Graph const& G, int v, int radius) {
  assert(0 <= v && v < G.number_of_nodes());
  assert(radius >= 0);
  typedef std::pair<int, int> PI;

  std::unordered_set<int> visited;
  std::queue<PI> q;
  q.push(std::make_pair(radius, v));

  // run BFS
  while (!q.empty()) {
    auto p = q.front();
    q.pop();
    auto x = p.first;
    auto y = p.second;

    if (contains(visited, y)) continue;
    visited.insert(y);
    if (x == 0) continue;

    for (auto u : G.neighbors(y)) {
      if (!contains(visited, u)) {
        // advance the frontier
        q.push(std::make_pair(x - 1, u));
      }
    }
  }
  return visited;
}

static inline int encode_distance(int start, int distance) { return ((start + 1) << 4) + distance; }

static inline int decode_distance(int x) { return x & ((1 << 4) - 1); }

static inline int node_visited(int x, int v) { return (x >> 4) == v + 1; }

template <typename Graph>
static void compute_all_closed_neighborhood_impl(Graph const& G, std::vector<std::vector<int>>& cnbrs, int v,
                                                 int radius, int buffer[]) {
  assert(0 <= v && v < G.number_of_nodes());
  assert(radius >= 0);

  // limitations
  if (G.number_of_nodes() > 100000000) abort();
  if (radius > 15) abort();

  std::queue<int> q;
  q.push(v);
  buffer[v] = encode_distance(v, 0);

  // run BFS
  while (!q.empty()) {
    auto parent = q.front();
    q.pop();

    for (auto u : G.neighbors(parent)) {
      if (!node_visited(buffer[u], v)) {
        // advance the frontier
        auto distance = decode_distance(buffer[parent]) + 1;
        buffer[u] = encode_distance(v, distance);

        // do not call `add_edge()` as it requires inter-thread locks
        cnbrs[v].push_back(u);
        if (distance < radius) q.push(u);
      }
    }
  }
}

template <typename Graph>
static void compute_congestion_sum_impl(Graph const& G, long long& count, int v, int radius, int buffer[]) {
  assert(0 <= v && v < G.number_of_nodes());
  assert(radius >= 0);

  // limitations
  if (G.number_of_nodes() > 100000000) abort();
  if (radius > 15) abort();

  std::queue<int> q;
  q.push(v);
  buffer[v] = encode_distance(v, 0);

  // run BFS
  while (!q.empty()) {
    auto parent = q.front();
    q.pop();

    for (auto u : G.neighbors(parent)) {
      if (!node_visited(buffer[u], v)) {
        // advance the frontier
        auto distance = decode_distance(buffer[parent]) + 1;
        buffer[u] = encode_distance(v, distance);

        // do not call `add_edge()` as it requires inter-thread locks
        ++count;
        if (distance < radius) q.push(u);
      }
    }
  }
}

#ifdef USE_SLOW_IMPLEMENTATION
template <typename Graph>
Graph distance_closure(Graph const& G, int radius, int num_threads) {
  assert(radius >= 0);
  typedef std::pair<int, int> PI;

  // special cases
  if (radius == 1) return G;

  auto n = G.number_of_nodes();
  Graph H(n);
  if (radius == 0) return H;

  // main logic
  if (num_threads <= 1) {
    // single-threaded
    for (int i = 0; i < n; ++i) {
      for (auto j : get_closed_neighborhood(G, i, radius)) {
        if (i < j) { H.add_edge(i, j); }
      }
    }
  } else {
    // multi-threaded
    std::vector<PI>* results[num_threads];
    for (int i = 0; i < num_threads; ++i) { results[i] = new std::vector<PI>; }

#pragma omp parallel for schedule(auto) num_threads(num_threads)
    for (int i = 0; i < n; ++i) {
      for (auto j : get_closed_neighborhood(G, i, radius)) {
        if (i < j) { results[omp_get_thread_num()]->push_back(std::make_pair(i, j)); }
      }
    }

    // collect results
    for (int i = 0; i < num_threads; ++i) {
      for (auto& p : *results[i]) { H.add_edge(p.first, p.second); }
      delete results[i];
    }
  }
  return H;
}
#endif

template <typename Graph>
Graph distance_closure(Graph const& G, int radius, int num_threads) {
  assert(radius >= 0);
  assert(num_threads >= 1);

  // special cases
  if (radius == 1) return G;

  auto n = G.number_of_nodes();
  Graph H(n);
  if (radius == 0) return H;

#pragma omp parallel num_threads(num_threads)
  {
    // create thread-local buffer
    int* buffer = (int*)std::calloc(n, sizeof(int));

    // main logic
#pragma omp for schedule(auto)
    for (int i = 0; i < n; ++i) { compute_all_closed_neighborhood_impl(G, H.edges_, i, radius, buffer); }

    // free buffer
    std::free(buffer);
  }

  return H;
}

template <typename Graph>
std::vector<std::vector<int>> compute_all_closed_neighborhood(Graph const& G, int radius, int num_threads) {
  assert(radius >= 0);
  assert(num_threads >= 1);

  auto n = G.number_of_nodes();

  std::vector<std::vector<int>> ret(n);

  // add vertices themselves
  for (int i = 0; i < n; ++i) ret[i].push_back(i);
  if (radius == 0) return ret;

#pragma omp parallel num_threads(num_threads)
  {
    // create thread-local buffer
    int* buffer = (int*)std::calloc(n, sizeof(int));

    // main logic
#pragma omp for schedule(auto)
    for (int i = 0; i < n; ++i) { compute_all_closed_neighborhood_impl(G, ret, i, radius, buffer); }

    // free buffer
    std::free(buffer);
  }

  return ret;
}

template <typename Graph>
long long compute_congestion_sum(Graph const& G, std::vector<int> const& landmarks, int radius, int num_threads) {
  assert(radius >= 0);
  assert(num_threads >= 1);

  auto n = G.number_of_nodes();

  long long ret = landmarks.size();
  if (radius == 0) return ret;

  std::vector<long long> count(num_threads);

#pragma omp parallel num_threads(num_threads)
  {
    // create thread-local buffer
    int* buffer = (int*)std::calloc(n, sizeof(int));

    // main logic
#pragma omp for schedule(auto)
    for (auto u : landmarks) { compute_congestion_sum_impl(G, count[omp_get_thread_num()], u, radius, buffer); }

    // free buffer
    std::free(buffer);
  }

  // accumulate counts
  for (auto c : count) ret += c;
  return ret;
}

template <typename Graph>
double compute_average_congestion(Graph const& G, std::vector<int> const& landmarks, int radius, int num_threads) {
  auto n = G.number_of_nodes();
  return n == 0 ? 0 : static_cast<double>(compute_congestion_sum(G, landmarks, radius, num_threads)) / n;
}

// template <typename Graph>
// std::vector<std::unordered_set<int>> compute_all_closed_neighborhood_set(Graph const& G, int radius, int num_threads) {
//   assert(radius >= 0);
//   assert(num_threads >= 1);

//   auto n = G.number_of_nodes();
//   auto vret = compute_all_closed_neighborhood(G, radius, num_threads);
//   std::vector<std::unordered_set<int>> ret(n);

// #pragma omp parallel num_threads(num_threads)
//   {
// #pragma omp for schedule(auto)
//     for (int i = 0; i < n; ++i) { ret[i] = std::unordered_set<int>(vret[i].begin(), vret[i].end()); }
//   }

//   return ret;
// }

template <typename Graph>
std::vector<int> bfs_ordering(Graph const& G, std::vector<int> const& landmarks) {
  auto n = G.number_of_nodes();

  std::vector<int> ret;
  std::vector<bool> visited(n, false);
  std::queue<int> q;

  auto f = [&ret, &visited, &q](std::vector<int> const& ws) {
    for (auto w : ws) {
      if (!visited[w]) {
        ret.push_back(w);
        visited[w] = true;
        q.push(w);
      }
    }
  };

  f(landmarks);

  while (!q.empty()) {
    auto v = q.front();
    q.pop();
    f(G.neighbors(v));
  }
  return ret;
}

template <typename Graph>
std::vector<std::vector<int>> bfs_layering(Graph const& G, std::vector<int> const& landmarks) {
  auto n = G.number_of_nodes();

  std::vector<std::vector<int>> ret(1);
  std::queue<int> q;

  std::vector<int> levels(n, -1);
  for (auto u : landmarks) {
    levels[u] = 0;
    ret[0].push_back(u);
    q.push(u);
  }

  while (!q.empty()) {
    auto v = q.front();
    q.pop();
    for (auto w : G.neighbors(v)) {
      if (levels[w] == -1) {
        levels[w] = levels[v] + 1;
        if (levels[w] == ret.size()) {
          ret.push_back({w});  // new level
        } else {
          ret[levels[w]].push_back(w);  // same level
        }
        q.push(w);
      }
    }
  }

  return ret;
}

template <typename Graph>
CIntDiGraph neighborhood_kernel(Graph const& G, std::vector<int> const& landmarks) {
  auto n = G.number_of_nodes();

  CIntDiGraph H(n);

  // [1] Find level-ordering
  std::vector<int> vs = bfs_ordering(G, landmarks);  // vertices in level order

  // [2] Compute levels
  std::vector<int> level(n, -1);          // -1: unassigned
  for (auto u : landmarks) level[u] = 0;  // landmarks

  for (auto v : vs) {
    for (auto w : G.neighbors(v)) {
      if (level[w] == -1) level[w] = level[v] + 1;
      if (level[w] == level[v] + 1) H.add_edge(v, w);  // add a directed edge
    }
  }
  return H;
}

template <typename Graph>
std::pair<CIntDiGraph, std::vector<std::vector<int>>> compact_neighborhood_kernel(Graph const& G, std::vector<int> const& landmarks) {
  auto n = G.number_of_nodes();

  CIntDiGraph U(n);
  std::vector<std::vector<int>> contracted_vertices(n);

  // [1] Find level-ordering
  std::vector<int> vs = bfs_ordering(G, landmarks);  // vertices in level order

  // [2] Identify monochromatic regions
  auto H = neighborhood_kernel(G, landmarks);
  std::vector<std::unordered_set<int>> regions(n);  // set of vertices that should have the same color as v
  std::vector<bool> removed(n, false);

  for (auto v : vs) {
    if (removed[v]) continue;

    std::queue<int> q;
    std::unordered_set<int> visited;  // visited vertices
    regions[v].insert(v);
    q.push(v);

    while (!q.empty()) {
      auto u = q.front();
      q.pop();
      for (auto w : H.out_neighbors(u)) {
        if (!contains(visited, w)) {
          visited.insert(w);
          // validate w's in-neighbors
          auto xs = H.in_neighbors(w);
          if (std::all_of(xs.begin(), xs.end(), [&](int x) { return contains(regions[v], x); })) {
            removed[w] = true;
            regions[v].insert(w);
            q.push(w);
          }
        }
      }
    }
  }

  // [3] Contract monochromatic regions
  for (auto v : vs) {
    if (removed[v]) continue;

    std::unordered_set<int> connect_to;

    for (auto u : regions[v]) {
      contracted_vertices[v].push_back(u);
      for (auto w : H.out_neighbors(u)) {
        if (!removed[w]) connect_to.insert(w);  // use set to avoid double edges
      }
    }

    for (auto w : connect_to) { U.add_edge(v, w); }
  }

  return {U, contracted_vertices};
}

// explicit instantiation
#ifdef USE_SLOW_IMPLEMENTATION
template std::unordered_set<int> get_closed_neighborhood<CIntGraphSlow>(CIntGraphSlow const&, int, int);
#endif

template std::unordered_set<int> get_closed_neighborhood<CIntGraph>(CIntGraph const&, int, int);
template CIntGraph distance_closure<CIntGraph>(CIntGraph const&, int, int);
template std::vector<std::vector<int>> compute_all_closed_neighborhood<CIntGraph>(CIntGraph const&, int, int);
template long long compute_congestion_sum<CIntGraph>(CIntGraph const&, std::vector<int> const&, int, int);
template double compute_average_congestion<CIntGraph>(CIntGraph const&, std::vector<int> const&, int, int);

// template std::vector<std::unordered_set<int>> compute_all_closed_neighborhood_set<CIntGraph>(CIntGraph const&, int, int);

template std::vector<int> bfs_ordering<CIntGraph>(CIntGraph const&, std::vector<int> const&);
template std::vector<int> bfs_ordering<CIntDiGraph>(CIntDiGraph const&, std::vector<int> const&);
template std::vector<std::vector<int>> bfs_layering<CIntGraph>(CIntGraph const&, std::vector<int> const&);
template std::vector<std::vector<int>> bfs_layering<CIntDiGraph>(CIntDiGraph const&, std::vector<int> const&);
template CIntDiGraph neighborhood_kernel<CIntGraph>(CIntGraph const&, std::vector<int> const&);
template std::pair<CIntDiGraph, std::vector<std::vector<int>>> compact_neighborhood_kernel<CIntGraph>(CIntGraph const&,
                                                                                                      std::vector<int> const&);

}  // namespace graph
}  // namespace sparsedomsets
