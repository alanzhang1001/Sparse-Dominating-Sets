/**
 * Neighborhood partitioning algorithms.
 */

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <queue>
#include <random>

#include "../graph/CIntGraph.hpp"
#include "../graph/c_graph_ops.hpp"
#include "c_flow.hpp"
#include "c_partition.hpp"

namespace sparsedomsets {
namespace algorithms {

template <typename K, typename V>
static inline bool contains(std::unordered_map<K, V> const& m, K const& k) {
  return m.find(k) != m.end();
}

namespace partition {
enum class Strategy : uint32_t {
  WEIGHT_GREEDY = 1,
  LAYER_GREEDY = 2,
  BRANCH_EXACT = 3,
  RANDOMIZE = 4,
};
}

template <typename Graph>
static std::map<int, std::vector<int>> neighborhood_partition_weight_greedy(  //
    Graph const& G,                                                           //
    std::vector<int> const& landmarks,                                        //
    uint32_t strategy,                                                        //
    unsigned int seed                                                         //
) {
  // [1] Create the compact neighborhood kernel
  int n = G.number_of_nodes();
  auto p = compact_neighborhood_kernel(G, landmarks);
  auto& H = p.first;            // directed graph
  auto& contracted = p.second;  // vertex |-> list of contracted vertices

  // [2] Find level-ordering
  std::vector<int> vs = bfs_ordering(G, landmarks);  // vertices in level order (should be sorted by the original distance)

  // [3] Run greedy strategy
  std::vector<int> counter(n, 0);
  std::vector<int> assigned(n, -1);  // vertex |-> landmark; -1: unassined

  for (auto v : vs) {
    if (assigned[v] != -1) continue;  // already assigned

    int best_landmark = -1;
    int least_weight = n;

    // find a parent with the least weight
    for (auto w : H.in_neighbors_sorted(v)) {
      int u = assigned[w];
      int x = counter[u];
      if (x < least_weight) {
        best_landmark = u;
        least_weight = x;
      }
    }
    if (best_landmark < 0) best_landmark = v;

    // assign v
    for (auto w : contracted[v]) assigned[w] = best_landmark;
    counter[best_landmark] += contracted[v].size();
  }

  // [4] Aggregate result
  std::map<int, std::vector<int>> ret;
  for (int i = 0; i < n; ++i) {
    if (assigned[i] >= 0) ret[assigned[i]].push_back(i);
  }
  return ret;
}

/**
 * Solves the best-balanced assignment problem.
 * @return list of task -> agent; -1: unassinged
 */
std::vector<int> balanced_assignment(                  //
    int num_tasks,                                     //
    int num_agents,                                    //
    std::vector<std::vector<int>> const& choices,      // task -> list of agents
    std::vector<long long> const& num_exclusive_tasks  // agent -> number of exclusive tasks
) {
  std::vector<int> out_degrees(num_agents);  // out-degree at each agent

  // convert to a flow problem
  std::vector<flow::IntFlowCost> edges;
  int s = num_tasks + num_agents;
  int t = s + 1;

  for (int v = 0; v < num_tasks; ++v) {
    // tasks ---> sink
    edges.push_back({v, t, 1, 0});  // capacity=1, cost=0

    // agents ---> tasks
    for (auto u : choices[v]) {
      edges.push_back({num_tasks + u, v, 1, 0});  // capacity=1, cost=0
      ++out_degrees[u];
    }
  }

  for (int u = 0; u < num_agents; ++u) {
    // source ---> agents
    for (long long k = 0; k < out_degrees[u]; ++k) {
      edges.push_back({s, num_tasks + u, 1, 2 * (num_exclusive_tasks[u] + k) + 1});  // cost=2(w+k)+1
    }
  }

  auto flow_ret = flow::min_cost_max_flow(t + 1, edges, s, t);
  std::vector<int> ret(num_tasks, -1);
  for (auto flow : flow_ret.flow) {
    if (flow.to < num_tasks && flow.weight == 1) { ret[flow.to] = flow.from_ - num_tasks; }
  }
  return ret;
}

template <typename Graph>
static std::map<int, std::vector<int>> neighborhood_partition_layer_greedy(  //
    Graph const& G,                                                          //
    std::vector<int> const& landmarks,                                       //
    uint32_t strategy,                                                       //
    unsigned int seed                                                        //
) {
  // [1] Create the (not compact) neighborhood kernel
  int n = G.number_of_nodes();
  auto H = neighborhood_kernel(G, landmarks);

  // [2] Find level-layering
  auto layers = bfs_layering(G, landmarks);

  // [3] Run greedy strategy
  std::vector<int> assigned(n, -1);     // vertex |-> landmark; -1: unassined
  std::map<int, std::vector<int>> ret;  // landmark |-> list of vertices
  for (auto u : landmarks) {
    assigned[u] = u;
    ret[u].push_back(u);
  }

  for (int level = 1; level < layers.size(); ++level) {
    // create an assignment problem instance for this level
    std::vector<std::vector<int>> target_choices;
    std::vector<long long> target_count;
    std::vector<int> target_vertex_table;
    // std::unordered_map<int, int> target_vertex_table_rev;
    std::vector<int> target_landmark_table;
    std::unordered_map<int, int> target_landmark_table_rev;

    for (auto v : layers[level]) {
      std::unordered_set<int> choices;
      for (auto w : H.in_neighbors_sorted(v)) choices.insert(assigned[w]);
      if (choices.size() == 1) {  // no choice
        auto u = *choices.begin();
        assigned[v] = u;
        ret[u].push_back(v);
      } else {
        // register vertex v
        auto vertex_index = target_vertex_table.size();
        target_vertex_table.push_back(v);
        // target_vertex_table_rev[v] = vertex_index;
        target_choices.push_back({});

        for (auto u : choices) {
          // register landmark u
          auto landmark_index = -1;
          if (contains(target_landmark_table_rev, u)) {
            landmark_index = target_landmark_table_rev[u];
          } else {
            landmark_index = target_landmark_table.size();
            target_landmark_table.push_back(u);
            target_landmark_table_rev[u] = landmark_index;
          }
          target_choices[vertex_index].push_back(landmark_index);
        }
      }
    }

    for (int i = 0; i < target_landmark_table.size(); ++i) {
      target_count.push_back(ret[target_landmark_table[i]].size());
    }

    // run balanced assignment solver
    auto assign = balanced_assignment(target_vertex_table.size(), target_landmark_table.size(), target_choices, target_count);

    // apply level-wise results
    for (int i = 0; i < target_vertex_table.size(); ++i) {
      auto v = target_vertex_table[i];
      auto u = target_landmark_table[assign[i]];
      assigned[v] = u;
      ret[u].push_back(v);
    }
  }

  return ret;
}

template <typename Graph>
static std::map<int, std::vector<int>> neighborhood_partition_branching(  //
    Graph const& G,                                                       //
    std::vector<int> const& landmarks,                                    //
    uint32_t strategy,                                                    //
    unsigned int seed                                                     //
) {
  // [1] Create the compact neighborhood kernel
  int n = G.number_of_nodes();
  auto p = compact_neighborhood_kernel(G, landmarks);
  auto& H = p.first;            // directed graph
  auto& contracted = p.second;  // vertex |-> list of contracted vertices

  // [2] Find level-ordering
  std::vector<int> vs = bfs_ordering(G, landmarks);  // vertices in level order (should be sorted by the original distance)
  std::vector<int> internal_nodes, leaf_nodes;
  for (auto v : vs) {
    if (H.in_degree(v) == 0) {
      // landmarks
    } else if (contracted[v].size() == 1 && H.out_degree(v) == 0) {
      leaf_nodes.push_back(v);  // unit-weight leaf
    } else {
      internal_nodes.push_back(v);
    }
  }

  // [3] Prepare for the flow formulation (at this point, leaves are already fixed)
  std::vector<int> target_vertex_table;
  std::vector<int> target_landmark_table;
  std::unordered_map<int, int> target_landmark_table_rev;

  for (auto v : leaf_nodes) {
    auto vertex_index = target_vertex_table.size();
    target_vertex_table.push_back(v);
  }
  for (auto u : landmarks) {
    auto landmark_index = target_landmark_table.size();
    target_landmark_table.push_back(u);
    target_landmark_table_rev[u] = landmark_index;
  }

  // [4] Branch and bound

  // search context
  long long best = static_cast<long long>(n) * n + 1;
  std::vector<int> best_assigned(n);
  std::vector<int> counter(n, 0);
  std::vector<int> assigned(n, -1);
  long long num_assigned = 0;

  long long sofar = 0;
  std::vector<std::pair<bool, std::pair<int, int>>> stack;

  // process landmarks
  for (auto u : landmarks) {
    counter[u] += contracted[u].size();
    num_assigned += contracted[u].size();
    assigned[u] = u;
    sofar += contracted[u].size() * contracted[u].size();
  }

  // initial branching
  if (internal_nodes.empty()) {
    // immediately obtain the optimum
    stack.push_back({true, {0, -1}});
  } else {
    std::unordered_set<int> choices;
    for (auto w : H.in_neighbors(internal_nodes[0])) {
      if (assigned[w] < 0) abort();
      choices.insert(assigned[w]);
    }
    for (auto u : choices) {
      stack.push_back({false, {0, u}});
      stack.push_back({true, {0, u}});
    }
  }

  while (!stack.empty()) {
    auto p = stack.back();
    stack.pop_back();

    // printf("[%c:index=%d/%d,v=%d,assign=%d] sofar=%lld, best=%lld\n", p.first ? 'T' : 'F', p.second.first, internal_nodes.size(),
    //        p.second.first == internal_nodes.size() ? -1 : internal_nodes[p.second.first], p.second.second, sofar, best);

    if (p.first && p.second.first == internal_nodes.size()) {
      //----------------------------------------
      //    base case
      //----------------------------------------
      std::vector<long long> target_count;
      for (int ii = 0; ii < landmarks.size(); ++ii) { target_count.push_back(counter[target_landmark_table[ii]]); }

      std::vector<std::vector<int>> target_choices(leaf_nodes.size());
      for (int vv = 0; vv < leaf_nodes.size(); ++vv) {
        std::unordered_set<int> choices;
        for (auto w : H.in_neighbors(target_vertex_table[vv])) choices.insert(assigned[w]);
        for (auto u : choices) { target_choices[vv].push_back(target_landmark_table_rev[u]); }
      }

      // run balanced assignment solver
      auto assign = balanced_assignment(leaf_nodes.size(), landmarks.size(), target_choices, target_count);

      // apply results
      for (int i = 0; i < leaf_nodes.size(); ++i) {
        auto v = target_vertex_table[i];
        auto u = target_landmark_table[assign[i]];
        assigned[v] = u;  // this element might be dirty but no need to clean
        ++target_count[assign[i]];
      }

      long long result = 0;
      for (int j = 0; j < landmarks.size(); ++j) { result += target_count[j] * target_count[j]; }

      if (result < best) {
        best = result;
        best_assigned = assigned;  // copy all elements
      }
      continue;
    }

    auto v = internal_nodes[p.second.first];
    auto u = p.second.second;            // landmark
    long long x = contracted[v].size();  // weight of this node
    long long y = counter[u];            // current part size

    if (p.first) {
      // forward
      sofar += x * (2 * y + x);     // update value
      counter[u] += x;
      assigned[v] = u;
      num_assigned += x;

      long long remain = n - num_assigned;
      // branch cut
      if (sofar + remain * remain / landmarks.size() >= best) continue;

      int nxt = p.second.first + 1;
      if (nxt == internal_nodes.size()) {
        stack.push_back({true, {nxt, -1}});
      } else {
        // branching
        std::unordered_set<int> choices;
        for (auto w : H.in_neighbors(internal_nodes[nxt])) {
          if (assigned[w] < 0) abort();
          choices.insert(assigned[w]);
        }
        for (auto uu : choices) {
          stack.push_back({false, {nxt, uu}});
          stack.push_back({true, {nxt, uu}});
        }
      }
    } else {
      // traceback
      sofar -= x * (2 * y - x);
      counter[u] -= x;
      assigned[v] = -1;
      num_assigned -= x;
    }
  }

  // [5] Aggregate result
  std::map<int, std::vector<int>> ret;
  for (int i = 0; i < n; ++i) {
    if (best_assigned[i] >= 0) {
      for (auto w : contracted[i]) ret[best_assigned[i]].push_back(w);
    }
  }
  return ret;
}

template <typename Graph>
std::map<int, std::vector<int>> neighborhood_partition(  //
    Graph const& G,                                      //
    std::vector<int> const& landmarks,                   //
    uint32_t strategy,                                   //
    unsigned int seed                                    //
) {
  switch (static_cast<uint32_t>(strategy) & 3) {
    case static_cast<uint32_t>(partition::Strategy::WEIGHT_GREEDY):
      return neighborhood_partition_weight_greedy(G, landmarks, strategy, seed);
    case static_cast<uint32_t>(partition::Strategy::LAYER_GREEDY):
      return neighborhood_partition_layer_greedy(G, landmarks, strategy, seed);
    case static_cast<uint32_t>(partition::Strategy::BRANCH_EXACT):
      return neighborhood_partition_branching(G, landmarks, strategy, seed);
    default:  // not yet implemented
      abort();
  }
}

// explicit instantiation
template std::map<int, std::vector<int>> neighborhood_partition<graph::CIntGraph>(  //
    graph::CIntGraph const&,                                                        //
    std::vector<int> const&,                                                        //
    uint32_t, unsigned int                                                          //
);

}  // namespace algorithms
}  // namespace sparsedomsets
