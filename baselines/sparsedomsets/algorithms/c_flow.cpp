#include "c_flow.hpp"
#include <queue>
#include <cstdlib>

namespace sparsedomsets {
namespace algorithms {
namespace flow {

typedef std::vector<std::vector<std::pair<VertexId, EdgeId>>> EdgeInfo;

/**
 * Dinic's algorithm.
 */
namespace dinic_impl {
/**
 * Calculates the distances from vertex s to all connected vertices.
 *
 * @param edges    adjacent list
 * @param capacity map of edge id -> capacity
 * @param used     map of edge id -> used amount
 * @param s        source vertex of the flow
 * @return vertex vector of distance from the source vertex; -1: unreachable
 */
static std::vector<int> bfs(EdgeInfo const& edges, std::vector<Weight> const& capacity, std::vector<Weight> const& used, VertexId s) {
  std::vector<int> level(edges.size(), -1);

  std::queue<VertexId> q;
  level[s] = 0;
  q.push(s);

  while (!q.empty()) {
    auto p = q.front();
    q.pop();
    for (auto& e : edges[p]) {
      if ((capacity[e.second] - used[e.second]) > 0 && level[e.first] < 0) {
        level[e.first] = level[p] + 1;
        q.push(e.first);
      }
    }
  }
  return level;
}

/**
 * Searches an increasing path.
 *
 * @param edges    adjacent list
 * @param capacity map of edge id -> capacity
 * @param used     map of edge id -> used amount
 * @param level    level data (distances from the source)
 * @param iter     iteration data
 * @param v        vertex
 * @param t        vertex
 * @param f        flow
 */
static Weight dfs(EdgeInfo const& edges, std::vector<Weight> const& capacity, std::vector<Weight>& used,
                  std::vector<int> const& level, std::vector<int>& iter, VertexId v, VertexId t, Weight f) {
  if (v == t) return f;

  for (int& i = iter[v]; i < edges[v].size(); ++i) {
    auto& e = edges[v][i];
    auto remain = capacity[e.second] - used[e.second];
    if (remain > 0 && level[v] < level[e.first]) {
      auto d = dfs(edges, capacity, used, level, iter, e.first, t, std::min(f, remain));

      if (d > 0) {
        used[e.second] += d;
        used[e.second ^ 1] -= d;
        return d;
      }
    }
  }
  return 0;
}

/**
 * Dinic's maximum flow algorithm
 *
 * @param g   edge-weighted graph
 * @param capacity map of edge id -> capacity
 * @param src source vertex of the flow
 * @param dst destination vertex of the flow
 * @return maximum flow
 */
static Weight max_flow_impl(              //
    EdgeInfo const& edges,                //
    std::vector<Weight> const& capacity,  //
    VertexId src,                         //
    VertexId dst,                         //
    Weight infinity = 1e18                //
) {
  int n = (int)edges.size();
  int m = (int)capacity.size();

  // main loop
  std::vector<Weight> used(m);
  Weight flow = 0;
  for (;;) {
    auto level = dinic_impl::bfs(edges, capacity, used, src);
    if (level[dst] < 0) return flow;

    std::vector<int> iter(n, 0);  // iteration
    for (Weight f; (f = dinic_impl::dfs(edges, capacity, used, level, iter, src, dst, infinity)) > 0;) flow += f;
  }
}

}  // namespace dinic_impl

namespace bellman_ford_impl {

/**
 * Min-cost flow algorithm.
 */
static Weight min_cost_flow_impl(     //
    EdgeInfo const& edges,            // from -> [{to, edge ID}]
    std::vector<Weight>& capacity,    // edge ID -> capacity
    std::vector<Weight> const& cost,  // edge ID -> cost
    VertexId src,                     // source vertex
    VertexId dst,                     // destination vertex
    Weight flow,                      // desired flow amount
    Weight infinity = 1e18            //
) {
  int n = (int)edges.size();
  int m = (int)capacity.size();  // number of original (forward) edges

  // main logic
  Weight ret = 0;
  std::vector<std::pair<VertexId, EdgeId>> parent(n, {-1, -1});  // previous vertex and forward edge ID

  while (flow > 0) {
    // Bellman-Ford Algorithm
    std::vector<Weight> dist(n, infinity);
    dist[src] = 0;
    bool update = true;
    while (update) {
      update = false;
      for (int v = 0; v < n; ++v)
        if (dist[v] != infinity) {
          for (auto& e : edges[v]) {
            auto k = dist[v] + cost[e.second];
            if (capacity[e.second] > 0 && dist[e.first] > k) {
              dist[e.first] = k;
              parent[e.first] = {v, e.second};
              update = true;
            }
          }
        }
    }

    if (dist[dst] == infinity) return infinity;  // unreachable

    // update the current flow and capacity
    auto incr = flow;
    int v = dst;
    std::vector<int> aug;  // augmenting path

    while (v != src) {
      incr = std::min(incr, capacity[parent[v].second]);
      aug.push_back(v);
      v = parent[v].first;
    }

    for (auto v : aug) {
      capacity[parent[v].second] -= incr;
      capacity[parent[v].second ^ 1] += incr;  // reverse edge
    }

    flow -= incr;
    ret += incr * dist[dst];
  }

  return ret;
}

}  // namespace bellman_ford_impl

ResultFlowCost min_cost_max_flow(                    //
    int n,                                           // number of vertices
    std::vector<IntFlowCost> const& weighted_edges,  // edge, capacity, cost
    VertexId src,                                    // source vertex
    VertexId dst                                     // destination vertex
) {
  int m = weighted_edges.size();
  EdgeInfo edges(n);
  std::vector<Weight> capacity, cost;

  int i = 0;
  for (auto& e : weighted_edges) {
    // to-vertex, edge ID (supports multigraphs)
    edges[e.from_].push_back({e.to, i * 2});
    capacity.push_back(e.capacity);
    cost.push_back(e.cost);

    // add reverse edges
    edges[e.to].push_back({e.from_, i * 2 + 1});  // rev[i] = i ^ 1 for any i
    capacity.push_back(0);
    cost.push_back(-e.cost);
    ++i;
  }
  auto flow = dinic_impl::max_flow_impl(edges, capacity, src, dst);
  auto min_cost = bellman_ford_impl::min_cost_flow_impl(edges, capacity, cost, src, dst, flow);

  // collect resulting flow
  std::vector<IntFlow> flow_amount;
  for (int i = 0; i < m; ++i) {
    auto e = weighted_edges[i];
    flow_amount.push_back({e.from_, e.to, e.capacity - capacity[i * 2]});
  }

  return {flow, min_cost, flow_amount};
}

}  // namespace flow
}  // namespace algorithms
}  // namespace sparsedomsets