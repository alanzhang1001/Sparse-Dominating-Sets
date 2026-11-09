#include "CIntGraph.hpp"
#include <algorithm>
#include <cassert>

namespace sparsedomsets {
namespace graph {

CIntGraph::CIntGraph() {}

CIntGraph::CIntGraph(int n) { edges_.resize(n); }

void CIntGraph::add_edge(int u, int v) {
  assert(0 <= u && u < number_of_nodes());
  assert(0 <= v && v < number_of_nodes());
  assert(u != v);

  // never checks if the given edge already exists
  edges_[u].push_back(v);
  edges_[v].push_back(u);
}

int CIntGraph::number_of_nodes() const { return edges_.size(); }

long long CIntGraph::number_of_edges() const {
  long long ret = 0;
  for (int i = 0; i < number_of_nodes(); ++i) ret += edges_[i].size();
  return ret / 2;
}

std::vector<int> const& CIntGraph::neighbors(int v) const {
  assert(0 <= v && v < number_of_nodes());

  return edges_[v];
}

std::vector<int> CIntGraph::neighbors_sorted(int v) const {
  assert(0 <= v && v < number_of_nodes());

  std::vector<int> ret(edges_[v].begin(), edges_[v].end());
  std::sort(ret.begin(), ret.end());
  return ret;
}

// std::vector<int> CIntGraph::closed_neighborhood(int v) const {
//   assert(0 <= v && v < number_of_nodes());

//   std::vector<int> ret(edges_[v].begin(), edges_[v].end());
//   ret.push_back(v);  // include v itself
//   return ret;
// }

// std::vector<int> CIntGraph::closed_neighborhood_sorted(int v) const {
//   assert(0 <= v && v < number_of_nodes());

//   std::vector<int> ret(edges_[v].begin(), edges_[v].end());
//   ret.push_back(v);  // include v itself
//   std::sort(ret.begin(), ret.end());
//   return ret;
// }

int CIntGraph::degree(int v) const { return edges_[v].size(); }

}  // namespace graph
}  // namespace sparsedomsets