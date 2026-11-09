#include "CIntDiGraph.hpp"
#include <algorithm>
#include <cassert>

namespace sparsedomsets {
namespace graph {

CIntDiGraph::CIntDiGraph() {}

CIntDiGraph::CIntDiGraph(int n) {
  in_edges_.resize(n);
  out_edges_.resize(n);
}

void CIntDiGraph::add_edge(int u, int v) {
  assert(0 <= u && u < number_of_nodes());
  assert(0 <= v && v < number_of_nodes());
  assert(u != v);

  // never checks if the given edge already exists
  out_edges_[u].push_back(v);
  in_edges_[v].push_back(u);
}

int CIntDiGraph::number_of_nodes() const { return out_edges_.size(); }

long long CIntDiGraph::number_of_edges() const {
  long long ret = 0;
  for (int i = 0; i < number_of_nodes(); ++i) ret += out_edges_[i].size();
  return ret;
}

std::vector<int> const& CIntDiGraph::in_neighbors(int v) const {
  assert(0 <= v && v < number_of_nodes());
  return in_edges_[v];
}

std::vector<int> const& CIntDiGraph::out_neighbors(int v) const {
  assert(0 <= v && v < number_of_nodes());
  return out_edges_[v];
}

std::vector<int> const& CIntDiGraph::neighbors(int v) const { return out_neighbors(v); }

std::vector<int> CIntDiGraph::in_neighbors_sorted(int v) const {
  assert(0 <= v && v < number_of_nodes());

  std::vector<int> ret(in_edges_[v].begin(), in_edges_[v].end());
  std::sort(ret.begin(), ret.end());
  return ret;
  return in_edges_[v];
}

std::vector<int> CIntDiGraph::out_neighbors_sorted(int v) const {
  assert(0 <= v && v < number_of_nodes());

  std::vector<int> ret(out_edges_[v].begin(), out_edges_[v].end());
  std::sort(ret.begin(), ret.end());
  return ret;
}

int CIntDiGraph::in_degree(int v) const {
  assert(0 <= v && v < number_of_nodes());
  return in_edges_[v].size();
}

int CIntDiGraph::out_degree(int v) const {
  assert(0 <= v && v < number_of_nodes());
  return out_edges_[v].size();
}

}  // namespace graph
}  // namespace sparsedomsets