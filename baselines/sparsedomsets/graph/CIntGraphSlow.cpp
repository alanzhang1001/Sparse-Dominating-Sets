#include "CIntGraphSlow.hpp"
#include <algorithm>
#include <cassert>

namespace sparsedomsets {
namespace graph {

CIntGraphSlow::CIntGraphSlow() : m_(0) {}

CIntGraphSlow::CIntGraphSlow(int n) : m_(0) { edges_.resize(n); }

void CIntGraphSlow::add_edge(int u, int v) {
  assert(0 <= u && u < number_of_nodes());
  assert(0 <= v && v < number_of_nodes());
  assert(u != v);

  if (edges_[u].find(v) != edges_[u].end()) {
    // the given edge already exists
    return;
  }

  edges_[u].insert(v);
  edges_[v].insert(u);
  ++m_;
}

int CIntGraphSlow::number_of_nodes() const { return edges_.size(); }

int CIntGraphSlow::number_of_edges() const { return m_; }

std::unordered_set<int> const& CIntGraphSlow::neighbors(int v) const {
  assert(0 <= v && v < number_of_nodes());

  return edges_[v];
}

std::vector<int> CIntGraphSlow::neighbors_sorted(int v) const {
  assert(0 <= v && v < number_of_nodes());

  std::vector<int> ret(edges_[v].begin(), edges_[v].end());
  std::sort(ret.begin(), ret.end());
  return ret;
}

}  // namespace graph
}  // namespace sparsedomsets