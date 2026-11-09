#pragma once
#include <unordered_set>
#include <vector>

namespace sparsedomsets {
namespace graph {
class CIntGraph {
 public:
  CIntGraph();
  CIntGraph(int n);
  void add_edge(int u, int v);
  int number_of_nodes() const;
  long long number_of_edges() const;
  std::vector<int> neighbors_sorted(int v) const;
  std::vector<int> const& neighbors(int v) const;
  // std::vector<int> closed_neighborhood(int v) const;
  // std::vector<int> closed_neighborhood_sorted(int v) const;
  int degree(int v) const;

  std::vector<std::vector<int>> edges_;
};
}  // namespace graph
}  // namespace sparsedomsets
