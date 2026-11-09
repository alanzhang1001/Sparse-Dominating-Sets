#pragma once
#include <unordered_set>
#include <vector>

namespace sparsedomsets {
namespace graph {
class CIntDiGraph {
 public:
  CIntDiGraph();
  CIntDiGraph(int n);
  void add_edge(int u, int v);
  int number_of_nodes() const;
  long long number_of_edges() const;
  std::vector<int> const& in_neighbors(int v) const;
  std::vector<int> const& out_neighbors(int v) const;
  std::vector<int> const& neighbors(int v) const;  // alias of out_neighbors
  std::vector<int> in_neighbors_sorted(int v) const;
  std::vector<int> out_neighbors_sorted(int v) const;
  int in_degree(int v) const;
  int out_degree(int v) const;

  std::vector<std::vector<int>> in_edges_;
  std::vector<std::vector<int>> out_edges_;
};
}  // namespace graph
}  // namespace sparsedomsets
