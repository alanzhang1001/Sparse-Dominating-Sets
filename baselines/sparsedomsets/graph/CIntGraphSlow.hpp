#pragma once
#include <unordered_set>
#include <vector>

namespace sparsedomsets {
namespace graph {
class CIntGraphSlow {
 public:
  CIntGraphSlow();
  CIntGraphSlow(int n);
  void add_edge(int u, int v);
  int number_of_nodes() const;
  int number_of_edges() const;
  std::vector<int> neighbors_sorted(int v) const;
  std::unordered_set<int> const& neighbors(int v) const;

 private:
  int m_;
  std::vector<std::unordered_set<int>> edges_;
};
}  // namespace graph
}  // namespace sparsedomsets
