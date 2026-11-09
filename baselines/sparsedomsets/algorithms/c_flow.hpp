#pragma once
#include <tuple>
#include <utility>
#include <vector>

namespace sparsedomsets {
namespace algorithms {
namespace flow {

/** Aliases */
typedef int VertexId;
typedef int EdgeId;
typedef long long Weight;

/** Structs */
struct IntFlow {
  VertexId from_, to;  // edge from and to vertices
  Weight weight;      // used for either flow amount or capacity
};

struct IntFlowCost {
  VertexId from_, to;
  Weight capacity, cost;
};

struct ResultFlow {
  Weight max_flow;
  std::vector<IntFlow> flow;
};

struct ResultFlowCost {
  Weight max_flow;
  Weight min_cost;
  std::vector<IntFlow> flow;
};

// ResultFlow max_flow(                             //
//     int n,                                       // number of vertices
//     std::vector<IntFlow> const& weighted_edges,  // list of (from-vertex, to-vertex, capacity)
//     VertexId src,                                // source vertex
//     VertexId dst                                 // destination vertex
// );

ResultFlowCost min_cost_max_flow(                    //
    int n,                                           // number of vertices
    std::vector<IntFlowCost> const& weighted_edges,  // list of (from-vertex, to-vertex, capacity, cost)
    VertexId src,                                    // source vertex
    VertexId dst                                     // destination vertex
);

}  // namespace flow
}  // namespace algorithms
}  // namespace sparsedomsets