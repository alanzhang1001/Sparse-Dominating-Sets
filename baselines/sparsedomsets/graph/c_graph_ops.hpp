/**
 * Graph Operations.
 */
#pragma once

#include <cassert>
#include <omp.h>
#include <unordered_set>
#include <vector>

#include "CIntDiGraph.hpp"

namespace sparsedomsets {
namespace graph {

template <typename Graph>
std::unordered_set<int> get_closed_neighborhood(Graph const& G, int v, int radius);

template <typename Graph>
Graph distance_closure(Graph const& G, int radius, int num_threads = 1);

template <typename Graph>
std::vector<std::vector<int>> compute_all_closed_neighborhood(Graph const& G, int radius, int num_threads = 1);

template <typename Graph>
long long compute_congestion_sum(Graph const& G, std::vector<int> const& landmarks, int radius, int num_threads = 1);

template <typename Graph>
double compute_average_congestion(Graph const& G, std::vector<int> const& landmarks, int radius, int num_threads = 1);

template <typename Graph>
std::vector<int> bfs_ordering(Graph const& G, std::vector<int> const& landmarks);

template <typename Graph>
std::vector<std::vector<int>> bfs_layering(Graph const& G, std::vector<int> const& landmarks);

template <typename Graph>
CIntDiGraph neighborhood_kernel(Graph const& G, std::vector<int> const& landmarks);

template <typename Graph>
std::pair<CIntDiGraph, std::vector<std::vector<int>>> compact_neighborhood_kernel(Graph const& G, std::vector<int> const& landmarks);

// template <typename Graph>
// std::vector<std::unordered_set<int>> compute_all_closed_neighborhood_set(Graph const& G, int radius, int num_threads = 1);
}  // namespace graph
}  // namespace sparsedomsets
