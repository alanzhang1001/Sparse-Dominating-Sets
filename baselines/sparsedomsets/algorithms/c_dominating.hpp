/**
 * Algorithms to find dominating sets.
 */
#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

namespace sparsedomsets {
namespace algorithms {

template <typename Graph>
std::vector<int> find_dominating_set(Graph const& G, uint32_t strategy, unsigned int seed, int radius = 1, int num_threads = 1);

template <typename Graph>
std::vector<int> order_vertices(Graph const& G, uint32_t strategy, unsigned int seed, int radius = 1, int num_threads = 1);

}  // namespace algorithms
}  // namespace sparsedomsets
