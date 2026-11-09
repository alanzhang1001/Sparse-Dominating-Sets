/**
 * Algorithms to find neighborhood partitioning.
 */
#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

namespace sparsedomsets {
namespace algorithms {

std::vector<int> balanced_assignment(                  //
    int num_tasks,                                     //
    int num_agents,                                    //
    std::vector<std::vector<int>> const& choices,      // task -> list of agents
    std::vector<long long> const& num_exclusive_tasks  // agent -> number of exclusive tasks
);

template <typename Graph>
std::map<int, std::vector<int>> neighborhood_partition(  //
    Graph const& G,                                      //
    std::vector<int> const& landmarks,                   //
    uint32_t strategy,                                   //
    unsigned int seed                                    //
);

}  // namespace algorithms
}  // namespace sparsedomsets
