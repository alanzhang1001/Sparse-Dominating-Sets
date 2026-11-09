#pragma once
#include "../graph/CIntGraph.hpp"
#include <cstdio>
#include <vector>
#include <map>

#ifdef USE_SLOW_IMPLEMENTATION
#include "../graph/CIntGraphSlow.hpp"
#endif

namespace sparsedomsets {
namespace readwrite {

graph::CIntGraph read_gxt(char const *path);
void write_gxt(graph::CIntGraph const &G, char const *path, bool sorted = true);

std::vector<int> read_vertices(char const *path);
void write_vertices(std::vector<int> const &vs, char const *path, bool sorted = true);

void write_parts(std::map<int, std::vector<int>> const &parts, char const *path);

}  // namespace readwrite
}  // namespace sparsedomsets
