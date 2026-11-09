#include <algorithm>

#include "c_edgelist.hpp"

namespace sparsedomsets {
namespace readwrite {

graph::CIntGraph read_gxt(char const *path) {
  FILE *fp;
  if (!(fp = fopen(path, "r"))) { perror("IO Error"); }

  // parse the number of nodes
  int ret, n, u, v;
  ret = fscanf(fp, "%d", &n);
  if (ret != 1) { perror("invalid content"); }

  auto G = graph::CIntGraph(n);

  // parse edges
  for (;;) {
    ret = fscanf(fp, "%d %d", &u, &v);
    if (ret == 2) {
      if (u < v) {
        // assume the file contains each edge in both directions
        G.add_edge(u, v);
      }
    } else if (ret == EOF) {
      break;
    } else {
      perror("invalid content");
    }
  }

  fclose(fp);
  return G;
}

void write_gxt(graph::CIntGraph const &G, char const *path, bool sorted) {
  FILE *fp;
  if (!(fp = fopen(path, "w"))) { perror("IO Error"); }

  // write the number of nodes
  auto n = G.number_of_nodes();
  fprintf(fp, "%d\n", n);

  // write edges
  for (int v = 0; v < n; ++v) {
    auto &us = sorted ? G.neighbors_sorted(v) : G.neighbors(v);
    for (auto u : us) {
      // write each edge in both directions
      fprintf(fp, "%d %d\n", v, u);
    }
  }

  fclose(fp);
}

std::vector<int> read_vertices(char const *path) {
  FILE *fp;
  if (!(fp = fopen(path, "r"))) { perror("IO Error"); }

  std::vector<int> vs;
  int ret, v;

  for (;;) {
    ret = fscanf(fp, "%d", &v);
    if (ret == 1) {
      vs.push_back(v);
    } else if (ret == EOF) {
      break;
    } else {
      perror("invalid content");
    }
  }

  fclose(fp);
  return vs;
}

void write_vertices(std::vector<int> const &vs, char const *path, bool sorted) {
  FILE *fp;
  if (!(fp = fopen(path, "w"))) { perror("IO Error"); }

  if (sorted) {
    std::vector<int> xs = vs;  // copy the entire vector
    std::sort(xs.begin(), xs.end());
    for (auto x : xs) fprintf(fp, "%d\n", x);
  } else {
    for (auto v : vs) fprintf(fp, "%d\n", v);
  }

  fclose(fp);
}

void write_parts(std::map<int, std::vector<int>> const &parts, char const *path) {
  FILE *fp;
  if (!(fp = fopen(path, "w"))) { perror("IO Error"); }

  for (auto p : parts) {
    fprintf(fp, "%d", p.first);                     // landmark
    for (auto v : p.second) fprintf(fp, " %d", v);  // part members
    fprintf(fp, "\n");                              // newline
  }

  fclose(fp);
}

}  // namespace readwrite
}  // namespace sparsedomsets
