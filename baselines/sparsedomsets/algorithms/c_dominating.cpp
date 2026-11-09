/**
 * Dominating set algorithms.
 */

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <queue>
#include <random>

#include "../graph/CIntGraph.hpp"
#include "../graph/c_graph_ops.hpp"
#include "c_dominating.hpp"

namespace sparsedomsets {
namespace algorithms {

namespace dominating {
/**
 * bits
 *        8 7      0
 * xxxxxxxx xxxxxxxx
 *               ***  comparison method (1-4)
 *            ***     tie-breaking method (0-4)
 *           *        randomization (0: off, 1: on)
 *          *         aggressive deletion (0: off, 1: on)
 *        *           degree-one reduction (0: off, 1: on)
 *       *            reordering (0: on, 1: off)
 */
enum class Strategy : uint32_t {
  UNDOMINATED = 1,
  RATIO = 2,
  LEAST_VOLUME = 3,
  WEIGHT = 4,

  COMP_SHIFT = 3,
  COMP_MASK = (1 << COMP_SHIFT) - 1,

  RANDOMIZE = 1 << (COMP_SHIFT * 2),
  AGGRESSIVE_DELETION = RANDOMIZE << 1,
  DEG_ONE_REDUCTION = AGGRESSIVE_DELETION << 1,
  NO_REORDERING = DEG_ONE_REDUCTION << 1,
};
}  // namespace dominating

template <typename T>
static bool has_bit(T x, dominating::Strategy y) {
  return static_cast<uint32_t>(x) & static_cast<uint32_t>(y);
}

struct VertexState {
  /** Id of the vertex. */
  int id;

  /** The size of the closed neighborhood of this vertex. */
  int vol;

  /** Number of dominated vertices in the closed neighborhood. */
  int dom;

  bool has_leaf;

  /** The sum of the sizes of the closed neighborhood over this vertex's closed neighborhood. */
  long long weight;

  /** The value used for the last tie-breaking. */
  uint64_t salt;
};

// comparisons
template <typename T>
static inline int compare(T a, T b) {
  return a < b ? -1 : a > b ? 1 : 0;
}
static int compare_undominated(VertexState const& a, VertexState const& b) {  //
  return compare(a.vol - a.dom, b.vol - b.dom);
}
static int compare_ratio(VertexState const& a, VertexState const& b) {  //
  return compare(static_cast<long long>(a.vol) * b.dom, static_cast<long long>(a.dom) * b.vol);
}
static int compare_volume(VertexState const& a, VertexState const& b) {  //
  return compare(b.vol, a.vol);
}
static int compare_weight(VertexState const& a, VertexState const& b) {  //
  return compare(a.weight, b.weight);
}
static int compare_has_leaf(VertexState const& a, VertexState const& b) {  //
  return compare(a.has_leaf ? 1 : 0, b.has_leaf ? 1 : 0);
}
static int compare_salt(VertexState const& a, VertexState const& b) {
  // smallest first
  return compare(b.salt, a.salt);
}

static std::function<bool(VertexState const&, VertexState const&)> get_compare_func(dominating::Strategy strategy) {
  std::vector<decltype(&compare_salt)> metrics;

  if (has_bit(strategy, dominating::Strategy::DEG_ONE_REDUCTION)) metrics.push_back(compare_has_leaf);

  for (int j = 0; j < 2; ++j) {
    switch ((static_cast<uint32_t>(strategy) >> (j * static_cast<uint32_t>(dominating::Strategy::COMP_SHIFT))) &
            static_cast<uint32_t>(dominating::Strategy::COMP_MASK)) {
      case static_cast<uint32_t>(dominating::Strategy::UNDOMINATED): metrics.push_back(compare_undominated); break;
      case static_cast<uint32_t>(dominating::Strategy::RATIO): metrics.push_back(compare_ratio); break;
      case static_cast<uint32_t>(dominating::Strategy::LEAST_VOLUME): metrics.push_back(compare_volume); break;
      case static_cast<uint32_t>(dominating::Strategy::WEIGHT): metrics.push_back(compare_weight); break;
      default: break;  // do nothing
    }
  }

  metrics.push_back(compare_salt);

  auto f = [metrics](VertexState const& a, VertexState const& b) {
    for (auto m : metrics) {
      auto ret = m(a, b);
      if (ret < 0) return true;
      if (ret > 0) return false;
    }
    return false;
  };
  return f;
}

static std::vector<VertexState> initialize_vertex_state(std::vector<std::vector<int>> const& cnbrs) {
  auto n = cnbrs.size();
  std::vector<VertexState> vstates(n);

  for (int i = 0; i < n; ++i) {
    int s = cnbrs[i].size();
    vstates[i].id = i;
    vstates[i].vol = s;
    vstates[i].dom = 0;
    vstates[i].salt = i;  // default salt: vertex id

    // initialize weight
    for (auto u : cnbrs[i]) vstates[u].weight += s;

    // check for leaves
    if (s == 2) {
      for (auto u : cnbrs[i]) {
        if (u != i) vstates[u].has_leaf = true;
      }
    }
  }

  return vstates;
}

/** Initializes the sets of undominated vertices in the closed neighborhood. */
// template <typename Graph>
// static std::vector<std::unordered_set<int>> initialize_undoms(Graph const& G) {
//   auto n = G.number_of_nodes();
//   std::vector<std::unordered_set<int>> undoms(n);

//   for (int i = 0; i < n; ++i) {
//     undoms[i].insert(i);
//     for (auto u : G.neighbors(i)) undoms[i].insert(u);
//   }

//   return undoms;
// }

static void randomize_vertex_salt(std::vector<VertexState>& vstates, unsigned int seed) {
  std::mt19937_64 eng(seed);
  std::uniform_int_distribution<uint64_t> dist;
  for (auto& s : vstates) s.salt = dist(eng);
}

static std::vector<int> find_dominating_set_no_reordering(std::vector<VertexState>& vstates,
                                                          std::vector<std::vector<int>> const& cnbrs,
                                                          std::vector<std::unordered_set<int>>& undoms,
                                                          std::function<bool(VertexState const&, VertexState const&)> compare,
                                                          bool aggressive_deletion) {
  auto n = cnbrs.size();
  std::vector<bool> dominated(n);  // initialize to false
  std::vector<int> doms;           // chosen dominators

  std::sort(vstates.begin(), vstates.end(), compare);
  std::reverse(vstates.begin(), vstates.end());

  for (int i = 0; i < n; ++i) {
    auto v = vstates[i].id;
    if (aggressive_deletion && dominated[v]) continue;
    if (undoms[v].empty()) continue;

    // include this vertex in the dominating set
    doms.push_back(v);

    for (auto u : undoms[v]) {
      dominated[u] = true;
      for (auto w : cnbrs[u]) {
        if (w != v) undoms[w].erase(u);
      }
    }
  }
  return doms;
}

static std::vector<int> find_dominating_set_with_reordering(std::vector<VertexState>& vstates,
                                                            std::vector<std::vector<int>> const& cnbrs,
                                                            std::vector<std::unordered_set<int>>& undoms,
                                                            std::function<bool(VertexState const&, VertexState const&)> compare,
                                                            bool aggressive_deletion, bool deg_one_reduction) {
  auto n = cnbrs.size();
  std::vector<bool> dominated(n);  // initialize to false
  std::vector<int> doms;           // chosen dominators

  // initialize the priority queue
  typedef std::priority_queue<VertexState, std::vector<VertexState>, std::function<bool(VertexState const&, VertexState const&)>> PQ;
  PQ q(vstates.begin(), vstates.end(), compare);

  while (!q.empty()) {
    auto& st = q.top();
    auto v = st.id;
    auto vd = st.dom;

    // printf("Fetched: {id=%d, dom/vol=%d/%d, has_leaf=%d, weight=%lld, salt=%llu}\n", st.id, st.dom, st.vol,
    //  st.has_leaf ? 1 : 0, st.weight, st.salt);
    q.pop();

    if (aggressive_deletion && dominated[v]) continue;
    if (vd != vstates[v].vol - undoms[v].size()) continue;  // outdated information
    if (undoms[v].empty()) continue;  // all members of the closed neighborhood of this vertex are already dominated

    // printf("{id=%d, dom=%d, has_leaf=%d, vol=%d, weight=%lld, salt=%llu}\n", st.id, st.dom, st.has_leaf ? 1 : 0, st.vol,
    //  st.weight, st.salt);

    // add this vertex to the dominators
    doms.push_back(v);

    std::unordered_set<int> affected;

    for (auto u : undoms[v]) {
      dominated[u] = true;  // state change of u: undominated -> dominated (possible that u = v)

      // effects to u's closed neighborhood
      for (auto w : cnbrs[u]) {
        affected.insert(w);
        if (w != v) undoms[w].erase(u);
      }
    }

    // degree-one reduction
    if (deg_one_reduction) {
      std::unordered_set<int> aff = affected;  // copy the entire set
      for (auto a : aff) {
        if (!dominated[a] && undoms[a].size() == 2) {
          for (auto b : undoms[a]) {
            if (b != a && !vstates[b].has_leaf) {
              vstates[b].has_leaf = true;  // update has_leaf
              affected.insert(b);
            }
          }
        }
      }
    }

    // update heap
    for (auto a : affected) {
      if (a != v) {
        auto d = vstates[a].vol - static_cast<int>(undoms[a].size());

        // printf("Inserted: {id=%d, dom/vol=%d/%d, has_leaf=%d, weight=%lld, salt=%llu}\n", a, d, vstates[a].vol,
        //  vstates[a].has_leaf ? 1 : 0, vstates[a].weight, vstates[a].salt);
        q.push(VertexState({a, vstates[a].vol, d, vstates[a].has_leaf, vstates[a].weight, vstates[a].salt}));
      }
    }
  }
  return doms;
}

template <typename Graph>
std::vector<int> find_dominating_set(Graph const& G, uint32_t strategy, unsigned int seed, int radius, int num_threads) {
  auto st = static_cast<dominating::Strategy>(strategy);

  // initialize vertex states
  auto n = G.number_of_nodes();
  auto cnbrs = graph::compute_all_closed_neighborhood(G, radius, num_threads);  // multi-threaded

  // initialize the sets to empty
  std::vector<std::unordered_set<int>> undoms(n);

#pragma omp parallel num_threads(num_threads)
  {
#pragma omp for schedule(auto)
    for (int i = 0; i < n; ++i) undoms[i] = std::unordered_set<int>(cnbrs[i].begin(), cnbrs[i].end());
  }

  auto vstates = initialize_vertex_state(cnbrs);

  // randomize salts if needed
  if (has_bit(st, dominating::Strategy::RANDOMIZE)) randomize_vertex_salt(vstates, seed);

  // pick a comparison function
  auto comp = get_compare_func(st);

  // main logic
  if (has_bit(st, dominating::Strategy::NO_REORDERING)) {
    return find_dominating_set_no_reordering(vstates, cnbrs, undoms, comp, has_bit(st, dominating::Strategy::AGGRESSIVE_DELETION));
  } else {
    return find_dominating_set_with_reordering(vstates, cnbrs, undoms, comp, has_bit(st, dominating::Strategy::AGGRESSIVE_DELETION),
                                               has_bit(st, dominating::Strategy::DEG_ONE_REDUCTION));
  }
}

/** For debugging purpose. */
template <typename Graph>
std::vector<int> order_vertices(Graph const& G, uint32_t strategy, unsigned int seed, int radius, int num_threads) {
  auto st = static_cast<dominating::Strategy>(strategy);
  auto comp = get_compare_func(st);
  auto vstates = initialize_vertex_state(graph::compute_all_closed_neighborhood(G, radius, num_threads));
  if (has_bit(st, dominating::Strategy::RANDOMIZE)) randomize_vertex_salt(vstates, seed);
  std::sort(vstates.begin(), vstates.end(), comp);
  std::reverse(vstates.begin(), vstates.end());

  std::vector<int> ret;
  for (auto& v : vstates) {
    // printf("{id=%d, dom/vol=%d/%d, has_leaf=%d, weight=%lld, salt=%llu}\n", v.id, v.dom, v.vol, v.has_leaf ? 1 : 0,
    //  v.weight, v.salt);
    ret.push_back(v.id);
  }
  return ret;
}

// explicit instantiation
template std::vector<int> find_dominating_set<graph::CIntGraph>(graph::CIntGraph const&, uint32_t, unsigned int, int, int);
template std::vector<int> order_vertices<graph::CIntGraph>(graph::CIntGraph const&, uint32_t, unsigned int, int, int);

}  // namespace algorithms
}  // namespace sparsedomsets