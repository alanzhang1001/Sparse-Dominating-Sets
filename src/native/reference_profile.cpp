// Compile with -I<snapshot>/sparsedomsets/cpp. The included reference source is
// copied byte-for-byte and hash-checked by the runner; its greedy routine is
// called directly. Only the wrapper splits preparation/selection/I/O timing.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "algorithms/c_dominating.cpp"
#include "readwrite/c_edgelist.hpp"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: reference_profile GRAPH R STRATEGY OUTPUT");
        int radius = std::stoi(argv[2]), strategy = std::stoi(argv[3]);
        if (radius < 1 || (strategy != 1 && strategy != 2 && strategy != 10 && strategy != 17))
            throw std::runtime_error("unsupported radius/strategy");
        using namespace sparsedomsets;
        using namespace sparsedomsets::algorithms;
        auto started = Clock::now(), phase = started;
        auto G = readwrite::read_gxt(argv[1]);
        double read_seconds = elapsed(phase); phase = Clock::now();
        const auto n = G.number_of_nodes();
        auto neighborhoods = graph::compute_all_closed_neighborhood(G, radius, 1);
        std::vector<std::unordered_set<int>> uncovered(n);
        for (int i=0; i<n; ++i) uncovered[i] = std::unordered_set<int>(neighborhoods[i].begin(), neighborhoods[i].end());
        auto states = initialize_vertex_state(neighborhoods);
        auto comp = get_compare_func(static_cast<dominating::Strategy>(strategy));
        double preprocessing_seconds = elapsed(phase); phase = Clock::now();
        auto selected = find_dominating_set_with_reordering(states, neighborhoods, uncovered, comp, false, false);
        double greedy_seconds = elapsed(phase); phase = Clock::now();
        readwrite::write_vertices(selected, argv[4], false);
        double write_seconds = elapsed(phase);
        const double total_seconds = elapsed(started);
        long long cost = 0, entries = 0;
        for (int v : selected) cost += neighborhoods[v].size();
        for (const auto& row : neighborhoods) entries += row.size();
        std::cout << std::setprecision(17)
          << "{\"implementation\":\"published-reference-profiled\",\"n\":" << n
          << ",\"radius\":" << radius << ",\"strategy\":" << strategy
          << ",\"total_cost\":" << cost << ",\"selected\":" << selected.size()
          << ",\"neighborhood_entries\":" << entries
          << ",\"read_seconds\":" << read_seconds << ",\"preprocessing_seconds\":" << preprocessing_seconds
          << ",\"greedy_seconds\":" << greedy_seconds << ",\"write_seconds\":" << write_seconds
          << ",\"internal_total_seconds\":" << total_seconds << "}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
    return 0;
}
