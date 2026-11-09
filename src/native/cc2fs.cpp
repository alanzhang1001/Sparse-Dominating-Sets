// Standalone reimplementation of the published CC2FS pseudocode.
// (Wang, Cai, Yin, IJCAI 2017) for radius-1 weighted dominating set.
// Weights are the congestion objective w(u)=|N[u]|.
// Audit v2: update last-change timestamps on BOTH addition and removal.
// A timestamp is an operation index; minimum means least recently changed.
// Remaining ties use vertex ID; this implementation has no random restarts.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

struct GainEntry {
    long long gain;
    int vertex;
};

struct GainLess {
    bool operator()(const GainEntry& a, const GainEntry& b) const {
        if (a.gain != b.gain) return a.gain < b.gain;
        return a.vertex > b.vertex;
    }
};

static double seconds_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: cc2fs GRAPH.gxt TIME_LIMIT_SECONDS OUTPUT.vertices.txt [--max-iterations N] [--trace FILE] [--history FILE] [--legacy-age]\n";
        return 2;
    }
    const std::string graph_path = argv[1];
    const double time_limit = std::stod(argv[2]);
    const std::string output_path = argv[3];
    long long maximum_iterations = std::numeric_limits<long long>::max();
    bool legacy_age = false;
    std::string trace_path, history_path;
    for (int i=4; i<argc; ++i) {
        std::string option = argv[i];
        if (option == "--legacy-age") legacy_age = true;
        else if (option == "--max-iterations" && i+1<argc) maximum_iterations = std::stoll(argv[++i]);
        else if (option == "--trace" && i+1<argc) trace_path = argv[++i];
        else if (option == "--history" && i+1<argc) history_path = argv[++i];
        else { std::cerr << "invalid option: " << option << '\n'; return 2; }
    }
    if (maximum_iterations < 0) { std::cerr << "iteration limit must be nonnegative\n"; return 2; }
    std::ofstream trace, history;
    if (!trace_path.empty()) { trace.open(trace_path); if (!trace) return 2; }
    if (!history_path.empty()) { history.open(history_path); if (!history) return 2; }
    if (!(time_limit > 0.0)) {
        std::cerr << "time limit must be positive\n";
        return 2;
    }

    std::ifstream input(graph_path);
    int n = 0;
    if (!(input >> n) || n <= 0) {
        std::cerr << "invalid graph header\n";
        return 2;
    }
    std::vector<std::vector<int>> adj(static_cast<size_t>(n));
    int u = 0, v = 0;
    while (input >> u >> v) {
        if (u < 0 || u >= n || v < 0 || v >= n) {
            std::cerr << "edge endpoint outside graph range\n";
            return 2;
        }
        adj[static_cast<size_t>(u)].push_back(v);
    }
    for (int x = 0; x < n; ++x) {
        auto& row = adj[static_cast<size_t>(x)];
        row.push_back(x);  // closed neighborhood
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }

    const auto started = Clock::now();
    std::vector<int> weight(static_cast<size_t>(n));
    for (int x = 0; x < n; ++x) weight[static_cast<size_t>(x)] = static_cast<int>(adj[static_cast<size_t>(x)].size());

    // InitGreedyConstruction from the paper: repeatedly choose the vertex
    // covering the largest number of currently uncovered vertices.
    std::vector<unsigned char> in_solution(static_cast<size_t>(n), 0);
    std::vector<int> cover_count(static_cast<size_t>(n), 0);
    std::vector<int> uncovered_count(static_cast<size_t>(1), n);
    std::vector<long long> greedy_gain(static_cast<size_t>(n));
    std::priority_queue<GainEntry, std::vector<GainEntry>, GainLess> greedy_heap;
    for (int x = 0; x < n; ++x) {
        greedy_gain[static_cast<size_t>(x)] = static_cast<long long>(adj[static_cast<size_t>(x)].size());
        greedy_heap.push({greedy_gain[static_cast<size_t>(x)], x});
    }
    std::vector<int> members;
    std::vector<long long> age(static_cast<size_t>(n), 0);
    long long clock_tick = 1;
    long long current_cost = 0;
    while (uncovered_count[0] > 0) {
        while (!greedy_heap.empty()) {
            const auto top = greedy_heap.top();
            if (top.gain == greedy_gain[static_cast<size_t>(top.vertex)] && !in_solution[static_cast<size_t>(top.vertex)]) break;
            greedy_heap.pop();
        }
        if (greedy_heap.empty()) {
            std::cerr << "greedy construction failed to cover graph\n";
            return 3;
        }
        const int pick = greedy_heap.top().vertex;
        greedy_heap.pop();
        in_solution[static_cast<size_t>(pick)] = 1;
        age[static_cast<size_t>(pick)] = clock_tick++;
        members.push_back(pick);
        current_cost += weight[static_cast<size_t>(pick)];
        for (int x : adj[static_cast<size_t>(pick)]) {
            if (cover_count[static_cast<size_t>(x)]++ != 0) continue;
            --uncovered_count[0];
            for (int y : adj[static_cast<size_t>(x)]) {
                --greedy_gain[static_cast<size_t>(y)];
                greedy_heap.push({greedy_gain[static_cast<size_t>(y)], y});
            }
        }
    }
    const long long initial_cost = current_cost;
    const double initialization_seconds = seconds_since(started);

    std::vector<unsigned char> conf_change(static_cast<size_t>(n), 1);
    std::set<int> add_candidates;
    for (int x = 0; x < n; ++x) if (!in_solution[static_cast<size_t>(x)]) add_candidates.insert(x);
    std::vector<unsigned char> forbid(static_cast<size_t>(n), 0);
    std::vector<long long> frequency(static_cast<size_t>(n), 1);
    std::vector<int> best_members = members;
    long long best_cost = current_cost;
    long long iterations = 0;
    long long additions = 0;
    long long removals = 0;
    auto trace_state = [&](const char* phase, int vertex) {
        if (!trace.is_open()) return;
        auto emit = [&](const char* key, const auto& values) {
            trace << ",\"" << key << "\":[";
            for (size_t i=0; i<values.size(); ++i) { if (i) trace << ','; trace << static_cast<long long>(values[i]); }
            trace << ']';
        };
        trace << "{\"phase\":\"" << phase << "\",\"vertex\":" << vertex
              << ",\"iteration\":" << iterations << ",\"cost\":" << current_cost
              << ",\"best_cost\":" << best_cost << ",\"uncovered\":" << uncovered_count[0];
        emit("in_solution", in_solution); emit("cover_count", cover_count);
        emit("conf_change", conf_change); emit("frequency", frequency);
        emit("age", age); emit("forbid", forbid);
        trace << "}\n";
    };
    auto save_history = [&](const char* event) {
        if (history.is_open()) history << std::setprecision(17)
            << "{\"event\":\"" << event << "\",\"seconds\":" << seconds_since(started)
            << ",\"cost\":" << best_cost << ",\"selected\":" << best_members.size()
            << ",\"iterations\":" << iterations << "}\n";
    };
    trace_state("initial", -1);
    save_history("initial");

    auto mark_configuration_after_change = [&](int changed, bool removed) {
        if (removed) {
            conf_change[static_cast<size_t>(changed)] = 0;
            add_candidates.erase(changed);
        }
        // Rule 2/3: every vertex at distance 1 or 2 has a changed two-level
        // configuration. Self is omitted, matching N_2(v)=N_1(v) union N_2(v).
        for (int first : adj[static_cast<size_t>(changed)]) {
            for (int affected : adj[static_cast<size_t>(first)]) {
                if (affected == changed) continue;
                if (!conf_change[static_cast<size_t>(affected)]) {
                    conf_change[static_cast<size_t>(affected)] = 1;
                }
                if (!in_solution[static_cast<size_t>(affected)]) add_candidates.insert(affected);
            }
        }
    };

    auto current_score_numerator = [&](int x) -> long long {
        long long score = 0;
        if (!in_solution[static_cast<size_t>(x)]) {
            for (int y : adj[static_cast<size_t>(x)]) {
                if (cover_count[static_cast<size_t>(y)] == 0) score += frequency[static_cast<size_t>(y)];
            }
        } else {
            for (int y : adj[static_cast<size_t>(x)]) {
                if (cover_count[static_cast<size_t>(y)] == 1) score -= frequency[static_cast<size_t>(y)];
            }
        }
        return score;
    };

    auto score_better = [&](int a, int b) {
        const long long na = current_score_numerator(a);
        const long long nb = current_score_numerator(b);
        const __int128 left = static_cast<__int128>(na) * weight[static_cast<size_t>(b)];
        const __int128 right = static_cast<__int128>(nb) * weight[static_cast<size_t>(a)];
        if (left != right) return left > right;
        if (age[static_cast<size_t>(a)] != age[static_cast<size_t>(b)]) return age[static_cast<size_t>(a)] < age[static_cast<size_t>(b)];
        return a < b;
    };

    auto best_to_remove = [&](bool respect_forbid) -> int {
        int best = -1;
        for (int x : members) {
            if (!in_solution[static_cast<size_t>(x)]) continue;
            if (respect_forbid && forbid[static_cast<size_t>(x)]) continue;
            if (best < 0 || score_better(x, best)) best = x;
        }
        return best;
    };

    auto best_to_add = [&]() -> int {
        int best = -1;
        for (int x : add_candidates) {
            if (in_solution[static_cast<size_t>(x)] || !conf_change[static_cast<size_t>(x)]) continue;
            if (best < 0 || score_better(x, best)) best = x;
        }
        return best;
    };

    auto remove_vertex = [&](int x) {
        in_solution[static_cast<size_t>(x)] = 0;
        if (!legacy_age) age[static_cast<size_t>(x)] = clock_tick++;
        current_cost -= weight[static_cast<size_t>(x)];
        ++removals;
        auto it = std::find(members.begin(), members.end(), x);
        if (it != members.end()) {
            *it = members.back();
            members.pop_back();
        }
        for (int y : adj[static_cast<size_t>(x)]) {
            if (--cover_count[static_cast<size_t>(y)] == 0) ++uncovered_count[0];
        }
        mark_configuration_after_change(x, true);
    };

    auto add_vertex = [&](int x) {
        in_solution[static_cast<size_t>(x)] = 1;
        add_candidates.erase(x);
        age[static_cast<size_t>(x)] = clock_tick++;
        members.push_back(x);
        current_cost += weight[static_cast<size_t>(x)];
        ++additions;
        for (int y : adj[static_cast<size_t>(x)]) {
            if (cover_count[static_cast<size_t>(y)]++ == 0) --uncovered_count[0];
        }
        mark_configuration_after_change(x, false);
        // CC2FS increases the frequency of each vertex still uncovered after
        // every addition in the restoration phase.
        for (int y = 0; y < n; ++y) if (cover_count[static_cast<size_t>(y)] == 0) ++frequency[static_cast<size_t>(y)];
    };

    while (seconds_since(started) < time_limit && iterations < maximum_iterations) {
        ++iterations;
        if (uncovered_count[0] == 0) {
            if (current_cost < best_cost) {
                best_cost = current_cost;
                best_members = members;
                save_history("improvement");
            }
            const int remove = best_to_remove(false);  // REMOVE-RULE1
            if (remove < 0) break;
            remove_vertex(remove);
            trace_state("remove-feasible", remove);
            continue;
        }

        const int remove = best_to_remove(true);  // REMOVE-RULE2
        if (remove < 0) break;
        remove_vertex(remove);
        std::fill(forbid.begin(), forbid.end(), 0);
        trace_state("remove-infeasible", remove);

        bool restoration_stalled = false;
        while (uncovered_count[0] > 0 && seconds_since(started) < time_limit) {
            const int add = best_to_add();  // ADD-RULE over CCV2
            if (add < 0) {
                restoration_stalled = true;
                break;
            }
            add_vertex(add);
            forbid[static_cast<size_t>(add)] = 1;
            trace_state("add", add);
        }
        if (restoration_stalled || seconds_since(started) >= time_limit) break;
    }
    if (uncovered_count[0] == 0 && current_cost < best_cost) {
        best_cost = current_cost;
        best_members = members;
        save_history("improvement-final");
    }
    save_history("finished");

    std::sort(best_members.begin(), best_members.end());
    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "cannot open output file\n";
        return 2;
    }
    for (int x : best_members) output << x << '\n';
    output.close();
    long long verified_cost = 0;
    for (int x : best_members) verified_cost += weight[static_cast<size_t>(x)];
    std::vector<int> verify_cover(static_cast<size_t>(n), 0);
    for (int x : best_members) for (int y : adj[static_cast<size_t>(x)]) ++verify_cover[static_cast<size_t>(y)];
    const bool feasible = std::all_of(verify_cover.begin(), verify_cover.end(), [](int c) { return c > 0; });
    std::cout << "{\"status\":\"" << (seconds_since(started) >= time_limit ? "timeout_with_feasible_solution" : "ok")
              << "\",\"feasible\":" << (feasible ? "true" : "false")
              << ",\"initial_cost\":" << initial_cost
              << ",\"best_cost\":" << verified_cost
              << ",\"selected\":" << best_members.size()
              << ",\"iterations\":" << iterations
              << ",\"additions\":" << additions
              << ",\"removals\":" << removals
              << ",\"revision\":\"cc2fs-audit-v2\",\"legacy_age\":" << (legacy_age ? "true" : "false")
              << ",\"random_restarts\":0,\"time_budget_seconds\":" << time_limit
              << ",\"initialization_seconds\":" << initialization_seconds
              << ",\"wall_seconds\":" << seconds_since(started) << "}\n";
    return feasible ? 0 : 3;
}
