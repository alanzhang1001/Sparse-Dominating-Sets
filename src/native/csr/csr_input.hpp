#pragma once
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Both implementation wrappers use exactly this loader and validation.
// Raw files are the existing little-endian uint64 offsets / uint32 neighbors.
inline std::vector<std::vector<int>> read_csr_adj(const std::string& dir) {
    const auto offset_file = dir+"/indptr.u64", index_file = dir+"/indices.u32";
    auto bytes = std::filesystem::file_size(offset_file);
    if (bytes < 16 || bytes%8) throw std::runtime_error("bad CSR offsets size");
    const uint64_t n = bytes/8-1;
    if (n >= INT32_MAX) throw std::runtime_error("too many vertices");
    std::vector<uint64_t> ptr(n+1);
    std::ifstream p(offset_file, std::ios::binary), q(index_file, std::ios::binary);
    if (!p.read(reinterpret_cast<char*>(ptr.data()), bytes) || !q || ptr[0] != 0 ||
        ptr.back()*4 != std::filesystem::file_size(index_file)) throw std::runtime_error("bad CSR files");
    std::vector<std::vector<int>> a(n);
    for (uint64_t v=0; v<n; ++v) {
        if (ptr[v] > ptr[v+1] || ptr[v+1] > ptr.back()) throw std::runtime_error("nonmonotone CSR");
        auto& row=a[v]; row.resize(ptr[v+1]-ptr[v]);
        if (!q.read(reinterpret_cast<char*>(row.data()), row.size()*4)) throw std::runtime_error("short CSR row");
        int previous=-1;
        for (int u:row) {
            if (u<0 || uint64_t(u)>=n || u==int(v) || u<=previous) throw std::runtime_error("invalid CSR neighbor order");
            previous=u;
        }
    }
    for (uint64_t v=0; v<n; ++v) for (int u:a[v])
        if (!std::binary_search(a[u].begin(),a[u].end(),int(v))) throw std::runtime_error("asymmetric CSR");
    return a;
}
