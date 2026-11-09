// Dedicated data conversion. No changes to either dominating-set algorithm.
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
static void progress(const char* phase) { std::cerr << "{\"phase\":\"" << phase << "\"}\n"; }
struct DSU {
    std::vector<uint32_t> parent;
    std::vector<uint8_t> rank;
    explicit DSU(size_t n):parent(n),rank(n,0) { std::iota(parent.begin(),parent.end(),0); }
    uint32_t find(uint32_t v) {
        uint32_t root=v;
        while (parent[root]!=root) root=parent[root];
        while (parent[v]!=v) { auto next=parent[v]; parent[v]=root; v=next; }
        return root;
    }
    void unite(uint32_t a,uint32_t b) {
        a=find(a); b=find(b); if(a==b)return;
        if(rank[a]<rank[b])std::swap(a,b);
        parent[b]=a; if(rank[a]==rank[b])rank[a]++;
    }
};
template<class T> static void binary(const std::string& path,const std::vector<T>& data) {
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
    if(!f)throw std::runtime_error("write failed: "+path);
}
class TextWriter {
    std::ofstream out;
    std::vector<char> buffer;
    size_t used=0;
public:
    explicit TextWriter(const std::string& path):out(path,std::ios::binary),buffer(16*1024*1024) {
        if(!out)throw std::runtime_error("cannot open "+path);
    }
    void number(uint32_t x,char suffix) {
        if(buffer.size()-used<32)flush();
        auto result=std::to_chars(buffer.data()+used,buffer.data()+buffer.size(),x);
        used=result.ptr-buffer.data();buffer[used++]=suffix;
    }
    void flush() { out.write(buffer.data(),used);used=0;if(!out)throw std::runtime_error("GXT write failed"); }
    void close() { flush();out.close();if(!out)throw std::runtime_error("GXT close failed"); }
};
int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("usage: convert_csr N CANONICAL_EDGES.bin OUTPUT_DIR");
        const uint32_t n=std::stoul(argv[1]);
        if(!n || n>=INT32_MAX)throw std::runtime_error("unsupported graph size");
        const uint32_t endian=1;
        if(*reinterpret_cast<const uint8_t*>(&endian)!=1)throw std::runtime_error("little-endian host required");
        const std::string output=argv[3];std::filesystem::create_directories(output);
        auto start=Clock::now(),phase=start;
        auto bytes=std::filesystem::file_size(argv[2]);
        if(bytes%8)throw std::runtime_error("truncated input");
        std::vector<uint64_t> edges(bytes/8);
        std::ifstream input(argv[2],std::ios::binary);
        input.read(reinterpret_cast<char*>(edges.data()),bytes);
        if(!input)throw std::runtime_error("read failed");
        const uint64_t records=edges.size();uint64_t loops=0;
        for(auto edge:edges) {
            uint32_t u=edge>>32,v=edge;
            if(u>=n || v>=n || u>v)throw std::runtime_error("invalid canonical input");
            loops+=u==v;
        }
        edges.erase(std::remove_if(edges.begin(),edges.end(),[](uint64_t e){return uint32_t(e>>32)==uint32_t(e);}),edges.end());
        double read_s=seconds(phase);phase=Clock::now();progress("sort_and_deduplicate");
        std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
        const uint64_t unique_edges=edges.size();double sort_s=seconds(phase);phase=Clock::now();progress("undirected_components");
        DSU components(n);
        for(auto edge:edges)components.unite(edge>>32,uint32_t(edge));
        std::vector<uint32_t> sizes(n,0);
        for(uint32_t v=0;v<n;v++)sizes[components.find(v)]++;
        const uint32_t largest_root=std::max_element(sizes.begin(),sizes.end())-sizes.begin();
        const uint32_t retained_n=sizes[largest_root];
        const uint32_t component_count=std::count_if(sizes.begin(),sizes.end(),[](uint32_t x){return x>0;});
        std::vector<int32_t> mapping(n,-1);std::vector<uint32_t> ids;ids.reserve(retained_n);
        for(uint32_t v=0;v<n;v++)if(components.find(v)==largest_root){mapping[v]=ids.size();ids.push_back(v);}
        std::vector<uint64_t> offsets(retained_n+1,0);uint64_t retained_m=0;
        for(auto edge:edges) {
            int32_t u=mapping[edge>>32],v=mapping[uint32_t(edge)];
            if(u<0) { if(v>=0)throw std::runtime_error("component inconsistency");continue; }
            if(v<0)throw std::runtime_error("component inconsistency");
            offsets[u+1]++;offsets[v+1]++;retained_m++;
        }
        std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
        std::vector<uint32_t> neighbors(offsets.back());std::vector<uint64_t> cursor(offsets);
        for(auto edge:edges) {
            int32_t u=mapping[edge>>32],v=mapping[uint32_t(edge)];if(u<0)continue;
            neighbors[cursor[u]++]=v;neighbors[cursor[v]++]=u;
        }
        edges.clear();edges.shrink_to_fit();
        uint64_t max_degree=0;
        for(uint32_t v=0;v<retained_n;v++) {
            auto begin=neighbors.begin()+offsets[v],end=neighbors.begin()+offsets[v+1];
            std::sort(begin,end);
            if(std::adjacent_find(begin,end)!=end || std::binary_search(begin,end,v))throw std::runtime_error("nonsimple CSR row");
            max_degree=std::max(max_degree,offsets[v+1]-offsets[v]);
        }
        double convert_s=seconds(phase);phase=Clock::now();progress("write_csr_and_gxt");
        binary(output+"/indptr.u64",offsets);binary(output+"/indices.u32",neighbors);binary(output+"/original_ids.u32",ids);
        TextWriter text(output+"/graph.gxt");text.number(retained_n,'\n');
        for(uint32_t v=0;v<retained_n;v++)for(uint64_t j=offsets[v];j<offsets[v+1];j++){text.number(v,' ');text.number(neighbors[j],'\n');}
        text.close();
        std::string result="{\"raw_nodes\":"+std::to_string(n)+",\"input_records\":"+std::to_string(records)
          +",\"input_loops_removed\":"+std::to_string(loops)+",\"undirected_edges_before_lcc\":"+std::to_string(unique_edges)
          +",\"undirected_components\":"+std::to_string(component_count)+",\"n\":"+std::to_string(retained_n)
          +",\"m\":"+std::to_string(retained_m)+",\"csr_entries\":"+std::to_string(offsets.back())
          +",\"r1_closed_neighborhood_entries\":"+std::to_string(offsets.back()+retained_n)
          +",\"maximum_degree\":"+std::to_string(max_degree)+",\"read_seconds\":"+std::to_string(read_s)
          +",\"sort_seconds\":"+std::to_string(sort_s)+",\"conversion_seconds\":"+std::to_string(convert_s)
          +",\"write_seconds\":"+std::to_string(seconds(phase))+",\"total_seconds\":"+std::to_string(seconds(start))+"}";
        std::ofstream receipt(output+"/metadata.json");receipt<<result<<'\n';if(!receipt)throw std::runtime_error("receipt write failed");
        std::cout<<result<<'\n';
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 2; }
}
