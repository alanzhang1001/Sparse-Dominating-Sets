// Release wrapper around the unchanged reference, including its final-tie RNG.
#include <iostream>
#include <stdexcept>
#include "algorithms/c_dominating.cpp"
#include "readwrite/c_edgelist.hpp"
int main(int argc,char**argv){try{
 if(argc!=6)throw std::runtime_error("GRAPH R STRATEGY SEED OUTPUT");
 using namespace sparsedomsets;
 auto g=readwrite::read_gxt(argv[1]);int r=std::stoi(argv[2]);
 auto d=algorithms::find_dominating_set(g,std::stoul(argv[3]),std::stoul(argv[4]),r,1);
 readwrite::write_vertices(d,argv[5],false);
 auto nb=graph::compute_all_closed_neighborhood(g,r,1);long long cost=0;
 for(int v:d)cost+=nb[v].size();
 std::cout<<"{\"total_cost\":"<<cost<<",\"selected\":"<<d.size()<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
