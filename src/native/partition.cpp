// Argument-only wrapper for the unchanged Prt-Weight reference.
#include <cstdlib>
#include "algorithms/c_partition.hpp"
#include "readwrite/c_edgelist.hpp"
int main(int argc,char**argv){
 if(argc!=6)return 2;
 using namespace sparsedomsets;
 auto g=readwrite::read_gxt(argv[1]);auto d=readwrite::read_vertices(argv[2]);
 auto parts=algorithms::neighborhood_partition(g,d,std::atoi(argv[3]),std::atoi(argv[4]));
 readwrite::write_parts(parts,argv[5]);return 0;
}
