// Query-layer pilot only. Frozen selectors, pieces and boundary indexes are inputs.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
using I=int32_t;using L=int64_t;using U=uint64_t;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
template<class T>std::vector<T> load(std::string p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(p);auto n=f.tellg();if(n%sizeof(T))throw std::runtime_error("size "+p);std::vector<T>v(n/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);if(!f)throw std::runtime_error("read "+p);return v;}
template<class T>T read(std::ifstream&f){T v;f.read((char*)&v,sizeof(v));if(!f)throw std::runtime_error("expected EOF");return v;}
struct Answer{std::vector<I>v;std::vector<U>e;bool operator==(const Answer&o)const{return v==o.v&&e==o.e;}};
struct Query{I id,h;std::vector<I>seeds;Answer expected;};
U hash_answer(const Answer&a){U h=1469598103934665603ULL;for(auto v:a.v){h^=U(v);h*=1099511628211ULL;}h^=a.v.size();for(auto e:a.e){h^=e;h*=1099511628211ULL;}return h;}
L physical_read_bytes(){std::ifstream f("/proc/self/io");std::string s;L n;while(f>>s>>n)if(s=="read_bytes:")return n;return -1;}
struct Reader{
 std::vector<L>ptr,pos;std::vector<I>resident;std::vector<const I*>row;
 std::vector<uint32_t>cached;uint32_t gen=0;std::vector<std::vector<I>>chunks;
 bool file;int fd=-1;L bytes=0,requests=0;double io_ms=0;
 Reader(std::string graph,std::string layout,bool disk):ptr(load<L>(graph+"/ptr.bin")),pos(load<L>(layout+"/pos.bin")),row(pos.size()),cached(pos.size()),file(disk){
  std::string p=layout+"/adj.bin";if(file){fd=open(p.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error(p);
   // Identical whole-file prewarming, outside online timings.
   std::vector<char>b(1<<20);while(::read(fd,b.data(),b.size())>0){}lseek(fd,0,SEEK_SET);
  }else resident=load<I>(p);
 }
 ~Reader(){if(fd>=0)close(fd);}
 void reset(){chunks.clear();if(++gen==0){std::fill(cached.begin(),cached.end(),0);gen=1;}bytes=requests=0;io_ms=0;}
 L degree(I v)const{return ptr[v+1]-ptr[v];}
 void fetch(const std::vector<I>&vertices){
  std::vector<I>todo;todo.reserve(vertices.size());
  for(I v:vertices)if(cached[v]!=gen){cached[v]=gen;todo.push_back(v);}
  if(!file){for(I v:todo){row[v]=resident.data()+pos[v];bytes+=degree(v)*sizeof(I);}return;}
  std::sort(todo.begin(),todo.end(),[&](I a,I b){return pos[a]!=pos[b]?pos[a]<pos[b]:a<b;});
  for(size_t i=0;i<todo.size();){size_t j=i+1;L start=pos[todo[i]],end=start+degree(todo[i]);
   while(j<todo.size()&&pos[todo[j]]==end){end+=degree(todo[j]);++j;}
   chunks.emplace_back(end-start);auto&buf=chunks.back();L need=(end-start)*sizeof(I),done=0;auto t=Clock::now();
   while(done<need){auto got=pread(fd,(char*)buf.data()+done,need-done,start*sizeof(I)+done);if(got<=0)throw std::runtime_error("pread");done+=got;++requests;}
   io_ms+=ms(t);bytes+=need;for(size_t k=i;k<j;++k)row[todo[k]]=buf.data()+(pos[todo[k]]-start);i=j;
  }
 }
};
struct Engine{
 Reader reader;std::vector<I>owner,members,bi;std::vector<L>off,bp;
 std::vector<uint32_t>seen,allowed,ps;uint32_t gen=0;bool partition;
 L candidates=0;double candidate_ms=0,refine_ms=0,output_ms=0;
 Engine(std::string g,std::string l,bool file,bool part):reader(g,l,file),seen(reader.pos.size()),allowed(reader.pos.size()),partition(part){
  if(part){owner=load<I>(l+"/owner.bin");members=load<I>(l+"/members.bin");off=load<L>(l+"/offsets.bin");bp=load<L>(l+"/boundary_indptr.bin");bi=load<I>(l+"/boundary_indices.bin");ps.resize(off.size()-1);}
 }
 Answer run(const Query&q){
  reader.reset();if(++gen==0){std::fill(seen.begin(),seen.end(),0);std::fill(allowed.begin(),allowed.end(),0);std::fill(ps.begin(),ps.end(),0);gen=1;}
  candidate_ms=0;candidates=0;auto t=Clock::now();
  if(partition){
   std::vector<I>front=q.seeds;std::sort(front.begin(),front.end());front.erase(std::unique(front.begin(),front.end()),front.end());
   if(q.h==2){reader.fetch(front);size_t count=front.size();for(size_t i=0;i<count;++i){I v=front[i];auto d=reader.degree(v);if(d)front.insert(front.end(),reader.row[v],reader.row[v]+d);}std::sort(front.begin(),front.end());front.erase(std::unique(front.begin(),front.end()),front.end());}
   std::vector<I>pieces;auto add=[&](I p){if(ps[p]!=gen){ps[p]=gen;pieces.push_back(p);}};
   for(I v:front){add(owner[v]);for(L j=bp[v];j<bp[v+1];++j)add(bi[j]);}
   std::vector<I>cand;for(I p:pieces)for(L j=off[p];j<off[p+1];++j){I v=members[j];allowed[v]=gen;cand.push_back(v);}
   candidates=cand.size();reader.fetch(cand);candidate_ms=ms(t);
  }
  t=Clock::now();std::vector<I>queue;for(I v:q.seeds)if(seen[v]!=gen){if(partition&&allowed[v]!=gen)throw std::runtime_error("seed not covered");seen[v]=gen;queue.push_back(v);}
  size_t begin=0,end=queue.size();
  for(I depth=0;depth<q.h;++depth){std::vector<I>front(queue.begin()+begin,queue.begin()+end);reader.fetch(front);
   for(size_t i=begin;i<end;++i){I v=queue[i];for(L j=0;j<reader.degree(v);++j){I u=reader.row[v][j];if(seen[u]!=gen&&(!partition||allowed[u]==gen)){seen[u]=gen;queue.push_back(u);}}}begin=end;end=queue.size();
  }
  refine_ms=ms(t);t=Clock::now();Answer a;a.v=std::move(queue);std::sort(a.v.begin(),a.v.end());reader.fetch(a.v);
  for(I v:a.v)for(L j=0;j<reader.degree(v);++j){I u=reader.row[v][j];if(u>v&&seen[u]==gen)a.e.push_back((U(uint32_t(v))<<32)|uint32_t(u));}
  std::sort(a.e.begin(),a.e.end());output_ms=ms(t);if(!partition)candidates=a.v.size();return a;
 }
};
int main(int argc,char**argv){try{
 if(argc!=8)throw std::runtime_error("graph layout memory|file partition|bfs rep output seed");
 std::string graph=argv[1],layout=argv[2];bool file=std::string(argv[3])=="file",part=std::string(argv[4])=="partition";int rep=std::stoi(argv[5]);
 std::ifstream f(graph+"/queries.bin",std::ios::binary);I count=read<I>(f);std::vector<Query>queries;
 for(int i=0;i<count;++i){Query q;q.id=read<I>(f);q.h=read<I>(f);I k=read<I>(f);for(int j=0;j<k;++j)q.seeds.push_back(read<I>(f));L nv=read<L>(f),ne=read<L>(f);q.expected.v.resize(nv);q.expected.e.resize(ne);f.read((char*)q.expected.v.data(),nv*sizeof(I));f.read((char*)q.expected.e.data(),ne*sizeof(U));if(!f)throw std::runtime_error("answers read");queries.push_back(std::move(q));}
 std::mt19937 rng(std::stoul(argv[7]));std::shuffle(queries.begin(),queries.end(),rng);Engine engine(graph,layout,file,part);std::ofstream out(argv[6]);out<<std::setprecision(12);
 for(auto&q:queries){auto warm_start=Clock::now();auto warm=engine.run(q);double warm_ms=ms(warm_start);if(!(warm==q.expected))throw std::runtime_error("warm correctness q="+std::to_string(q.id));
  int inner=std::max(1,std::min(32,int(2/std::max(.001,warm_ms))));L bytes=0,requests=0,candidates=0;double cand=0,ref=0,construct=0,io=0;Answer answer;L before=physical_read_bytes();auto start=Clock::now();
  for(int k=0;k<inner;++k){answer=engine.run(q);bytes+=engine.reader.bytes;requests+=engine.reader.requests;candidates+=engine.candidates;cand+=engine.candidate_ms;ref+=engine.refine_ms;construct+=engine.output_ms;io+=engine.reader.io_ms;}
  double total=ms(start)/inner;L disk=physical_read_bytes()-before;if(!(answer==q.expected))throw std::runtime_error("correctness q="+std::to_string(q.id));
  out<<"{\"query\":"<<q.id<<",\"repeat\":"<<rep<<",\"inner\":"<<inner<<",\"total_ms\":"<<total<<",\"candidate_stage_ms\":"<<cand/inner<<",\"refine_ms\":"<<ref/inner<<",\"output_ms\":"<<construct/inner<<",\"pread_ms\":"<<io/inner<<",\"requested_bytes\":"<<double(bytes)/inner<<",\"read_calls\":"<<double(requests)/inner<<",\"physical_read_bytes_batch\":"<<disk<<",\"candidates\":"<<double(candidates)/inner<<",\"vertices\":"<<answer.v.size()<<",\"edges\":"<<answer.e.size()<<",\"answer_hash64\":\""<<hash_answer(answer)<<"\",\"correct\":true}\n";out.flush();
 }
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
