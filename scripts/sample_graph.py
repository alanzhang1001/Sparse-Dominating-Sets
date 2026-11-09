"""Create a BFS-induced sample from NPZ or a raw CSR directory."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from congestionlab.graphs import sample_bfs_graph
from reproduction.csr_sampling import sample_csr
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--graph', type=Path, required=True)
p.add_argument('--nodes', type=int, required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
if a.graph.is_dir():
    print(sample_csr(a.graph, a.nodes, a.seed, a.output))
else:
    print(sample_bfs_graph(a.graph, a.nodes, a.seed, a.output)['graph_file'])
