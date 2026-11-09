"""Portable candidate and exact-subgraph reproduction pipeline."""
from pathlib import Path
import os
import sys
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from reproduction.query_pipeline import main
if __name__ == '__main__':
    main()
