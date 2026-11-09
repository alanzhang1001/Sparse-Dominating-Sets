"""Generate inputs and run selection, solver, radius and stability experiments."""
from pathlib import Path
import argparse
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
from reproduction.experiments import register

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT / 'data/generated')
    sub = parser.add_subparsers(dest='command', required=True)
    register(sub)
    args = parser.parse_args()
    args.function(args)

if __name__ == '__main__':
    main()
