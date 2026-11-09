"""Recompute paired medians and 2,000-resample stratified confidence intervals."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from reproduction.query_summary import aggregate
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--skip-bootstrap', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
print(aggregate(a.runs, a.output, not a.skip_bootstrap))
