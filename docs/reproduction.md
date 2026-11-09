# Reproduction guide

Run commands from the repository root in the experiment environment described in the README. Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`; experiment entry points set these automatically. Use Linux/GCC/OpenMP for performance comparisons. Use `--serial-reference` to build the reference implementations on macOS.

Every command writes to a new output directory. Downloaded data and generated results belong under ignored `data/inputs/`, `data/generated/` or `results/local/`.

## 1. Prepare graphs

SNAP sources, original checksums and deterministic preprocessing are recorded in `data/datasets.json` and `data/dataset_sources.json`.

```sh
python scripts/prepare_data.py web-Stanford --output data/inputs/stanford
python scripts/prepare_data.py web-Google --output data/inputs/google
python scripts/prepare_data.py soc-Epinions1 --output data/inputs/epinions
python scripts/prepare_data.py com-Amazon --output data/inputs/amazon
python scripts/prepare_data.py com-Youtube --output data/inputs/youtube
```

`--raw PATH` uses an existing compressed source instead of downloading it. `--no-proxy` bypasses proxy environment variables. Preparation removes loops, merges reciprocal/duplicate edges and retains the largest undirected connected component with original-ID mapping. Uncompressed NPZ array hashes are checked because ZIP compression bytes can vary across zlib versions.

For the LAW WebGraph inputs, install Java/Javac 17+ and a C++17 compiler:

```sh
python scripts/prepare_webgraph.py in-2004 --output data/inputs/in-2004
python scripts/prepare_webgraph.py eu-2005 --output data/inputs/eu-2005
python scripts/prepare_webgraph.py uk-2002 --output data/inputs/uk-2002
```

This downloads the official `.graph`, `.properties` and `.md5sums` files, checks source checksums, downloads version-pinned WebGraph jars with checksums, decodes arcs, and constructs the same simple undirected LCC. `--raw-dir` and `--jars` support existing offline inputs. `in-2004`/`eu-2005` also produce NPZ files. The full `uk-2002` graph uses binary CSR directly. Its conversion needs substantial RAM and tens of GB of temporary/output disk space. The generated small-graph demo uses Python and C++.

Use `--raw-dir` for downloaded source files, `--jars` for existing Java dependencies, and `--download-base` to select a source endpoint.

Public graph sources: [SNAP](https://snap.stanford.edu/data/), [LAW graph collection](https://law.di.unimi.it/datasets.php).

## 2. Synthetic, sampled and exact-solver instances

```sh
python scripts/reproduce.py --workspace data/inputs generate \
  --suite main --output data/generated/main
python scripts/reproduce.py --workspace data/inputs generate \
  --suite theory --output data/generated/theory
python scripts/reproduce.py --workspace data/generated/main exact \
  --suite main --output results/local/exact-main
python scripts/reproduce.py --workspace data/generated/theory exact \
  --suite theory --output results/local/exact-theory
python scripts/reproduce.py certificates --output results/local/certificates
```

`configs/instances.json` contains 160 main radius conditions and 140 theory conditions. Generation preserves the original seeded graph constructors and BFS discovery order. The main suite needs the downloaded parent graphs; the theory suite is synthetic. `--limit N` is available for a smaller run. LP/IP use SciPy/HiGHS. Set `--lp-seconds` and `--ip-seconds` to choose the time limits; results include the solver status, objective, bounds and optimality status.

An arbitrary BFS sample can also be generated directly:

```sh
python scripts/sample_graph.py \
  --graph data/inputs/stanford/processed/web-Stanford.lcc.npz \
  --nodes 10000 --seed 20261010 --output data/generated/stanford-sample
```

`--graph` can instead name the processed `uk-2002` CSR directory. Sampling reads its arrays through memory maps and avoids allocating a second full adjacency matrix.

## 3. Selection quality, implementation and tie stability

```sh
python scripts/reproduce.py --workspace data/inputs quality \
  --graphs soc-Epinions1 --output results/local/quality
python scripts/reproduce.py --workspace data/inputs implementation \
  --graphs soc-Epinions1 --output results/local/implementation
python scripts/reproduce.py --workspace data/inputs stability \
  --graphs soc-Epinions1 --output results/local/stability
python scripts/reproduce.py --workspace data/inputs radius-cohort \
  --cohort paper --output results/local/paper-radius
python scripts/reproduce.py --workspace data/inputs radius-cohort \
  --cohort report --output results/local/report-radius
```

Comma-separated `--graphs` and `--methods` select conditions. All four greedy strategies are available. Quality runs retain corrected CC2FS at the requested budgets; `--budgets 1 10 60 300` is the default. Implementation runs check ordered old/new sequence equality. Stability uses ten seeded final-tie orders. Radius cohorts automatically generate missing BFS samples from their parent graph; the exact conditions are in `data/rerun_cohorts.json`.

The full `uk-2002` quality/implementation entry supports radius one using the CSR adapters. Full Stanford reference runs at radius three previously reached approximately 251 GiB peak RSS; use the recorded samples or a suitably provisioned machine. CC2FS uses a wall-clock search budget; each run records its elapsed time and result.

## 4. Generate the complete retrieval and exact-subgraph inputs

```sh
python scripts/query_pipeline.py prepare \
  --graph data/inputs/stanford/processed/web-Stanford.lcc.npz \
  --graph-name web-Stanford --cc-seconds 60 \
  --output data/generated/stanford-query
```

Repeat with `web-Google`, `in-2004`, and `eu-2005`. Their processed paths follow the preparation scripts' output structure. The named graph selects the original workload seeds, pilot-start exclusions and three published byte caps. The process:

1. Runs Ratio, Ratio+, Degree, corrected CC2FS and five Random-Prune seeds.
2. Uses the original Prt-Weight partitioner and constructs the boundary index.
3. Calibrates three byte budgets using graph information only, then creates a fixed-count control using one common count across all nine selectors. The common augmentation permutation is fixed.
4. Locks index choices before generating queries. Matched-budget experiments require index occupancy in the 95–100% range; the manifest records the occupancy and match status.
5. Generates candidate dev/test workloads (6,800 test queries on each full graph), then 2,800 separate exact-confirmation queries. Confirmation starts exclude all old dev/test and pilot starts.
6. Generates independent full vertex/edge answers, original adjacency layout, and all nine partition layouts. Every layout is charged the same `8*n` mapping bytes.

`index_lock.json`, `candidate_queries.json`, `exact/queries.bin`, `exact/layouts/` and a manifest are produced locally. `configs/query_workloads.json` supplies the seeds and start-exclusion lists for the experiment sampling splits.

Calibration starts from the configured landmark counts and searches by index size. Use `--refine-steps` and `--budgets` to configure this search. The index manifest records the attempted counts, selected indexes and budget occupancy before query evaluation.

For a custom graph, omit `--graph-name`; caps are derived from native index sizes. `--quick` reduces query counts. Without `--graph`, preparation generates the 256-vertex example and always labels it a quick demonstration. For a pure method demonstration, select `--controls native,fixed-count --exact-control fixed-count`, as in the README.

## 5. Run queries and summarize

```sh
python scripts/query_pipeline.py candidates \
  --inputs data/generated/stanford-query --output results/local/stanford-candidates
python scripts/query_pipeline.py subgraphs \
  --inputs data/generated/stanford-query --output results/local/stanford-subgraphs
python scripts/summarize_queries.py \
  --runs results/local/stanford-subgraphs --output results/local/stanford-summary
```

Candidate evaluation verifies full recall and the exact union-of-pieces candidate count against an independent BFS answer. Exact extraction compares complete vertex/edge outputs, runs memory and file modes, retains original-layout and same-layout BFS, and randomizes job order. Three repetitions are the default. `strata.json` keeps all ten query strata. Summary code collapses repetitions by per-query median and computes paired stratified 2,000-resample intervals, including resampling the five random-selector seeds. Primary one-hop/single-seed results and context strata remain separate.

File-mode experiments prewarm the adjacency files. Requested bytes and read calls measure logical adjacency access. Per-run outputs are written to the selected output directory.
