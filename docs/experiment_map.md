# Experiment map

| Paper/report content | Input generation | Run entry | Public result material |
| --- | --- | --- | --- |
| Small-instance quality vs LP/optimum | `reproduce.py generate --suite main` | `reproduce.py exact --suite main` | Per-instance display points; solver status/bounds |
| Theory families and construction certificates | `generate --suite theory`; built-in client-hub constructor | `exact --suite theory`; `certificates` | Theory display points and certificates |
| Real-graph quality and CC2FS budgets | `prepare_data.py`; `prepare_webgraph.py` | `reproduce.py quality` | Aggregate quality/time summaries |
| Old/new implementation and large CSR graph | Same public graph preparation | `reproduce.py implementation` | Phase, time and RSS summaries |
| Main/report radius conditions | Automatic seeded BFS sampling of prepared parents | `reproduce.py radius-cohort --cohort paper/report` | All twelve conditions in each cohort |
| Candidate retrieval and matched-count controls | `query_pipeline.py prepare` | `query_pipeline.py candidates` | All 144 published configuration summaries |
| Exact subgraph extraction and both BFS controls | Same preparation, independent query answers and layouts | `query_pipeline.py subgraphs`; `summarize_queries.py` | All methods/modes/strata and uncertainty |
| Final-tie stability | Prepared public graph; seeds 0–9 | `reproduce.py stability` | Four strategies, five graphs, ten-seed summaries |
| Paper/report figures and tables | Packaged small display inputs | `regenerate_displays.py`; `supplementary/build.py` | PDF, figures, CSV and LaTeX tables |

See [reproduction.md](reproduction.md) for complete commands, input preparation and output formats.
