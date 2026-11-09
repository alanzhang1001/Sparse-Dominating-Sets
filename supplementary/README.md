# Supplementary technical report

[Read the report](supplementary_report.pdf) for implementation details, retrieval controls, exact-subgraph comparisons, uncertainty estimates and final-tie stability.

This directory contains the PDF, LaTeX source, figure and table inputs, generated displays and a hash manifest. Algorithm implementations and experiment entry points are in `src/` and `scripts/` at the repository root.

## Build the report

Install `environment/requirements-display.txt` in the display environment and provide pdfLaTeX. From the repository root:

```sh
python supplementary/render.py
python supplementary/build.py
```

The renderer generates the figures, LaTeX tables and CSV counterparts. The builder compiles the report and checks references and layout.

See the [reproduction guide](../docs/reproduction.md) to prepare experiment inputs and run the algorithms.
